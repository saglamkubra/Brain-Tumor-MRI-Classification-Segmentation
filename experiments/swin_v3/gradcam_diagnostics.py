"""Read-only diagnostics for the deployed CNN and Swin attribution layers."""
import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path
import random
import sys
from zipfile import ZipFile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import EigenGradCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
import timm
import torch

from backend.models import CLASSES, Models
from backend.segmentation import foreground_region, swin_reshape

SEED = 20261119
FRACTIONS = np.linspace(0, 1, 11)
METHODS = {"GradCAM": GradCAM, "GradCAM++": GradCAMPlusPlus, "EigenGradCAM": EigenGradCAM}
ORDERS = ("cam", "random", "reverse")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample(split, archive):
    assert split.parent.name == "veri_bolunmesi_v3" and split.name == "test.csv"
    with split.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["dosya"] for row in rows}) == 1579
    assert set(row["sinif"] for row in rows) == set(CLASSES)
    rng = random.Random(SEED)
    seen = set()
    selected = []
    for label in CLASSES:
        pool = sorted(row["dosya"] for row in rows if row["sinif"] == label)
        rng.shuffle(pool)
        count = 0
        for key in pool:
            with Image.open(BytesIO(archive.read(key))) as raw:
                image = raw.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            selected.append((dict(sample=len(selected)+1, file=key, truth=label, pixel_sha256=digest), image))
            count += 1
            if count == 10:
                break
        assert count == 10
    return selected


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def attribution(model, tensor, target, layer, reshape, method):
    with method(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        result = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target)], aug_smooth=True)[0]
    assert result.shape == (224, 224) and np.isfinite(result).all()
    return result


def compare(a, b):
    rho = float(spearmanr(a.ravel(), b.ravel()).statistic) if np.ptp(a) and np.ptp(b) else np.nan
    # Gaussian-window SSIM (11 pixels, sigma 1.5), valid interior, range [0, 1].
    a, b = a.astype(np.float64), b.astype(np.float64)
    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)
    x, y = blur(a), blur(b)
    ssim = ((2*x*y + .01**2) * (2*(blur(a*b)-x*y) + .03**2) /
            ((x*x+y*y+.01**2) * (blur(a*a)-x*x+blur(b*b)-y*y+.03**2)))
    return rho, float(ssim[5:-5, 5:-5].mean())


@torch.inference_mode()
def scores(model, tensor, target):
    return model(tensor).softmax(1)[:, target].cpu().numpy()


def intervention_scores(model, tensor, target, masks, insert=False):
    baseline = tensor.mean(dim=(2, 3), keepdim=True)
    values = []
    for start in range(0, len(masks), 4):
        mask = torch.as_tensor(np.stack(masks[start:start+4]), device=tensor.device)[:, None]
        a, b = (tensor, baseline) if insert else (baseline, tensor)
        values.extend(scores(model, torch.where(mask, a, b), target))
    return np.asarray(values, dtype=float)


def curves(model, tensor, target, cam, random_order, endpoints, random_curves):
    # Stable ordering within activation ties uses the same shuffled indices for both CAM controls.
    values = cam.ravel()[random_order]
    permutations = (random_order[np.argsort(-values, kind="stable")], random_order,
                    random_order[np.argsort(values, kind="stable")])
    result = {}
    for label, order in zip(ORDERS, permutations):
        if label == "random" and random_curves:
            result.update(random_curves)
            continue
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        masks = [(ranks < round(len(order) * fraction)).reshape(cam.shape) for fraction in FRACTIONS[1:-1]]
        for kind in ("deletion", "insertion"):
            insert = kind == "insertion"
            start, end = endpoints[::-1] if insert else endpoints
            path = np.r_[start, intervention_scores(model, tensor, target, masks, insert), end]
            result[f"{kind}_{label}"] = path
            if label == "random":
                random_curves[f"{kind}_{label}"] = path
    return result


def layers_and_groups(name, model):
    if name == "CNN":
        return {"features.3.block.3": (model.features[3].block[3], None)}, [
            ("head", ("classifier.9",)), ("classifier", ("classifier",)),
            *((f"features.{i}", (f"features.{i}",)) for i in reversed(range(4)))]
    return {"Aşama 3": (model.layers[2], swin_reshape),
            "Aşama 4": (model.layers[3].blocks[-1].norm1, swin_reshape)}, [
        ("head", ("head",)), ("norm+stage4", ("norm", "layers.3")),
        *((f"stage{i+1}", (f"layers.{i}",)) for i in reversed(range(3))),
        ("patch_embed", ("patch_embed",))]


def diagnostic_model(models, name, items, metrics, curve_rows, panels):
    model = models.models[name]
    layers, groups = layers_and_groups(name, model)
    torch.manual_seed(SEED)
    fresh = (type(model)() if name == "CNN" else
             timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=4)).state_dict()
    weights = model.state_dict()
    assert fresh.keys() == weights.keys()
    touched = set()
    start_maps = {}
    records = {}
    inputs = {}
    with models.on_device(model, models.device):
        for info, source in items:
            sid = info["sample"]
            tensor = models.tensor(name, source).to(models.device)
            with torch.inference_mode():
                probabilities = model(tensor).softmax(1)[0]
                target = int(probabilities.argmax())
                score = float(probabilities[target])
            inputs[sid] = (tensor.cpu(), target)
            rgb = np.asarray(source.resize((224, 224), Image.Resampling.BILINEAR))
            head = foreground_region(rgb)
            bottom = np.indices(head.shape)[0] >= 168
            outside = ~head
            union = outside | bottom
            ablated = intervention_scores(model, tensor, target, [outside, bottom, union, ~union])
            random_order = np.random.default_rng(SEED + sid).permutation(head.size)
            mean_input = tensor.mean(dim=(2, 3), keepdim=True).expand_as(tensor).contiguous()
            endpoints = (score, float(scores(model, mean_input, target)[0]))
            random_curves = {}
            keys = []
            for stage, (layer, reshape) in layers.items():
                for method_name, method in METHODS.items():
                    cam = attribution(model, tensor, target, layer, reshape, method)
                    key = sid, stage, method_name
                    keys.append(key)
                    start_maps[key] = cam
                    base = dict(sample=sid, truth=info["truth"], model=name, stage=stage, method=method_name,
                                prediction=CLASSES[target], correct=CLASSES[target] == info["truth"])
                    record = dict(base, score=score, flat=not bool(np.ptp(cam)), foreground_fraction=head.mean())
                    for label, value in zip(("outside", "bottom", "outside_or_bottom", "keep_only_outside_or_bottom"), ablated):
                        record[f"score_{label}"] = value
                        record[f"drop_{label}"] = score-value
                    for label, region in (("outside", outside), ("bottom", bottom)):
                        record[f"energy_{label}"] = cam[region].sum()/cam.sum() if cam.sum() else np.nan
                    paths = curves(model, tensor, target, cam, random_order, endpoints, random_curves)
                    for label, points in paths.items():
                        record[f"auc_{label}"] = np.dot(np.diff(FRACTIONS), (points[1:] + points[:-1]) / 2)
                        curve_rows.extend(dict(base, curve=label, fraction=f, score=p) for f, p in zip(FRACTIONS, points))
                    records[key] = record
                    metrics.append(record)
                    if method_name == "GradCAM":
                        panels[(name, sid, stage)] = (rgb, cam, paths, record)
            if len(layers) == 2:
                for method in METHODS:
                    a, b = (start_maps[sid, stage, method] for stage in layers)
                    rho, _ = compare(a, b)
                    fa = head & (a >= .70 * a.max()) & (a > 0)
                    fb = head & (b >= .70 * b.max()) & (b > 0)
                    union_size = (fa | fb).sum()
                    iou = (fa & fb).sum()/union_size if union_size else np.nan
                    for stage in layers:
                        records[sid, stage, method].update(stage_spearman=rho, stage_focus_iou=iou)
        print(name, "baseline and interventions", len(items), flush=True)
        previous = set()
        for label, prefixes in groups:
            selected = {key for key in fresh if any(key.startswith(prefix + ".") for prefix in prefixes)}
            with torch.no_grad():
                current = model.state_dict()
                for key in selected - touched:
                    current[key].copy_(fresh[key])
            touched |= selected
            assert touched - previous
            previous = touched.copy()
            for (sid, stage, method_name), trained in start_maps.items():
                tensor, target = inputs[sid]
                layer, reshape = layers[stage]
                cam = attribution(model, tensor.to(models.device), target, layer, reshape, METHODS[method_name])
                rho, ssim = compare(trained, cam)
                records[sid, stage, method_name].update({
                    f"rho_{label}": rho, f"ssim_{label}": ssim, f"flat_{label}": not bool(np.ptp(cam))})
            print(name, "randomization", label, flush=True)
        assert {key for key, _ in model.named_parameters()} <= touched, "Learned weights remain"
    # These model objects were constructed exclusively for this offline process.
    del weights


def make_panels(out, panels):
    wrong = sorted({sid for (_, sid, _), (_, _, _, row) in panels.items() if not row["correct"]})
    correct = [sid for sid in sorted({sid for _, sid, _ in panels}) if sid not in wrong]
    chosen = wrong[:4] + correct[:4]
    chosen += [sid for sid in wrong[4:] + correct[4:] if sid not in chosen][:8-len(chosen)]
    for position, sid in enumerate(chosen, 1):
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
        configs = ((name, stage) for name, sample_id, stage in panels if sample_id == sid)
        for (name, stage), (original, overlay, graph) in zip(configs, axes):
            rgb, cam, paths, row = panels[name, sid, stage]
            original.imshow(rgb)
            original.set_title(f"{name} | {stage}", fontsize=8)
            overlay.imshow(rgb)
            overlay.imshow(cam, cmap="magma", alpha=.65*cam, vmin=0, vmax=1)
            overlay.set_title(f"{row['prediction']} ({'doğru' if row['correct'] else 'yanlış'})", fontsize=9)
            original.axis("off")
            overlay.axis("off")
            for order in ORDERS:
                graph.plot(FRACTIONS * 100, paths[f"deletion_{order}"], label=order)
            graph.set(xlim=(0, 100), ylim=(0, 1), xlabel="Silinen piksel (%)", ylabel="Softmax")
            graph.legend(fontsize=7)
        fig.suptitle(f"Örnek {sid} | gerçek sınıf: {row['truth']} | Grad-CAM")
        fig.savefig(out / f"panel_{position:02d}.png", dpi=115)
        plt.close(fig)
    return chosen


def inventory():
    roots = (Path("backend"), Path("frontend"), Path("models"))
    return {path.as_posix(): sha(path) for root in roots for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def aggregate(rows, field, spread=False):
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "—"
    if spread:
        return f"{values.mean():.3f} ± {values.std(ddof=1):.3f} (n={len(values)})" if len(values)>1 else f"{values[0]:.3f} (n=1)"
    return f"{values.mean():.3f}"


def write_report(out, args, models, metrics, chosen):
    command = " ".join(f'"{arg}"' if " " in arg else arg for arg in sys.argv[1:])
    lines = [
        "# Grad-CAM tanı çalışması", "",
        f"Komut `.venv/Scripts/python.exe -B -m experiments.swin_v3.gradcam_diagnostics {command}`.",
        f"Cihaz: {models.device}. PyTorch: {torch.__version__}. GPU: {torch.cuda.get_device_name(0) if models.device == 'cuda' else '—'}.",
        f"Test CSV SHA-256: `{sha(args.split)}`. Veri kaynağı: {args.archive.name}.",
        "CNN V2 kaynağı: dağıtım paketi. Swin: swin_tiny_v3_best.pth. İki modelin kendi eğitimden gelen tahmini sabit hedef.",
        "",
        "## Örneklem ve yöntem", "",
        "V3 test listesinin 1.579 benzersiz dosya yolundan sınıf başına 10 görüntü seçildi. Sınıf içinde dosya yolları sıralandı, 20261119 ile karıştırıldı, boyut ve RGB piksel hash’i aynı olan görüntüler atlandı. Tahmine göre eleme yapılmadı.",
        "Üretimdeki model yükleyici ve ön işleme doğrudan kullanıldı. Hedefler CNN features.3.block.3, Swin layers[2] ve Swin layers[3].blocks[-1].norm1. Üç yöntemde de üretimdeki gibi aug_smooth=True. Ölçümler dolgusuz 224×224 model girişinde yapıldı.",
        "a. Sınıflandırıcıdan başlayarak grupların ağırlıkları aynı mimarinin sabit tohumla yeni başlatılmış ağırlıklarıyla kümülatif değiştirildi. CNN batch norm durumları da yeni başlangıçtandır. Tüm öğrenilebilir parametrelerin değişim kapsamına girdiği denetlendi. Spearman sabit haritalarda tanımsızdır ve ortalamaya girmez. SSIM [0,1] aralıklı giriş haritalarında 11×11 Gauss penceresi (σ=1,5) ile hesaplandı.",
        "b/e. Deletion ve insertion, %0–100 arasında 10 puanlık adımlarla, görüntünün kanal başına uzamsal ortalaması kullanılarak hesaplandı. En yüksek, rastgele ve en düşük aktivasyon sıraları karşılaştırıldı. Aynı görüntünün aynı rastgele sırası tüm yöntemlerde kullanıldı. Eşit sıralar aynı rastgele sıra ile çözülür. Eğri uçları gerçek özgün ve sabit ortalama görüntünün tekil ileri geçişleriyle ölçüldü ve sıralamalar arasında yeniden kullanıldı. AUC trapez integrali, küçük deletion ve büyük insertion lehine.",
        "c. Kafa ön planı üretimdeki foreground_region işleviyle hesaplandı. 5×5 Gauss, max(5, %95 yüzdelik × 0,09) eşiği, 7×7 kapama, en büyük dış konturun dolgusu ve 5×5 erozyon kullanıldı. Alt bant son 56 satırdır. Toplam aktivasyon sıfırsa enerji oranı tanımsızdır.",
        "d. Kafa dışı, alt bant ve bunların birleşimi ayrı ayrı ortalamayla değiştirildi. Böylece birleşim testinde sadece kafa maskesinin üst %75 içindeki kısmı korundu. Ters kontrolde sadece birleşim korundu. Bu bölgeler anatomik beyin maskesi değildir, alt bant beyin dokusu içerebilir. Farklar orijinal hedef skoru eksi müdahale sonrası skordur.",
        "f. Swin aşamaları için tüm piksellerde Spearman ve kafa ön planında her haritanın kendi tepesinin %70’ine ulaşan pozitif odakların kesişim/birleşim oranı kullanıldı.",
        "", "## Çıktı özeti ve doğrulama", "",
        f"Yukarıdaki komut a–f ölçümlerini tek çalıştırmada üretti. {len(metrics)} görüntü/model/aşama/yöntem kaydı, 40 görüntü ve {len(chosen)} panel. Eğri başına 11 skor. Ölçüm ayrıntıları CSV dosyalarında.",
        "Profil veya checkpoint değişiklikleri ve küçük deney örneklemi dışında daha geniş veri setine genelleme doğrulanmadı. Hasta düzeyinde ayrım doğrulanmadı. Referans segmentasyon maskesi bulunmuyor.",
        "", "## a. Kümülatif rastgeleleştirme", "",
        "| Model | Katman | Yöntem | Son grup | Spearman | SSIM | Sabit rastgele harita |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    groups = {}
    for row in metrics:
        key = row["model"], row["stage"], row["method"]
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        for field in (k for k in rows[0] if k.startswith("rho_")):
            label = field[4:]
            flat = sum(row[f"flat_{label}"] for row in rows)
            lines.append("| " + " | ".join([*key, label, aggregate(rows, field, True),
                          aggregate(rows, f"ssim_{label}", True), str(flat)]) + " |")
    lines += ["", "## b/e. Sadakat AUC ortalamaları", "",
              "| Model | Katman | Yöntem | Silme CAM | Silme rastgele | Silme ters | Ekleme CAM | Ekleme rastgele | Ekleme ters |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, rows in groups.items():
        columns = [aggregate(rows, f"auc_{kind}_{order}") for kind in ("deletion", "insertion") for order in ORDERS]
        lines.append("| " + " | ".join([*key, *columns]) + " |")
    lines += ["", "## c. Enerji oranları ve f. aşamalar arası örtüşme", "",
              "| Model | Katman | Yöntem | Sınıf | Kafa dışı | Alt bant | Aşama Spearman | Ortak odak IoU |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for key, rows in groups.items():
        for label in (*CLASSES, "Tümü"):
            subset = [row for row in rows if row["truth"] == label or label == "Tümü"]
            values = [aggregate(subset, field, True) if field in subset[0] else "—"
                      for field in ("energy_outside", "energy_bottom", "stage_spearman", "stage_focus_iou")]
            lines.append("| " + " | ".join([*key, label, *values]) + " |")
    lines += ["", "## d. Bölge müdahalesi", "",
              "Bu ölçüm CAM yönteminden ve aşamadan bağımsızdır. Aynı görüntü/model için bir kez ölçüldü.", "",
              "| Model | Sınıf | Kafa dışı Δ | Alt bant Δ | İkisi birlikte Δ | Sadece ikisi korununca Δ |",
              "|---|---|---:|---:|---:|---:|"]
    for name in models.models:
        rows = list({row["sample"]: row for row in metrics if row["model"] == name}.values())
        for label in (*CLASSES, "Tümü"):
            subset = [row for row in rows if row["truth"] == label or label == "Tümü"]
            values = [aggregate(subset, f"drop_{field}", True) for field in
                      ("outside", "bottom", "outside_or_bottom", "keep_only_outside_or_bottom")]
            lines.append("| " + " | ".join([name, label, *values]) + " |")
        lines += ["", f"{name}: örneklemde {sum(row['correct'] for row in rows)}/40 doğru tahmin.", ""]
    lines += ["", "## Paneller", "",
              "Görsel seçim kuralı ilk dört hatalı görüntü (en az bir model), ardından her iki modelin doğru tahmin ettiği ilk dört görüntüdür. Eksik kontenjan sıra numarasına göre doldurulur. Bu seçim ölçüm örneklemini değiştirmez.", ""]
    for i, sid in enumerate(chosen, 1):
        lines.append(f"![Örnek {sid} Grad-CAM ve deletion eğrileri](panel_{i:02d}.png)")
    lines += ["", "## Yorum ve sınırlamalar", "",
              "Aşağıdaki karşılaştırmalar aynı görüntülerin ortalamalarıdır. AUC farkları puan cinsinden verilmiştir.", ""]
    for key, rows in groups.items():
        last = "features.0" if key[0] == "CNN" else "patch_embed"
        del_gap = 100 * np.mean([r["auc_deletion_random"]-r["auc_deletion_cam"] for r in rows])
        ins_gap = 100 * np.mean([r["auc_insertion_cam"]-r["auc_insertion_random"] for r in rows])
        joint_wins = sum(r["auc_deletion_cam"] < min(r["auc_deletion_random"], r["auc_deletion_reverse"])
                         and r["auc_insertion_cam"] > max(r["auc_insertion_random"], r["auc_insertion_reverse"])
                         for r in rows)
        lines.append(f"- {' / '.join(key)}. Tam rastgeleleştirme Spearman {aggregate(rows, f'rho_{last}', True)}, "
                     f"SSIM {aggregate(rows, f'ssim_{last}', True)}. "
                     f"Rastgele sıraya karşı silme avantajı {del_gap:+.1f}, ekleme avantajı {ins_gap:+.1f} puan. "
                     f"Her iki kontrolden her iki ölçümde de iyi: {joint_wins}/{len(rows)}.")
    lines += ["",
              "Rastgeleleştirmede yüksek SSIM tek başına modelden bağımsız kenar haritasını kanıtlamaz. İki haritanın ortak boş alanları SSIM’i yükseltebilir. Spearman, sabit harita sayısı ve SSIM birlikte okunmalıdır.",
              "Bir sıralamanın rastgele sırayı geçmesi sadece bu örneklem ve bu piksel müdahalesi altında sadakat kanıtıdır. Ortalama dolgusu dağılım dışı görüntüler oluşturabilir. Alt bant veya kafa dışı müdahalesine duyarlılık, tek başına tümör dışı ipucu kullanımını doğrulamaz. Ortak odak yalnızca iki haritanın uzamsal uyumunu ölçer.",
              "Bir örneklem çekimi, bir başlangıç tohumu ve bir rastgele sıralama kullanıldı. Klinik olasılık kalibrasyonu veya tümör yerleşim doğruluğu ölçülmedi.", "",
              "Yöntem kaynakları: [Adebayo ve ark.](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html), [Petsiuk ve ark.](https://arxiv.org/abs/1806.07421).", "",
              "## Dosyalar ve değişmezlik", "",
              "Üretim ve model dosyalarının çalışma öncesi/sonrası SHA-256 değerleri [integrity.csv](integrity.csv) dosyasında eşleştirildi.",
              "Eklenenler: deney betiği, summary.md, metrics.csv, sample.csv, curves.csv, integrity.csv ve panel_01.png–panel_08.png. GitHub’a push yapılmadı."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def validate_measurements(metrics, curve_rows):
    """Check full coverage, endpoint identities and consistency with exported AUCs."""
    keys = ("sample", "model", "stage", "method")
    indexed_metrics = {tuple(r[k] for k in keys): r for r in metrics}
    assert len(indexed_metrics) == len(metrics) == 360
    traces = {}
    for row in curve_rows:
        key = tuple(row[k] for k in keys)
        traces.setdefault((key, row["curve"]), []).append((float(row["fraction"]), float(row["score"])))
    assert len(traces) == 2160
    for key, row in indexed_metrics.items():
        for order in ORDERS:
            deletion, insertion = [np.asarray(traces[key, f"{kind}_{order}"]) for kind in ("deletion", "insertion")]
            for kind, trace in zip(("deletion", "insertion"), (deletion, insertion)):
                assert trace.shape == (11, 2) and np.allclose(trace[:, 0], FRACTIONS)
                points = trace[:, 1]
                assert np.all((points >= 0) & (points <= 1))
                integral = np.dot(np.diff(FRACTIONS), (points[1:] + points[:-1]) / 2)
                assert np.isclose(integral, row[f"auc_{kind}_{order}"])
            assert np.allclose(deletion[[0, -1], 1], insertion[[-1, 0], 1], atol=1e-5)
            assert np.isclose(deletion[0, 1], row["score"], atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before = inventory()
    args.out.mkdir(parents=True, exist_ok=True)
    models = Models()
    assert models.swin_profile == "v3" and not models.errors, models.errors
    assert models.cnn_integrity["source"] == "deployment", models.cnn_integrity
    assert models.device == "cuda", "CUDA unavailable, experiment stopped"
    with ZipFile(args.archive) as archive:
        items = sample(args.split, archive)
    csv_write(args.out / "sample.csv", [row for row, _ in items])
    metrics, curve_rows, panels = [], [], {}
    print("Sample", len(items), "device", models.device, flush=True)
    for name in models.models:
        diagnostic_model(models, name, items, metrics, curve_rows, panels)
        csv_write(args.out / "metrics.csv", metrics)
        csv_write(args.out / "curves.csv", curve_rows)
    assert len(curve_rows) == 23760
    validate_measurements(metrics, curve_rows)
    chosen = make_panels(args.out, panels)
    after = inventory()
    assert before == after, "Production or model files changed"
    csv_write(args.out / "integrity.csv", [dict(file=path, before=digest, after=after[path])
                                            for path, digest in before.items()])
    write_report(args.out, args, models, metrics, chosen)
    print(f"Completed: {len(metrics)} metrics, {len(curve_rows)} curve points, {len(chosen)} panels")


if __name__ == "__main__":
    main()
