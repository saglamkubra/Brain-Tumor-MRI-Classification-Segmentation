"""Compare the historical and current Grad-CAM/SAM on the same local archive."""
import argparse
import base64
from io import BytesIO
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from backend.models import Models, ROOT
from backend import segmentation as current

BASELINE = "46847fad8dcd0d97b4f09fa7178c96ea5fce99ce"
CASES = [
    "Testing/glioma/Te-gl_22.jpg", "Testing/meningioma/Te-aug-me_7.jpg",
    "Testing/glioma/Te-gl_1.jpg", "Testing/meningioma/Te-me_263.jpg",
]


def image_from_uri(uri):
    return Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def time_segment(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    result["elapsed_s"] = round(perf_counter() - start, 3)
    return result


def check_hooks(models, image, expected):
    model = models.models["Swin-Tiny"]
    assert not model.training
    rgb = current.sam_image(image)
    verified = {}
    for stage in (3, 4):
        block = model.layers[stage - 1]
        target = block if stage == 3 else block.blocks[-1].norm1
        observed, gradients = [], []
        before = len(target._forward_hooks)

        def hook(_module, _inputs, output):
            if output.requires_grad:
                observed.append(list(output.shape))
                output.register_hook(lambda gradient: gradients.append(list(gradient.shape)))

        handle = target.register_forward_hook(hook)
        try:
            results = [current.explain(models, image, rgb, stage) for _ in range(2)]
        finally:
            handle.remove()
        shape = [1, 14, 14, 384] if stage == 3 else [1, 7, 7, 768]
        assert observed == gradients == [shape] * 12, (stage, observed, gradients)
        assert len(target._forward_hooks) == before
        assert np.allclose(results[0][1], results[1][1], atol=1e-6)
        for label, cam, _, _, layer in results:
            assert label == expected
            assert cam.shape == (224, 224) and np.isfinite(cam).all()
            assert 0 <= cam.min() <= cam.max() <= 1
            alpha = np.asarray(image_from_uri(layer))[:, :, 3]
            assert np.all(alpha[rgb.max(axis=2) <= 8] == 0)
        verified[str(stage)] = {"activation_shape": shape, "gradient_shape": shape, "hook_cleanup": True, "stable_repeat": True}
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_source = subprocess.check_output(["git", "-C", str(args.repository), "show", f"{BASELINE}:backend/segmentation.py"])
    old = ModuleType("backend.segmentation_baseline")
    exec(compile(old_source, BASELINE, "exec"), old.__dict__)
    args.output.mkdir(parents=True, exist_ok=True)
    models = Models()
    current.load_sam(models)
    assert not models.errors, models.errors
    assert models.versions["Swin-Tiny"] == "swin_tiny_v3_best.pth"
    report = {"baseline": BASELINE, "device": models.device, "cases": []}
    report["checkpoints_sha256"] = {
        name: hashlib.file_digest((ROOT / "models" / weights).open("rb"), "sha256").hexdigest()
        for name, weights in models.versions.items()
    }

    with ZipFile(args.archive) as archive:
        names = archive.namelist()
        paths = CASES + [next(p for p in names if p.startswith(prefix) and p.endswith(".jpg"))
                         for prefix in ("Testing/pituitary/", "Testing/notumor/")]
        fig, axes = plt.subplots(len(paths), 6, figsize=(15, 2.65 * len(paths)), layout="constrained")
        for row, path in enumerate(paths):
            with Image.open(BytesIO(archive.read(path))) as raw:
                image = raw.convert("RGB")
            rgb = current.sam_image(image)
            legacy_rgb = old.sam_image(image)
            if not row:
                models.predict("Swin-Tiny", image)  # CUDA warm-up before comparisons
                models.predict("CNN", image)
            predictions = [models.predict(name, image) for name in ("CNN", "Swin-Tiny")]
            previous = time_segment(old.segment, models, image, None)
            updated = time_segment(current.segment, models, image, None, allow_fallback=True)
            default = time_segment(current.segment, models, image, None)
            fourth = current.explain(models, image, rgb, 4)
            assert all(label == predictions[1]["class"] for label in
                       (previous["classification"], updated["classification"], fourth[0]))
            if not row:
                report["hooks"] = check_hooks(models, image, predictions[1]["class"])
                strict = current.segment(models, image, None, allow_fallback=False)
                assert "box_stages" not in strict or strict["box_stages"] == [3]
                report["fallback_off"] = {"status": strict["status"], "reason": strict.get("box_reason")}
                if models.device == "cuda":
                    reference = current.explain(models, image, rgb, 3)
                    models.device = "cpu"
                    try:
                        cpu = current.explain(models, image, rgb, 3)
                    finally:
                        models.device = "cuda"
                    assert reference[0] == cpu[0]
                    report["cpu_cuda"] = {"class": cpu[0], "max_abs_cam_difference": float(np.max(np.abs(reference[1] - cpu[1]))),
                                          "mean_abs_cam_difference": float(np.mean(np.abs(reference[1] - cpu[1])))}

            def panel(ax, title, uri=None, box=None, empty=None, background=rgb):
                ax.imshow(image_from_uri(uri) if uri else background)
                if box and not uri:
                    x, y, X, Y = box
                    ax.add_patch(plt.Rectangle((x, y), X-x, Y-y, fill=False, edgecolor="gold", linewidth=1.6))
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                if empty:
                    ax.text(.5, .06, empty, transform=ax.transAxes, ha="center", color="white",
                            bbox={"facecolor": "black", "alpha": .85, "pad": 3}, fontsize=7)

            source = {"gradcam_selected":"Aşama 3", "gradcam_other_stage":"Diğer aşama",
                      "gradcam_agreement":"Ortak odak"}.get(updated.get("box_source"), "Kutu yok")
            titles = [f"{Path(path).name}\nOrijinal MR", "Eski Grad-CAM · Aşama 3", "Güncel Grad-CAM · Aşama 3",
                      "Güncel Grad-CAM · Aşama 4", "Eski otomatik SAM", f"Güncel SAM (alternatif açık)\n{source}"]
            uris = [None, previous["gradcam"], updated["gradcam"], fourth[3], previous.get("overlay"), updated.get("overlay")]
            for i, (ax, title, uri) in enumerate(zip(axes[row], titles, uris)):
                result = previous if i == 4 else updated
                panel(ax, title, uri, result.get("box") if i >= 4 else None,
                      result["status"] if i >= 4 and not uri else None, legacy_rgb if i == 4 else rgb)
            fields = ("status", "box", "box_reason", "box_source", "box_stages", "pixels", "elapsed_s", "native_box")
            record = {"image": Path(path).name, "dataset_label": path.split("/")[1],
                      "native_size": list(image.size), "preview_rect": list(current.PreviewGeometry(*image.size).rect),
                      "predictions": {p["model"]: p["class"] for p in predictions},
                      "old": {key: previous.get(key) for key in fields},
                      "current": {key: updated.get(key) for key in fields}}
            record["default"] = {key: default.get(key) for key in fields}
            if previous.get("box"):
                record["old"]["box_source"] = "gradcam_stage_3"
            report["cases"].append(record)
            print(record, flush=True)
        report["status_counts"] = {key: dict(Counter(case[key]["status"] for case in report["cases"]))
                                   for key in ("old", "default", "current")}
        fig.savefig(args.output / "gradcam_sam_once_sonra.png", dpi=140)
        plt.close(fig)
    (args.output / "measurements.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
