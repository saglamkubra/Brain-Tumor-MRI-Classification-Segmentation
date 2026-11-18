import argparse
import csv
import hashlib
from io import BytesIO
import json
import random

import numpy as np
from pathlib import Path
from zipfile import ZipFile

import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
ARCH = "swin_tiny_patch4_window7_224"
SEED = 42
EPOCHS = 5
BATCH = 8
EFFECTIVE_BATCH = 16


class MRDataset(Dataset):
    def __init__(self, archive, rows, training=False):
        self.archive = archive
        self.rows = rows
        operations = [transforms.Resize((224, 224))]
        if training:
            operations.append(transforms.RandomRotation(10))
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(BytesIO(self.archive.read(row["dosya"]))) as raw:
            tensor = self.transform(raw.convert("RGB"))
        return tensor, CLASSES.index(row["sinif"])


def read_splits(directory, archive, requests, seen=None):
    seen = set() if seen is None else seen
    available = set(archive.namelist())
    data = {}
    for name, size in requests:
        with (directory / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        paths = {row["dosya"] for row in rows}
        if len(rows) != size or len(paths) != size or paths & seen or not paths <= available:
            raise ValueError(f"Geçersiz v3 veri listesi: {name}")
        for row in rows:
            folder, label, _ = row["dosya"].split("/")
            if label != row["sinif"] or label not in CLASSES or folder != ("Testing" if name == "test" else "Training"):
                raise ValueError(f"Veri etiketi veya bölmesi uyuşmuyor: {row['dosya']}")
        seen.update(paths)
        data[name] = rows
    return data, seen


def epoch_pass(model, loader, loss_fn, optimizer=None, scaler=None, class_weights=None):
    training = optimizer is not None
    model.train(training)
    actual, predicted = [], []
    loss_total = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    group_weight = 0.0
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=training):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Sonlu olmayan kayıp oluştu.")
            if training:
                group_weight += class_weights[labels].sum().item()
                scaler.scale(loss).backward()
                if (step + 1) * BATCH % EFFECTIVE_BATCH == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_weight)
                    group_weight = 0.0
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            loss_total += loss.item()
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).tolist())
            if training and ((step + 1) % 100 == 0 or step + 1 == len(loader)):
                print(f"  Eğitim grubu {step+1}/{len(loader)} | VRAM {torch.cuda.max_memory_reserved()/2**20:.0f} MiB", flush=True)
    glioma = [(a, b) for a, b in zip(actual, predicted) if a == 0]
    return {"loss": loss_total / len(actual), "accuracy": accuracy_score(actual, predicted),
            "glioma_recall": sum(b == 0 for _, b in glioma) / len(glioma),
            "glioma_to_notumor": sum(b == 2 for _, b in glioma)}, actual, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("models/swin_tiny_v3_best.pth"))
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.results / "metrics.json"
    checkpoint = args.output / "swin_tiny_v3_finetuned_best.pth"
    if report_path.exists() or checkpoint.exists():
        raise FileExistsError("Bu konumda önceki v3 çıktıları var. Yeni çıktı klasörü seçin.")
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Eğitim için CUDA bulunamadı.")
    args.results.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | timm {timm.__version__}", flush=True)

    with ZipFile(args.archive) as archive:
        splits, seen = read_splits(args.splits, archive, [("train", 3870), ("val", 994)])
        print("v3 veri sayıları:", {key: len(rows) for key, rows in splits.items()}, flush=True)
        base = torch.load(args.base, map_location="cpu", weights_only=True)
        if "model_state_dict" in base and tuple(base.get("class_names", ())) != CLASSES:
            raise ValueError("Başlangıç checkpoint sınıf sırası uyuşmuyor.")
        model = timm.create_model(ARCH, pretrained=False, num_classes=len(CLASSES))
        model.load_state_dict(base.get("model_state_dict", base), strict=True)
        del base
        model.cuda()
        metadata = {
            "architecture": ARCH,
            "class_names": list(CLASSES),
            "base_checkpoint": args.base.name,
            "learning_rate": 5e-6, "weight_decay": 0.01, "class_weights": [1.5, 1., 1., 1.],
            "selection": "validation accuracy; ties: glioma recall, then fewer glioma-to-notumor errors",
            "patience": 2, "max_epochs": EPOCHS,
            "split_sha256": {name: hashlib.sha256((args.splits / f"{name}.csv").read_bytes()).hexdigest()
                             for name in splits},
            "seed": SEED, "amp": "float16", "microbatch": BATCH, "effective_batch": EFFECTIVE_BATCH,
        }
        loaders = {name: DataLoader(MRDataset(archive, rows, name == "train"),
                                    batch_size=BATCH, shuffle=name == "train", pin_memory=True)
                   for name, rows in splits.items()}
        loss = nn.CrossEntropyLoss(reduction="sum")
        weights = torch.tensor(metadata["class_weights"], device="cuda")
        train_loss = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        history, best, stale = [], None, 0
        for epoch in range(1, EPOCHS+1):
            print(f"Epoch {epoch}/{EPOCHS}", flush=True)
            train, _, _ = epoch_pass(model, loaders["train"], train_loss, optimizer, scaler, weights)
            val, _, _ = epoch_pass(model, loaders["val"], loss)
            history.append({"epoch": epoch, "train": train, "validation": val})
            print(history[-1], flush=True)
            key = (val["accuracy"], val["glioma_recall"], -val["glioma_to_notumor"])
            if best is None or key > best:
                best, stale = key, 0
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(dict(metadata, model_state_dict=model.state_dict(), epoch=epoch,
                                val_accuracy=val["accuracy"], val_glioma_recall=val["glioma_recall"]), temporary)
                temporary.replace(checkpoint)
            else:
                stale += 1
                if stale >= metadata["patience"]:
                    print("Early stopping: iki epoch boyunca doğrulama ölçütü iyileşmedi.", flush=True)
                    break
        del optimizer, scaler
        selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(selected.pop("model_state_dict"), strict=True)
        print("Eğitim tamamlandı. Seçilen checkpoint ile v3 test listesi açılıyor.", flush=True)
        test_split, _ = read_splits(args.splits, archive, [("test", 1579)], seen)
        metadata["split_sha256"]["test"] = hashlib.sha256((args.splits / "test.csv").read_bytes()).hexdigest()
        test_loader = DataLoader(MRDataset(archive, test_split["test"]), batch_size=BATCH, pin_memory=True)
        test, actual, predicted = epoch_pass(model, test_loader, loss)
        test["classification"] = classification_report(actual, predicted, labels=range(len(CLASSES)),
                                                        target_names=CLASSES, output_dict=True, zero_division=0)
        test["confusion_matrix"] = confusion_matrix(actual, predicted, labels=range(len(CLASSES))).tolist()
        test["n"] = len(actual)
        report_path.write_text(json.dumps(dict(metadata=selected, test_csv_sha256=metadata["split_sha256"]["test"], epochs=history, test=test), indent=2)+"\n", encoding="utf-8")
        print(f"Seçilen epoch {selected['epoch']} | v3 test accuracy {test['accuracy']:.4%} "
              f"| macro F1 {test['classification']['macro avg']['f1-score']:.4%}", flush=True)


if __name__ == "__main__":
    main()
