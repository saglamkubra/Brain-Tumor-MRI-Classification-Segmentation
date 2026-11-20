from contextlib import contextmanager
from pathlib import Path
from os import environ
from threading import Lock
from time import perf_counter
import math
import hashlib
import json

from .cnn_architecture import CNN_V2

import numpy as np

import timm
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SWIN_PROFILES = {"legacy": "swin_tiny_best.pth", "v3": "swin_tiny_v3_best.pth",
                 "v3-finetuned": "swin_tiny_v3_finetuned_best.pth"}
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


class Models:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = Lock()
        self.models = {}
        self.errors = {}
        self.versions = {}
        self.cnn_integrity = None
        self.cnn_preprocess = None
        self.cpu_only = set()
        self.swin_stage3_target = environ.get("NEURO_SWIN_STAGE3_TARGET", "stage_output")
        if self.swin_stage3_target not in {"block_norm1", "stage_output"}:
            raise ValueError("NEURO_SWIN_STAGE3_TARGET: block_norm1 veya stage_output seçin.")
        self.swin_profile = environ.get("NEURO_SWIN_PROFILE", "v3")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        for name, filename, factory in [
            ("CNN", "cnn_v2_best.pth", CNN_V2),
            ("Swin-Tiny", SWIN_PROFILES.get(self.swin_profile), lambda: timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=False, num_classes=4)),
        ]:
            self.load(name, filename, factory)

    def load(self, name, filename, factory, raw=False):
        if filename is None:
            self.errors[name] = "Geçersiz NEURO_SWIN_PROFILE. legacy, v3 veya v3-finetuned kullanın."
            return
        self.versions[name] = filename
        path = ROOT / "models" / filename
        try:
            if name == "CNN":
                package = ROOT / "models" / "CNN_V2"
                class_order = dict(zip(CLASSES, range(len(CLASSES))))
                expected = "ae6501b19f70a1479374d9111cc954a76a2e1fa77a6863ddf159c3b03bf1fa98"
                using_package = package.exists()
                if using_package:
                    required = (filename, "deployment_config.json", "cnn_v2_architecture.py", "inference.py")
                    missing = [item for item in required if not (package / item).is_file()]
                    if missing:
                        raise ValueError("CNN paketi bulunamadı veya eksik. Eksik dosyalar: " + ", ".join(missing))
                    config = json.loads((package / "deployment_config.json").read_text(encoding="utf-8-sig"))
                    if (config["input_height"], config["input_width"], config["input_channels"],
                            config["image_mode"], config["resize_interpolation"], config["class_to_idx"],
                            config["gradcam_target_layer"]) != (224,224,1,"L","bilinear",class_order,"features.3.block.3"):
                        raise ValueError("CNN dağıtım yapılandırması uyuşmuyor.")
                    path = package / filename
                    expected = config["checkpoint_sha256"].lower()
                elif environ.get("NEURO_ALLOW_LOCAL_CNN") != "1":
                    raise ValueError("CNN paketi bulunamadı. Yerel ağırlık varsayılan olarak kapalı.")
                with path.open("rb") as checkpoint:
                    if hashlib.file_digest(checkpoint, "sha256").hexdigest() != expected:
                        raise ValueError("CNN checkpoint SHA-256 uyuşmuyor.")
                if using_package:
                    weights = torch.load(path, map_location="cpu", weights_only=True)
                    prep = dict(image_size=config["input_width"], mean=config["normalization_mean"][0],
                                std=config["normalization_std"][0])
                else:
                    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
                    with torch.serialization.safe_globals(allowed):
                        weights = torch.load(path, map_location="cpu", weights_only=True)
                    prep = weights["preprocessing_config"]
                if weights.get("class_to_idx") != class_order:
                    raise ValueError("CNN sınıf sırası uyuşmuyor.")
                mean, std = float(prep["mean"]), float(prep["std"])
                if prep["image_size"] != 224 or not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                    raise ValueError("CNN V2 ön işleme ayarları geçersiz.")
                self.cnn_preprocess = (mean, std)
                self.cnn_integrity = {
                    "source": "deployment" if using_package else "pinned_local",
                    "file": path.relative_to(ROOT).as_posix(),
                    "sha256": expected,
                    "weights_only": True,
                    "numpy_allowlist": not using_package,
                }
            else:
                weights = torch.load(path, map_location="cpu", weights_only=True)
            # The supplied V3 export contains only tensors and uses the documented class order.
            bare_v3 = name == "Swin-Tiny" and self.swin_profile == "v3" and all(
                torch.is_tensor(value) for value in weights.values())
            if name == "Swin-Tiny" and not bare_v3 and tuple(weights.get("class_names", ())) != CLASSES:
                raise ValueError("Checkpoint sınıf sırası uyuşmuyor.")
            model = factory()
            model.load_state_dict(weights if raw or bare_v3 else weights["model_state_dict"], strict=True)
            if name == "CNN" and sum(parameter.numel() for parameter in model.parameters()) != 3_337_764:
                raise ValueError("CNN parametre sayısı uyuşmuyor.")
            self.models[name] = model.eval()
        except FileNotFoundError:
            self.errors[name] = f"{filename} bulunamadı."
        except Exception as exc:
            self.errors[name] = f"{filename} yüklenemedi: {exc}"

    def tensor(self, name, image):
        gray = name == "CNN"
        if gray and self.cnn_preprocess is None:
            raise ValueError(self.errors.get(name, "CNN V2 ön işlemesi yüklenmedi."))
        mean, std = self.cnn_preprocess if gray else (None, None)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([mean] if gray else RGB_MEAN, [std] if gray else RGB_STD),
        ])
        return transform(image.convert("L" if gray else "RGB")).unsqueeze(0)

    @contextmanager
    def on_device(self, model, device):
        try:
            yield model.to(device)
        finally:
            if device == "cuda":
                model.zero_grad(set_to_none=True)
                model.to("cpu")
                torch.cuda.empty_cache()

    def run(self, name, operation):
        if name not in self.models:
            raise ValueError(self.errors.get(name, f"{name} yüklenmedi."))
        with self.lock:
            model = self.models[name]
            device = "cpu" if name in self.cpu_only else self.device
            try:
                with self.on_device(model, device):
                    return operation(model, device), device
            except torch.cuda.OutOfMemoryError:
                if device != "cuda":
                    raise
            torch.cuda.empty_cache()
            self.cpu_only.add(name)
            return operation(model, "cpu"), "cpu"

    def predict(self, name, image):
        tensor = self.tensor(name, image)

        @torch.inference_mode()
        def infer(model, device):
            inputs = tensor.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            logits = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (perf_counter() - started) * 1000
            if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError(f"{name} geçerli sınıf skorları üretemedi.")
            return logits.softmax(dim=1)[0].tolist(), elapsed

        (scores, elapsed), device = self.run(name, infer)
        best = max(range(len(scores)), key=scores.__getitem__)
        return {"model": name, "class": CLASSES[best], "probabilities": dict(zip(CLASSES, scores)),
                "device": device, "inference_ms": elapsed, "checkpoint": self.versions[name],
                "profile": self.swin_profile if name == "Swin-Tiny" else "v2"}
