from contextlib import contextmanager
from pathlib import Path
from os import environ
from threading import Lock

import timm
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16

ROOT = Path(__file__).resolve().parents[1]
SWIN_PROFILES = {"legacy": "swin_tiny_best.pth", "v3": "swin_tiny_v3_best.pth",
                 "v3-finetuned": "swin_tiny_v3_finetuned_best.pth"}
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for source, target in [(1, 32), (32, 64), (64, 128)]:
            layers.extend([nn.Conv2d(source, target, 3, padding=1),
                           nn.BatchNorm2d(target), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 128),
                                        nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 4))

    def forward(self, tensor):
        return self.classifier(self.pool(self.features(tensor)))


class Models:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = Lock()
        self.models = {}
        self.errors = {}
        self.cpu_only = set()
        self.swin_profile = environ.get("NEURO_SWIN_PROFILE", "legacy")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        for name, filename, factory in [
            ("CNN", "cnn_best.pth", BrainTumorCNN),
            ("ViT", "vit_best.pth", lambda: vit_b_16(weights=None, num_classes=4)),
            ("Swin-Tiny", SWIN_PROFILES.get(self.swin_profile), lambda: timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=False, num_classes=4)),
        ]:
            self.load(name, filename, factory)

    def load(self, name, filename, factory, raw=False):
        if filename is None:
            self.errors[name] = "Geçersiz NEURO_SWIN_PROFILE. legacy, v3 veya v3-finetuned kullanın."
            return
        path = ROOT / "models" / filename
        try:
            weights = torch.load(path, map_location="cpu", weights_only=True)
            # The supplied V3 export contains only tensors and uses the documented class order.
            bare_v3 = name == "Swin-Tiny" and self.swin_profile == "v3" and all(
                torch.is_tensor(value) for value in weights.values())
            if name == "Swin-Tiny" and not bare_v3 and tuple(weights.get("class_names", ())) != CLASSES:
                raise ValueError("Checkpoint sınıf sırası uyuşmuyor.")
            model = factory()
            model.load_state_dict(weights if raw or bare_v3 else weights["model_state_dict"], strict=True)
            self.models[name] = model.eval()
        except FileNotFoundError:
            self.errors[name] = f"{filename} bulunamadı."
        except Exception as exc:
            self.errors[name] = f"{filename} yüklenemedi: {exc}"

    @staticmethod
    def tensor(name, image):
        gray = name == "CNN"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] if gray else RGB_MEAN, [0.5] if gray else RGB_STD),
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
            logits = model(tensor.to(device))
            if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError(f"{name} geçerli sınıf skorları üretemedi.")
            return logits.softmax(dim=1)[0].tolist()

        scores, device = self.run(name, infer)
        best = max(range(len(scores)), key=scores.__getitem__)
        return {"model": name, "class": CLASSES[best], "probabilities": dict(zip(CLASSES, scores)),
                "device": device}
