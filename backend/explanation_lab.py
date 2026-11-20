"""Interactive probes that reuse the deployed preprocessing and models."""
import math

import cv2
import numpy as np
import torch
from PIL import Image

from .models import CLASSES
from .segmentation import PreviewGeometry, explain, png, sam_image

RANDOM_SEED = 20261119


def input_previews(models, image):
    previews = []
    for name in ("CNN", "Swin-Tiny"):
        tensor = models.tensor(name, image)[0].permute(1, 2, 0).numpy()
        low, high = tensor.min(), tensor.max()
        scaled = np.rint((tensor - low) * (255 / (high - low))).astype(np.uint8) if high > low else np.zeros_like(tensor, dtype=np.uint8)
        previews.append(dict(model=name, image=png(scaled.squeeze(2) if name == "CNN" else scaled)))
    return previews


def validate_box(image, value):
    if not value:
        raise ValueError("Bir müdahale kutusu çizin.")
    try:
        x, y, X, Y = map(float, value.split(","))
    except ValueError:
        raise ValueError("Geçersiz müdahale kutusu.") from None
    geo = PreviewGeometry(*image.size)
    left, top, width, height = geo.rect
    if (not all(math.isfinite(v) for v in (x, y, X, Y)) or
            not (left <= x < X <= left + width and top <= y < Y <= top + height) or X-x < 4 or Y-y < 4):
        raise ValueError("Geçersiz müdahale kutusu.")
    if (X-x)*(Y-y) > width*height/2:
        raise ValueError("Müdahale kutusu görüntü alanının yarısını aşamaz.")
    native = geo.native_points([(x, y), (X, Y)])
    return tuple(map(int, np.r_[np.floor(native[0]), np.ceil(native[1])]))


def random_pixels(width, height, count):
    """Return a connected patch of exactly count pixels at a seeded position."""
    rng = np.random.default_rng(RANDOM_SEED)
    cols = min(width, max(math.ceil(count / height), round(math.sqrt(count * width / height))))
    rows = math.ceil(count / cols)
    left = int(rng.integers(width - cols + 1))
    top = int(rng.integers(height - rows + 1))
    offsets = np.arange(count)
    return (top + offsets // cols) * width + left + offsets % cols


def region_mask(models, image, mode, box, stage):
    mask = np.zeros((image.height, image.width), dtype=bool)
    if mode == "box":
        x, y, X, Y = box
        mask[y:Y, x:X] = True
    else:
        count = max(1, round(mask.size * .10))
        if mode == "random":
            if box is not None:
                x, y, X, Y = box
                count = (X-x)*(Y-y)
            indices = random_pixels(image.width, image.height, count)
        else:
            _, cam, _, _, _ = explain(models, image, sam_image(image), stage)
            x, y, w, h = PreviewGeometry(*image.size).rect
            activation = cv2.resize(cam[y:y+h, x:x+w], image.size, interpolation=cv2.INTER_LINEAR).ravel()
            if np.ptp(activation) <= 0:
                raise ValueError("Grad-CAM haritasında sıralanabilir bir odak bulunamadı.")
            indices = np.argpartition(-activation, count-1)[:count]
        mask.ravel()[indices] = True
    return mask


def intervention(models, image, mode, fill, box, stage):
    mask = region_mask(models, image, mode, box, stage)
    original = np.asarray(image)
    if fill == "gray":
        replacement = np.uint8(round(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).mean()))
    else:
        sigma = max(image.size) * .035
        replacement = cv2.GaussianBlur(original, (0, 0), sigma)[mask]
    altered = original.copy()
    altered[mask] = replacement
    modified = Image.fromarray(altered)
    results = []
    for name in ("CNN", "Swin-Tiny"):
        pair = torch.cat([models.tensor(name, original_image) for original_image in (image, modified)])

        @torch.inference_mode()
        def infer(model, device):
            logits = torch.cat([model(tensor[None].to(device)) for tensor in pair], dim=0)
            if logits.shape != (2, len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError("Müdahale için geçerli model skorları üretilemedi.")
            target = int(logits[0].argmax())
            probabilities = logits.softmax(dim=1)[:, target]
            rivals = logits.clone()
            rivals[:, target] = -torch.inf
            margins = logits[:, target] - rivals.max(dim=1).values
            return dict(target=CLASSES[target], before_class=CLASSES[target], after_class=CLASSES[int(logits[1].argmax())],
                        scores=probabilities.tolist(), margins=margins.tolist())

        values, device = models.run(name, infer)
        results.append(dict(model=name, device=device, **values))
    return dict(results=results, preview=png(sam_image(modified)), pixels=int(mask.sum()), total=int(mask.size))
