import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import SamPredictor, sam_model_registry

from .models import CLASSES
from .lesion_candidate import intensity_candidate

SIZE = 224


def swin_reshape(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        side = int(tokens ** 0.5)
        if side * side == tokens:
            return tensor.transpose(1, 2).reshape(batch, channels, side, side)
    raise ValueError("Swin aktivasyonunun boyutu beklenen biçimde değil.")


def automatic_box(cam_map):
    heatmap = cv2.resize(cam_map.astype(np.float32), (SIZE, SIZE))
    yy, xx = np.ogrid[:SIZE, :SIZE]
    inside = ((xx - 112) / 75) ** 2 + ((yy - 112) / 78) ** 2 <= 1
    # A zero map must not turn the whole ellipse into an activation.
    if not np.isfinite(heatmap).all() or heatmap[inside].max() <= 0:
        return None
    threshold = np.percentile(heatmap[inside], 90)
    binary = ((heatmap >= threshold) & inside).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 10:
            clean[labels == index] = 1
    ys, xs = np.where(clean > 0)
    if not len(xs):
        return None
    return [max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12),
            min(223, int(xs.max()) + 12), min(223, int(ys.max()) + 12)]


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getbuffer()).decode("ascii")


def load_sam(models):
    models.load("SAM ViT-B", "sam_vit_b_01ec64.pth", sam_model_registry["vit_b"], raw=True)


def explain(models, image, rgb, stage=3):
    tensor = models.tensor("Swin-Tiny", image)

    def infer(model, device):
        device_input = tensor.to(device)
        with torch.no_grad():
            logits = model(device_input)
            if not torch.isfinite(logits).all():
                raise ValueError("Swin-Tiny geçerli sınıf skorları üretemedi.")
            index = int(logits.argmax(dim=1).item())
        with GradCAM(model=model, target_layers=[model.layers[stage - 1].blocks[-1].norm1],
                     reshape_transform=swin_reshape) as cam:
            heatmap = cam(input_tensor=device_input, targets=[ClassifierOutputTarget(index)])[0]
        if not np.isfinite(heatmap).all():
            raise ValueError("Grad-CAM sonlu bir aktivasyon haritası üretemedi.")
        return index, heatmap

    (index, heatmap), device = models.run("Swin-Tiny", infer)
    overlay = show_cam_on_image(rgb.astype(np.float32) / 255, heatmap, use_rgb=True)
    return CLASSES[index], heatmap, device, png(overlay)


def sam_image(image):
    return np.asarray(image.resize((SIZE, SIZE)), dtype=np.uint8).copy()


def segment(models, image, manual_box, method='gradcam', stage=3):
    rgb = sam_image(image)
    response = {"status": "hazır", "image_size": [SIZE, SIZE]}
    if manual_box is None:
        label, heatmap, device, overlay = explain(models, image, rgb, stage)
        response.update(classification=label, gradcam=overlay, gradcam_device=device, stage=stage, method=method)
        if method == "intensity" and label == "notumor":
            response["warning"] = "Sınıflandırıcı tümör yok dedi. Görüntü işleme kutusu olası bir parlak bölgedir; tanı değildir."
        if label == "notumor" and method == "gradcam":
            return dict(response, status="atlandı", message="Swin-Tiny tümör yok sınıfını seçti. Otomatik SAM atlandı.")
        box = intensity_candidate(rgb) if method == "intensity" else automatic_box(heatmap)
        if box is None:
            return dict(response, status="kutu_yok", message="Otomatik yöntem uygun bir aday kutu üretemedi. Manuel kutu çizin.")
    else:
        box = manual_box

    @torch.inference_mode()
    def infer(model, device):
        predictor = SamPredictor(model)
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
        if not np.isfinite(scores).all():
            raise ValueError("SAM sonlu maske skorları üretemedi.")
        best = int(scores.argmax())
        return masks[best], float(scores[best])

    try:
        (mask, score), device = models.run("SAM ViT-B", infer)
    except (RuntimeError, ValueError) as exc:
        return dict(response, status="hata", box=box, message=f"SAM maskesi üretilemedi: {exc}")
    area = int(mask.sum())
    if area == 0 or area == SIZE * SIZE:
        return dict(response, status="maske_yok", box=box, device=device,
                    message="SAM kullanılabilir bir maske üretemedi. Kutuyu değiştirip tekrar deneyin.")
    overlay = rgb.copy()
    overlay[mask] = (0.5 * rgb[mask] + 0.5 * np.array([64, 245, 180])).astype(np.uint8)
    cv2.rectangle(overlay, tuple(map(round, box[:2])), tuple(map(round, box[2:])), (245, 221, 109), 1)
    return dict(response, box=box, mask=png(mask.astype(np.uint8) * 255), overlay=png(overlay),
                score=score, pixels=area, device=device)
