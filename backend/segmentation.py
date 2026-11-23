import base64
from io import BytesIO
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import SamPredictor, sam_model_registry

from .models import CLASSES
from .lesion_candidate import intensity_candidate

SIZE = 224


@dataclass(frozen=True)
class PreviewGeometry:
    native_width: int
    native_height: int

    @property
    def rect(self):
        scale = SIZE / max(self.native_width, self.native_height)
        width = max(1, round(self.native_width * scale))
        height = max(1, round(self.native_height * scale))
        return (SIZE - width) // 2, (SIZE - height) // 2, width, height

    def project(self, array, interpolation):
        x, y, width, height = self.rect
        frame = np.zeros((SIZE, SIZE, *array.shape[2:]), dtype=array.dtype)
        frame[y:y + height, x:x + width] = cv2.resize(array, (width, height), interpolation=interpolation)
        return frame

    def contains(self, x, y):
        left, top, width, height = self.rect
        return left <= x < left + width and top <= y < top + height

    def native_points(self, points):
        left, top, width, height = self.rect
        values = np.asarray(points, dtype=np.float32)
        return (values - [left, top]) * [self.native_width / width, self.native_height / height]


def swin_reshape(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        side = int(tokens ** 0.5)
        if side * side == tokens:
            return tensor.transpose(1, 2).reshape(batch, channels, side, side)
    raise ValueError("Swin aktivasyonunun boyutu beklenen biçimde değil.")


def foreground_region(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = max(5, float(np.percentile(smooth, 95)) * 0.09)
    binary = (smooth > threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(binary)
    if contours:
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        region = cv2.erode(region, np.ones((5, 5), dtype=np.uint8))
    return region.astype(bool)


def box_diagnostic(cam_map, foreground):
    heatmap = cv2.resize(cam_map.astype(np.float32), (SIZE, SIZE))
    if foreground.sum() < 64 or not np.isfinite(heatmap).all():
        return None, 'ön plan veya Grad-CAM verisi yetersiz'
    foreground = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if foreground.sum() < 64:
        return None, 'görüntü kenarı elendikten sonra ön plan yetersiz'
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    activations = heatmap[foreground]
    peak = float(activations.max())
    contrast = peak - float(np.median(activations))
    # Keep the original gate for strong maps; accept a weaker peak only
    # when it still stands clearly above the foreground baseline.
    if contrast < 0.15 and (peak < 0.025 or contrast < 0.35 * peak):
        return None, 'aktivasyonlar birbirinden yeterince ayrışmadı'
    threshold = max(float(np.percentile(activations, 90)), 0.60 * peak)
    binary = (foreground & (heatmap >= threshold)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 10]
    if not candidates:
        return None, 'yeterince büyük bağlantılı bölge bulunmadı'
    # Rank coherent hotspots; a distant component never expands the selected box.
    ranked = sorted(((float(heatmap[components == index].max()), index) for index in candidates), reverse=True)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.08 * peak:
        return None, 'benzer güçte birden fazla odak bulundu'
    selected = ranked[0][1]
    x, y, width, height, _ = map(int, stats[selected])
    padding = max(5, min(12, round(max(width, height) * 0.18)))
    box = [max(0, x - padding), max(0, y - padding),
           min(SIZE - 1, x + width - 1 + padding), min(SIZE - 1, y + height - 1 + padding)]
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.28 * foreground.sum():
        return None, 'önerilen kutu ön plana göre çok geniş'
    if min(box[2] - box[0], box[3] - box[1]) < 12:
        return None, 'önerilen kutu çok dar'
    return box, None


def automatic_box(cam_map, foreground):
    return box_diagnostic(cam_map, foreground)[0]


def shared_cam_focus(first, second, foreground):
    # Agreement is computed on the Grad-CAM outputs and used only for box selection.
    a, b = (np.maximum(cam, 0) * foreground for cam in (first, second))
    peaks = (float(a.max()), float(b.max()))
    if min(peaks) <= 0:
        return None
    a /= peaks[0]
    b /= peaks[1]
    # Both stages must strongly support the same pixels.
    agreement = np.sqrt(a * b)
    focus = (a >= 0.70) & (b >= 0.70)
    if np.count_nonzero(focus) < 16:
        return None
    return automatic_box(np.where(focus, agreement, 0), foreground)


def select_mask(masks, scores, box, foreground, prompts=()):
    x1, y1, x2, y2 = map(round, box)
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    valid = []
    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if not 0 < area < SIZE * SIZE:
            continue
        overlap = int(mask[y1:y2 + 1, x1:x2 + 1].sum())
        if overlap < 0.75 * area or overlap > 0.95 * box_area:
            continue
        window = mask[y1:y2 + 1, x1:x2 + 1]
        edge_coverage = [window[0].mean(), window[-1].mean(), window[:, 0].mean(), window[:, -1].mean()]
        if overlap > 0.78 * box_area and sum(edge > 0.40 for edge in edge_coverage) >= 3:
            continue
        if any(bool(mask[y, x]) != bool(label) for x, y, label in prompts):
            continue
        if foreground is not None and (area > 0.22 * foreground.sum() or (mask & foreground).sum() < 0.85 * area):
            continue
        valid.append((mask, float(score)))
    return max(valid, key=lambda candidate: candidate[1], default=None)


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getbuffer()).decode("ascii")


def load_sam(models):
    models.load("SAM ViT-B", "sam_vit_b_01ec64.pth", sam_model_registry["vit_b"], raw=True)


def render_cam(rgb, heatmap):
    # Visualization only. The original Grad-CAM values still drive box selection.
    foreground = foreground_region(rgb) & (rgb.max(axis=2) > 8)
    peak = float(np.where(foreground, heatmap, 0).max())
    levels = np.clip(heatmap / peak, 0, 1) if peak > 0 else np.zeros_like(heatmap)
    alpha = np.clip((levels - 0.25) / 0.75, 0, 1) * foreground
    color_bgr = cv2.applyColorMap(np.rint(levels * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    layer = np.dstack((color, np.rint(alpha * 255).astype(np.uint8)))
    composite = np.rint(rgb * (1 - 0.75 * alpha[..., None]) + color * (0.75 * alpha[..., None]))
    return png(layer), png(composite.astype(np.uint8))


def swin_stage3_layer_name(models):
    return ("layers[2]" if models.swin_stage3_target == "stage_output"
            else "layers[2].blocks[-1].norm1")


def explain(models, image, rgb, stage=3, name="Swin-Tiny"):
    tensor = models.tensor(name, image)

    def infer(model, device):
        device_input = tensor.to(device)
        with torch.no_grad():
            logits = model(device_input)
            if not torch.isfinite(logits).all():
                raise ValueError(f"{name} geçerli sınıf skorları üretemedi.")
            index = int(logits.argmax(dim=1).item())
        if name == "CNN":
            target_layer = model.features[3].block[3]
            reshape = None
        else:
            swin_stage = model.layers[stage - 1]
            target_layer = swin_stage.blocks[-1].norm1
            if stage == 3 and models.swin_stage3_target == "stage_output":
                target_layer = swin_stage
            reshape = swin_reshape
        with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape) as cam:
            heatmap = cam(input_tensor=device_input, targets=[ClassifierOutputTarget(index)], aug_smooth=True)[0]
        if not np.isfinite(heatmap).all():
            raise ValueError("Grad-CAM sonlu bir aktivasyon haritası üretemedi.")
        return index, heatmap

    (index, heatmap), device = models.run(name, infer)
    heatmap = PreviewGeometry(*image.size).project(heatmap, cv2.INTER_LINEAR)
    layer, overlay = render_cam(rgb, heatmap)
    return CLASSES[index], heatmap, device, overlay, layer


def sam_image(image):
    return PreviewGeometry(*image.size).project(np.asarray(image.convert("RGB")), cv2.INTER_LINEAR)


def segment(models, image, manual_box, method='gradcam', stage=3, prompts=(), allow_fallback=False, model='swin'):
    geometry = PreviewGeometry(*image.size)
    native_rgb = np.asarray(image.convert("RGB"))
    rgb = geometry.project(native_rgb, cv2.INTER_LINEAR)
    response = {"status": "hazır", "image_size": [SIZE, SIZE]}
    foreground = foreground_region(rgb) if manual_box is None else None
    if manual_box is None:
        cnn_source = method == "gradcam" and model == "cnn"
        classifier = "CNN" if cnn_source else "Swin-Tiny"
        label, heatmap, device, overlay, layer = explain(models, image, rgb, stage, classifier)
        response.update(classification=label, gradcam=overlay, gradcam_layer=layer,
                        gradcam_device=device, stage=None if cnn_source else stage, method=method,
                        model=classifier)
        if method == "intensity" and label == "notumor":
            response["warning"] = "Sınıflandırıcı tümör yok dedi. Görüntü işleme kutusu olası bir parlak bölgedir; tanı değildir."
        if label == "notumor" and method == "gradcam":
            return dict(response, status="atlandı", message=f"{classifier} tümör yok sınıfını seçti. Otomatik SAM atlandı.")
        if method == "intensity":
            box = intensity_candidate(rgb)
            if box is not None:
                response["box_source"] = "intensity"
        else:
            box, reason = box_diagnostic(heatmap, foreground)
            if cnn_source:
                if box is not None:
                    response["box_source"] = "gradcam_cnn"
            elif box is not None:
                response.update(box_source="gradcam_selected", box_stage=stage, box_stages=[stage])
            else:
                response["box_reason"] = f"Aşama {stage}: {reason}."
        if box is None and method == "gradcam" and allow_fallback and not cnn_source:
            other_stage = 4 if stage == 3 else 3
            other_label, other_cam, _, _, _ = explain(models, image, rgb, other_stage)
            if other_label == label:
                box = shared_cam_focus(heatmap, other_cam, foreground)
                if box is not None:
                    response.update(box_source="gradcam_agreement", box_stages=[3, 4])
                if box is None:
                    response["box_warning"] = "İki aşama aynı bölgede yeterince güçlü ortak odak göstermedi."
        if box is None and cnn_source:
            return dict(response, status="kutu_yok",
                        message="CNN Grad-CAM haritasından uygun kutu çıkmadı. Swin-Tiny seçebilir veya manuel kutu çizebilirsiniz.")
        if box is None:
            return dict(response, status="kutu_yok", message="Seçilen aşamadan uygun aday kutu oluşturulamadı. Alternatif aşamayı deneyebilir veya manuel kutu çizebilirsiniz.")
    else:
        box = manual_box
        response["box_source"] = "manual"

    left, top, width, height = geometry.rect
    box = [max(box[0], left), max(box[1], top),
           min(box[2], left + width - 1), min(box[3], top + height - 1)]
    if box[2] - box[0] < 4 or box[3] - box[1] < 4 or any(
        not geometry.contains(x, y) for x, y, _ in prompts
    ):
        return dict(response, status="kutu_yok",
                    message="Kutu ve işaretleri MR görüntüsünün içinde tutun.")
    native_box = geometry.native_points([box[:2], box[2:]]).reshape(4).astype(np.float32)
    native_prompts = geometry.native_points([(x, y) for x, y, _ in prompts]).astype(np.float32) if prompts else None

    @torch.inference_mode()
    def infer(model, device):
        predictor = SamPredictor(model)
        predictor.set_image(native_rgb)
        masks, scores, _ = predictor.predict(
            box=native_box,
            point_coords=native_prompts,
            point_labels=np.asarray([label for _, _, label in prompts], dtype=np.int32) if prompts else None,
            multimask_output=True,
        )
        if not np.isfinite(scores).all():
            raise ValueError("SAM sonlu maske skorları üretemedi.")
        projected = [geometry.project(mask.astype(np.uint8), cv2.INTER_NEAREST_EXACT).astype(bool)
                     for mask in masks]
        return select_mask(projected, scores, box, foreground, prompts)

    try:
        candidate, device = models.run("SAM ViT-B", infer)
    except (RuntimeError, ValueError) as exc:
        return dict(response, status="hata", box=box, message=f"SAM maskesi üretilemedi: {exc}")
    if candidate is None:
        message = ("Otomatik maske aday bölgeye yeterince odaklanmadı. Orijinal MR üzerinde daha dar bir kutu çizin."
                   if manual_box is None else "Kutuya ve işaretlere uyan maske bulunamadı. Dahil etme veya çıkarma noktası ekleyip tekrar deneyin.")
        return dict(response, status="maske_yok", box=box, device=device, message=message)
    mask, score = candidate
    area = int(mask.sum())
    overlay = rgb.copy()
    overlay[mask] = (0.5 * rgb[mask] + 0.5 * np.array([64, 245, 180])).astype(np.uint8)
    cv2.rectangle(overlay, tuple(map(round, box[:2])), tuple(map(round, box[2:])), (245, 221, 109), 1)
    return dict(response, box=box, mask=png(mask.astype(np.uint8) * 255), overlay=png(overlay),
                score=score, pixels=area, device=device, native_box=native_box.tolist())
