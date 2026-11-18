import cv2
import numpy as np


def intensity_candidate(rgb):
    """224x224 MR'da parlak ve yerel kontrastı yüksek tek bir kutu öner."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    foreground = (gray >= max(12, np.percentile(gray, 70) * 0.25)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    region = np.zeros_like(gray)
    cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    inside = cv2.erode(region, kernel).astype(bool)
    if inside.sum() < 1500 or np.ptp(gray[inside]) < 20:
        return None
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 7)
    contrast = gray.astype(np.float32) - blur
    bright = (gray >= np.percentile(gray[inside], 86)) & inside
    candidates = bright & (contrast >= max(9, np.percentile(contrast[inside], 86)))
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE, close)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary * inside.astype(np.uint8))
    valid = [i for i in range(1, count) if 30 <= stats[i, cv2.CC_STAT_AREA] <= inside.sum() * 0.15]
    if not valid:
        return None
    best = max(valid, key=lambda i: float(np.maximum(contrast[labels == i], 0).sum()))
    x, y, w, h = map(int, stats[best, :4])
    pad = 8
    return [max(0, x-pad), max(0, y-pad), min(gray.shape[1]-1, x+w+pad),
            min(gray.shape[0]-1, y+h+pad)]
