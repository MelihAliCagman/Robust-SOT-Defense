"""Kutu yardımcıları. Varsayılan format: [x, y, w, h] (sol üst köşe, genişlik, yükseklik)."""
import numpy as np


def xywh_to_xyxy(box):
    """[x, y, w, h] -> [x1, y1, x2, y2]"""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    """[x1, y1, x2, y2] -> [x, y, w, h]"""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def center(box):
    """box: [x, y, w, h] -> (cx, cy)"""
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def from_center(cx, cy, w, h):
    """-> [x, y, w, h]"""
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def iou(a, b):
    """Kesişim / birleşim oranı. a, b: [x, y, w, h]"""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def center_error(a, b):
    """İki kutunun merkezleri arasındaki uzaklık (piksel). a, b: [x, y, w, h]"""
    (ax, ay), (bx, by) = center(a), center(b)
    return float(np.hypot(ax - bx, ay - by))


def clip_box(box, width, height):
    """[x, y, w, h] kutusunu görüntü sınırlarına sığdırır (en az 1x1)."""
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return [x, y, w, h]
