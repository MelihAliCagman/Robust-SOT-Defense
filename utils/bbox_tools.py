"""Kutu yardımcıları. Varsayılan format: [x, y, w, h] (sol-üst köşe, genişlik, yükseklik)."""
import numpy as np

def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]

def center(b):
    """b: [x, y, w, h] -> (cx, cy)"""
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

def from_center(cx, cy, w, h):
    """-> [x, y, w, h]"""
    return [cx - w / 2.0, cy - h / 2.0, w, h]

def iou(a, b):
    """a, b: [x, y, w, h]"""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def center_error(a, b):
    ca, cb = center(a), center(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

def clip_box(b, W, H):
    x, y, w, h = b
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return [x, y, w, h]
