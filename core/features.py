"""Kare başına güven özellikleri (öğrenilmiş güven ağı için girdi). Pencere: son K kare."""
import numpy as np, cv2
from utils.bbox_tools import center

def frame_features(score, box, prev_box, ncc):
    """score: takipçi skoru; box/prev_box: [x,y,w,h]; ncc: şablon-aday benzerliği (yoksa 0)."""
    if prev_box is None or box is None or box[2] <= 0:
        jump, area_ratio = 0.0, 1.0
    else:
        ca, cb = center(box), center(prev_box)
        diag = max(1.0, np.hypot(prev_box[2], prev_box[3]))
        jump = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / diag)
        area_ratio = float((box[2] * box[3]) / max(1.0, prev_box[2] * prev_box[3]))
    return np.array([score, jump, area_ratio, ncc], dtype=np.float32)

def window(feats, K=8):
    """feats: list[np.ndarray(4)] -> (K,4) dolgulu pencere + skor farkı kanalı (K,5)."""
    a = np.stack(feats[-K:]) if feats else np.zeros((1, 4), np.float32)
    if len(a) < K:
        a = np.concatenate([np.repeat(a[:1], K - len(a), 0), a])
    d = np.diff(a[:, 0], prepend=a[0, 0])[:, None]
    return np.concatenate([a, d], 1).astype(np.float32)

def ncc_to_template(frame, box, templ):
    """Takipçi kutusundaki görüntünün şablonla normalize korelasyonu (-1..1)."""
    H, W = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2)); w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
    c = cv2.resize(frame[y:y + h, x:x + w], (templ.shape[1], templ.shape[0]))
    return float(cv2.matchTemplate(c, templ, cv2.TM_CCOEFF_NORMED)[0, 0])
