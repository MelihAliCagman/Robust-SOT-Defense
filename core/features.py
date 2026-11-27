"""Güven modelinin girdileri.

Her kare için 4 sayı: [skor, kutu sıçraması, alan oranı, şablon benzerliği].
Pencerede buna skor farkı eklenir; modele (K, 5) boyutunda dizi verilir.
"""
import cv2
import numpy as np

from utils.bbox_tools import center


def frame_features(score, box, prev_box, ncc):
    """score: takipçi skoru; box, prev_box: [x, y, w, h]; ncc: şablon benzerliği (yoksa 0)."""
    if prev_box is None or box is None or box[2] <= 0:
        jump, area_ratio = 0.0, 1.0
    else:
        (cx, cy), (prev_cx, prev_cy) = center(box), center(prev_box)
        prev_diagonal = max(1.0, np.hypot(prev_box[2], prev_box[3]))
        jump = float(np.hypot(cx - prev_cx, cy - prev_cy) / prev_diagonal)  # kutu boyutuna göre
        area_ratio = float((box[2] * box[3]) / max(1.0, prev_box[2] * prev_box[3]))
    return np.array([score, jump, area_ratio, ncc], dtype=np.float32)


def window(feats, K=8):
    """Son K karenin özellikleri -> (K, 5). Geçmiş kısaysa ilk kare tekrarlanarak doldurulur."""
    recent = np.stack(feats[-K:]) if feats else np.zeros((1, 4), np.float32)
    if len(recent) < K:
        padding = np.repeat(recent[:1], K - len(recent), 0)
        recent = np.concatenate([padding, recent])
    score_diff = np.diff(recent[:, 0], prepend=recent[0, 0])[:, None]
    return np.concatenate([recent, score_diff], 1).astype(np.float32)


def ncc_to_template(frame, box, templ):
    """Kutudaki görüntünün şablonla normalize korelasyonu (-1..1). box: [x, y, w, h]"""
    height, width = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, min(x, width - 2)), max(0, min(y, height - 2))
    w, h = max(2, min(w, width - x)), max(2, min(h, height - y))
    crop = cv2.resize(frame[y:y + h, x:x + w], (templ.shape[1], templ.shape[0]))
    return float(cv2.matchTemplate(crop, templ, cv2.TM_CCOEFF_NORMED)[0, 0])
