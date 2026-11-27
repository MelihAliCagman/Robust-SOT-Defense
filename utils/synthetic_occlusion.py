"""Gerçek videolara sentetik kapanma basar. Siyah kutu DEĞİL: dokulu yama, gürültü, yarı saydam 'duman'."""
import numpy as np, cv2

def _texture(h, w, rng, kind):
    if kind == "texture":
        t = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        return cv2.GaussianBlur(t, (0, 0), 3)
    if kind == "smoke":
        t = np.full((h, w, 3), 200, np.uint8)
        n = cv2.GaussianBlur(rng.integers(0, 80, (h, w, 3), dtype=np.uint8), (0, 0), 8)
        return cv2.add(t, n)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)  # noise

def apply_occlusion(frames, gt_boxes, start, length, kind="texture", mode="static", seed=0, alpha=1.0):
    """
    frames: list[H,W,3]; gt_boxes: list[[x,y,w,h]]
    mode='static': perde, start karesindeki hedef konumunun ortasında, hedeften büyük, sabit.
    mode='follow': yama hedefle birlikte hareket eder (kapanma tüm süre boyunca tam).
    alpha<1: yarı saydam (duman).
    Döner: yeni_frames, labels (1=kapalı, 0=görünür) -- görünür kutu alanının >%70'i örtülüyse 1.
    """
    rng = np.random.default_rng(seed)
    H, W = frames[0].shape[:2]
    out, labels = [], []
    x0, y0, w0, h0 = gt_boxes[min(start, len(gt_boxes) - 1)]
    pw, ph = int(w0 * 1.6) + 20, int(h0 * 1.6) + 20
    tex = _texture(ph, pw, rng, kind)
    for t, (f, g) in enumerate(zip(frames, gt_boxes)):
        f = f.copy(); lab = 0
        if start <= t < start + length:
            if mode == "follow":
                px, py = int(g[0] + g[2] / 2 - pw / 2), int(g[1] + g[3] / 2 - ph / 2)
            else:
                px, py = int(x0 + w0 / 2 - pw / 2), int(y0 + h0 / 2 - ph / 2)
            xa, ya, xb, yb = max(px, 0), max(py, 0), min(px + pw, W), min(py + ph, H)
            if xb > xa and yb > ya:
                tx, ty = xa - px, ya - py
                patch = tex[ty:ty + (yb - ya), tx:tx + (xb - xa)]
                roi = f[ya:yb, xa:xb]
                f[ya:yb, xa:xb] = cv2.addWeighted(patch, alpha, roi, 1 - alpha, 0)
                ix = max(0, min(xb, g[0] + g[2]) - max(xa, g[0]))
                iy = max(0, min(yb, g[1] + g[3]) - max(ya, g[1]))
                cover = (ix * iy) / max(1.0, g[2] * g[3])
                lab = int(cover > 0.7 and alpha >= 0.6)
        out.append(f); labels.append(lab)
    return out, labels
