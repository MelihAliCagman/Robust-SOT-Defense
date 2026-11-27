"""Videolara sentetik kapanma basar. Düz siyah kutu değil: dokulu yama, gürültü, yarı saydam duman."""
import cv2
import numpy as np

KINDS = ("texture", "noise", "smoke")
MODES = ("static", "follow")

COVER_THRESHOLD = 0.7  # hedef alanının bu orandan fazlası örtülüyse kare "kapalı" sayılır
MIN_OPAQUE_ALPHA = 0.6  # örtücü bundan saydamsa kare "kapalı" sayılmaz


def alpha_for(kind):
    """Eğitim ve değerlendirmede kullanılan opaklık: duman yarı saydam, diğerleri tam opak."""
    return 0.85 if kind == "smoke" else 1.0


def _texture(h, w, rng, kind):
    if kind == "texture":
        noise = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (0, 0), 3)
    if kind == "smoke":
        base = np.full((h, w, 3), 200, np.uint8)
        variation = cv2.GaussianBlur(rng.integers(0, 80, (h, w, 3), dtype=np.uint8), (0, 0), 8)
        return cv2.add(base, variation)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)  # noise


def apply_occlusion(frames, gt_boxes, start, length, kind="texture", mode="static", seed=0, alpha=1.0):
    """[start, start + length) aralığındaki karelere örtücü yama basar.

    frames: list[H, W, 3]; gt_boxes: list[[x, y, w, h]]
    mode="static": yama, start karesindeki hedefin üstünde sabit durur (hedeften büyüktür).
    mode="follow": yama hedefle birlikte hareket eder.
    alpha < 1: yarı saydam.
    Döner: (yeni_kareler, etiketler); etiket 1 = kapalı, 0 = görünür.
    """
    rng = np.random.default_rng(seed)
    height, width = frames[0].shape[:2]
    x0, y0, w0, h0 = gt_boxes[min(start, len(gt_boxes) - 1)]
    patch_w, patch_h = int(w0 * 1.6) + 20, int(h0 * 1.6) + 20
    texture = _texture(patch_h, patch_w, rng, kind)

    occluded_frames, labels = [], []
    for t, (frame, gt) in enumerate(zip(frames, gt_boxes)):
        frame, label = frame.copy(), 0
        if start <= t < start + length:
            # yamanın sol üst köşesi: hedefi (ya da başlangıç konumunu) ortalar
            tx, ty, tw, th = gt if mode == "follow" else (x0, y0, w0, h0)
            px, py = int(tx + tw / 2 - patch_w / 2), int(ty + th / 2 - patch_h / 2)
            # görüntü içinde kalan bölüm
            left, top = max(px, 0), max(py, 0)
            right, bottom = min(px + patch_w, width), min(py + patch_h, height)
            if right > left and bottom > top:
                patch = texture[top - py:bottom - py, left - px:right - px]
                region = frame[top:bottom, left:right]
                frame[top:bottom, left:right] = cv2.addWeighted(patch, alpha, region, 1 - alpha, 0)
                overlap_w = max(0, min(right, gt[0] + gt[2]) - max(left, gt[0]))
                overlap_h = max(0, min(bottom, gt[1] + gt[3]) - max(top, gt[1]))
                cover = (overlap_w * overlap_h) / max(1.0, gt[2] * gt[3])
                label = int(cover > COVER_THRESHOLD and alpha >= MIN_OPAQUE_ALPHA)
        occluded_frames.append(frame)
        labels.append(label)
    return occluded_frames, labels
