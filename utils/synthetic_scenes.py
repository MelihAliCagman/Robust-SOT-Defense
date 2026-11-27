"""Eğitim için rastgele sentetik sahneler: hareketli kamera, dokulu hedef, bazen benzer bir "çeldirici".

Dikkat: aynı tohumun aynı sahneyi üretmesi için rng çağrılarının SIRASI değiştirilmemelidir.
"""
import cv2
import numpy as np

CAMERA_RANGE = 500  # arka plan görüntüden bu kadar geniştir; kamera bu aralıkta kayar


def _random_color(rng):
    return tuple(int(v) for v in rng.integers(0, 255, 3))


def _make_background(rng, width, height):
    noise = rng.integers(0, 255, (height, width + CAMERA_RANGE, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), float(rng.uniform(3, 10)))
    contrast, brightness = rng.uniform(0.6, 1.2), rng.uniform(-30, 30)
    return np.clip(blurred.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)


def _make_object_texture(rng, size, width=None):
    """Çerçeveli ve ortasında daire olan dokulu nesne. width verilmezse rastgele seçilir."""
    if width is None:
        width = int(size * rng.uniform(0.8, 2.2))
    texture = cv2.GaussianBlur(rng.integers(30, 255, (size, width, 3), dtype=np.uint8), (0, 0), 1.2)
    cv2.rectangle(texture, (0, 0), (texture.shape[1] - 1, texture.shape[0] - 1), _random_color(rng), 3)
    cv2.circle(texture, (texture.shape[1] // 2, texture.shape[0] // 2), size // 5, _random_color(rng), -1)
    return texture


def make_scene(seed, n=150, W=640, H=360):
    """n karelik bir sahne üretir. Döner: (kareler, hedefin gerçek kutuları [x, y, w, h])."""
    rng = np.random.default_rng(seed)
    background = _make_background(rng, W, H)

    size = int(rng.integers(40, 85))
    target = _make_object_texture(rng, size)
    th, tw = target.shape[:2]

    # çeldirici hedefe BENZER ama aynı değil: %70 hedef dokusu + %30 aynı boyutta yeni doku
    has_distractor = rng.random() < 0.5
    distractor = cv2.addWeighted(target, 0.7, _make_object_texture(rng, size, tw), 0.3, 0) if has_distractor else None

    # hedef hareketi: yatayda sabit hız, düşeyde sinüs salınımı
    speed = rng.uniform(1.5, 6.0) * rng.choice([-1, 1])
    amplitude = rng.uniform(10, 50)
    period = rng.uniform(10, 25)
    camera_speed = rng.uniform(-1.0, 2.0)
    start_x = int(rng.integers(60, 300) if speed > 0 else rng.integers(250, 480))
    start_y = int(rng.integers(100, H - 100 - th))
    # çeldirici hedefin tersi yönde gider
    distractor_speed = -speed * rng.uniform(0.5, 1.2)
    distractor_x = int(rng.integers(0, 400))
    distractor_y = int(rng.integers(20, H - 20 - th))

    frames, gts = [], []
    for t in range(n):
        camera_x = int(np.clip(150 + camera_speed * t, 0, CAMERA_RANGE))
        frame = background[:, camera_x:camera_x + W].copy()
        if has_distractor:
            dx = int(distractor_x + distractor_speed * t) - camera_x
            if 0 <= dx < W - tw and 0 <= distractor_y < H - th:
                frame[distractor_y:distractor_y + th, dx:dx + tw] = distractor
        world_x = start_x + speed * t
        x = int(np.clip(int(world_x) - camera_x, 0, W - tw))
        y = int(np.clip(int(start_y + amplitude * np.sin(t / period)), 0, H - th))
        frame[y:y + th, x:x + tw] = target  # hedef en son çizilir (çeldiricinin üstünde)
        frames.append(frame)
        gts.append([x, y, tw, th])
    return frames, gts
