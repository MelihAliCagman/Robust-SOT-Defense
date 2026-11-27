"""Rastgele sentetik sahneler (eğitim verisi için): hareketli kamera, dokulu hedef, bazen benzer bir 'çeldirici'."""
import numpy as np, cv2

def make_scene(seed, n=150, W=640, H=360):
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 500, 3), dtype=np.uint8), (0, 0), float(rng.uniform(3, 10)))
    big = np.clip(big.astype(np.float32) * rng.uniform(0.6, 1.2) + rng.uniform(-30, 30), 0, 255).astype(np.uint8)
    sz = int(rng.integers(40, 85))
    def make_tex(width=None):
        if width is None: width = int(sz * rng.uniform(0.8, 2.2))
        t = cv2.GaussianBlur(rng.integers(30, 255, (sz, width, 3), dtype=np.uint8), (0, 0), 1.2)
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), tuple(int(v) for v in rng.integers(0, 255, 3)), 3)
        cv2.circle(t, (t.shape[1] // 2, t.shape[0] // 2), sz // 5, tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        return t
    tex = make_tex(); th, tw = tex.shape[:2]
    use_d = rng.random() < 0.5
    # çeldirici hedefe BENZER ama aynı değil: %70 hedef dokusu + %30 aynı boyutta yeni doku
    dtex = cv2.addWeighted(tex, 0.7, make_tex(tw), 0.3, 0) if use_d else None
    speed = rng.uniform(1.5, 6.0) * rng.choice([-1, 1]); amp = rng.uniform(10, 50); per = rng.uniform(10, 25)
    cam_v = rng.uniform(-1.0, 2.0); wx0 = int(rng.integers(60, 300) if speed > 0 else rng.integers(250, 480))
    y0 = int(rng.integers(100, H - 100 - th))
    dspeed = -speed * rng.uniform(0.5, 1.2); dx0 = int(rng.integers(0, 400)); dy = int(rng.integers(20, H - 20 - th))
    frames, gts = [], []
    for t in range(n):
        off = int(np.clip(150 + cam_v * t, 0, 500))
        f = big[:, off:off + W].copy()
        if use_d:
            dx = int(dx0 + dspeed * t) - off; 
            if 0 <= dx < W - tw and 0 <= dy < H - th: f[dy:dy + th, dx:dx + tw] = dtex
        wx = wx0 + speed * t; y = int(y0 + amp * np.sin(t / per)); x = int(wx) - off
        x = int(np.clip(x, 0, W - tw)); y = int(np.clip(y, 0, H - th))
        f[y:y + th, x:x + tw] = tex
        frames.append(f); gts.append([x, y, tw, th])
    return frames, gts
