"""GPU gerektirmeyen kurulum testi: sabit bir sentetik sahne + sentetik kapanma.

Kullanım: python scripts/selftest.py vit.onnx
Çıktı: üç sistemin metrikleri ve selftest_out.mp4 (yan yana video).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.pipeline import run_baseline, run_pipeline
from evaluation.metrics import per_frame, precision_at, recovery_rate, success_auc
from utils.drawing import draw_result
from utils.synthetic_occlusion import apply_occlusion

WIDTH, HEIGHT, TARGET_SIZE = 640, 360, 60
BASELINE, FULL_SYSTEM = "A0 baseline", "A3 +yeniden yakalama"


def make_selftest_scene(n=140, seed=0):
    """Sabit arka plan üzerinde dalgalı yol izleyen kare hedef. Döner: (kareler, gerçek kutular)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(noise, (0, 0), 6)
    target = cv2.GaussianBlur(rng.integers(40, 255, (TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8), (0, 0), 1.5)
    cv2.rectangle(target, (0, 0), (59, 59), (0, 0, 255), 3)
    cv2.circle(target, (30, 30), 12, (255, 255, 0), -1)
    frames, gts = [], []
    for t in range(n):
        x, y = int(40 + t * 3.5), int(150 + 30 * np.sin(t / 15))
        frame = background.copy()
        frame[y:y + TARGET_SIZE, x:x + TARGET_SIZE] = target
        frames.append(frame)
        gts.append([x, y, TARGET_SIZE, TARGET_SIZE])
    return frames, gts


def write_comparison_video(path, frames, results):
    """Solda takipçi tek başına, sağda tam sistem."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH * 2, HEIGHT))
    for i, frame in enumerate(frames):
        panels = [draw_result(frame, results[name][i], name, (10, 25), 0.7) for name in (BASELINE, FULL_SYSTEM)]
        writer.write(np.hstack(panels))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Sentetik sahnede kurulum testi.")
    parser.add_argument("model", help="vit.onnx dosyasının yolu")
    args = parser.parse_args()

    frames, gts = make_selftest_scene()
    frames, labels = apply_occlusion(frames, gts, start=45, length=30, kind="texture", mode="follow", seed=1)
    cfg = Config()

    def new_tracker():
        return OpenCVTracker("vit", args.model)

    results = {
        BASELINE: run_baseline(frames, gts[0], new_tracker()),
        "A1 skor+Kalman": run_pipeline(frames, gts[0], new_tracker(), cfg, redetect=False),
        FULL_SYSTEM: run_pipeline(frames, gts[0], new_tracker(), cfg, redetect=True),
    }
    print(f"kapanma kareleri: {sum(labels)}/{len(labels)}")
    for name, result in results.items():
        ious, errors = per_frame([r["box"] for r in result], gts)
        print(f"{name:24s} AUC={success_auc(ious):.3f}  P@20={precision_at(errors):.3f}  "
              f"Recovery={recovery_rate(ious, labels):.2f}")
    write_comparison_video("selftest_out.mp4", frames, results)


if __name__ == "__main__":
    main()
