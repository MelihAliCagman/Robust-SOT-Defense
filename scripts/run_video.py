"""Bir videoda takipçi tek başına ile bizim sistemi karşılaştırır.

Kullanım:
    python scripts/run_video.py --video v.mp4 --box X Y W H --model vit.onnx [--start 0] [--out out]
--video bir jpg/png kare klasörü de olabilir (OTB biçimi).
Çıktılar: <out>_side_by_side.mp4, <out>_score.png, <out>.csv
Gerçek etiket kullanılmaz; çıktılar görsel karşılaştırma içindir.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # pencere açmadan dosyaya çizim
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.pipeline import OCCLUDED, run_baseline, run_pipeline
from utils.drawing import draw_result
from utils.video_io import read_frames

OUTPUT_FPS = 25
SYSTEM_NAME = "Bizim sistem"


def parse_args():
    parser = argparse.ArgumentParser(description="Takipçi tek başına ve bizim sistem, yan yana.")
    parser.add_argument("--video", required=True, help="video dosyası veya kare klasörü")
    parser.add_argument("--box", type=float, nargs=4, required=True, metavar=("X", "Y", "W", "H"),
                        help="başlangıç karesindeki hedef kutusu")
    parser.add_argument("--model", default="vit.onnx")
    parser.add_argument("--start", type=int, default=0, help="kutunun verildiği kare (öncesi atlanır)")
    parser.add_argument("--max_frames", type=int, default=400)
    parser.add_argument("--scale", type=float, default=1.0, help="kareleri ve kutuyu ölçekler")
    parser.add_argument("--out", default="out", help="çıktı dosyalarının ön adı")
    return parser.parse_args()


def panel_title(name, result):
    if name == SYSTEM_NAME and result["state"] == OCCLUDED:
        return f"{name}  KAPALI (takipci durduruldu, Kalman tahmini)"  # OpenCV Türkçe harf çizemez
    return f"{name}  skor={result['score']:.2f}"


def write_video(path, frames, baseline, system):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (width * 2, height))
    for i, frame in enumerate(frames):
        panels = [draw_result(frame, results[i], panel_title(name, results[i]), (10, 28), 0.8)
                  for name, results in (("Baseline", baseline), (SYSTEM_NAME, system))]
        writer.write(np.hstack(panels))
    writer.release()


def plot_scores(path, baseline, system):
    """Takipçi tek başınayken skoru; kırmızı gölge: bizim sistemin kapanma kararı."""
    scores = [r["score"] for r in baseline]
    occluded = np.array([r["state"] for r in system]) == OCCLUDED
    plt.figure(figsize=(9, 3))
    plt.plot(scores, label="takipçi güven skoru")
    plt.fill_between(range(len(occluded)), 0, 1, where=occluded, alpha=0.2, color="r",
                     label="OCCLUDED (bizim sistem)")
    plt.legend()
    plt.xlabel("kare")
    plt.tight_layout()
    plt.savefig(path, dpi=120)


def write_csv(path, system):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "score", "state", "x", "y", "w", "h"])
        for i, r in enumerate(system):
            writer.writerow([i, r["score"], r["state"], *[round(v, 1) for v in r["box"]]])


def main():
    args = parse_args()
    frames = read_frames(args.video, start=args.start, max_frames=args.max_frames, scale=args.scale)
    box = [v * args.scale for v in args.box]

    def new_tracker():
        return OpenCVTracker("vit", args.model)

    baseline = run_baseline(frames, box, new_tracker())
    system = run_pipeline(frames, box, new_tracker(), Config())

    video_path, plot_path, csv_path = args.out + "_side_by_side.mp4", args.out + "_score.png", args.out + ".csv"
    write_video(video_path, frames, baseline, system)
    plot_scores(plot_path, baseline, system)
    write_csv(csv_path, system)
    print("bitti:", video_path, plot_path, csv_path)


if __name__ == "__main__":
    main()
