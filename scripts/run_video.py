"""Gerçek video üzerinde baseline vs bizim sistem. Çıktı: yan yana mp4 + skor grafiği (png) + csv.
Kullanım: python scripts/run_video.py --video v.mp4 (veya jpg klasörü) --box x y w h --model vit.onnx [--start 0] [--max_frames 400] [--scale 1.0] [--out out]"""
import sys, argparse, csv, numpy as np, cv2
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.pipeline import run_baseline, run_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--box", type=float, nargs=4, required=True)
ap.add_argument("--model", default="vit.onnx"); ap.add_argument("--max_frames", type=int, default=400)
ap.add_argument("--start", type=int, default=0, help="kutunun verildiği kare numarası (önceki kareler atlanır)")
ap.add_argument("--out", default="out"); ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()
import os, glob
frames = []
if os.path.isdir(a.video):  # OTB tarzı jpg kare klasörü (örn. Jogging/img)
    for p in sorted(glob.glob(os.path.join(a.video, "*.jpg")) + glob.glob(os.path.join(a.video, "*.png")))[:a.max_frames + a.start]:
        f = cv2.imread(p); frames.append(cv2.resize(f, None, fx=a.scale, fy=a.scale) if a.scale != 1 else f)
else:
    cap = cv2.VideoCapture(a.video)
    while len(frames) < a.max_frames + a.start:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(f, None, fx=a.scale, fy=a.scale) if a.scale != 1 else f)
frames = frames[a.start:]
box = [v * a.scale for v in a.box]
cfg = Config(); mk = lambda: OpenCVTracker("vit", a.model)
r0 = run_baseline(frames, box, mk()); r1 = run_pipeline(frames, box, mk(), cfg)
H, W = frames[0].shape[:2]
vw = cv2.VideoWriter(a.out + "_side_by_side.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (W * 2, H))
for i, f in enumerate(frames):
    ps = []
    for name, r in (("Baseline", r0), ("Bizim sistem", r1)):
        p = f.copy(); d = r[i]; b = [int(v) for v in d["box"]]
        col = (0, 200, 0) if d["state"] == 0 else (0, 0, 255)
        if b[2] > 0: cv2.rectangle(p, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), col, 2)
        if name != "Baseline" and d["state"] == 1:
            txt = f"{name}  KAPALI (takipci durduruldu, Kalman tahmini)"
        else:
            txt = f"{name}  skor={d['score']:.2f}"
        cv2.putText(p, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ps.append(p)
    vw.write(np.hstack(ps))
vw.release()
s0 = [d["score"] for d in r0]; occ = [d["state"] for d in r1]
plt.figure(figsize=(9, 3)); plt.plot(s0, label="takipçi güven skoru")
plt.fill_between(range(len(occ)), 0, 1, where=np.array(occ) == 1, alpha=0.2, color="r", label="OCCLUDED (bizim sistem)")
plt.legend(); plt.xlabel("kare"); plt.tight_layout(); plt.savefig(a.out + "_score.png", dpi=120)
with open(a.out + ".csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "score", "state", "x", "y", "w", "h"])
    for i, d in enumerate(r1): w.writerow([i, d["score"], d["state"], *[round(v, 1) for v in d["box"]]])
print("bitti:", a.out + "_side_by_side.mp4", a.out + "_score.png", a.out + ".csv")
