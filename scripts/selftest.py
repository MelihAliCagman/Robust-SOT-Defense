"""GPU'suz kendi kendine test: sentetik sahne + sentetik kapanma. Kullanım: python scripts/selftest.py vit.onnx"""
import sys, numpy as np, cv2
sys.path.insert(0, ".")
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.pipeline import run_baseline, run_pipeline
from utils.synthetic_occlusion import apply_occlusion
from evaluation.metrics import per_frame, success_auc, precision_at, recovery_rate

def make_scene(n=140, seed=0):
    rng = np.random.default_rng(seed); H, W = 360, 640
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    tex = cv2.GaussianBlur(rng.integers(40, 255, (60, 60, 3), dtype=np.uint8), (0, 0), 1.5)
    cv2.rectangle(tex, (0, 0), (59, 59), (0, 0, 255), 3); cv2.circle(tex, (30, 30), 12, (255, 255, 0), -1)
    fr, gt = [], []
    for t in range(n):
        x, y = int(40 + t * 3.5), int(150 + 30 * np.sin(t / 15))
        f = bg.copy(); f[y:y + 60, x:x + 60] = tex; fr.append(f); gt.append([x, y, 60, 60])
    return fr, gt

if __name__ == "__main__":
    model = sys.argv[1]
    frames, gts = make_scene()
    frames, labels = apply_occlusion(frames, gts, start=45, length=30, kind="texture", mode="follow", seed=1)
    cfg = Config()
    mk = lambda: OpenCVTracker("vit", model)
    res = {"A0 baseline": run_baseline(frames, gts[0], mk()),
           "A1 skor+Kalman": run_pipeline(frames, gts[0], mk(), cfg, redetect=False),
           "A3 +yeniden yakalama": run_pipeline(frames, gts[0], mk(), cfg, redetect=True)}
    print(f"kapanma kareleri: {sum(labels)}/{len(labels)}")
    for k, r in res.items():
        ious, ces = per_frame([x['box'] for x in r], gts)
        print(f"{k:24s} AUC={success_auc(ious):.3f}  P@20={precision_at(ces):.3f}  Recovery={recovery_rate(ious, labels):.2f}")
    # yan yana video
    vw = cv2.VideoWriter("selftest_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (640 * 2, 360))
    for i, f in enumerate(frames):
        panels = []
        for k in ("A0 baseline", "A3 +yeniden yakalama"):
            p = f.copy(); d = res[k][i]; b = [int(v) for v in d['box']]
            col = (0, 200, 0) if d['state'] == 0 else (0, 0, 255)
            if b[2] > 0: cv2.rectangle(p, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), col, 2)
            cv2.putText(p, k, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(p)
        vw.write(np.hstack(panels))
    vw.release()
