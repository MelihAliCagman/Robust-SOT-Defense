"""A1 (skor eşiği) vs A2 (öğrenilmiş ağ) uçtan uca. Sentetik held-out + gerçek video (sözde-GT)."""
import sys, numpy as np, cv2
sys.path.insert(0, ".")
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.pipeline import run_baseline, run_pipeline
from core.confidence_net import ConfidenceNet, LearnedGate
from evaluation.metrics import per_frame, success_auc, precision_at, recovery_rate
from utils.synthetic_scenes import make_scene
from utils.synthetic_occlusion import apply_occlusion
from scripts.train_eval_gate import red_boxes

def evaluate(fr, gts, labels, model, net, cfg, thr_p):
    vis = [i for i, g in enumerate(gts) if g is not None]
    res = {}
    for name, fn in (("A0 baseline", lambda: run_baseline(fr, gts[0], OpenCVTracker("vit", model))),
                     ("A3 eşik-kapı", lambda: run_pipeline(fr, gts[0], OpenCVTracker("vit", model), cfg)),
                     ("A3 öğrenilmiş-kapı", lambda: run_pipeline(fr, gts[0], OpenCVTracker("vit", model), cfg, gate=LearnedGate(net, thr_p)))):
        r = fn(); I, C = per_frame([r[i]['box'] for i in vis], [gts[i] for i in vis])
        full = np.zeros(len(gts)); full[vis] = I
        res[name] = (success_auc(I), precision_at(C), recovery_rate(full, labels))
    return res

if __name__ == "__main__":
    model, vid, st = sys.argv[1], sys.argv[2], int(sys.argv[3]); box = [float(v) for v in sys.argv[4:8]]
    net = ConfidenceNet.load("confidence_net.pkl"); cfg = Config()
    th = np.load("gate_thresholds.npz")  # train_eval_gate.py'nin doğrulamada seçtiği eşikler (iki kapı için de)
    cfg.score_thr, thr_p = float(th["thr_s"]), float(th["thr_p"])
    print(f"eşikler: skor {cfg.score_thr:.2f}, ağ {thr_p:.2f}")
    agg = {}
    rng = np.random.default_rng(7); NV = int(sys.argv[8]) if len(sys.argv) > 8 else 12
    for v in range(NV):
        fr, gt = make_scene(5000 + v)
        kind = str(rng.choice(["texture", "noise", "smoke"])); mode = str(rng.choice(["static", "follow"]))
        fr, lab = apply_occlusion(fr, gt, int(rng.integers(35, 80)), int(rng.integers(15, 45)), kind, mode, v, alpha=0.85 if kind == "smoke" else 1.0)
        for k, val in evaluate(fr, gt, lab, model, net, cfg, thr_p).items(): agg.setdefault(k, []).append(val)
    print(f"[sentetik, {NV} yeni video, ortalama]  (AUC, P@20, Recovery)")
    for k, v in agg.items(): print(f"  {k:20s} AUC={np.nanmean([a[0] for a in v]):.3f} P@20={np.nanmean([a[1] for a in v]):.3f} Rec={np.nanmean([a[2] for a in v]):.2f}")
    cap = cv2.VideoCapture(vid); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    fr = fr[st:]; gts = red_boxes(fr); lab = [1 if g is None else 0 for g in gts]
    print("[gerçek video (sözde-GT)]")
    for k, v in evaluate(fr, [g if g else None for g in gts], lab, model, net, cfg, thr_p).items():
        print(f"  {k:20s} AUC={v[0]:.3f} P@20={v[1]:.3f} Rec={v[2]:.2f}")
