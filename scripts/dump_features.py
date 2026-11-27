"""Takipçiyi videolar üzerinde serbest koştur, kare başına özellik + etiket kaydet.
Etiket: 1 = takipçi kutusu doğru (IoU>=0.5) VE hedef görünür; 0 = aksi.  Kullanım: python scripts/dump_features.py vit.onnx 60 out.npz"""
import sys, numpy as np, cv2
sys.path.insert(0, ".")
from core.backbone import OpenCVTracker
from core.features import frame_features, window, ncc_to_template
from utils.bbox_tools import iou
from utils.synthetic_scenes import make_scene
from utils.synthetic_occlusion import apply_occlusion

def extract(frames, init_box, tracker, gts=None, vis_label=None, K=8):
    """gts: kare başına GT kutu ya da None (hedef görünmüyor). Döner: X (N,K,5), y, ok_mask, score."""
    tracker.init(frames[0], init_box)
    x, y, w, h = [int(v) for v in init_box]; templ = frames[0][y:y + h, x:x + w].copy()
    feats, X, Y, OK, S = [], [], [], [], []
    prev, last = None, list(init_box)
    for t in range(1, len(frames)):
        ok, b, s = tracker.update(frames[t])
        box = b if ok else last
        n = ncc_to_template(frames[t], box, templ) if ok and box[2] > 1 else -1.0
        f = frame_features(s, box, prev, n); feats.append(f); prev = box
        if ok: last = box
        X.append(window(feats, K)); OK.append(ok); S.append(s)
        g = gts[t] if gts is not None else None
        lab = 0
        if ok and g is not None and (vis_label is None or vis_label[t] == 0): lab = int(iou(box, g) >= 0.5)
        Y.append(lab)
    return np.array(X), np.array(Y), np.array(OK), np.array(S)

if __name__ == "__main__":
    model, nvid, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    Xs, Ys, Os, Ss, G = [], [], [], [], []
    rng = np.random.default_rng(123)
    for v in range(nvid):
        fr, gt = make_scene(1000 + v)
        kind = str(rng.choice(["texture", "noise", "smoke"])); mode = str(rng.choice(["static", "follow"]))
        length = int(rng.integers(8, 45)); start = int(rng.integers(30, 90))
        fr, lab = apply_occlusion(fr, gt, start, length, kind, mode, int(v), alpha=0.85 if kind == "smoke" else 1.0)
        X, Y, O, S = extract(fr, gt[0], OpenCVTracker("vit", model), gt, lab)
        Xs.append(X); Ys.append(Y); Os.append(O); Ss.append(S); G.append(np.full(len(Y), v))
        if v % 10 == 0: print("video", v, "poz oranı", Y.mean().round(2), flush=True)
    np.savez(out, X=np.concatenate(Xs), y=np.concatenate(Ys), ok=np.concatenate(Os), score=np.concatenate(Ss), vid=np.concatenate(G))
    print("kaydedildi", out)
