"""Güven ağını eğit (sentetik, video bazında eğitim/doğrulama/test), sentetik testte ve GERÇEK videoda (renk maskesinden sözde-GT) ölç.
Çıktı: confidence_net.pkl ve gate_thresholds.npz (doğrulamada seçilen eşikler; compare_gates.py bunları okur).
Kullanım: python scripts/train_eval_gate.py feats_synth.npz vit.onnx video.mp4 START x y w h"""
import sys, numpy as np, cv2
sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score, f1_score
from core.confidence_net import ConfidenceNet
from core.backbone import OpenCVTracker
from scripts.dump_features import extract

def red_boxes(frames):
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (0, 120, 90), (8, 255, 255)) | cv2.inRange(hsv, (172, 120, 90), (180, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, l, st, _ = cv2.connectedComponentsWithStats(m); b = None
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if a > 300 and (b is None or a > b[4]): b = (int(x), int(y), int(w), int(h), int(a))
        out.append(list(b[:4]) if b else None)
    return out

def split_by_video(vid):
    """Video bazında ~%60 eğitim / %20 doğrulama / %20 test maskeleri."""
    return vid % 5 >= 2, vid % 5 == 1, vid % 5 == 0

def best_thr(s, y):
    ts = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    return max(ts, key=lambda t: f1_score(y, s >= t))

if __name__ == "__main__":
    d = np.load(sys.argv[1]); X, y, ok, sc, vid = d["X"], d["y"], d["ok"], d["score"], d["vid"]
    m = ok.astype(bool); X, y, sc, vid = X[m], y[m], sc[m], vid[m]
    tr, va, te = split_by_video(vid)
    print(f"kare sayısı: eğitim {tr.sum()}, doğrulama {va.sum()}, test {te.sum()}, eğitim pozitif oranı {y[tr].mean():.2f}")
    # erken durdurma doğrulaması eğitim videolarının içinden, video bazında ayrılır
    net = ConfidenceNet().fit(X[tr], y[tr], groups=vid[tr]); net.save("confidence_net.pkl")
    print(f"erken durdurma: en iyi epoch {net.best_epoch}")
    # eşikler yalnızca doğrulamada seçilir, sonuç yalnızca testte raporlanır
    thr_s = best_thr(sc[va], y[va]); thr_p = best_thr(net.predict_proba(X[va]), y[va])
    p_te = net.predict_proba(X[te])
    print(f"[sentetik test] AUROC skor={roc_auc_score(y[te], sc[te]):.3f} ağ={roc_auc_score(y[te], p_te):.3f} | F1 skor-eşik={f1_score(y[te], sc[te] >= thr_s):.3f} ağ={f1_score(y[te], p_te >= thr_p):.3f}")
    # GERÇEK video
    cap = cv2.VideoCapture(sys.argv[3]); fr = []
    while True:
        ok_, f = cap.read()
        if not ok_: break
        fr.append(f)
    st = int(sys.argv[4]); box = [float(v) for v in sys.argv[5:9]]; fr = fr[st:]
    gts = red_boxes(fr)
    Xg, Yg, Og, Sg = extract(fr, box, OpenCVTracker("vit", sys.argv[2]), gts, None)
    mg = Og.astype(bool); Xg, Yg, Sg = Xg[mg], Yg[mg], Sg[mg]
    pg = net.predict_proba(Xg)
    print(f"[GERÇEK video] kare {len(Yg)}, pozitif oranı {Yg.mean():.2f}")
    print(f"  AUROC skor={roc_auc_score(Yg, Sg):.3f} ağ={roc_auc_score(Yg, pg):.3f}")
    print(f"  F1 (sentetik doğrulama eşikleriyle) skor-eşik={f1_score(Yg, Sg >= thr_s):.3f} ağ={f1_score(Yg, pg >= thr_p):.3f}  (eşikler: skor {thr_s:.2f}, ağ {thr_p:.2f})")
    np.savez("gate_thresholds.npz", thr_s=thr_s, thr_p=thr_p)
