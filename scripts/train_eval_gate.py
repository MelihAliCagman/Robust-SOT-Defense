"""Güven ağını sentetik veride eğitir (video bazında eğitim/doğrulama/test) ve ölçer.

Ardından bir videoda, kırmızı hedefin renk maskesinden çıkan YAKLAŞIK etiketle sınar.
Kullanım: python scripts/train_eval_gate.py feats_synth.npz vit.onnx video.mp4 START X Y W H
Çıktı: confidence_net.pkl ve gate_thresholds.npz (doğrulamada seçilen eşikler; compare_gates.py okur).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backbone import OpenCVTracker
from core.confidence_net import ConfidenceNet
from scripts.dump_features import extract
from utils.pseudo_labels import red_boxes
from utils.video_io import read_frames


def split_by_video(vid):
    """Video bazında yaklaşık %60 eğitim / %20 doğrulama / %20 test maskeleri."""
    return vid % 5 >= 2, vid % 5 == 1, vid % 5 == 0


def best_threshold(scores, labels):
    """F1'i en yüksek yapan eşik; adaylar skor dağılımının yüzdelikleridir."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    return max(candidates, key=lambda t: f1_score(labels, scores >= t))


def train_on_synthetic(feature_file):
    """Eğitir, eşikleri doğrulamada seçer, testte raporlar. Döner: (ağ, skor eşiği, ağ eşiği)."""
    data = np.load(feature_file)
    tracked = data["ok"].astype(bool)  # takipçinin başarısız olduğu kareler değerlendirilmez
    X, y, score, vid = (data[key][tracked] for key in ("X", "y", "score", "vid"))

    train, val, test = split_by_video(vid)
    print(f"kare sayısı: eğitim {train.sum()}, doğrulama {val.sum()}, test {test.sum()}, "
          f"eğitim pozitif oranı {y[train].mean():.2f}")
    # erken durdurma parçası eğitim videolarının içinden, yine video bazında ayrılır
    net = ConfidenceNet().fit(X[train], y[train], groups=vid[train])
    net.save("confidence_net.pkl")
    print(f"erken durdurma: en iyi epoch {net.best_epoch}")

    score_thr = best_threshold(score[val], y[val])
    net_thr = best_threshold(net.predict_proba(X[val]), y[val])
    p_test, y_test, score_test = net.predict_proba(X[test]), y[test], score[test]
    print(f"[sentetik test] AUROC skor={roc_auc_score(y_test, score_test):.3f} "
          f"ağ={roc_auc_score(y_test, p_test):.3f} | "
          f"F1 skor-eşik={f1_score(y_test, score_test >= score_thr):.3f} "
          f"ağ={f1_score(y_test, p_test >= net_thr):.3f}")
    return net, score_thr, net_thr


def evaluate_on_video(net, score_thr, net_thr, frames, init_box, tracker):
    """Videoda kare düzeyinde karşılaştırma (yaklaşık etiketle)."""
    X, y, ok, score = extract(frames, init_box, tracker, red_boxes(frames), None)
    tracked = ok.astype(bool)
    X, y, score = X[tracked], y[tracked], score[tracked]
    probability = net.predict_proba(X)
    print(f"[GERÇEK video] kare {len(y)}, pozitif oranı {y.mean():.2f}")
    print(f"  AUROC skor={roc_auc_score(y, score):.3f} ağ={roc_auc_score(y, probability):.3f}")
    print(f"  F1 (sentetik doğrulama eşikleriyle) skor-eşik={f1_score(y, score >= score_thr):.3f} "
          f"ağ={f1_score(y, probability >= net_thr):.3f}  (eşikler: skor {score_thr:.2f}, ağ {net_thr:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Güven ağını eğitir ve ölçer.")
    parser.add_argument("features", help="dump_features.py çıktısı (.npz)")
    parser.add_argument("model", help="vit.onnx dosyasının yolu")
    parser.add_argument("video", help="kırmızı hedefli, kutu çizilmemiş video")
    parser.add_argument("start", type=int, help="kutunun verildiği kare")
    parser.add_argument("box", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()

    net, score_thr, net_thr = train_on_synthetic(args.features)
    frames = read_frames(args.video, start=args.start)
    evaluate_on_video(net, score_thr, net_thr, frames, args.box, OpenCVTracker("vit", args.model))
    np.savez("gate_thresholds.npz", thr_s=score_thr, thr_p=net_thr)


if __name__ == "__main__":
    main()
