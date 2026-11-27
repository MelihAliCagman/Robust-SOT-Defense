"""Güven ağını sentetik veride eğitir (video bazında eğitim/doğrulama/test) ve ölçer.

Video verilirse ayrıca o videoda, kırmızı hedefin renk maskesinden çıkan YAKLAŞIK etiketle sınar.
Kullanım:
    python scripts/train_eval_gate.py feats_synth.npz
    python scripts/train_eval_gate.py feats_synth.npz vit.onnx video.mp4 START X Y W H
Çıktı: confidence_net.pkl ve gate_thresholds.npz (doğrulamada seçilen eşikler; compare_gates.py okur).
İki dosya da video bölümünden ÖNCE kaydedilir.
Her koşu ayrıca experiments/ altında kaydedilir (ayarlar, sonuçlar, öğrenme eğrisi, ROC, model).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backbone import OpenCVTracker
from core.confidence_net import ConfidenceNet
from evaluation.experiment import Experiment
from evaluation.plots import plot_learning_curve, plot_roc
from scripts.dump_features import extract
from utils.pseudo_labels import red_boxes
from utils.video_args import USAGE, parse_video_spec
from utils.video_io import read_frames


def split_by_video(vid):
    """Video bazında yaklaşık %60 eğitim / %20 doğrulama / %20 test maskeleri."""
    return vid % 5 >= 2, vid % 5 == 1, vid % 5 == 0


def best_threshold(scores, labels):
    """F1'i en yüksek yapan eşik; adaylar skor dağılımının yüzdelikleridir."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    return max(candidates, key=lambda t: f1_score(labels, scores >= t))


def train_on_synthetic(feature_file, run):
    """Eğitir, eşikleri doğrulamada seçer, testte raporlar ve kaydeder. Döner: (ağ, skor eşiği, ağ eşiği)."""
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
    metrics = {
        "test_auroc_score": roc_auc_score(y_test, score_test),
        "test_auroc_net": roc_auc_score(y_test, p_test),
        "test_f1_score": f1_score(y_test, score_test >= score_thr),
        "test_f1_net": f1_score(y_test, p_test >= net_thr),
    }
    print(f"[sentetik test] AUROC skor={metrics['test_auroc_score']:.3f} ağ={metrics['test_auroc_net']:.3f} | "
          f"F1 skor-eşik={metrics['test_f1_score']:.3f} ağ={metrics['test_f1_net']:.3f}")

    run.log_metrics({
        "frames_train": train.sum(), "frames_val": val.sum(), "frames_test": test.sum(),
        "positive_rate_train": y[train].mean(), "best_epoch": net.best_epoch, "epochs_run": len(net.history),
        "threshold_score": score_thr, "threshold_net": net_thr, **metrics,
    })
    run.log_table("history", net.history)
    plot_learning_curve(net.history, run.path("learning_curve.png"), net.best_epoch)
    plot_roc(y_test, {"skor": score_test, "ağ": p_test}, run.path("roc_test.png"))
    return net, score_thr, net_thr


def evaluate_on_video(net, score_thr, net_thr, frames, init_box, tracker, run):
    """Videoda kare düzeyinde karşılaştırma (yaklaşık etiketle)."""
    X, y, ok, score = extract(frames, init_box, tracker, red_boxes(frames), None)
    tracked = ok.astype(bool)
    X, y, score = X[tracked], y[tracked], score[tracked]
    probability = net.predict_proba(X)
    print(f"[GERÇEK video] kare {len(y)}, pozitif oranı {y.mean():.2f}")
    print(f"  AUROC skor={roc_auc_score(y, score):.3f} ağ={roc_auc_score(y, probability):.3f}")
    print(f"  F1 (sentetik doğrulama eşikleriyle) skor-eşik={f1_score(y, score >= score_thr):.3f} "
          f"ağ={f1_score(y, probability >= net_thr):.3f}  (eşikler: skor {score_thr:.2f}, ağ {net_thr:.2f})")
    run.log_metrics({"video_pseudo_labels": {
        "frames": len(y), "positive_rate": y.mean(),
        "auroc_score": roc_auc_score(y, score), "auroc_net": roc_auc_score(y, probability),
        "f1_score": f1_score(y, score >= score_thr), "f1_net": f1_score(y, probability >= net_thr),
    }})


def main():
    parser = argparse.ArgumentParser(description="Güven ağını eğitir ve ölçer.")
    parser.add_argument("features", help="dump_features.py çıktısı (.npz)")
    parser.add_argument("model", nargs="?", help="vit.onnx dosyasının yolu (yalnızca video bölümü için)")
    parser.add_argument("video_spec", nargs="*", metavar="VIDEO_BİLGİSİ",
                        help=f"isteğe bağlı, 6 değer: {USAGE} (kırmızı hedefli, kutu çizilmemiş video)")
    parser.add_argument("--run_name", default="train_gate", help="deney kaydının adı")
    parser.add_argument("--log_dir", help="deney kayıtlarının kök klasörü (varsayılan: experiments/)")
    args = parser.parse_args()
    video = parse_video_spec(parser, args.video_spec)  # hatalı argüman eğitimden önce yakalanır

    model = ConfidenceNet()  # ayarların kaydı için varsayılan değerler
    params = {"model": "MLP", "window": model.K, "hidden": model.m.hidden_layer_sizes, "alpha_l2": model.m.alpha,
              "max_epochs": model.max_epochs, "patience": model.patience, "val_frac": model.val_frac,
              "seed": model.seed, "split": "vid % 5: 0 test, 1 doğrulama, 2-4 eğitim"}
    with Experiment(args.run_name, params, args.log_dir) as run:
        run.add_input("features", args.features)
        net, score_thr, net_thr = train_on_synthetic(args.features, run)
        np.savez("gate_thresholds.npz", thr_s=score_thr, thr_p=net_thr)
        print(f"kaydedildi: confidence_net.pkl, gate_thresholds.npz  (eşikler: skor {score_thr:.2f}, ağ {net_thr:.2f})")
        # her koşunun modeli saklanır; sonraki koşu üzerine yazsa da kaybolmaz
        net.save(run.path("confidence_net.pkl"))
        np.savez(run.path("gate_thresholds.npz"), thr_s=score_thr, thr_p=net_thr)
        print(f"deney kaydı: {run.dir}")

        if video is not None:
            run.add_input("video", video.path)
            frames = read_frames(video.path, start=video.start)
            tracker = OpenCVTracker("vit", args.model)
            evaluate_on_video(net, score_thr, net_thr, frames, video.box, tracker, run)


if __name__ == "__main__":
    main()
