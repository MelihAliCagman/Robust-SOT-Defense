"""Skor eşiği kapısı ile öğrenilmiş kapıyı uçtan uca karşılaştırır.

Veri: eğitimde kullanılmayan sentetik videolar + bir videoda yaklaşık (sözde) etiket.
Kullanım: python scripts/compare_gates.py vit.onnx video.mp4 START X Y W H [VIDEO_SAYISI]
Gerekli: çalışma klasöründe train_eval_gate.py'nin ürettiği confidence_net.pkl ve gate_thresholds.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.confidence_net import ConfidenceNet, LearnedGate
from core.pipeline import run_baseline, run_pipeline
from evaluation.metrics import per_frame, precision_at, recovery_rate, success_auc
from utils.pseudo_labels import red_boxes
from utils.synthetic_occlusion import KINDS, MODES, alpha_for, apply_occlusion
from utils.synthetic_scenes import make_scene
from utils.video_io import read_frames

FIRST_SCENE_SEED = 5000  # eğitim tohumlarından (1000+) farklı


def evaluate(frames, gts, labels, model, net, cfg, net_thr):
    """Üç sistemi aynı videoda koşturur. gts: kutu ya da None.

    Döner: {sistem adı: (Success AUC, Precision@20, Recovery)}. Metrikler yalnızca kutusu olan karelerde.
    """
    def new_tracker():
        return OpenCVTracker("vit", model)

    runners = {
        "A0 baseline": lambda: run_baseline(frames, gts[0], new_tracker()),
        "A3 eşik-kapı": lambda: run_pipeline(frames, gts[0], new_tracker(), cfg),
        "A3 öğrenilmiş-kapı": lambda: run_pipeline(frames, gts[0], new_tracker(), cfg,
                                                   gate=LearnedGate(net, net_thr)),
    }
    visible = [i for i, gt in enumerate(gts) if gt is not None]
    scores = {}
    for name, run in runners.items():
        results = run()
        ious, errors = per_frame([results[i]["box"] for i in visible], [gts[i] for i in visible])
        all_ious = np.zeros(len(gts))  # Recovery zaman sırasına bakar: tüm kareler gerekir
        all_ious[visible] = ious
        scores[name] = (success_auc(ious), precision_at(errors), recovery_rate(all_ious, labels))
    return scores


def evaluate_synthetic(num_videos, model, net, cfg, net_thr):
    collected = {}
    rng = np.random.default_rng(7)
    for video in range(num_videos):
        frames, gts = make_scene(FIRST_SCENE_SEED + video)
        # rng çağrı sırası veriyi belirler: tür, kip, başlangıç, süre
        kind = str(rng.choice(KINDS))
        mode = str(rng.choice(MODES))
        start = int(rng.integers(35, 80))
        length = int(rng.integers(15, 45))
        frames, labels = apply_occlusion(frames, gts, start, length, kind, mode, video, alpha=alpha_for(kind))
        for name, values in evaluate(frames, gts, labels, model, net, cfg, net_thr).items():
            collected.setdefault(name, []).append(values)

    print(f"[sentetik, {num_videos} yeni video, ortalama]  (AUC, P@20, Recovery)")
    for name, rows in collected.items():
        auc, precision, recovery = (np.nanmean([row[i] for row in rows]) for i in range(3))
        print(f"  {name:20s} AUC={auc:.3f} P@20={precision:.3f} Rec={recovery:.2f}")


def evaluate_video(frames, model, net, cfg, net_thr):
    gts = red_boxes(frames)
    labels = [1 if gt is None else 0 for gt in gts]  # kırmızı görünmüyorsa "kapalı" say
    print("[gerçek video (sözde-GT)]")
    for name, (auc, precision, recovery) in evaluate(frames, gts, labels, model, net, cfg, net_thr).items():
        print(f"  {name:20s} AUC={auc:.3f} P@20={precision:.3f} Rec={recovery:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Skor eşiği ve öğrenilmiş kapıyı karşılaştırır.")
    parser.add_argument("model", help="vit.onnx dosyasının yolu")
    parser.add_argument("video", help="kırmızı hedefli, kutu çizilmemiş video")
    parser.add_argument("start", type=int, help="kutunun verildiği kare")
    # X Y W H değerleri geriye uyumluluk için alınır; kutular renk maskesinden çıkarılır
    parser.add_argument("box", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="kullanılmıyor")
    parser.add_argument("num_videos", type=int, nargs="?", default=12)
    args = parser.parse_args()

    net = ConfidenceNet.load("confidence_net.pkl")
    thresholds = np.load("gate_thresholds.npz")  # iki kapının da eşiği doğrulamada seçilmiştir
    cfg = Config()
    cfg.score_thr, net_thr = float(thresholds["thr_s"]), float(thresholds["thr_p"])
    print(f"eşikler: skor {cfg.score_thr:.2f}, ağ {net_thr:.2f}")

    evaluate_synthetic(args.num_videos, args.model, net, cfg, net_thr)
    evaluate_video(read_frames(args.video, start=args.start), args.model, net, cfg, net_thr)


if __name__ == "__main__":
    main()
