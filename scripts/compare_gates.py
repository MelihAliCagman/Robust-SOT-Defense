"""Skor eşiği kapısı ile öğrenilmiş kapıyı uçtan uca karşılaştırır.

Veri: eğitimde kullanılmayan sentetik videolar; video verilirse ayrıca onda yaklaşık (sözde) etiket.
Kullanım:
    python scripts/compare_gates.py vit.onnx [--num_videos 12]
    python scripts/compare_gates.py vit.onnx video.mp4 START X Y W H [VIDEO_SAYISI]
Gerekli: çalışma klasöründe train_eval_gate.py'nin ürettiği confidence_net.pkl ve gate_thresholds.npz
Her koşu experiments/ altında kaydedilir; per_video.csv her videonun sonucunu tek tek içerir.
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.tracker_config import Config
from core.backbone import OpenCVTracker
from core.confidence_net import ConfidenceNet, LearnedGate
from core.pipeline import run_baseline, run_pipeline
from evaluation.experiment import Experiment
from evaluation.metrics import per_frame, precision_at, recovery_rate, success_auc
from utils.pseudo_labels import red_boxes
from utils.synthetic_occlusion import KINDS, MODES, alpha_for, apply_occlusion
from utils.synthetic_scenes import make_scene
from utils.video_args import USAGE, parse_video_spec
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


def result_row(video, name, values, **extra):
    auc, precision, recovery = values
    return {"video": video, "system": name, "success_auc": auc, "precision20": precision, "recovery": recovery, **extra}


def evaluate_synthetic(num_videos, model, net, cfg, net_thr, run):
    collected, rows_per_video = {}, []
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
            rows_per_video.append(result_row(f"synthetic_{FIRST_SCENE_SEED + video}", name, values,
                                             kind=kind, mode=mode, start=start, length=length))

    print(f"[sentetik, {num_videos} yeni video, ortalama]  (AUC, P@20, Recovery)")
    means = {}
    for name, rows in collected.items():
        auc, precision, recovery = (np.nanmean([row[i] for row in rows]) for i in range(3))
        print(f"  {name:20s} AUC={auc:.3f} P@20={precision:.3f} Rec={recovery:.2f}")
        means[name] = {"success_auc": auc, "precision20": precision, "recovery": recovery}

    # ortalama yanıltabilir: sistemin takipçi tek başınadan KÖTÜ olduğu videolar ayrıca sayılır
    for name, rows in collected.items():
        if name != "A0 baseline":
            worse = sum(row[0] < base[0] for row, base in zip(rows, collected["A0 baseline"]))
            means[name]["videos_worse_than_baseline"] = worse
    run.log_table("per_video", rows_per_video)
    run.log_metrics({"synthetic_videos": num_videos, "synthetic_mean": means})


def evaluate_video(frames, model, net, cfg, net_thr, run):
    gts = red_boxes(frames)
    labels = [1 if gt is None else 0 for gt in gts]  # kırmızı görünmüyorsa "kapalı" say
    print("[gerçek video (sözde-GT)]")
    results = evaluate(frames, gts, labels, model, net, cfg, net_thr)
    for name, (auc, precision, recovery) in results.items():
        print(f"  {name:20s} AUC={auc:.3f} P@20={precision:.3f} Rec={recovery:.2f}")
    run.log_metrics({"video_pseudo_labels": {
        name: {"success_auc": auc, "precision20": precision, "recovery": recovery}
        for name, (auc, precision, recovery) in results.items()}})


def main():
    parser = argparse.ArgumentParser(description="Skor eşiği ve öğrenilmiş kapıyı karşılaştırır.")
    parser.add_argument("model", help="vit.onnx dosyasının yolu")
    parser.add_argument("video_spec", nargs="*", metavar="VIDEO_BİLGİSİ",
                        help=f"isteğe bağlı: {USAGE} [VIDEO_SAYISI]. X Y W H kullanılmaz (kutular renkten çıkar)")
    parser.add_argument("--num_videos", type=int, default=12, help="sentetik test videosu sayısı")
    parser.add_argument("--run_name", default="compare_gates", help="deney kaydının adı")
    parser.add_argument("--log_dir", help="deney kayıtlarının kök klasörü (varsayılan: experiments/)")
    args = parser.parse_args()

    values, num_videos = args.video_spec, args.num_videos
    if len(values) == 7:  # eski kullanım: video sayısı en sonda
        if not values[6].isdigit():
            parser.error("son değer (video sayısı) tam sayı olmalı")
        values, num_videos = values[:6], int(values[6])
    video = parse_video_spec(parser, values)

    net = ConfidenceNet.load("confidence_net.pkl")
    thresholds = np.load("gate_thresholds.npz")  # iki kapının da eşiği doğrulamada seçilmiştir
    cfg = Config()
    cfg.score_thr, net_thr = float(thresholds["thr_s"]), float(thresholds["thr_p"])
    print(f"eşikler: skor {cfg.score_thr:.2f}, ağ {net_thr:.2f}")

    params = {"tracker_config": asdict(cfg), "threshold_net": net_thr, "num_videos": num_videos,
              "first_scene_seed": FIRST_SCENE_SEED, "rng_seed": 7}
    with Experiment(args.run_name, params, args.log_dir) as run:
        run.add_input("net", "confidence_net.pkl")
        run.add_input("thresholds", "gate_thresholds.npz")
        run.add_input("tracker_model", args.model)
        evaluate_synthetic(num_videos, args.model, net, cfg, net_thr, run)
        if video is not None:
            run.add_input("video", video.path)
            evaluate_video(read_frames(video.path, start=video.start), args.model, net, cfg, net_thr, run)
        print(f"deney kaydı: {run.dir}")


if __name__ == "__main__":
    main()
