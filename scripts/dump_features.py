"""Takipçiyi sentetik videolarda serbest koşturup kare başına özellik ve etiket kaydeder.

Etiket 1: takipçi kutusu doğru (IoU >= 0.5) VE hedef görünür; aksi halde 0.
Kullanım: python scripts/dump_features.py vit.onnx 70 feats_synth.npz
Takipçiyi koşturmak pahalı adımdır; bir kez yapılır, eğitim kayıtlı dosyadan yürür.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backbone import OpenCVTracker
from core.features import frame_features, ncc_to_template, window
from utils.bbox_tools import iou
from utils.synthetic_occlusion import KINDS, MODES, alpha_for, apply_occlusion
from utils.synthetic_scenes import make_scene

CORRECT_IOU = 0.5
FIRST_SCENE_SEED = 1000  # karşılaştırma betiğinin tohumlarıyla (5000+) çakışmaz


def extract(frames, init_box, tracker, gts=None, vis_label=None, K=8):
    """Bir videodaki özellikleri çıkarır. Kutular [x, y, w, h].

    gts: kare başına gerçek kutu ya da None (hedef görünmüyor).
    vis_label: kare başına 1 = kapalı, 0 = görünür.
    Döner: X (N, K, 5), y, takipçi_başarılı_mı, skor. N = kare sayısı - 1.
    """
    tracker.init(frames[0], init_box)
    x, y, w, h = [int(v) for v in init_box]
    templ = frames[0][y:y + h, x:x + w].copy()

    history, windows, labels, tracked, scores = [], [], [], [], []
    previous, last_good = None, list(init_box)
    for t in range(1, len(frames)):
        ok, box, score = tracker.update(frames[t])
        box = box if ok else last_good
        ncc = ncc_to_template(frames[t], box, templ) if ok and box[2] > 1 else -1.0
        history.append(frame_features(score, box, previous, ncc))
        previous = box
        if ok:
            last_good = box

        gt = gts[t] if gts is not None else None
        visible = gt is not None and (vis_label is None or vis_label[t] == 0)
        labels.append(int(iou(box, gt) >= CORRECT_IOU) if ok and visible else 0)
        windows.append(window(history, K))
        tracked.append(ok)
        scores.append(score)
    return np.array(windows), np.array(labels), np.array(tracked), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Güven modeli için sentetik eğitim verisi üretir.")
    parser.add_argument("model", help="vit.onnx dosyasının yolu")
    parser.add_argument("num_videos", type=int)
    parser.add_argument("output", help="çıktı .npz dosyası")
    args = parser.parse_args()

    columns = {"X": [], "y": [], "ok": [], "score": [], "vid": []}
    rng = np.random.default_rng(123)
    for video in range(args.num_videos):
        frames, gts = make_scene(FIRST_SCENE_SEED + video)
        # rng çağrı sırası veriyi belirler: tür, kip, süre, başlangıç
        kind = str(rng.choice(KINDS))
        mode = str(rng.choice(MODES))
        length = int(rng.integers(8, 45))
        start = int(rng.integers(30, 90))
        frames, occluded = apply_occlusion(frames, gts, start, length, kind, mode, video, alpha=alpha_for(kind))

        X, y, ok, score = extract(frames, gts[0], OpenCVTracker("vit", args.model), gts, occluded)
        for key, value in zip(columns, (X, y, ok, score, np.full(len(y), video))):
            columns[key].append(value)
        if video % 10 == 0:
            print("video", video, "poz oranı", y.mean().round(2), flush=True)

    np.savez(args.output, **{key: np.concatenate(values) for key, values in columns.items()})
    print("kaydedildi", args.output)


if __name__ == "__main__":
    main()
