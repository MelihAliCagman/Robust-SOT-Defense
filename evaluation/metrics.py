"""Takip metrikleri. Kutular: [x, y, w, h]."""
import numpy as np

from utils.bbox_tools import center_error, iou

LOST_CENTER_ERROR = 1e9  # kayıp karede merkez hatası


def per_frame(preds, gts):
    """Kare başına (IoU dizisi, merkez hatası dizisi). Genişliği 0 olan tahmin "kayıp" sayılır."""
    ious, errors = [], []
    for pred, gt in zip(preds, gts):
        if pred[2] <= 0:
            ious.append(0.0)
            errors.append(LOST_CENTER_ERROR)
        else:
            ious.append(iou(pred, gt))
            errors.append(center_error(pred, gt))
    return np.array(ious), np.array(errors)


def success_auc(ious, n=21):
    """0..1 arasındaki n eşik için "IoU > eşik" oranlarının ortalaması."""
    thresholds = np.linspace(0, 1, n)
    return float(np.mean([(ious > t).mean() for t in thresholds]))


def precision_at(errors, px=20):
    """Merkez hatası en fazla px piksel olan karelerin oranı."""
    return float((errors <= px).mean())


def recovery_rate(ious, labels, within=10, thr=0.5):
    """Kapanma bitişlerinden kaçında, sonraki `within` kare içinde IoU > thr olan bir kare var.

    labels: 1 = kapalı, 0 = görünür. Hiç kapanma bitişi yoksa NaN.
    """
    labels = np.asarray(labels)
    ends = [i for i in range(1, len(labels)) if labels[i - 1] == 1 and labels[i] == 0]
    if not ends:
        return float("nan")
    return float(np.mean([(ious[end:end + within] > thr).any() for end in ends]))
