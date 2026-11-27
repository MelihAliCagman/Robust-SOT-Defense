import numpy as np
from utils.bbox_tools import iou, center_error

def per_frame(preds, gts):
    ious, ces = [], []
    for p, g in zip(preds, gts):
        if p[2] <= 0:
            ious.append(0.0); ces.append(1e9)
        else:
            ious.append(iou(p, g)); ces.append(center_error(p, g))
    return np.array(ious), np.array(ces)

def success_auc(ious, n=21):
    th = np.linspace(0, 1, n)
    return float(np.mean([(ious > t).mean() for t in th]))

def precision_at(ces, px=20):
    return float((ces <= px).mean())

def recovery_rate(ious, labels, within=10, thr=0.5):
    """Her kapanma bitişinden sonraki ilk `within` karede IoU>thr olan en az bir kare var mı."""
    labels = np.asarray(labels); ends = [i for i in range(1, len(labels)) if labels[i - 1] == 1 and labels[i] == 0]
    if not ends: return float('nan')
    ok = [(ious[e:e + within] > thr).any() for e in ends]
    return float(np.mean(ok))
