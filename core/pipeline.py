"""Durum makinesi: TRACKING / OCCLUDED. Kapanma kararı varsayılan olarak skor eşiğidir.
Öğrenilmiş güven ağı gate parametresiyle verilir."""
import numpy as np, cv2
from core.kalman_tracker import Kalman2D
from utils.bbox_tools import clip_box, center, iou
from core.features import ncc_to_template

TRACKING, OCCLUDED = 0, 1

def _crop(frame, box):
    H, W = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in clip_box(box, W, H)]
    return frame[y:y + h, x:x + w].copy()

def _match(frame, templ, center_xy, win_w, win_h):
    """Şablonu arama penceresinde eşle. Döner: (ncc, [x,y,w,h]) veya (-1, None)."""
    H, W = frame.shape[:2]
    th, tw = templ.shape[:2]
    cx, cy = center_xy
    x1, y1 = int(max(0, cx - win_w / 2)), int(max(0, cy - win_h / 2))
    x2, y2 = int(min(W, cx + win_w / 2)), int(min(H, cy + win_h / 2))
    if x2 - x1 < tw + 1 or y2 - y1 < th + 1 or tw < 4 or th < 4:
        return -1.0, None
    res = cv2.matchTemplate(frame[y1:y2, x1:x2], templ, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), [x1 + loc[0], y1 + loc[1], tw, th]

def run_baseline(frames, init_box, tracker):
    """A0: hazır takipçi. Takipçi başarısız olursa kutu [0,0,0,0] (kayıp)."""
    tracker.init(frames[0], init_box)
    out = [dict(box=list(init_box), state=TRACKING, score=1.0)]
    for f in frames[1:]:
        ok, b, s = tracker.update(f)
        out.append(dict(box=b if ok else [0, 0, 0, 0], state=TRACKING, score=s))
    return out

def run_pipeline(frames, init_box, tracker, cfg, redetect=True, gate=None):
    """A1/A3. gate(score, ncc, box, prev_box) -> True ise 'görünür'. Varsayılan: score >= cfg.score_thr."""
    tracker.init(frames[0], init_box)
    kf = Kalman2D(init_box, cfg.kalman_q_pos, cfg.kalman_q_vel, cfg.kalman_r, cfg.vel_decay)
    templ = _crop(frames[0], init_box); templ0 = templ.copy()
    if gate is not None and hasattr(gate, 'reset'): gate.reset()
    state, occ_n, last_good = TRACKING, 0, list(init_box)
    out = [dict(box=list(init_box), state=TRACKING, score=1.0, ncc=1.0)]
    for f in frames[1:]:
        if state == TRACKING:
            ok, b, s = tracker.update(f)
            visible = ok and (gate(s, ncc_to_template(f, b, templ0), b, last_good) if gate else s >= cfg.score_thr)
            if visible:
                kb = kf.predict(); kb = kf.update(b)
                last_good = b
                if s >= cfg.template_update_thr:
                    c = _crop(f, b)
                    if c.size: templ = c
                out.append(dict(box=b, state=TRACKING, score=s, ncc=0.0))
            else:
                state, occ_n = OCCLUDED, 0
                out.append(dict(box=kf.predict(damp=True), state=OCCLUDED, score=s, ncc=0.0))
        else:
            occ_n += 1
            if occ_n > cfg.max_occluded_frames:  # pes et: artık arama yok, kutu [0,0,0,0] (kayıp)
                out.append(dict(box=[0, 0, 0, 0], state=OCCLUDED, score=0.0, ncc=-1.0))
                continue
            kb = kf.predict(damp=True)
            ncc = -1.0
            if redetect:
                sg = kf.pos_sigma()
                ww = cfg.search_base * kf.w + cfg.search_k * sg * 2 + 40
                hh = cfg.search_base * kf.h + cfg.search_k * sg * 2 + 40
                ncc, cand = _match(f, templ, center(kb), ww, hh)
                if cand is not None and ncc >= cfg.redetect_thr:
                    tracker.init(f, cand)           # takipçiyi adayda yeniden başlat
                    ok, b2, s2 = tracker.update(f)  # hemen doğrula
                    if ok and s2 >= cfg.redetect_score_thr:
                        kf.update(cand); last_good = cand; state = TRACKING
                        # takipçi yeniden başladı: kapı da kapanma öncesi geçmişi unutsun
                        if gate is not None and hasattr(gate, 'reset'): gate.reset()
                        out.append(dict(box=cand, state=TRACKING, score=s2, ncc=ncc)); continue
            out.append(dict(box=kb, state=OCCLUDED, score=0.0, ncc=ncc))
    return out
