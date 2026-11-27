"""Takip durum makinesi.

TRACKING: takipçi çalışır. "Kapı" çıktıya güvenirse Kalman güncellenir.
OCCLUDED: takipçi durdurulur, kutu Kalman tahminidir; büyüyen pencerede hedef aranır.
          Aday bulunur ve takipçi onu doğrularsa TRACKING durumuna dönülür.

Her kare için sonuç sözlüğü: box [x, y, w, h], state, score, ncc.
Kapanma kararı varsayılan olarak skor eşiğidir; öğrenilmiş güven ağı gate parametresiyle verilir.
"""
import cv2

from core.features import ncc_to_template
from core.kalman_tracker import Kalman2D
from utils.bbox_tools import center, clip_box

TRACKING, OCCLUDED = 0, 1
LOST_BOX = [0, 0, 0, 0]  # hedef kayıp
SEARCH_MARGIN = 40       # arama penceresine eklenen sabit pay (piksel)
MIN_TEMPLATE_SIDE = 4    # bundan küçük şablonla arama yapılmaz (piksel)


def _result(box, state, score, ncc):
    return dict(box=box, state=state, score=score, ncc=ncc)


def _crop(frame, box):
    height, width = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in clip_box(box, width, height)]
    return frame[y:y + h, x:x + w].copy()


def _match(frame, templ, center_xy, win_w, win_h):
    """Şablonu arama penceresinde eşler. Döner: (ncc, [x, y, w, h]); aranamazsa (-1, None)."""
    height, width = frame.shape[:2]
    th, tw = templ.shape[:2]
    cx, cy = center_xy
    x1, y1 = int(max(0, cx - win_w / 2)), int(max(0, cy - win_h / 2))
    x2, y2 = int(min(width, cx + win_w / 2)), int(min(height, cy + win_h / 2))
    too_small_window = x2 - x1 < tw + 1 or y2 - y1 < th + 1
    if too_small_window or tw < MIN_TEMPLATE_SIDE or th < MIN_TEMPLATE_SIDE:
        return -1.0, None
    response = cv2.matchTemplate(frame[y1:y2, x1:x2], templ, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, (best_x, best_y) = cv2.minMaxLoc(response)
    return float(best_score), [x1 + best_x, y1 + best_y, tw, th]


def run_baseline(frames, init_box, tracker):
    """A0: hazır takipçi tek başına. Takipçi başarısız olursa kutu [0,0,0,0] (kayıp)."""
    tracker.init(frames[0], init_box)
    results = [dict(box=list(init_box), state=TRACKING, score=1.0)]
    for frame in frames[1:]:
        ok, box, score = tracker.update(frame)
        results.append(dict(box=box if ok else list(LOST_BOX), state=TRACKING, score=score))
    return results


def _try_reacquire(frame, tracker, kalman, templ, predicted_box, cfg):
    """Tahminin çevresinde hedefi arar; takipçiyi adayda başlatıp doğrular.

    Döner: (aday_kutu veya None, takipçi skoru, ncc). None ise doğrulama başarısızdır.
    """
    sigma = kalman.pos_sigma()
    win_w = cfg.search_base * kalman.w + cfg.search_k * sigma * 2 + SEARCH_MARGIN
    win_h = cfg.search_base * kalman.h + cfg.search_k * sigma * 2 + SEARCH_MARGIN
    ncc, candidate = _match(frame, templ, center(predicted_box), win_w, win_h)
    if candidate is None or ncc < cfg.redetect_thr:
        return None, 0.0, ncc
    tracker.init(frame, candidate)           # takipçiyi adayda yeniden başlat
    ok, _, score = tracker.update(frame)     # hemen doğrula
    if ok and score >= cfg.redetect_score_thr:
        return candidate, score, ncc
    return None, 0.0, ncc


def run_pipeline(frames, init_box, tracker, cfg, redetect=True, gate=None):
    """Güvenlik katmanlı takip (A1: redetect=False, A3: redetect=True).

    gate(score, ncc, box, prev_box) -> True ise "güvenilir".
    gate verilmezse karar score >= cfg.score_thr ile alınır.
    """
    tracker.init(frames[0], init_box)
    kalman = Kalman2D(init_box, cfg.kalman_q_pos, cfg.kalman_q_vel, cfg.kalman_r, cfg.vel_decay)
    templ = _crop(frames[0], init_box)  # yeniden yakalamada aranan görüntü (güncellenir)
    first_templ = templ.copy()          # güven özelliği için sabit ilk görüntü
    if gate is not None and hasattr(gate, "reset"):
        gate.reset()

    state, occluded_count, last_good = TRACKING, 0, list(init_box)
    results = [_result(list(init_box), TRACKING, 1.0, 1.0)]

    for frame in frames[1:]:
        if state == TRACKING:
            ok, box, score = tracker.update(frame)
            if not ok:
                trusted = False
            elif gate is not None:
                trusted = gate(score, ncc_to_template(frame, box, first_templ), box, last_good)
            else:
                trusted = score >= cfg.score_thr

            if trusted:
                kalman.predict()
                kalman.update(box)
                last_good = box
                if score >= cfg.template_update_thr:  # şablon yalnızca çok güvenilen karede güncellenir
                    crop = _crop(frame, box)
                    if crop.size:
                        templ = crop
                results.append(_result(box, TRACKING, score, 0.0))
            else:
                state, occluded_count = OCCLUDED, 0
                results.append(_result(kalman.predict(damp=True), OCCLUDED, score, 0.0))
            continue

        # kapanma durumu
        occluded_count += 1
        if occluded_count > cfg.max_occluded_frames:  # pes et: arama yok, hedef kayıp
            results.append(_result(list(LOST_BOX), OCCLUDED, 0.0, -1.0))
            continue

        predicted_box = kalman.predict(damp=True)
        candidate, score, ncc = None, 0.0, -1.0
        if redetect:
            candidate, score, ncc = _try_reacquire(frame, tracker, kalman, templ, predicted_box, cfg)

        if candidate is None:
            results.append(_result(predicted_box, OCCLUDED, 0.0, ncc))
        else:
            kalman.update(candidate)
            last_good, state = candidate, TRACKING
            if gate is not None and hasattr(gate, "reset"):
                gate.reset()  # takipçi yeniden başladı: kapı kapanma öncesi geçmişi unutsun
            results.append(_result(candidate, TRACKING, score, ncc))
    return results
