"""Tüm eşikler burada. Hepsi BAŞLANGIÇ değeridir; doğrulama videolarında ayarlanmalıdır."""
from dataclasses import dataclass


@dataclass
class Config:
    # güven kararı
    score_thr: float = 0.45            # skor bunun altındaysa kapanma şüphesi
    template_update_thr: float = 0.6   # aranan görüntü yalnızca skor bunun üstündeyken güncellenir

    # yeniden yakalama
    redetect_thr: float = 0.55         # şablon eşleşmesi (NCC) eşiği
    redetect_score_thr: float = 0.5    # adayda yeniden başlatılan takipçinin doğrulama eşiği
    search_base: float = 1.5           # arama penceresi = search_base * boyut + 2 * search_k * sigma + sabit pay
    search_k: float = 3.0
    max_occluded_frames: int = 300     # bu kadar kareden sonra aramaktan vazgeçilir

    # Kalman filtresi
    kalman_q_pos: float = 1.0          # süreç gürültüsü (konum)
    kalman_q_vel: float = 0.5          # süreç gürültüsü (hız)
    kalman_r: float = 4.0              # ölçüm gürültüsü (piksel^2)
    vel_decay: float = 0.97            # kapanmada kare başına hız sönümü
