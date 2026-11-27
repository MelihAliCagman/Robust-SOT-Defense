"""Tüm eşikler burada. Hepsi BAŞLANGIÇ değeridir; doğrulama videolarından ayarlanmalıdır."""
from dataclasses import dataclass

@dataclass
class Config:
    score_thr: float = 0.45        # bu değerin altı -> kapanma şüphesi (ViT/Nano skor ölçeğine göre ayarla)
    template_update_thr: float = 0.6  # sadece skor bunun üstündeyken şablon (crop) güncellenir
    redetect_thr: float = 0.55     # şablon eşleşme (NCC) eşiği
    redetect_score_thr: float = 0.5  # yeniden başlatılan takipçinin ilk skorunun doğrulama eşiği
    kalman_q_pos: float = 1.0      # süreç gürültüsü (konum)
    kalman_q_vel: float = 0.5      # süreç gürültüsü (hız)
    kalman_r: float = 4.0          # ölçüm gürültüsü (piksel^2)
    vel_decay: float = 0.97        # kapanmada hız sönümü (kare başına)
    search_base: float = 1.5       # arama penceresi = search_base*boyut + search_k*sigma
    search_k: float = 3.0
    max_occluded_frames: int = 300 # bu kadar kare sonra pes et
