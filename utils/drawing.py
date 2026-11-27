"""Takip sonuçlarını kare üzerine çizer (BGR)."""
import cv2

GREEN = (0, 200, 0)  # takip
RED = (0, 0, 255)    # kapanma: kutu Kalman tahminidir
WHITE = (255, 255, 255)


def draw_result(frame, result, title, title_position, title_scale):
    """Karenin kopyasına sonucu çizer. result: pipeline'ın ürettiği tek karelik sözlük."""
    panel = frame.copy()
    x, y, w, h = [int(v) for v in result["box"]]
    if w > 0:  # [0,0,0,0] hedef kayıp demektir, çizilmez
        color = GREEN if result["state"] == 0 else RED
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
    cv2.putText(panel, title, title_position, cv2.FONT_HERSHEY_SIMPLEX, title_scale, WHITE, 2)
    return panel
