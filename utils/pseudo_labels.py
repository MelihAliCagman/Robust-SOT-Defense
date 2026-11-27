"""Etiketi olmayan videolar için renkten YAKLAŞIK (sözde) etiket. Gerçek etiket yerine geçmez."""
import cv2
import numpy as np

MIN_AREA = 300  # bundan küçük kırmızı lekeler yok sayılır (piksel)


def red_boxes(frames):
    """Her karedeki en büyük kırmızı lekenin kutusu [x, y, w, h]; leke yoksa None.

    Yalnızca hedefin kırmızı olduğu videolarda anlamlıdır.
    """
    boxes = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # kırmızı, ton ekseninin iki ucunda yer alır
        mask = cv2.inRange(hsv, (0, 120, 90), (8, 255, 255)) | cv2.inRange(hsv, (172, 120, 90), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        largest = None
        for label in range(1, count):  # 0 arka plandır
            area = stats[label][cv2.CC_STAT_AREA]
            if area > MIN_AREA and (largest is None or area > largest[cv2.CC_STAT_AREA]):
                largest = stats[label]
        boxes.append([int(v) for v in largest[:4]] if largest is not None else None)
    return boxes
