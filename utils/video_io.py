"""Video okuma. Kaynak bir video dosyası ya da kare görüntülerinden oluşan bir klasör olabilir."""
import glob
import os

import cv2


def _resize(frame, scale):
    return cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame


def _iter_folder(folder):
    """Klasördeki jpg/png dosyalarını ada göre sıralı okur (OTB biçimi)."""
    paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
    for path in sorted(paths):
        yield cv2.imread(path)


def _iter_video(path):
    capture = cv2.VideoCapture(path)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def read_frames(source, start=0, max_frames=None, scale=1.0):
    """BGR kare listesi döndürür.

    start: baştan atlanacak kare sayısı.
    max_frames: start'tan sonra en fazla kaç kare okunacağı (None: tümü).
    Hiç kare okunamazsa SystemExit ile anlaşılır bir hata verir.
    """
    if not os.path.exists(source):
        raise SystemExit(f"Video bulunamadı: {source}")
    reader = _iter_folder(source) if os.path.isdir(source) else _iter_video(source)
    frames = []
    for index, frame in enumerate(reader):
        if max_frames is not None and index >= start + max_frames:
            break
        if index >= start:
            frames.append(_resize(frame, scale))
    if not frames:
        raise SystemExit(f"Videodan kare okunamadı: {source} (başlangıç karesi: {start})")
    return frames
