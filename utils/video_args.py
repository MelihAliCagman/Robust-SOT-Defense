"""Eğitim ve karşılaştırma betiklerinin isteğe bağlı video argümanları: VIDEO START X Y W H"""
from typing import NamedTuple

USAGE = "VIDEO START X Y W H"


class VideoSpec(NamedTuple):
    path: str
    start: int  # kutunun verildiği kare
    box: list   # başlangıç karesindeki hedef [x, y, w, h]


def parse_video_spec(parser, values):
    """values boşsa None; 6 değerse VideoSpec. Aksi halde parser üzerinden hata verir."""
    if not values:
        return None
    if len(values) != 6:
        parser.error(f"video için tam olarak 6 değer gerekir: {USAGE} (verilen: {len(values)})")
    try:
        return VideoSpec(values[0], int(values[1]), [float(v) for v in values[2:]])
    except ValueError:
        parser.error(f"START tam sayı, X Y W H sayı olmalı: {USAGE}")
