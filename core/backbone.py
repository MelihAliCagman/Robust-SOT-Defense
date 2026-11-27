"""Hazır (donuk) takipçi sarmalayıcısı: OpenCV ViT veya Nano.

Arayüz: init(frame, box) ve update(frame) -> (ok, box, score). Bu arayüz korunursa takipçi değiştirilebilir.
Not: OpenCV tepki haritasını dışarı vermez, bu yüzden PSR hesaplanamaz.
"""
import cv2


class OpenCVTracker:
    def __init__(self, kind="vit", model_path="vit.onnx", nano_backbone=None, nano_head=None):
        self.kind, self.model_path = kind, model_path
        self.nano_backbone, self.nano_head = nano_backbone, nano_head
        self.tracker = None

    def _create(self):
        if self.kind == "vit":
            params = cv2.TrackerVit_Params()
            params.net = self.model_path
            return cv2.TrackerVit_create(params)
        params = cv2.TrackerNano_Params()
        params.backbone, params.neckhead = self.nano_backbone, self.nano_head
        return cv2.TrackerNano_create(params)

    def init(self, frame, box):
        """Takipçiyi sıfırdan oluşturup verilen kutuya kilitler. box: [x, y, w, h]"""
        self.tracker = self._create()
        self.tracker.init(frame, tuple(int(round(v)) for v in box))

    def update(self, frame):
        """Döner: (başarılı mı, [x, y, w, h], güven skoru)."""
        ok, box = self.tracker.update(frame)
        return bool(ok), [float(v) for v in box], float(self.tracker.getTrackingScore())
