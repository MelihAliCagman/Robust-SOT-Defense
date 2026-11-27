"""Donuk takipçi sarmalayıcı. Şimdilik OpenCV ViT/Nano (güven skoru: getTrackingScore).
NOT: OpenCV response map'i dışarı VERMEZ -> PSR hesaplanamaz. PSR/response map için pytracking (DiMP) veya PySOT gerekir.
Aynı arayüz (init/update -> ok, box, score) korunursa takipçi sonradan değiştirilebilir."""
import cv2

class OpenCVTracker:
    def __init__(self, kind="vit", model_path="vit.onnx", nano_backbone=None, nano_head=None):
        self.kind, self.model_path = kind, model_path
        self.nano_backbone, self.nano_head = nano_backbone, nano_head
        self.t = None

    def _make(self):
        if self.kind == "vit":
            p = cv2.TrackerVit_Params(); p.net = self.model_path
            return cv2.TrackerVit_create(p)
        p = cv2.TrackerNano_Params()
        p.backbone, p.neckhead = self.nano_backbone, self.nano_head
        return cv2.TrackerNano_create(p)

    def init(self, frame, box):
        self.t = self._make()
        self.t.init(frame, tuple(int(round(v)) for v in box))

    def update(self, frame):
        ok, b = self.t.update(frame)
        return bool(ok), [float(v) for v in b], float(self.t.getTrackingScore())
