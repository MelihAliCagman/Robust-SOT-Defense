"""2B sabit hızlı Kalman filtresi. Durum: [cx, cy, vx, vy]. Genişlik ve yükseklik ayrı tutulur."""
import numpy as np

SIZE_SMOOTHING = 0.2  # her güncellemede yeni ölçülen boyutun ağırlığı


class Kalman2D:
    def __init__(self, box, q_pos=1.0, q_vel=0.5, r=4.0, vel_decay=0.97):
        """box: başlangıç kutusu [x, y, w, h]. q_*: süreç gürültüsü, r: ölçüm gürültüsü."""
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        self.w, self.h = float(box[2]), float(box[3])
        self.x = np.array([cx, cy, 0.0, 0.0])        # durum
        self.P = np.diag([10.0, 10.0, 50.0, 50.0])   # durum belirsizliği
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)  # geçiş
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)  # yalnızca konum ölçülür
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * r
        self.vel_decay = vel_decay

    def predict(self, damp=False):
        """Bir kare ileri tahmin. damp=True (kapanma): hız yavaşça söner, aşırı sapma önlenir."""
        if damp:
            self.x[2:] *= self.vel_decay
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        """Güvenilen bir ölçümle düzeltme. box: [x, y, w, h]"""
        measurement = np.array([box[0] + box[2] / 2.0, box[1] + box[3] / 2.0])
        self.w = (1 - SIZE_SMOOTHING) * self.w + SIZE_SMOOTHING * box[2]
        self.h = (1 - SIZE_SMOOTHING) * self.h + SIZE_SMOOTHING * box[3]
        innovation = measurement - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.P = (np.eye(4) - gain @ self.H) @ self.P
        return self.box()

    def box(self):
        """Şu anki tahmin [x, y, w, h]."""
        return [self.x[0] - self.w / 2.0, self.x[1] - self.h / 2.0, self.w, self.h]

    def pos_sigma(self):
        """Konum belirsizliği (piksel). Yeniden yakalama arama penceresini büyütmek için."""
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
