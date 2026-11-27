"""2D sabit hızlı Kalman. Durum: [cx, cy, vx, vy]. w,h ayrı tutulur (kapanmada sabit)."""
import numpy as np

class Kalman2D:
    def __init__(self, box, q_pos=1.0, q_vel=0.5, r=4.0, vel_decay=0.97):
        """box: [x, y, w, h]"""
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        self.w, self.h = float(box[2]), float(box[3])
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 50.0, 50.0])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * r
        self.vel_decay = vel_decay

    def predict(self, damp=False):
        if damp:  # kapanmada hız yavaşça sönsün (aşırı sapmayı önler)
            self.x[2:] *= self.vel_decay
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        z = np.array([box[0] + box[2] / 2.0, box[1] + box[3] / 2.0])
        self.w = 0.8 * self.w + 0.2 * box[2]
        self.h = 0.8 * self.h + 0.2 * box[3]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.box()

    def box(self):
        return [self.x[0] - self.w / 2.0, self.x[1] - self.h / 2.0, self.w, self.h]

    def pos_sigma(self):
        """Konum belirsizliği (piksel). Yeniden yakalama arama penceresi için."""
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
