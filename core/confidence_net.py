"""Öğrenilmiş güven ağı: son K kare özellikleri -> p(takipçi kutusu doğru ve hedef görünür).
Küçük MLP (scikit-learn). Girdi: (K,5) pencere düzleştirilir. Yerine PyTorch GRU konabilir (aynı arayüz)."""
import copy
import numpy as np, pickle
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from core.features import frame_features, window

class ConfidenceNet:
    def __init__(self, K=8, hidden=(32, 16), seed=0, max_epochs=300, patience=10, val_frac=0.15):
        self.K, self.seed = K, seed
        self.max_epochs, self.patience, self.val_frac = max_epochs, patience, val_frac
        self.sc = StandardScaler()
        self.m = MLPClassifier(hidden, activation="relu", alpha=1e-3, random_state=seed)
        self.best_epoch = None

    def fit(self, X, y, groups=None):
        """groups: kare başına video numarası. Verilirse erken durdurma için doğrulama VİDEO bazında ayrılır
        (örtüşen pencereler iki tarafa sızmaz). Verilmezse erken durdurma yok, max_epochs kadar eğitilir."""
        X = X.reshape(len(X), -1)
        tr, va = np.arange(len(X)), None
        if groups is not None:
            split = GroupShuffleSplit(n_splits=1, test_size=self.val_frac, random_state=self.seed)
            tr, va = next(split.split(X, y, groups))
        Xtr = self.sc.fit_transform(X[tr])
        Xva = self.sc.transform(X[va]) if va is not None else None
        best_loss, best_m, wait = np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.m.partial_fit(Xtr, y[tr], classes=[0, 1])  # bir epoch
            if va is None: continue
            loss = log_loss(y[va], self.m.predict_proba(Xva), labels=[0, 1])
            if loss < best_loss - 1e-4:
                best_loss, best_m, wait, self.best_epoch = loss, copy.deepcopy(self.m), 0, epoch
            else:
                wait += 1
                if wait >= self.patience: break
        if best_m is not None: self.m = best_m
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X.reshape(len(X), -1)))[:, 1]
    def save(self, p): pickle.dump(self, open(p, "wb"))
    @staticmethod
    def load(p): return pickle.load(open(p, "rb"))

class LearnedGate:
    """pipeline.run_pipeline(gate=...) ile kullanılır. Durumlu: özellik geçmişini kendisi tutar.
    reset(): takip başında ve yeniden yakalamadan sonra pipeline çağırır."""
    def __init__(self, net, thr=0.5):
        self.net, self.thr, self.hist, self.prev = net, thr, [], None
    def reset(self): self.hist, self.prev = [], None
    def __call__(self, score, ncc, box, prev_box):
        f = frame_features(score, box, self.prev if self.prev is not None else prev_box, ncc)
        self.hist.append(f); self.prev = box
        p = float(self.net.predict_proba(window(self.hist, self.net.K)[None])[0])
        return p >= self.thr
