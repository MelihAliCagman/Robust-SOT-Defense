"""Öğrenilmiş güven ağı: son K karenin özellikleri -> p(takipçi kutusu doğru ve hedef görünür).

Küçük bir tam bağlı ağ (scikit-learn MLP). Girdi: (K, 5) pencerenin düzleştirilmiş hali.
Aynı arayüz (fit, predict_proba) korunursa yerine PyTorch modeli konabilir.
"""
import copy
import pickle

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from core.features import frame_features, window

MIN_IMPROVEMENT = 1e-4  # erken durdurmada "iyileşme" sayılan en küçük kayıp düşüşü


def _flatten(X):
    return X.reshape(len(X), -1)


class ConfidenceNet:
    def __init__(self, K=8, hidden=(32, 16), seed=0, max_epochs=300, patience=10, val_frac=0.15):
        self.K, self.seed = K, seed
        self.max_epochs, self.patience, self.val_frac = max_epochs, patience, val_frac
        self.sc = StandardScaler()
        self.m = MLPClassifier(hidden, activation="relu", alpha=1e-3, random_state=seed)  # alpha: L2 cezası
        self.best_epoch = None
        self.history = []  # epoch başına {"epoch", "train_loss", "val_loss"}

    def fit(self, X, y, groups=None):
        """groups: kare başına video numarası.

        Verilirse erken durdurmanın doğrulama parçası VİDEO bazında ayrılır (örtüşen pencereler iki tarafa sızmaz).
        Verilmezse erken durdurma yapılmaz, max_epochs kadar eğitilir.
        """
        X = _flatten(X)
        train_idx, val_idx = np.arange(len(X)), None
        if groups is not None:
            split = GroupShuffleSplit(n_splits=1, test_size=self.val_frac, random_state=self.seed)
            train_idx, val_idx = next(split.split(X, y, groups))
        X_train = self.sc.fit_transform(X[train_idx])  # ölçekleme yalnızca eğitim parçasından öğrenilir
        X_val = self.sc.transform(X[val_idx]) if val_idx is not None else None

        best_loss, best_model, wait = np.inf, None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            self.m.partial_fit(X_train, y[train_idx], classes=[0, 1])  # bir epoch
            record = {"epoch": epoch, "train_loss": float(self.m.loss_), "val_loss": None}
            self.history.append(record)
            if val_idx is None:
                continue
            loss = log_loss(y[val_idx], self.m.predict_proba(X_val), labels=[0, 1])
            record["val_loss"] = float(loss)
            if loss < best_loss - MIN_IMPROVEMENT:
                best_loss, best_model, wait = loss, copy.deepcopy(self.m), 0
                self.best_epoch = epoch
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_model is not None:
            self.m = best_model
        return self

    def predict_proba(self, X):
        """X: (N, K, 5) -> (N,) olasılık."""
        return self.m.predict_proba(self.sc.transform(_flatten(X)))[:, 1]

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        """Yalnızca kendi ürettiğiniz dosyaları yükleyin: pickle, dosyadaki kodu çalıştırabilir."""
        with open(path, "rb") as handle:
            return pickle.load(handle)


class LearnedGate:
    """run_pipeline(gate=...) ile kullanılır. Özellik geçmişini kendisi tutar.

    reset(): takip başında ve yeniden yakalamadan sonra pipeline tarafından çağrılır.
    """

    def __init__(self, net, thr=0.5):
        self.net, self.thr = net, thr
        self.reset()

    def reset(self):
        self.hist, self.prev = [], None

    def __call__(self, score, ncc, box, prev_box):
        previous = self.prev if self.prev is not None else prev_box
        self.hist.append(frame_features(score, box, previous, ncc))
        self.prev = box
        probability = float(self.net.predict_proba(window(self.hist, self.net.K)[None])[0])
        return probability >= self.thr
