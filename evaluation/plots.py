"""Deney grafikleri (pencere açmadan dosyaya)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_learning_curve(history, path, best_epoch=None):
    """history: [{"epoch": 1, "train_loss": .., "val_loss": ..}, ...]; val_loss olmayabilir."""
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [row["train_loss"] for row in history], label="eğitim kaybı (L2 cezası dahil)")
    if history and history[0].get("val_loss") is not None:
        axis.plot(epochs, [row["val_loss"] for row in history], label="doğrulama kaybı")
    if best_epoch is not None:
        axis.axvline(best_epoch, linestyle="--", color="gray", label=f"en iyi epoch ({best_epoch})")
    axis.set_xlabel("epoch")
    axis.set_ylabel("kayıp")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_roc(labels, scores_by_name, path):
    """labels: 0/1; scores_by_name: {model adı: skor dizisi}."""
    figure, axis = plt.subplots(figsize=(5, 5))
    for name, scores in scores_by_name.items():
        false_positive, true_positive, _ = roc_curve(labels, scores)
        axis.plot(false_positive, true_positive, label=f"{name} (AUROC {roc_auc_score(labels, scores):.3f})")
    axis.plot([0, 1], [0, 1], linestyle=":", color="gray")
    axis.set_xlabel("yanlış pozitif oranı")
    axis.set_ylabel("doğru pozitif oranı")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
