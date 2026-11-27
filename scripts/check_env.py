"""Kurulum doğrulaması: paket sürümleri, ekran kartında eğitim adımı, takipçi ve pencere desteği.

Kullanım: python scripts/check_env.py [vit.onnx]
Zorunlu bir kontrol başarısızsa çıkış kodu 1 olur. GPU yoksa uyarı verilir (proje CPU'da da çalışır).
"""
import argparse
import sys

import cv2
import matplotlib
import numpy as np
import sklearn


def check_versions():
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | opencv {cv2.__version__} | "
          f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")
    return True


def check_gpu():
    """PyTorch'u içe aktarır ve cihazda küçük bir ileri/geri geçiş yapar."""
    try:
        import torch
        import torchvision
    except ImportError as error:
        print(f"[HATA] PyTorch yüklenemedi: {error}")
        return False
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if has_gpu else "cpu")
    if has_gpu:
        properties = torch.cuda.get_device_properties(0)
        print(f"[TAMAM] GPU: {properties.name}, {properties.total_memory / 1024**3:.1f} GB")
    else:
        print("[UYARI] CUDA kullanılamıyor; eğitim CPU'da yapılır.")

    # evrişim + GRU + geri yayılım: projede kullanılacak işlem türleri
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 8, 3).to(device)
    gru = torch.nn.GRU(8, 4, batch_first=True).to(device)
    features = conv(torch.randn(16, 3, 32, 32, device=device)).mean(dim=(2, 3))
    output, _ = gru(features.view(2, 8, 8))
    output.pow(2).mean().backward()
    gradient_ok = bool(torch.isfinite(conv.weight.grad).all())
    print(f"[{'TAMAM' if gradient_ok else 'HATA'}] {device.type} üzerinde ileri ve geri geçiş")

    # torchvision'ın derlenmiş işlemleri torch ile uyumlu mu (sürüm uyumsuzluğu burada ortaya çıkar)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    ops_ok = kept.tolist() == [0]
    print(f"[{'TAMAM' if ops_ok else 'HATA'}] torchvision işlemleri")
    return gradient_ok and ops_ok


def check_tracker(model_path):
    if not hasattr(cv2, "TrackerVit_create"):
        print("[HATA] cv2.TrackerVit_create yok (opencv-contrib-python gerekli)")
        return False
    if model_path is None:
        print("[ATLANDI] takipçi denemesi (model yolu verilmedi)")
        return True
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.backbone import OpenCVTracker

    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    tracker = OpenCVTracker("vit", model_path)
    tracker.init(frame, [100, 80, 60, 40])
    ok, _, score = tracker.update(frame)
    print(f"[{'TAMAM' if ok else 'HATA'}] takipçi çalışıyor (aynı karede skor {score:.2f})")
    return ok


def check_window():
    """Demo için OpenCV'nin pencere açabilmesi gerekir ("headless" paket açamaz)."""
    try:
        cv2.namedWindow("kontrol", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("kontrol")
    except cv2.error:
        print("[HATA] OpenCV pencere açamıyor (opencv-python-headless kurulu olabilir)")
        return False
    print("[TAMAM] OpenCV pencere desteği")
    return True


def main():
    parser = argparse.ArgumentParser(description="Kurulum doğrulaması.")
    parser.add_argument("model", nargs="?", help="vit.onnx yolu (takipçi denemesi için)")
    args = parser.parse_args()
    results = [check_versions(), check_gpu(), check_tracker(args.model), check_window()]
    print("SONUÇ:", "kurulum hazır" if all(results) else "eksik var")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
