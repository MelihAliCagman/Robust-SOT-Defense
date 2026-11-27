"""Deney kaydı: her koşu için ayrı klasör.

experiments/<tarih-saat>_<ad>/
    run.json       ayarlar, tohum, komut, kod sürümü, paket sürümleri, girdi dosyalarının özeti, durum
    metrics.json   ölçülen sayılar
    *.csv, *.png, model dosyaları
experiments/summary.jsonl   koşu başına bir satır (karşılaştırma için)

Kullanım:
    with Experiment("ad", params={...}) as run:
        run.add_input("veri", "dosya.npz")
        run.log_metrics({"auc": 0.9})
        run.log_table("history", satirlar)
        grafik_yolu = run.path("egri.png")
Blok hata ile biterse koşu "failed" olarak kaydedilir, hata yine yükselir.
"""
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "experiments"
PACKAGES = ("numpy", "scikit-learn", "opencv-contrib-python", "torch")


def _jsonable(value):
    """NumPy türlerini ve NaN'i geçerli JSON'a çevirir (NaN -> null)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    elif hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _write_json(path, data):
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _git(*args):
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _code_version():
    """dirty=True: commit edilmemiş değişiklik var, sonuç yalnızca commit numarasıyla tekrarlanamaz."""
    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def file_digest(path):
    """Dosyanın SHA-256 özeti."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Experiment:
    def __init__(self, name, params=None, root=None):
        self.name = name
        self.root = Path(root) if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        started = datetime.now()
        self.dir = self._make_unique_dir(f"{started:%Y%m%d-%H%M%S}_{name}")
        self.metrics = {}
        self.info = {
            "name": name,
            "started": started.isoformat(timespec="seconds"),
            "status": "running",
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "params": params or {},
            "inputs": {},
            "code": _code_version(),
            "environment": _environment(),
        }
        self._save_info()

    def _make_unique_dir(self, base):
        """Aynı saniyede başlayan koşular birbirinin üzerine yazmaz."""
        suffix = 1
        while True:
            candidate = self.root / (base if suffix == 1 else f"{base}-{suffix}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                suffix += 1

    def _save_info(self):
        _write_json(self.dir / "run.json", self.info)

    def path(self, filename):
        return self.dir / filename

    def add_input(self, label, file_path):
        """Girdi dosyasının adını ve özetini kaydeder (hangi veriyle koşulduğu belli olsun)."""
        file_path = Path(file_path)
        entry = {"file": file_path.name}
        if file_path.is_file():
            entry["sha256"] = file_digest(file_path)
        self.info["inputs"][label] = entry
        self._save_info()

    def log_metrics(self, values):
        self.metrics.update(values)
        _write_json(self.dir / "metrics.json", self.metrics)

    def log_table(self, name, rows):
        """rows: aynı anahtarlara sahip sözlük listesi -> <name>.csv"""
        if not rows:
            return
        with open(self.dir / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def finish(self, status="finished", error=None):
        self.info["status"] = status
        self.info["finished"] = datetime.now().isoformat(timespec="seconds")
        if error:
            self.info["error"] = error
        self._save_info()
        summary = {"run": self.dir.name, "name": self.name, "status": status, "params": self.info["params"],
                   "code": self.info["code"], "metrics": self.metrics}
        with open(self.root / "summary.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(summary), ensure_ascii=False) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.finish()
        else:
            self.finish("failed", f"{error_type.__name__}: {error}")
        return False  # hatayı yutma
