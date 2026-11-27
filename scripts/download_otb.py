"""OTB100 sekanslarını indirir ve açar.

Resmî site (cvlab.hanyang.ac.kr/tracker_benchmark) artık 404 verdiği için Internet Archive'daki
arşiv kopyası kullanılır. Arşivdeki genel tarihli adres bazı sekanslarda zip yerine HTML sayfası
döndürdüğü için, gerçek zip kopyalarının tarihleri önce arşiv dizininden (CDX) okunur.
Yarıda kalırsa tekrar çalıştırın: inmiş sekanslar atlanır.

Kullanım (kod kökünden):
    python scripts/download_otb.py                    # tüm liste
    python scripts/download_otb.py Jogging Subway     # yalnızca seçilenler
"""
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "cvlab.hanyang.ac.kr/tracker_benchmark/seq/"
CDX_URL = ("https://web.archive.org/cdx/search/cdx?url=" + SOURCE_URL +
           "&matchType=prefix&filter=statuscode:200&filter=mimetype:application/zip"
           "&fl=original,timestamp&output=json&limit=1000")
ROOT = Path(__file__).resolve().parent.parent / "datasets" / "OTB100"
ZIP_DIR = ROOT / "zips"
SEQ_DIR = ROOT / "sequences"

# Kaynak liste: github.com/prosti221/OTB-dataset (files.txt), 98 arşiv
SEQUENCES = """
Basketball Biker Bird1 BlurBody BlurCar2 BlurFace BlurOwl Bolt Box Car1 Car4 CarDark CarScale
ClifBar Couple Crowds David Deer Diving DragonBaby Dudek Football Freeman4 Girl Human3 Human4
Human6 Human9 Ironman Jump Jumping Liquor Matrix MotorRolling Panda RedTeam Shaking Singer2
Skating1 Skating2 Skiing Soccer Surfer Sylvester Tiger2 Trellis Walking Walking2 Woman Bird2
BlurCar1 BlurCar3 BlurCar4 Board Bolt2 Boy Car2 Car24 Coke Coupon Crossing Dancer Dancer2 David2
David3 Dog Dog1 Doll FaceOcc1 FaceOcc2 Fish FleetFace Football1 Freeman1 Freeman3 Girl2 Gym
Human2 Human5 Human7 Human8 Jogging KiteSurf Lemming Man Mhyang MountainBike Rubik Singer1 Skater
Skater2 Subway Suv Tiger1 Toy Trans Twinnings Vase
""".split()


def is_valid_zip(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def open_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "otb-downloader"})
    return urllib.request.urlopen(request, timeout=300)


def find_snapshots():
    """{sekans adı: ham zip kopyasının indirme adresi}"""
    with open_url(CDX_URL) as response:
        rows = json.load(response)[1:]  # ilk satır sütun adları
    snapshots = {}
    for original, timestamp in sorted(rows, key=lambda row: row[1]):
        match = re.search(r"/seq/([A-Za-z0-9]+)\.zip$", original)
        if match:  # "id_": Wayback sayfası eklenmeden dosyanın kendisi
            snapshots[match.group(1)] = f"https://web.archive.org/web/{timestamp}id_/{original}"
    return snapshots


def download(name, url, attempts=3):
    """Arşivi indirir. Başarılıysa zip yolunu, değilse None döndürür."""
    target = ZIP_DIR / f"{name}.zip"
    if target.exists() and is_valid_zip(target):
        return target
    partial = target.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        try:
            with open_url(url) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
            if is_valid_zip(partial):
                partial.replace(target)
                return target
            print(f"  {name}: bozuk arşiv (deneme {attempt})", flush=True)
        except Exception as error:  # ağ hataları: bekleyip tekrar dene
            print(f"  {name}: {error} (deneme {attempt})", flush=True)
        time.sleep(15 * attempt)
    partial.unlink(missing_ok=True)
    return None


def extract(zip_path):
    done_marker = SEQ_DIR / f".{zip_path.stem}.done"
    if done_marker.exists():
        return
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if "__MACOSX" not in m and not m.endswith(".DS_Store")]
        archive.extractall(SEQ_DIR, members)
    done_marker.touch()


def main():
    names = sys.argv[1:] or SEQUENCES
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = find_snapshots()
    missing = [name for name in names if name not in snapshots]
    if missing:
        print(f"arşivde zip kopyası yok ({len(missing)}): {' '.join(missing)}", flush=True)
    names = [name for name in names if name in snapshots]
    failed = []
    for index, name in enumerate(names, 1):
        zip_path = download(name, snapshots[name])
        if zip_path is None:
            failed.append(name)
        else:
            extract(zip_path)
            print(f"[{index}/{len(names)}] {name} hazır ({zip_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    print(f"bitti: {len(names) - len(failed)} başarılı, {len(failed)} başarısız: {' '.join(failed)}")


if __name__ == "__main__":
    main()
