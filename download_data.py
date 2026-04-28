import gzip
import shutil
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"
ANNEES = [2024, 2025]
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

echecs = []
for annee in ANNEES:
    cible_csv = DATA_DIR / f"full_{annee}.csv"
    if cible_csv.exists():
        print(f"[OK]   full_{annee}.csv deja present, on passe.")
        continue

    url = f"{BASE_URL}/{annee}/full.csv.gz"
    tmp_gz = DATA_DIR / f"full_{annee}.csv.gz"
    print(f"[DL]   Telechargement de {url}...")
    try:
        urllib.request.urlretrieve(url, tmp_gz)
    except urllib.error.HTTPError as e:
        print(f"[ERR]  Echec {annee} : {e}. On passe.")
        echecs.append(annee)
        if tmp_gz.exists():
            tmp_gz.unlink()
        continue

    print(f"[UNZIP] Decompression -> {cible_csv.name}")
    with gzip.open(tmp_gz, "rb") as src, open(cible_csv, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_gz.unlink()

print("\nTermine. Fichiers disponibles :")
for f in sorted(DATA_DIR.glob("full_*.csv")):
    print(f"  {f.name} : {f.stat().st_size / 1e6:.1f} MB")

if echecs:
    print(f"\nAnnees non telechargees : {echecs}")
