"""Download raw datasets ke data/raw/ (reproducible, idempotent).

Sumber (lihat CLAUDE.md sec.3). Hardware/latency diabaikan; ini hanya unduh data.

- smsa  : HF indonlp/indonlu  @ refs/convert/parquet -> smsa/{train,validation,test}
- nusax : HF indonlp/NusaX-senti @ refs/convert/parquet -> ind/{train,validation,test}  (Indonesia saja)
- ugm   : GitHub ridife/dataset-idsa "Indonesian Sentiment Twitter Dataset Labeled.csv" (TSV, label -1/0/1)
          *finance & ugm menunggu konfirmasi user sebelum di-download (CLAUDE.md sec.3, sec.10).

Jalankan dari root repo:
    python src/datasets/download.py --datasets smsa nusax
"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

PARQUET_SOURCES = {
    "smsa": {
        "repo_id": "indonlp/indonlu",
        "config": "smsa",
        "splits": {"train": "train", "valid": "validation", "test": "test"},
    },
    "nusax": {
        "repo_id": "indonlp/NusaX-senti",
        "config": "ind",  # Indonesia saja
        "splits": {"train": "train", "valid": "validation", "test": "test"},
    },
}

UGM_SOURCE = {
    "url": "https://raw.githubusercontent.com/ridife/dataset-idsa/master/"
           "Indonesian%20Sentiment%20Twitter%20Dataset%20Labeled.csv",
    "filename": "Indonesian Sentiment Twitter Dataset Labeled.csv",
}

# ID-SMSA: Indonesian Stock Market Sentiment (Data in Brief 2025), Mendeley Data
# DOI 10.17632/tn4vzs8tdw.3 -- 3.288 tweet (pos/neu/neg), CSV. JANGAN bingung
# dengan key "smsa" (IndoNLU SmSA) yang berbeda total.
FINANCE_SOURCE = {
    "url": "https://data.mendeley.com/public-files/datasets/tn4vzs8tdw/files/"
           "bff2eae3-92d9-4af8-93cd-589ca7ee41b7/file_downloaded",
    "filename": "IDSMSA.csv",
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f"  {dest}  ({dest.stat().st_size} bytes)")


def download_parquet(key: str) -> None:
    src = PARQUET_SOURCES[key]
    out_dir = RAW / key
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, hf_split in src["splits"].items():
        remote = f"{src['config']}/{hf_split}/0000.parquet"
        local = hf_hub_download(
            repo_id=src["repo_id"],
            repo_type="dataset",
            revision="refs/convert/parquet",
            filename=remote,
        )
        dest = out_dir / f"{split}.parquet"
        dest.write_bytes(Path(local).read_bytes())
        print(f"[{key}] {split:5s} -> {dest}  ({dest.stat().st_size} bytes)")


def download_ugm() -> None:
    print("[ugm] labeled ->")
    _download_url(UGM_SOURCE["url"], RAW / "ugm" / "labeled.tsv")


def download_finance() -> None:
    print("[finance] IDSMSA ->")
    _download_url(FINANCE_SOURCE["url"], RAW / "finance" / "IDSMSA.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["smsa", "nusax"],
                    choices=["smsa", "nusax", "ugm", "finance"])
    args = ap.parse_args()
    for key in args.datasets:
        if key in PARQUET_SOURCES:
            download_parquet(key)
        elif key == "ugm":
            download_ugm()
        elif key == "finance":
            download_finance()


if __name__ == "__main__":
    main()
