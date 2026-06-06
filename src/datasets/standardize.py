"""Standardisasi 4 dataset -> skema seragam {negative, neutral, positive}.

Output: data/processed/<key>/{train,valid,test}.parquet  (kolom: id, text, label, orig_label)
        data/processed/<key>/golden_test_ids.txt          (id test terkunci)
        data/processed/<key>/summary.json                 (statistik + config reproducible)

Aturan (CLAUDE.md sec.3, sec.6):
- Mapping label EKSPLISIT & terverifikasi dari sumber resmi (lihat di bawah). smsa & nusax
  punya urutan integer BERLAWANAN -- jangan diasumsikan sama.
- Baris gagal-map / teks kosong DITOLAK & dicatat (rejected.csv), tidak diam-diam ke default.
- smsa & nusax: pakai split resmi (test = golden, terkunci dari sananya).
- ugm & finance: belum bersplit -> stratified split terkunci (seed tetap). Duplikat teks
  persis di-drop SEBELUM split untuk cegah kebocoran train<->test.

Jalankan dari root repo:
    python src/datasets/standardize.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SEED = 42
SPLIT_FRACS = (0.8, 0.1, 0.1)  # train / valid / test untuk dataset tanpa split resmi
CANON = ("negative", "neutral", "positive")

# --- Mapping label terverifikasi dari sumber resmi -------------------------
# smsa : indonlu loader -> label_classes=["positive","neutral","negative"]
SMSA_MAP = {0: "positive", 1: "neutral", 2: "negative"}
# nusax: NusaX-senti loader -> ClassLabel(names=["negative","neutral","positive"])
NUSAX_MAP = {0: "negative", 1: "neutral", 2: "positive"}
# ugm  : ridife/dataset-idsa -> -1/0/1 (dikonfirmasi via contoh isi data)
UGM_MAP = {-1: "negative", 0: "neutral", 1: "positive"}
# finance: ID-SMSA -> kolom Sentiment sudah string Positive/Neutral/Negative
FIN_MAP = {"positive": "positive", "neutral": "neutral", "negative": "negative"}


# --- util ------------------------------------------------------------------
def _clean_text(s: object) -> str:
    return "" if pd.isna(s) else str(s).strip()


def _finalize(df: pd.DataFrame, key: str, split: str, rejects: list) -> pd.DataFrame:
    """df punya kolom: id, text, orig_label, label. Validasi + saring."""
    bad_text = df["text"].str.len() == 0
    bad_label = ~df["label"].isin(CANON)
    bad = bad_text | bad_label
    if bad.any():
        rej = df[bad].copy()
        rej["__reason"] = np.where(bad_text[bad], "empty_text", "unmapped_label")
        rej["__split"] = split
        rejects.append(rej)
    return df[~bad].reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets = {"train": [], "valid": [], "test": []}
    for _, grp in df.groupby("label", sort=True):
        idx = grp.index.to_numpy()
        idx = rng.permutation(idx)
        n = len(idx)
        n_tr = int(round(n * SPLIT_FRACS[0]))
        n_va = int(round(n * SPLIT_FRACS[1]))
        buckets["train"].append(idx[:n_tr])
        buckets["valid"].append(idx[n_tr:n_tr + n_va])
        buckets["test"].append(idx[n_tr + n_va:])
    return {k: df.loc[np.concatenate(v)].sort_index().reset_index(drop=True)
            for k, v in buckets.items()}


# --- loaders per dataset (return: dict split -> standardized df) ------------
def _from_parquet(key: str, label_map: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ("train", "valid", "test"):
        df = pd.read_parquet(RAW / key / f"{split}.parquet")
        std = pd.DataFrame({
            "id": [f"{key}-{split}-{i}" for i in range(len(df))],
            "text": df["text"].map(_clean_text),
            "orig_label": df["label"],
            "label": df["label"].map(label_map),
        })
        out[split] = std
    return out


def _from_flat(key: str, df: pd.DataFrame, text_col: str, label_col: str,
               label_map: dict, normalize_label=None) -> dict[str, pd.DataFrame]:
    orig = df[label_col]
    keymapped = orig.map(normalize_label) if normalize_label else orig
    std = pd.DataFrame({
        "id": [f"{key}-{i}" for i in range(len(df))],
        "text": df[text_col].map(_clean_text),
        "orig_label": orig,
        "label": keymapped.map(label_map),
    })
    # drop duplikat teks persis SEBELUM split (cegah kebocoran train<->test)
    n_before = len(std)
    std = std.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    n_dups = n_before - len(std)
    splits = _stratified_split(std)
    splits["__n_dups_dropped"] = n_dups  # type: ignore
    return splits


def standardize(key: str) -> dict:
    rejects: list = []
    if key == "smsa":
        splits = _from_parquet("smsa", SMSA_MAP)
    elif key == "nusax":
        splits = _from_parquet("nusax", NUSAX_MAP)
    elif key == "ugm":
        df = pd.read_csv(RAW / "ugm" / "labeled.tsv", sep="\t")
        splits = _from_flat("ugm", df, text_col="Tweet", label_col="sentimen",
                            label_map=UGM_MAP)
    elif key == "finance":
        df = pd.read_csv(RAW / "finance" / "IDSMSA.csv")
        splits = _from_flat("finance", df, text_col="Sentence", label_col="Sentiment",
                            label_map=FIN_MAP, normalize_label=lambda s: str(s).strip().lower())
    else:
        raise ValueError(key)

    n_dups = splits.pop("__n_dups_dropped", 0)
    out_dir = PROC / key
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"dataset": key, "seed": SEED, "n_dups_dropped": int(n_dups),
               "splits": {}, "official_split": key in ("smsa", "nusax")}
    for split, df in splits.items():
        df = _finalize(df, key, split, rejects)
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
        summary["splits"][split] = {
            "n": int(len(df)),
            "label_dist": {k: int(v) for k, v in df["label"].value_counts().items()},
        }

    # golden test ids terkunci
    test_ids = pd.read_parquet(out_dir / "test.parquet")["id"].tolist()
    (out_dir / "golden_test_ids.txt").write_text("\n".join(test_ids), encoding="utf-8")

    if rejects:
        rej = pd.concat(rejects, ignore_index=True)
        rej.to_csv(out_dir / "rejected.csv", index=False, encoding="utf-8")
        summary["n_rejected"] = int(len(rej))
    else:
        summary["n_rejected"] = 0

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return summary


def main() -> None:
    overall = {}
    for key in ("smsa", "nusax", "ugm", "finance"):
        s = standardize(key)
        overall[key] = s
        dist = " | ".join(f"{sp}:{d['n']}" for sp, d in s["splits"].items())
        print(f"[{key:7s}] {dist}  | dups_dropped={s['n_dups_dropped']} rejected={s['n_rejected']}")
    (PROC / "manifest.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"\nmanifest -> {PROC / 'manifest.json'}")


if __name__ == "__main__":
    main()
