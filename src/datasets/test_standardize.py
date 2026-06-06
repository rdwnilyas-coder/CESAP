"""Test loader & standardisasi. Runnable langsung (tanpa pytest) atau via pytest.

    python src/datasets/test_standardize.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from standardize import (PROC, RAW, CANON, SMSA_MAP, NUSAX_MAP, standardize)

KEYS = ("smsa", "nusax", "ugm", "finance")


def _load(key, split):
    return pd.read_parquet(PROC / key / f"{split}.parquet")


def test_labels_valid_and_text_nonempty():
    for key in KEYS:
        for split in ("train", "valid", "test"):
            df = _load(key, split)
            assert set(df["label"]).issubset(set(CANON)), f"{key}/{split} label asing"
            assert (df["text"].str.len() > 0).all(), f"{key}/{split} ada teks kosong"


def test_mapping_orientation_locked():
    """Kunci orientasi int->str dari hitungan mentah. smsa & nusax terbalik."""
    # smsa: 0=positive, 1=neutral, 2=negative
    raw = pd.read_parquet(RAW / "smsa" / "train.parquet")
    proc = _load("smsa", "train")
    for i, name in SMSA_MAP.items():
        assert (proc["label"] == name).sum() == (raw["label"] == i).sum(), \
            f"smsa mapping int {i}->{name} tidak konsisten"
    # nusax: 0=negative, 1=neutral, 2=positive
    raw = pd.read_parquet(RAW / "nusax" / "train.parquet")
    proc = _load("nusax", "train")
    for i, name in NUSAX_MAP.items():
        assert (proc["label"] == name).sum() == (raw["label"] == i).sum(), \
            f"nusax mapping int {i}->{name} tidak konsisten"
    # sanity: orientasi memang berlawanan
    assert SMSA_MAP[0] == "positive" and NUSAX_MAP[0] == "negative"


def test_no_leakage_between_splits():
    for key in KEYS:
        tr, va, te = (_load(key, s) for s in ("train", "valid", "test"))
        ids = [set(d["id"]) for d in (tr, va, te)]
        assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]), \
            f"{key}: id antar-split tumpang tindih"
        # untuk dataset yang kita split sendiri, teks tak boleh bocor antar split
        if key in ("ugm", "finance"):
            assert set(tr["text"]).isdisjoint(set(te["text"])), f"{key}: teks bocor train<->test"
            assert set(tr["text"]).isdisjoint(set(va["text"])), f"{key}: teks bocor train<->valid"


def test_golden_test_locked():
    for key in KEYS:
        ids_file = (PROC / key / "golden_test_ids.txt").read_text(encoding="utf-8").split("\n")
        te = _load(key, "test")
        assert ids_file == te["id"].tolist(), f"{key}: golden_test_ids.txt != test.parquet"


def test_deterministic_resplit():
    """Re-run standardize untuk ugm harus hasilkan test ids identik (seed terkunci)."""
    before = (PROC / "ugm" / "golden_test_ids.txt").read_text(encoding="utf-8")
    standardize("ugm")
    after = (PROC / "ugm" / "golden_test_ids.txt").read_text(encoding="utf-8")
    assert before == after, "split ugm tidak deterministik"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
