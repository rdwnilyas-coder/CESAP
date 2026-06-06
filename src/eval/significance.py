"""Uji signifikansi beda akurasi (McNemar + bootstrap Macro-F1) dari prediksi
per-sampel tersimpan (golden test, id selaras). Tanpa inferensi baru.

Pasangan kunci per dataset:
  - best open-source vs best LLM  (apakah bayar LLM signifikan?)
  - indobert-base vs lightweight  (anomali smsa lite>base?)
  - distill(Opus-2k) vs gold-2k   (apakah label Opus signifikan menurunkan?)

    python src/eval/significance.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
GA = ROOT / "results" / "groupA"
GB = ROOT / "results" / "groupB"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
LABELS = ["negative", "neutral", "positive"]
GA_ARMS = ["indobert-base", "indobert-large", "indobertweet", "lightweight"]


def preds(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)[["id", "true_label", "pred_label"]]


def best_oss(ds):
    best, bf = None, -1
    for a in GA_ARMS:
        f = json.load(open(GA / ds / a / "metrics.json", encoding="utf-8"))["test_metrics"]["macro_f1"]
        if f > bf:
            best, bf = a, f
    return best, GA / ds / best / "predictions.parquet"


def best_llm(ds):
    best, bf, bp = None, -1, None
    for d in (GB / ds).iterdir():
        mj = d / "metrics.json"
        if mj.exists() and json.load(open(mj, encoding="utf-8")).get("backend") == "max":
            f = json.load(open(mj, encoding="utf-8"))["macro_f1"]
            if f > bf:
                best, bf, bp = d.name, f, d / "predictions.parquet"
    return best, bp


def mcnemar(a, b):
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    ca = (m.pred_label_a == m.true_label_a).values
    cb = (m.pred_label_b == m.true_label_b).values
    b_ = int((ca & ~cb).sum()); c_ = int((~ca & cb).sum())
    if b_ + c_ == 0:
        return 1.0, b_, c_
    stat = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
    return float(chi2.sf(stat, 1)), b_, c_


def boot_diff(a, b, n=2000, seed=0):
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    ta = m.true_label_a.values; pa = m.pred_label_a.values; pb = m.pred_label_b.values
    rng = np.random.default_rng(seed); idx = np.arange(len(m)); d = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        d[i] = (f1_score(ta[s], pa[s], labels=LABELS, average="macro", zero_division=0)
                - f1_score(ta[s], pb[s], labels=LABELS, average="macro", zero_division=0))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cmp(name, pa, pb):
    a, b = preds(pa), preds(pb)
    p, bb, cc = mcnemar(a, b)
    md, lo, hi = boot_diff(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {name:42s} McNemar p={p:.4f} {sig:3s} (b={bb},c={cc}) | "
          f"dMacroF1={md:+.4f} [95% {lo:+.3f},{hi:+.3f}]")


def main():
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        bo, bop = best_oss(ds); bl, blp = best_llm(ds)
        cmp(f"{bo}(OSS) vs {bl}(LLM)", bop, blp)
        cmp("indobert-base vs lightweight", GA / ds / "indobert-base" / "predictions.parquet",
            GA / ds / "lightweight" / "predictions.parquet")
        dis = GA / f"{ds}_distill" / "indobert-base" / "predictions.parquet"
        g2 = GA / f"{ds}_gold2k" / "indobert-base" / "predictions.parquet"
        if dis.exists() and g2.exists():
            cmp("distill(Opus-2k) vs gold-2k", dis, g2)
    print("\n(* p<.05, ** p<.01, *** p<.001; dMacroF1 = arm1 - arm2; CI bootstrap 2000x)")


if __name__ == "__main__":
    main()
