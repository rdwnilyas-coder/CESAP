"""C2 — Uncertainty cascade Haiku -> Sonnet -> Opus berbasis confidence (verbalized).
Post-hoc dari label+conf Haiku & Sonnet (results/groupC/c2) + label Opus (Group B).

Aturan (threshold tunggal tau): terima Haiku bila conf>=tau; else Sonnet bila conf>=tau;
else Opus. cost/1k = haiku_cpk + sonnet_cpk*frac_ke_sonnet + opus_cpk*frac_ke_opus.

    python src/models/hybrid/c2_cascade.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
GB = ROOT / "results" / "groupB"
C2 = ROOT / "results" / "groupC" / "c2"
OUT = ROOT / "results" / "hybrid"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
LABELS = ["negative", "neutral", "positive"]
TAUS = [50, 60, 70, 80, 90, 95]


def merge_conf(d: Path, model: str) -> dict:
    m = {}
    for f in sorted(d.glob(f"{model}_b*.json")):
        try:
            m.update(json.loads(f.read_text(encoding="utf-8-sig")))
        except Exception as e:
            print(f"  (skip {f.name}: {e})")
    return m


def cpk(ds, tag):
    return json.loads((GB / ds / tag / "metrics.json").read_text(encoding="utf-8"))["cost_per_1k_usd_modeled"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        d = C2 / ds
        haiku = merge_conf(d, "haiku"); sonnet = merge_conf(d, "sonnet")
        gold = pd.read_parquet(GB / ds / "opus-few-max" / "predictions.parquet")[["id", "true_label"]]
        opus = pd.read_parquet(GB / ds / "opus-few-max" / "predictions.parquet").set_index("id")["pred_label"].to_dict()
        ids = [i for i in gold["id"] if i in haiku and i in sonnet and i in opus]
        miss = len(gold) - len(ids)
        if miss:
            print(f"[{ds}] BELUM lengkap: missing={miss} (haiku/sonnet) — repair dulu"); continue
        truth = gold.set_index("id")["true_label"].to_dict()
        ch = cpk(ds, "haiku-few-max"); cs = cpk(ds, "sonnet-few-max"); co = cpk(ds, "opus-few-max")

        def conf(m, i): return float(m[i].get("conf", 0)) if isinstance(m[i], dict) else 0.0
        def lab(m, i): return m[i].get("label") if isinstance(m[i], dict) else m[i]

        for tau in TAUS:
            preds, to_s, to_o = [], 0, 0
            for i in ids:
                if conf(haiku, i) >= tau:
                    preds.append(lab(haiku, i))
                elif conf(sonnet, i) >= tau:
                    preds.append(lab(sonnet, i)); to_s += 1
                else:
                    preds.append(opus[i]); to_o += 1; to_s += 1   # ke opus berarti lewat sonnet juga
            yt = [truth[i] for i in ids]
            f1 = f1_score(yt, preds, labels=LABELS, average="macro", zero_division=0)
            n = len(ids)
            cost = ch + cs * (to_s / n) + co * (to_o / n)
            rows.append({"dataset": ds, "tau": tau, "macro_f1": round(f1, 4),
                         "frac_sonnet": round(to_s / n, 3), "frac_opus": round(to_o / n, 3),
                         "cost_per_1k": round(cost, 4)})
        print(f"[{ds}] cascade dihitung (haiku$={ch} sonnet$={cs} opus$={co})")

    if not rows:
        print("Belum ada dataset lengkap."); return
    out = OUT / "c2_cascade_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'dataset':8s} {'tau':>4s} {'macroF1':>8s} {'%Sonnet':>8s} {'%Opus':>6s} {'$/1k':>6s}")
    print("-" * 46)
    for r in rows:
        print(f"{r['dataset']:8s} {r['tau']:>4d} {r['macro_f1']:8.4f} {r['frac_sonnet']*100:7.1f}% "
              f"{r['frac_opus']*100:5.1f}% {r['cost_per_1k']:6.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
