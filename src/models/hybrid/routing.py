"""C1 — Confidence routing: IndoBERT (router, $0) tangani kasus yakin; eskalasi ke
Sonnet bila softmax confidence < tau. Dihitung post-hoc dari prediksi Group A + B
(tanpa inferensi baru). Sweep tau; ukur Macro-F1, %traffic ke LLM, cost/1k.

cost_hibrida/1k = proporsi_eskalasi * cost_sonnet/1k  (bagian open-source = $0)

    python src/models/hybrid/routing.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
GA = ROOT / "results" / "groupA"
GB = ROOT / "results" / "groupB"
OUT = ROOT / "results" / "hybrid"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
ROUTER = "indobert-base"
LLM = "sonnet-few-max"
TAUS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]   # 0=all router, 1=all LLM


def macro(yt, yp):
    return f1_score(yt, yp, labels=["negative", "neutral", "positive"], average="macro", zero_division=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        r = pd.read_parquet(GA / ds / ROUTER / "predictions.parquet")[
            ["id", "true_label", "pred_label", "confidence"]].rename(
            columns={"pred_label": "router_pred"})
        l = pd.read_parquet(GB / ds / LLM / "predictions.parquet")[
            ["id", "pred_label"]].rename(columns={"pred_label": "llm_pred"})
        df = r.merge(l, on="id")
        sonnet_cost = json.loads((GB / ds / LLM / "metrics.json").read_text(encoding="utf-8"))["cost_per_1k_usd_modeled"]
        f1_router = macro(df["true_label"], df["router_pred"])   # tau=0 (semua router)
        f1_llm = macro(df["true_label"], df["llm_pred"])         # tau=1 (semua LLM)

        for tau in TAUS:
            esc = df["confidence"] < tau if tau > 0 else pd.Series(False, index=df.index)
            if tau >= 1.0:
                esc = pd.Series(True, index=df.index)
            pred = np.where(esc, df["llm_pred"], df["router_pred"])
            rows.append({"dataset": ds, "tau": tau,
                         "esc_frac": round(float(esc.mean()), 4),
                         "macro_f1": round(macro(df["true_label"], pred), 4),
                         "cost_per_1k": round(float(esc.mean()) * sonnet_cost, 4)})
        print(f"[{ds}] router({ROUTER})={f1_router:.4f}  LLM({LLM})={f1_llm:.4f}  sonnet$/1k={sonnet_cost}")

    out = OUT / "routing_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "tau", "esc_frac", "macro_f1", "cost_per_1k"])
        w.writeheader(); w.writerows(rows)

    # cetak ringkas per dataset
    print(f"\n{'dataset':8s} {'tau':>4s} {'%->LLM':>7s} {'macroF1':>8s} {'$/1k':>6s}")
    print("-" * 40)
    for r in rows:
        print(f"{r['dataset']:8s} {r['tau']:>4.2f} {r['esc_frac']*100:6.1f}% {r['macro_f1']:8.4f} {r['cost_per_1k']:6.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
