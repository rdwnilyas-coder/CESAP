"""Pareto frontier cost/1k vs Macro-F1 menggabungkan Group A (open-source, $0) +
Group B (LLM via Max, biaya modeled). Output: results/pareto_table.csv + paper/fig_pareto.png/pdf.

    python src/eval/plot_pareto.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GA = ROOT / "results" / "groupA"
GB = ROOT / "results" / "groupB"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
TEAL, ORNG, SLATE = "#2a9d8f", "#e76f51", "#64748b"


def load():
    """rows: dataset, group, arm, cost_per_1k, macro_f1."""
    rows = []
    # Group A (open-source, cost 0) dari summary_table.csv
    ga = GA / "summary_table.csv"
    if ga.exists():
        for r in csv.DictReader(open(ga, encoding="utf-8")):
            rows.append({"dataset": r["dataset"], "group": "open-source",
                         "arm": r["arm"], "cost": 0.0, "f1": float(r["macro_f1"])})
    # Group B (LLM, modeled cost) dari metrics.json backend=max
    for mj in GB.rglob("metrics.json"):
        m = json.load(open(mj, encoding="utf-8"))
        if m.get("backend") == "max":
            rows.append({"dataset": m["dataset"], "group": "LLM",
                         "arm": f"{m['arm']}-{m['mode']}", "cost": m["cost_per_1k_usd_modeled"],
                         "f1": m["macro_f1"]})
    # Group C1 (hybrid routing) dari routing_curve.csv (tau antara 0 dan 1)
    rc = ROOT / "results" / "hybrid" / "routing_curve.csv"
    if rc.exists():
        for r in csv.DictReader(open(rc, encoding="utf-8")):
            t = float(r["tau"])
            if 0 < t < 1:
                rows.append({"dataset": r["dataset"], "group": "hybrid (C1)",
                             "arm": f"t{t}", "cost": float(r["cost_per_1k"]), "f1": float(r["macro_f1"])})
    return rows


def pareto_front(pts):
    """pts: list (cost, f1). Kembalikan subset non-dominated, urut cost naik."""
    s = sorted(pts, key=lambda p: (p[0], -p[1]))
    front, best = [], -1
    for c, f in s:
        if f > best:
            front.append((c, f)); best = f
    return front


def main():
    rows = load()
    out = ROOT / "results" / "pareto_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "group", "arm", "cost", "f1"])
        w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ds in zip(axes.flat, DATASETS):
        dr = [r for r in rows if r["dataset"] == ds]
        oss = [r for r in dr if r["group"] == "open-source"]
        llm = [r for r in dr if r["group"] == "LLM"]
        hyb = [r for r in dr if r["group"] == "hybrid (C1)"]
        ax.scatter([r["cost"] for r in hyb], [r["f1"] for r in hyb], s=38, c="#9467bd",
                   marker="^", edgecolor="white", zorder=3, label="Hybrid C1 routing")
        ax.scatter([r["cost"] for r in oss], [r["f1"] for r in oss], s=70, c=TEAL,
                   edgecolor="white", zorder=4, label="Open-source ($0)")
        ax.scatter([r["cost"] for r in llm], [r["f1"] for r in llm], s=70, c=ORNG,
                   edgecolor="white", zorder=4, label="LLM (Max, modeled $)")
        for r in oss + llm:    # anotasi hanya titik utama (hindari berantakan)
            ax.annotate(r["arm"].replace("indobert-", "ib-").replace("-max", ""),
                        (r["cost"], r["f1"]), fontsize=6.5, color="#333",
                        xytext=(3, 3), textcoords="offset points")
        front = pareto_front([(r["cost"], r["f1"]) for r in dr])
        ax.plot([c for c, _ in front], [f for _, f in front], "--", color=SLATE,
                lw=1.3, zorder=2, label="Pareto frontier")
        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("Cost / 1k predictions (USD)")
        ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Cost vs Accuracy — Indonesian Sentiment (Group A open-source $0 + Group B LLM)\n"
                 "LLM cost = modeled (token x harga resmi); inferensi via Claude Code Max",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = ROOT / "paper" / "fig_pareto.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "paper" / "fig_pareto.pdf", bbox_inches="tight", facecolor="white")
    print(f"rows={len(rows)} -> {out}\nfigure -> {png}")


if __name__ == "__main__":
    main()
