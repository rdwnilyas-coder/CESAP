"""Kurva routing C1: %traffic ke LLM vs Macro-F1 per dataset (cari knee).
Input results/hybrid/routing_curve.csv -> paper/fig_routing.png/pdf.

    python src/eval/plot_routing.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["smsa", "nusax", "ugm", "finance"]


def main():
    rc = ROOT / "results" / "hybrid" / "routing_curve.csv"
    data = defaultdict(list)
    for r in csv.DictReader(open(rc, encoding="utf-8")):
        data[r["dataset"]].append((float(r["tau"]), float(r["esc_frac"]) * 100,
                                   float(r["macro_f1"]), float(r["cost_per_1k"])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, ds in zip(axes.flat, DATASETS):
        pts = sorted(data[ds])
        x = [p[1] for p in pts]      # % ke LLM
        y = [p[2] for p in pts]      # macro-F1
        ax.plot(x, y, "-o", color="#9467bd", ms=4, lw=1.4)
        # garis acuan: open-source (tau0) & LLM penuh (tau1)
        f_oss = pts[0][2]; f_llm = pts[-1][2]
        ax.axhline(f_oss, ls=":", c="#2a9d8f", lw=1, label=f"open-source $0 ({f_oss:.3f})")
        ax.axhline(f_llm, ls=":", c="#e76f51", lw=1, label=f"Sonnet penuh ({f_llm:.3f})")
        # knee = F1 maksimum
        best = max(pts, key=lambda p: p[2])
        ax.scatter([best[1]], [best[2]], s=90, facecolor="none", edgecolor="black",
                   lw=1.5, zorder=5)
        ax.annotate(f"knee t={best[0]:.2f}\n{best[2]:.3f} @ {best[1]:.0f}% (${best[3]:.2f}/1k)",
                    (best[1], best[2]), fontsize=7, xytext=(8, -22), textcoords="offset points")
        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("% traffic dieskalasi ke LLM")
        ax.set_ylabel("Macro-F1")
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("C1 Confidence Routing — IndoBERT-base ($0) -> Sonnet (uncertain cases)\n"
                 "Akurasi mendekati/melebihi LLM penuh dgn sebagian kecil traffic & biaya",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = ROOT / "paper" / "fig_routing.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "paper" / "fig_routing.pdf", bbox_inches="tight", facecolor="white")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
