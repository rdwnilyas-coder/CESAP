"""Orchestrator Group A: jalankan finetune.py untuk kombinasi dataset x arm
secara berurutan (subprocess per-run -> GPU dibebaskan tiap run), lalu agregasi
semua metrics.json jadi tabel ringkas.

    python src/models/opensource/run_groupA.py --datasets smsa --arms indobert-base lightweight indobertweet indobert-large
    python src/models/opensource/run_groupA.py            # default: semua 4x4
    python src/models/opensource/run_groupA.py --aggregate-only
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "groupA"
FINETUNE = ROOT / "src" / "models" / "opensource" / "finetune.py"

DATASETS = ["smsa", "nusax", "ugm", "finance"]
ARMS = ["indobert-base", "lightweight", "indobertweet", "indobert-large"]


def aggregate() -> None:
    rows = []
    for mj in sorted(RESULTS.rglob("metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        t = m["test_metrics"]
        rows.append({
            "dataset": m["dataset"], "arm": m["arm"], "method": m["method"],
            "accuracy": round(t["accuracy"], 4), "macro_f1": round(t["macro_f1"], 4),
            "weighted_f1": round(t["weighted_f1"], 4),
            "f1_neg": round(t["per_class"]["negative"]["f1"], 3),
            "f1_neu": round(t["per_class"]["neutral"]["f1"], 3),
            "f1_pos": round(t["per_class"]["positive"]["f1"], 3),
            "train_secs": m["timing_secs"]["train"], "peak_mib": m.get("peak_gpu_mib"),
        })
    if not rows:
        print("(belum ada metrics.json)")
        return
    import csv
    out = RESULTS / "summary_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # cetak tabel
    hdr = f"{'dataset':8s} {'arm':14s} {'meth':5s} {'acc':>6s} {'macroF1':>8s} {'wF1':>6s} {'neg':>5s} {'neu':>5s} {'pos':>5s} {'sec':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:8s} {r['arm']:14s} {r['method']:5s} {r['accuracy']:6.3f} "
              f"{r['macro_f1']:8.4f} {r['weighted_f1']:6.3f} {r['f1_neg']:5.2f} "
              f"{r['f1_neu']:5.2f} {r['f1_pos']:5.2f} {r['train_secs']:6.0f}")
    print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    total = len(args.datasets) * len(args.arms)
    i = 0
    for ds in args.datasets:
        for arm in args.arms:
            i += 1
            print(f"\n{'='*70}\n[{i}/{total}] {arm} x {ds}\n{'='*70}", flush=True)
            r = subprocess.run([sys.executable, str(FINETUNE),
                                "--dataset", ds, "--arm", arm])
            if r.returncode != 0:
                print(f"!! GAGAL: {arm} x {ds} (exit {r.returncode}) — lanjut ke berikutnya", flush=True)
    print("\n=== AGGREGATE ===")
    aggregate()


if __name__ == "__main__":
    main()
