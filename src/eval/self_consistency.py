"""(4) Self-consistency & kalibrasi confidence LLM, dari data tersimpan (tanpa inferensi baru).

- Stabilitas: 2 run independen tiap model (Group B 'few' vs re-run C2) -> % label sama.
  (Catatan: prompt C2 menambah permintaan confidence -> ini stabilitas thd perturbasi prompt+sampling.)
- Kalibrasi: bin confidence C2 (haiku/sonnet) vs akurasi aktual -> jelaskan over-escalation C2.

    python src/eval/self_consistency.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GB = ROOT / "results" / "groupB"
C2 = ROOT / "results" / "groupC" / "c2"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
MODELS = ["haiku", "sonnet"]
BINS = [(0, 60), (60, 80), (80, 95), (95, 101)]


def merge_c2(d: Path, model: str) -> dict:
    m = {}
    for f in sorted(d.glob(f"{model}_b*.json")):
        m.update(json.loads(f.read_text(encoding="utf-8-sig")))
    return m


def main():
    print("=== Stabilitas label antar 2 run independen (% sama) ===")
    print(f"{'dataset':8s} {'haiku':>8s} {'sonnet':>8s}")
    print("-" * 26)
    stab = {}
    for ds in DATASETS:
        row = {}
        for model in MODELS:
            r1 = pd.read_parquet(GB / ds / f"{model}-few-max" / "predictions.parquet").set_index("id")["pred_label"].to_dict()
            r2 = {k: (v["label"] if isinstance(v, dict) else v) for k, v in merge_c2(C2 / ds, model).items()}
            common = [i for i in r1 if i in r2]
            row[model] = np.mean([r1[i] == r2[i] for i in common])
        stab[ds] = row
        print(f"{ds:8s} {row['haiku']*100:7.1f}% {row['sonnet']*100:7.1f}%")

    print("\n=== Kalibrasi confidence C2 (akurasi vs gold per-bin) ===")
    for model in MODELS:
        print(f"\n  [{model}]  bin-conf -> akurasi (n)")
        for ds in DATASETS:
            gold = pd.read_parquet(GB / ds / "opus-few-max" / "predictions.parquet").set_index("id")["true_label"].to_dict()
            c2 = merge_c2(C2 / ds, model)
            cells = []
            for lo, hi in BINS:
                items = [(c2[i]["label"], gold[i]) for i in c2 if i in gold and isinstance(c2[i], dict) and lo <= c2[i].get("conf", -1) < hi]
                if items:
                    acc = np.mean([p == t for p, t in items])
                    cells.append(f"{lo}-{hi if hi<101 else 100}:{acc:.2f}(n{len(items)})")
                else:
                    cells.append(f"{lo}-{hi if hi<101 else 100}:--")
            print(f"    {ds:8s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
