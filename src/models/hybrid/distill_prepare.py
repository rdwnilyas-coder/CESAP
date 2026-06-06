"""C3 distillation — prepare: ambil subset train per dataset utk dilabeli Opus (guru)
via Max. Tulis item + gold (utk ukur akurasi guru) + system prompt few-shot.

    python src/models/hybrid/distill_prepare.py --n_label 2000
Output: results/groupC/<ds>/{_items.json,_gold.json,system_prompt.txt,_meta.json}
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "groupC"
sys.path.insert(0, str(ROOT / "src" / "models" / "llm"))
from prompt import build_system, build_user, select_fewshot, PROMPT_VERSION  # noqa: E402

DATASETS = ["smsa", "nusax", "ugm", "finance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_label", type=int, default=2000)
    ap.add_argument("--n_shot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for ds in DATASETS:
        train = pd.read_parquet(PROC / ds / "train.parquet")
        fewshot = select_fewshot(train, args.n_shot, args.seed)
        # subset stratified utk dilabeli guru
        n = min(args.n_label, len(train))
        frac = n / len(train)
        parts = [g.sample(n=min(len(g), max(1, round(len(g) * frac))), random_state=args.seed)
                 for _, g in train.groupby("label")]
        sub = pd.concat(parts).reset_index(drop=True)

        d = OUT / ds
        d.mkdir(parents=True, exist_ok=True)
        (d / "_items.json").write_text(json.dumps(
            [{"id": r.id, "text": r.text} for r in sub.itertuples(index=False)], ensure_ascii=False), encoding="utf-8")
        (d / "_gold.json").write_text(json.dumps(
            {r.id: r.label for r in sub.itertuples(index=False)}, ensure_ascii=False), encoding="utf-8")
        (d / "system_prompt.txt").write_text(build_system(fewshot), encoding="utf-8")
        (d / "_meta.json").write_text(json.dumps(
            {"dataset": ds, "n_label": len(sub), "n_shot": args.n_shot, "seed": args.seed,
             "teacher": "opus", "model_id": "claude-opus-4-8", "prompt_version": PROMPT_VERSION},
            ensure_ascii=False), encoding="utf-8")
        print(f"{ds}: {len(sub)} item train -> {d}")


if __name__ == "__main__":
    main()
