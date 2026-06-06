"""Orkestrasi Group B: jalankan satu arm LLM (model x mode) pada golden test (atau subset PoC).

Menyimpan prediksi per-sampel + usage/biaya, metrik (akurasi, macro-F1, parse-rate),
dan config -> results/groupB/<dataset>/<arm>-<mode>/.

Contoh:
    # dry-run (tanpa API key) untuk validasi pipeline:
    python src/models/llm/run_llm.py --dataset nusax --arm haiku --mode few --limit 20 --dry_run
    # PoC nyata (butuh ANTHROPIC_API_KEY):
    python src/models/llm/run_llm.py --dataset nusax --arm sonnet --mode few --limit 300
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from client import LLMClient, MODELS, system_blocks
from prompt import PROMPT_VERSION, build_system, build_user, select_fewshot
from schema import LABELS

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "groupB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["smsa", "nusax", "ugm", "finance"])
    ap.add_argument("--arm", required=True, choices=list(MODELS))
    ap.add_argument("--mode", required=True, choices=["zero", "few"])
    ap.add_argument("--n_shot", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="subset stratified utk PoC/dry-run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    test = pd.read_parquet(PROC / args.dataset / "test.parquet")
    if args.limit and args.limit < len(test):
        frac = args.limit / len(test)
        parts = [g.sample(n=min(len(g), max(1, round(len(g) * frac))), random_state=args.seed)
                 for _, g in test.groupby("label")]
        test = pd.concat(parts).reset_index(drop=True)

    fewshot = None
    if args.mode == "few":
        train = pd.read_parquet(PROC / args.dataset / "train.parquet")
        fewshot = select_fewshot(train, args.n_shot, args.seed)
    sys_text = build_system(fewshot)
    sys_blocks = system_blocks(sys_text)

    client = LLMClient(args.arm, dry_run=args.dry_run)
    tag = f"{args.arm}-{args.mode}{'-DRY' if args.dry_run else ''}"
    print(f"=== {tag} x {args.dataset} | n={len(test)} | model={MODELS[args.arm]['id']} "
          f"| fewshot={args.n_shot if fewshot else 0} ===", flush=True)

    rows = []
    t0 = time.time()
    for i, r in enumerate(test.itertuples(index=False)):
        p = client.classify(sys_blocks, build_user(r.text))
        rows.append({
            "id": r.id, "text": r.text, "true_label": r.label,
            "pred_label": p.label, "parse_ok": p.parse_ok,
            "in_tok": p.usage["input_tokens"], "out_tok": p.usage["output_tokens"],
            "cache_read": p.usage["cache_read_input_tokens"],
            "cache_creation": p.usage["cache_creation_input_tokens"],
            "cost_usd": p.cost_usd, "raw": p.raw,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)
    secs = time.time() - t0

    df = pd.DataFrame(rows)
    out_dir = RESULTS / args.dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "predictions.parquet", index=False)

    parsed = df[df["parse_ok"]]
    parse_rate = float(df["parse_ok"].mean())
    metrics = {
        "dataset": args.dataset, "arm": args.arm, "mode": args.mode,
        "model_id": MODELS[args.arm]["id"], "prompt_version": PROMPT_VERSION,
        "dry_run": args.dry_run, "n": int(len(df)), "n_shot": args.n_shot if fewshot else 0,
        "seed": args.seed, "parse_success_rate": round(parse_rate, 4),
        "n_parse_fail": int((~df["parse_ok"]).sum()),
        "total_cost_usd": round(float(df["cost_usd"].sum()), 6),
        "cost_per_1k_usd": round(float(df["cost_usd"].sum()) / len(df) * 1000, 4),
        "avg_in_tok": round(float(df["in_tok"].mean()), 1),
        "avg_out_tok": round(float(df["out_tok"].mean()), 1),
        "cache_read_total": int(df["cache_read"].sum()),
        "secs": round(secs, 1),
    }
    if len(parsed):
        yt = parsed["true_label"].tolist()
        yp = parsed["pred_label"].tolist()
        metrics["accuracy"] = round(accuracy_score(yt, yp), 4)
        metrics["macro_f1"] = round(f1_score(yt, yp, labels=list(LABELS), average="macro", zero_division=0), 4)
        metrics["confusion"] = {"labels": list(LABELS),
                                "matrix": confusion_matrix(yt, yp, labels=list(LABELS)).tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "system_prompt.txt").write_text(sys_text, encoding="utf-8")

    print(f"\nparse_rate={parse_rate:.3f} | cost/1k=${metrics['cost_per_1k_usd']} "
          f"| total=${metrics['total_cost_usd']}", flush=True)
    if "macro_f1" in metrics:
        print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
