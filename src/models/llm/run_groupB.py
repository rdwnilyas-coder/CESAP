"""Orchestrator Group B (LLM): jalankan kombinasi dataset x arm via subprocess run_llm.py,
lalu agregasi metrics.json + cek gate PoC (parse>98%, biaya terhitung, tak ada kelas kolaps).

Arm (model x mode) default = arm Anthropic di CLAUDE.md sec.4:
  haiku-zero, haiku-few, sonnet-few, opus-few.
(competitor = Gemini Flash / GPT-4o-mini BELUM disertakan — perlu klien non-Anthropic terpisah.)

Default --limit 300 = protokol PoC Fase 1 (stratified/dataset). --limit 0 = full golden test.

    # PoC dry-run (tanpa API key), validasi alur:
    python src/models/llm/run_groupB.py --datasets nusax --limit 15 --dry_run
    # PoC nyata (butuh ANTHROPIC_API_KEY):
    python src/models/llm/run_groupB.py --datasets smsa nusax ugm finance
    python src/models/llm/run_groupB.py --aggregate-only
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "groupB"
RUN_LLM = ROOT / "src" / "models" / "llm" / "run_llm.py"

DATASETS = ["smsa", "nusax", "ugm", "finance"]
ARMS = ["haiku-zero", "haiku-few", "sonnet-few", "opus-few"]   # (model)-(mode)
PARSE_GATE = 0.98


def aggregate() -> None:
    rows = []
    for mj in sorted(RESULTS.rglob("metrics.json")):
        m = json.loads(mj.read_text(encoding="utf-8"))
        rows.append(m)
    if not rows:
        print("(belum ada metrics.json)")
        return
    import csv
    fields = ["dataset", "arm", "mode", "dry_run", "n", "parse_success_rate",
              "n_parse_fail", "cost_per_1k_usd", "total_cost_usd",
              "accuracy", "macro_f1", "avg_in_tok", "cache_read_total"]
    out = RESULTS / "summary_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'dataset':8s} {'arm-mode':12s} {'n':>4s} {'parse%':>6s} {'$/1k':>7s} {'$tot':>7s} {'acc':>6s} {'macroF1':>8s}"
    print(hdr); print("-" * len(hdr))
    total_cost = 0.0
    fails = []
    for r in rows:
        am = f"{r['arm']}-{r['mode']}" + ("(DRY)" if r.get("dry_run") else "")
        total_cost += r.get("total_cost_usd", 0)
        pr = r.get("parse_success_rate", 0)
        if not r.get("dry_run") and pr < PARSE_GATE:
            fails.append(f"{r['dataset']}/{am} parse={pr:.3f}")
        print(f"{r['dataset']:8s} {am:12s} {r['n']:>4d} {pr*100:5.1f}% "
              f"{r.get('cost_per_1k_usd',0):7.3f} {r.get('total_cost_usd',0):7.3f} "
              f"{r.get('accuracy',float('nan')):6.3f} {r.get('macro_f1',float('nan')):8.4f}")
    print(f"\nTOTAL biaya semua arm: ${total_cost:.4f}")
    print("GATE PoC parse>98%:", "LULUS" if not fails else f"GAGAL -> {fails}")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--limit", type=int, default=300, help="0 = full golden test")
    ap.add_argument("--n_shot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(); return

    total = len(args.datasets) * len(args.arms)
    i = 0
    for ds in args.datasets:
        for arm_mode in args.arms:
            i += 1
            arm, mode = arm_mode.split("-")
            print(f"\n{'='*70}\n[{i}/{total}] {arm_mode} x {ds}\n{'='*70}", flush=True)
            cmd = [sys.executable, str(RUN_LLM), "--dataset", ds, "--arm", arm,
                   "--mode", mode, "--n_shot", str(args.n_shot), "--seed", str(args.seed)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.dry_run:
                cmd += ["--dry_run"]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!! GAGAL: {arm_mode} x {ds} (exit {r.returncode}) — lanjut", flush=True)
    print("\n=== AGGREGATE ===")
    aggregate()


if __name__ == "__main__":
    main()
