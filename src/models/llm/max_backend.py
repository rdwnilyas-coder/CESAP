"""Backend Group B via inferensi Claude Code Max (tanpa API key).

Pemisahan tugas:
  prepare : Python bangun system prompt + daftar item + biaya modeled (tiktoken x harga
            resmi). Tulis _tasks.json (lengkap, utk scoring) & _classify_input.json
            (hanya id+text, TANPA label asli -> diberikan ke subagent classifier).
  [klasifikasi: DILAKUKAN ORCHESTRATOR (Claude Code) via subagent dgn model di-switch
   per-arm; tulis hasil ke _labels.json = {id: "negative|neutral|positive"}]
  score   : Python baca _tasks.json + _labels.json -> prediksi per-sampel + biaya
            modeled + metrik. Output kompatibel dgn results/groupB (backend="max").

    python src/models/llm/max_backend.py prepare --dataset nusax --arm haiku --mode few --limit 12
    # (orchestrator klasifikasi -> tulis _labels.json)
    python src/models/llm/max_backend.py score   --dataset nusax --arm haiku --mode few
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import tiktoken
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from prompt import PROMPT_VERSION, build_system, build_user, select_fewshot
from schema import LABELS

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "groupB"
ENC = tiktoken.get_encoding("cl100k_base")
tok = lambda s: len(ENC.encode(s))
OUT_TOK = tok('{"label": "negative"}')

# id model (dilaporkan) + harga per MTok resmi 2026
MODEL_META = {
    "haiku":  ("claude-haiku-4-5",  1.0, 5.0),
    "sonnet": ("claude-sonnet-4-6", 3.0, 15.0),
    "opus":   ("claude-opus-4-8",   5.0, 25.0),
}


def _tag(arm, mode):
    return f"{arm}-{mode}-max"


def _subset(test, limit, seed):
    if not limit or limit >= len(test):
        return test
    frac = limit / len(test)
    parts = [g.sample(n=min(len(g), max(1, round(len(g) * frac))), random_state=seed)
             for _, g in test.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def prepare(args):
    test = _subset(pd.read_parquet(PROC / args.dataset / "test.parquet"), args.limit, args.seed)
    fewshot = None
    if args.mode == "few":
        train = pd.read_parquet(PROC / args.dataset / "train.parquet")
        fewshot = select_fewshot(train, args.n_shot, args.seed)
    system_text = build_system(fewshot)
    prefix_tok = tok(system_text)
    model_id, pin, pout = MODEL_META[args.arm]

    items = []
    for r in test.itertuples(index=False):
        user = build_user(r.text)
        in_tok = prefix_tok + tok(user)
        items.append({"id": r.id, "text": r.text, "true_label": r.label,
                      "in_tok": in_tok,
                      "cost_usd": (in_tok * pin + OUT_TOK * pout) / 1e6})

    out_dir = RESULTS / args.dataset / _tag(args.arm, args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_tasks.json").write_text(json.dumps({
        "dataset": args.dataset, "arm": args.arm, "mode": args.mode,
        "model_id": model_id, "prompt_version": PROMPT_VERSION,
        "n_shot": args.n_shot if fewshot else 0, "seed": args.seed,
        "prefix_tok": prefix_tok, "items": items,
    }, ensure_ascii=False), encoding="utf-8")
    (out_dir / "system_prompt.txt").write_text(system_text, encoding="utf-8")
    # input untuk classifier: TANPA label asli (cegah kebocoran ke pengklasifikasi)
    (out_dir / "_classify_input.json").write_text(json.dumps(
        [{"id": it["id"], "text": it["text"]} for it in items], ensure_ascii=False),
        encoding="utf-8")
    print(f"prepared {len(items)} item -> {out_dir}")
    print(f"model={model_id} | prefix_tok={prefix_tok} | total_modeled_cost=${sum(i['cost_usd'] for i in items):.4f}")
    print(f"classifier input: {out_dir / '_classify_input.json'}")
    print(f"system prompt   : {out_dir / 'system_prompt.txt'}")


POC_ARMS = [("haiku", "zero"), ("haiku", "few"), ("sonnet", "few"), ("opus", "few")]


def bundle(args):
    """Siapkan SEMUA arm PoC sekaligus + _bundle.json (args utk Workflow)."""
    test = _subset(pd.read_parquet(PROC / args.dataset / "test.parquet"), args.limit, args.seed)
    train = pd.read_parquet(PROC / args.dataset / "train.parquet")
    fewshot = select_fewshot(train, args.n_shot, args.seed)
    sys_text = {"zero": build_system(None), "few": build_system(fewshot)}
    prefix = {k: tok(v) for k, v in sys_text.items()}
    items = [{"id": r.id, "text": r.text} for r in test.itertuples(index=False)]

    arms_meta = []
    for arm, mode in POC_ARMS:
        model_id, pin, pout = MODEL_META[arm]
        tag = f"{arm}-{mode}-max"
        rows = []
        for r in test.itertuples(index=False):
            it = prefix[mode] + tok(build_user(r.text))
            rows.append({"id": r.id, "text": r.text, "true_label": r.label,
                         "in_tok": it, "cost_usd": (it * pin + OUT_TOK * pout) / 1e6})
        od = RESULTS / args.dataset / tag
        od.mkdir(parents=True, exist_ok=True)
        (od / "_tasks.json").write_text(json.dumps({
            "dataset": args.dataset, "arm": arm, "mode": mode, "model_id": model_id,
            "prompt_version": PROMPT_VERSION, "n_shot": args.n_shot if mode == "few" else 0,
            "seed": args.seed, "prefix_tok": prefix[mode], "items": rows}, ensure_ascii=False), encoding="utf-8")
        (od / "system_prompt.txt").write_text(sys_text[mode], encoding="utf-8")
        arms_meta.append({"tag": tag, "model": arm, "mode": mode})

    bdir = RESULTS / args.dataset
    (bdir / "_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"bundle {args.dataset}: {len(items)} item x {len(arms_meta)} arm")
    print(f"  items -> {bdir / '_items.json'}")
    for a in arms_meta:
        print(f"  arm dir -> {RESULTS / args.dataset / a['tag']}  (model={a['model']})")


def _merge_labels(od) -> dict:
    """Gabung semua _labels_b*.json + _labels_rep*.json (perbaikan)."""
    merged = {}
    for pat in ("_labels_b*.json", "_labels_rep*.json"):
        for f in sorted(od.glob(pat)):
            try:
                merged.update(json.loads(f.read_text(encoding="utf-8-sig")))  # toleran BOM
            except Exception as e:
                print(f"  (skip {f.name}: {e})")
    return merged


def finalize(args):
    """Merge label batch+perbaikan -> _labels.json, lalu score semua arm."""
    import types
    for arm, mode in POC_ARMS:
        od = RESULTS / args.dataset / f"{arm}-{mode}-max"
        merged = _merge_labels(od)
        if not merged:
            print(f"!! {od.name}: tak ada label"); continue
        (od / "_labels.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        score(types.SimpleNamespace(dataset=args.dataset, arm=arm, mode=mode))


def missing(args):
    """Tulis _missing.json per arm = item yang belum punya label valid (utk repair)."""
    total_missing = 0
    for arm, mode in POC_ARMS:
        od = RESULTS / args.dataset / f"{arm}-{mode}-max"
        tasks = json.loads((od / "_tasks.json").read_text(encoding="utf-8"))
        cur = _merge_labels(od)
        miss = [{"id": it["id"], "text": it["text"]} for it in tasks["items"]
                if cur.get(it["id"]) not in LABELS]
        (od / "_missing.json").write_text(json.dumps(miss, ensure_ascii=False), encoding="utf-8")
        total_missing += len(miss)
        print(f"{od.name}: missing={len(miss)}")
    print(f"TOTAL missing={total_missing}")


def score(args):
    out_dir = RESULTS / args.dataset / _tag(args.arm, args.mode)
    tasks = json.loads((out_dir / "_tasks.json").read_text(encoding="utf-8"))
    labels = json.loads((out_dir / "_labels.json").read_text(encoding="utf-8"))

    rows = []
    for it in tasks["items"]:
        pred = labels.get(it["id"])
        ok = pred in LABELS
        rows.append({"id": it["id"], "text": it["text"], "true_label": it["true_label"],
                     "pred_label": pred if ok else None, "parse_ok": ok,
                     "in_tok": it["in_tok"], "out_tok": OUT_TOK, "cost_usd": it["cost_usd"]})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "predictions.parquet", index=False)

    parsed = df[df["parse_ok"]]
    m = {"dataset": args.dataset, "arm": args.arm, "mode": args.mode,
         "backend": "max", "model_id": tasks["model_id"], "prompt_version": tasks["prompt_version"],
         "n": len(df), "n_shot": tasks["n_shot"], "seed": tasks["seed"],
         "parse_success_rate": round(float(df["parse_ok"].mean()), 4),
         "n_parse_fail": int((~df["parse_ok"]).sum()),
         "total_cost_usd_modeled": round(float(df["cost_usd"].sum()), 6),
         "cost_per_1k_usd_modeled": round(float(df["cost_usd"].sum()) / len(df) * 1000, 4),
         "cost_note": "modeled = token(tiktoken proxy) x harga resmi; inferensi via Claude Code Max (billing nyata $0)"}
    if len(parsed):
        yt, yp = parsed["true_label"].tolist(), parsed["pred_label"].tolist()
        m["accuracy"] = round(accuracy_score(yt, yp), 4)
        m["macro_f1"] = round(f1_score(yt, yp, labels=list(LABELS), average="macro", zero_division=0), 4)
        m["confusion"] = {"labels": list(LABELS),
                          "matrix": confusion_matrix(yt, yp, labels=list(LABELS)).tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"parse_rate={m['parse_success_rate']} | cost/1k(modeled)=${m['cost_per_1k_usd_modeled']} "
          f"| acc={m.get('accuracy','-')} macroF1={m.get('macro_f1','-')}")
    print(f"saved -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "score"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=["smsa", "nusax", "ugm", "finance"])
        p.add_argument("--arm", required=True, choices=list(MODEL_META))
        p.add_argument("--mode", required=True, choices=["zero", "few"])
        p.add_argument("--n_shot", type=int, default=6)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--seed", type=int, default=42)
    pb = sub.add_parser("bundle")
    pb.add_argument("--dataset", required=True, choices=["smsa", "nusax", "ugm", "finance"])
    pb.add_argument("--n_shot", type=int, default=6)
    pb.add_argument("--limit", type=int, default=100)
    pb.add_argument("--seed", type=int, default=42)
    for nm in ("finalize", "missing"):
        pp = sub.add_parser(nm)
        pp.add_argument("--dataset", required=True, choices=["smsa", "nusax", "ugm", "finance"])
    args = ap.parse_args()
    {"prepare": prepare, "score": score, "bundle": bundle,
     "finalize": finalize, "missing": missing}[args.cmd](args)


if __name__ == "__main__":
    main()
