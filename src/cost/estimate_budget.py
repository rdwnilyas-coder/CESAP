"""Estimasi biaya API eksperimen LLM (Group B + C) — token NYATA x harga resmi.

Token dihitung dengan tiktoken (cl100k) atas prompt asli (instruksi + few-shot dari
prompts/sentiment_id_v1.md) dan teks asli tiap dataset. Proxy tokenizer Anthropic
(~+/-10-15%); kalibrasi pasti dari API count_tokens saat PoC. Caching diasumsikan
TIDAK aktif (prefix few-shot < 4096 tok minimum Haiku/Opus) — angka = tanpa-cache.

    python src/cost/estimate_budget.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src" / "models" / "llm"))
from prompt import build_system, build_user, select_fewshot  # noqa: E402

DATASETS = ["smsa", "nusax", "ugm", "finance"]
ENC = tiktoken.get_encoding("cl100k_base")
tok = lambda s: len(ENC.encode(s))

# Harga per 1 JUTA token (USD), 2026. price_out = 5x price_in.
PRICE = {"haiku": (1.0, 5.0), "sonnet": (3.0, 15.0), "opus": (5.0, 25.0),
         "competitor": (0.15, 0.60)}
OUT_TOK = tok('{"label": "negative"}')   # output terstruktur ~konstan
N_SHOT = 6


def stats():
    s = {}
    for ds in DATASETS:
        tr = pd.read_parquet(PROC / ds / "train.parquet")
        te = pd.read_parquet(PROC / ds / "test.parquet")
        fewshot = select_fewshot(tr, N_SHOT, 42)
        prefix_few = tok(build_system(fewshot))      # instruksi + few-shot (cacheable block)
        prefix_zero = tok(build_system(None))        # instruksi saja
        # rata-rata token pesan user (test & train), sampel utk kecepatan
        te_s = te["text"].sample(min(200, len(te)), random_state=42)
        tr_s = tr["text"].sample(min(200, len(tr)), random_state=42)
        s[ds] = {
            "n_train": len(tr), "n_test": len(te),
            "sample_tok": sum(tok(build_user(t)) for t in te_s) / len(te_s),
            "sample_tok_tr": sum(tok(build_user(t)) for t in tr_s) / len(tr_s),
            "prefix_few": prefix_few, "prefix_zero": prefix_zero,
        }
    return s


def cost(model, n_req, sample_tok, prefix_tok):
    """No-cache: tiap request bayar prefix (instruksi+few-shot) + teks penuh."""
    pin, pout = PRICE[model]
    in_tok = n_req * (prefix_tok + sample_tok)
    return (in_tok * pin + n_req * OUT_TOK * pout) / 1e6


def main():
    s = stats()
    print(f"{'ds':8s} {'n_test':>6s} {'n_train':>7s} {'prefix_few':>10s} {'prefix_zero':>11s} {'sample_tok':>10s}")
    for ds in DATASETS:
        d = s[ds]
        print(f"{ds:8s} {d['n_test']:>6d} {d['n_train']:>7d} {d['prefix_few']:>10.0f} "
              f"{d['prefix_zero']:>11.0f} {d['sample_tok']:>10.1f}")
    print(f"(OUT_TOK={OUT_TOK}, few-shot={N_SHOT}; caching OFF: prefix_few < 4096 min)\n")

    # Group B: haiku-zero, haiku-few, sonnet-few, opus-few, competitor-few
    b_poc = b_full = 0.0
    for ds in DATASETS:
        d = s[ds]; N = d["n_test"]; poc = min(300, N)
        for model, pref in [("haiku", d["prefix_zero"]), ("haiku", d["prefix_few"]),
                            ("sonnet", d["prefix_few"]), ("opus", d["prefix_few"]),
                            ("competitor", d["prefix_few"])]:
            b_full += cost(model, N, d["sample_tok"], pref)
            b_poc += cost(model, poc, d["sample_tok"], pref)

    c1 = c2 = c3_opus = c3_sonnet = c3_haiku = 0.0
    for ds in DATASETS:
        d = s[ds]; N = d["n_test"]; tr = d["n_train"]
        c1 += cost("sonnet", int(0.30 * N), d["sample_tok"], d["prefix_few"])
        c2 += cost("haiku", N, d["sample_tok"], d["prefix_few"])
        c2 += cost("sonnet", int(0.40 * N), d["sample_tok"], d["prefix_few"])
        c2 += cost("opus", int(0.15 * N), d["sample_tok"], d["prefix_few"])
        c3_opus += cost("opus", tr, d["sample_tok_tr"], d["prefix_few"])
        c3_sonnet += cost("sonnet", tr, d["sample_tok_tr"], d["prefix_few"])
        c3_haiku += cost("haiku", tr, d["sample_tok_tr"], d["prefix_few"])

    print(f"Group B  PoC (~300/ds)               : ${b_poc:8.2f}")
    print(f"Group B  Full test                   : ${b_full:8.2f}")
    print(f"Group C1 routing (~30% Sonnet)       : ${c1:8.2f}")
    print(f"Group C2 cascade (H->S->O)           : ${c2:8.2f}")
    print(f"Group C3 distillation Opus (22k)     : ${c3_opus:8.2f}")
    print(f"Group C3 distillation Sonnet (22k)   : ${c3_sonnet:8.2f}")
    print(f"Group C3 distillation Haiku  (22k)   : ${c3_haiku:8.2f}")
    print("-" * 46)
    core = b_poc + b_full + c1 + c2
    print(f"B + C1 + C2 (TANPA C3)               : ${core:8.2f}")
    print(f"  + C3-Opus   = GOLD                 : ${core + c3_opus:8.2f}")
    print(f"  + C3-Sonnet = SEIMBANG             : ${core + c3_sonnet:8.2f}")
    print(f"  + C3-Haiku                          : ${core + c3_haiku:8.2f}")
    print(f"\nCATATAN: C3 = pelabelan train oleh model 'guru'. Bila guru dijalankan via")
    print(f"inferensi Claude Code Max (bukan API metered), komponen C3 -> $0 biaya nyata")
    print(f"(biaya tetap dilaporkan sebagai modeled = token x harga resmi di atas).")


if __name__ == "__main__":
    main()
