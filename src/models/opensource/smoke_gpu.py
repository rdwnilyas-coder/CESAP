"""Smoke test GPU + IndoBERT (operasional, BUKAN benchmark riset).

Tujuan: pastikan device bisa load IndoBERT & forward pass di GPU.
Head klasifikasi masih acak (belum fine-tune) -> prediksi TIDAK bermakna,
yang dicek hanya: CUDA aktif, tensor di GPU, logits shape benar [N,3].

    python src/models/opensource/smoke_gpu.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MODEL = "indobenchmark/indobert-base-p1"
N = 5
LABELS = ["negative", "neutral", "positive"]


def main() -> None:
    print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
    cuda = torch.cuda.is_available()
    print("cuda available:", cuda)
    if not cuda:
        raise SystemExit("CUDA tidak terdeteksi — cek instalasi torch/driver.")
    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
    torch.cuda.reset_peak_memory_stats()

    texts = pd.read_parquet(ROOT / "data/processed/smsa/test.parquet")["text"].head(N).tolist()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), ignore_mismatched_sizes=True)
    model.to(dev).eval()

    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**enc).logits

    print("\ninput_ids device:", enc["input_ids"].device)
    print("model device     :", next(model.parameters()).device)
    print("logits shape      :", tuple(logits.shape), "(harus", f"({N}, {len(LABELS)}))")
    preds = logits.argmax(-1).tolist()
    for t, p in zip(texts, preds):
        print(f"  [{LABELS[p]:8s}] {t[:70]}")
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\npeak GPU mem: {peak:.0f} MiB  (dari 4096 MiB)")
    assert logits.shape == (N, len(LABELS))
    assert enc["input_ids"].is_cuda and next(model.parameters()).is_cuda
    print("\nSMOKE TEST OK — IndoBERT jalan di GPU.")


if __name__ == "__main__":
    main()
