"""Bangun prompt sentimen ID v1: instruksi sistem (cacheable) + few-shot, dan
pesan user per-sampel. Few-shot diambil dari TRAIN (anti-kebocoran, CLAUDE.md sec.6).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "prompts" / "sentiment_id_v1.md"
PROMPT_VERSION = "sentiment_id_v1"
EX_MAXLEN = 240  # batasi panjang contoh agar prompt tak meledak


def _instruction() -> str:
    raw = TEMPLATE.read_text(encoding="utf-8")
    if "-->" in raw:                      # buang blok komentar HTML di awal
        raw = raw.split("-->", 1)[1]
    return raw.strip()


MIN_EX_LEN = 40  # buang contoh trivial/terpotong; pilih contoh informatif & utuh


def select_fewshot(train_df: pd.DataFrame, n_shot: int, seed: int) -> list[tuple[str, str]]:
    """n_shot contoh, seimbang antar kelas, deterministik (seed).
    Hanya contoh utuh & informatif (panjang MIN_EX_LEN..EX_MAXLEN) -> hindari
    contoh trivial/terpotong yang membiaskan model (lihat diagnosa few-shot)."""
    per_class = max(1, n_shot // 3)
    picked = []
    for label in ("negative", "neutral", "positive"):
        sub = train_df[train_df["label"] == label]
        good = sub[sub["text"].str.len().between(MIN_EX_LEN, EX_MAXLEN)]
        if len(good) >= per_class:
            sub = good
        sub = sub.sample(n=min(per_class, len(sub)), random_state=seed)
        for _, r in sub.iterrows():
            picked.append((str(r["text"])[:EX_MAXLEN], label))
    # acak urutan antar-kelas (deterministik)
    rng = __import__("random").Random(seed)
    rng.shuffle(picked)
    return picked[:n_shot] if n_shot else picked


def build_system(fewshot: list[tuple[str, str]] | None) -> str:
    parts = [_instruction()]
    if fewshot:
        parts.append("\nContoh:")
        for text, label in fewshot:
            parts.append(f'Teks: "{text}"\nLabel: {label}')
    return "\n".join(parts)


def build_user(text: str) -> str:
    return f'Klasifikasikan sentimen teks berikut.\nTeks: "{text}"'
