"""Skema output terstruktur untuk klasifikasi sentimen LLM (Group B).

Dipakai dengan client.messages.parse(output_format=SentimentOutput) sehingga
respons divalidasi otomatis ke salah satu dari 3 label kanonik. Output gagal-parse
ditolak & dicatat (CLAUDE.md sec.8) — tidak di-default diam-diam.
"""
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel

LABELS = ("negative", "neutral", "positive")


class SentimentOutput(BaseModel):
    label: Literal["negative", "neutral", "positive"]
