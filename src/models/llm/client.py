"""Klien LLM Group B (Anthropic). Structured output + prompt caching + akuntansi biaya.

- Model: Haiku 4.5 / Sonnet 4.6 / Opus 4.8 (harga per MTok 2026).
- temperature=0 untuk Haiku/Sonnet; DIHILANGKAN untuk Opus 4.8 (parameter sampling
  dihapus -> 400 bila dikirim). Determinisme Opus diandalkan dari prompt + schema.
- system+few-shot ditandai cache_control ephemeral (reuse antar request).
  CATATAN: caching hanya aktif bila prefix >= minimum model (Haiku/Opus 4096 tok,
  Sonnet 2048). Few-shot pendek (~700 tok) TAK akan ter-cache; perkaya prefix bila perlu.
- Mode dry-run: tanpa API key/SDK; label mock heuristik + usage estimasi, untuk uji pipeline.

Usage fields dibaca: input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens.
"""
from __future__ import annotations
from dataclasses import dataclass

from schema import SentimentOutput

# id + harga per 1 JUTA token (USD), 2026. price_out = 5x price_in.
MODELS = {
    "haiku":  {"id": "claude-haiku-4-5",  "price_in": 1.0, "price_out": 5.0,  "temp": True},
    "sonnet": {"id": "claude-sonnet-4-6", "price_in": 3.0, "price_out": 15.0, "temp": True},
    "opus":   {"id": "claude-opus-4-8",   "price_in": 5.0, "price_out": 25.0, "temp": False},  # Opus 4.8: no temperature
}
MAX_TOKENS = 64
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


@dataclass
class Prediction:
    label: str | None        # None bila gagal parse
    parse_ok: bool
    raw: str
    usage: dict              # in/out/cache_read/cache_creation
    cost_usd: float


def cost_of(arm: str, usage: dict) -> float:
    m = MODELS[arm]
    pin, pout = m["price_in"], m["price_out"]
    c = (usage.get("input_tokens", 0) * pin
         + usage.get("cache_read_input_tokens", 0) * pin * CACHE_READ_MULT
         + usage.get("cache_creation_input_tokens", 0) * pin * CACHE_WRITE_MULT
         + usage.get("output_tokens", 0) * pout)
    return c / 1e6


class LLMClient:
    def __init__(self, arm: str, dry_run: bool = False):
        if arm not in MODELS:
            raise ValueError(arm)
        self.arm = arm
        self.cfg = MODELS[arm]
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            import anthropic
            self._client = anthropic.Anthropic()  # baca ANTHROPIC_API_KEY dari env

    # --- mode dry-run: tanpa API ---
    @staticmethod
    def _mock(system_text: str, user_text: str) -> Prediction:
        t = user_text.lower()
        neg = sum(w in t for w in ("buruk", "jelek", "kecewa", "bosan", "gagal", "rugi", "benci"))
        pos = sum(w in t for w in ("bagus", "mantap", "senang", "puas", "suka", "keren", "terjangkau"))
        label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        # estimasi token kasar (char/3.5); few-shot tak ter-cache di mock
        in_tok = int((len(system_text) + len(user_text)) / 3.5) + 12
        usage = {"input_tokens": in_tok, "output_tokens": 8,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        return Prediction(label, True, f'{{"label": "{label}"}}', usage, 0.0)

    def classify(self, system_blocks: list[dict], user_text: str) -> Prediction:
        if self.dry_run:
            sys_text = "".join(b.get("text", "") for b in system_blocks)
            p = self._mock(sys_text, user_text)
            p.cost_usd = cost_of(self.arm, p.usage)  # $0 di mock, tapi jalur kalkulasi teruji
            return p

        kwargs = dict(
            model=self.cfg["id"],
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=[{"role": "user", "content": user_text}],
            output_format=SentimentOutput,
        )
        if self.cfg["temp"]:
            kwargs["temperature"] = 0
        resp = self._client.messages.parse(**kwargs)

        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        parsed = resp.parsed_output
        raw = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        if parsed is None or resp.stop_reason == "refusal":
            return Prediction(None, False, raw, usage, cost_of(self.arm, usage))
        return Prediction(parsed.label, True, raw, usage, cost_of(self.arm, usage))


def system_blocks(system_text: str) -> list[dict]:
    """Satu blok system dengan cache_control ephemeral (cacheable prefix)."""
    return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
