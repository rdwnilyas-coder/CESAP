# CESAP — Kode & Dataset

*Strategi Hemat Biaya untuk Analisis Sentimen Bahasa Indonesia Produksi.*
🇬🇧 [English version](README.md)

Repositori ini berisi **kode** dan **dataset terstandardisasi** untuk riset yang membandingkan
encoder open-source, LLM komersial, dan pipeline hibrida untuk analisis sentimen Bahasa Indonesia
atas dasar **akurasi vs biaya**. **Paper pendamping akan menyusul**; repositori ini dirujuk dari paper
tersebut. Hasil eksperimen/figur disimpan bersama paper dan *tidak* disertakan di sini.

## Isi
- `src/datasets/` — unduh + standardisasi label (ke `negative/neutral/positive`) + split golden test terkunci + test.
- `src/models/opensource/` — fine-tune IndoBERT (full + LoRA) dan prediksi.
- `src/models/llm/` — harness sentimen LLM (prompt, parsing output terstruktur, akuntansi biaya).
- `src/models/hybrid/` — confidence routing (C1), uncertainty cascade (C2), distillation (C3).
- `src/eval/`, `src/cost/` — metrik (Macro-F1), util Pareto/routing/signifikansi, estimator biaya.
- `data/` — empat dataset, distandardisasi ke skema 3-kelas seragam.

> Tidak ada di repo ini: bobot model hasil latih (besar; regenerate via skrip) dan paper-nya.

## Persiapan
```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11+
pip install transformers datasets scikit-learn pandas pyarrow torch pydantic anthropic tiktoken
```

## Menjalankan (ringkas)
```bash
python src/datasets/download.py --datasets smsa nusax ugm finance
python src/datasets/standardize.py
python src/models/opensource/run_groupA.py        # fine-tune open-source
# Arm LLM (Group B) butuh ANTHROPIC_API_KEY (dibaca dari env; jangan hardcode)
```

## Dataset & lisensi (atribusi)
| Key | Dataset | Sumber | Lisensi |
|-----|---------|--------|---------|
| `smsa` | IndoNLU **SmSA** | Wilie dkk., 2020 (IndoNLU) | sesuai IndoNLU |
| `nusax` | **NusaX** (Indonesia) | Winata dkk., 2023 | CC BY-SA 4.0 |
| `ugm` | **Indonesian Sentiment Twitter** | Ferdiana dkk., 2019 (`ridife/dataset-idsa`) | CC BY-NC 4.0 |
| `finance` | **ID-SMSA** (tweet saham) | Hartanto dkk., 2025, *Data in Brief* (Mendeley `10.17632/tn4vzs8tdw.3`) | CC BY |

Mohon sitasi penulis dataset asli. Data UGM bersifat **non-komersial (CC BY-NC)**; patuhi ketentuan tiap sumber.

## Lisensi
Kode di `src/` dirilis untuk **penggunaan riset**. Dataset di `data/` tetap di bawah lisensi aslinya (tabel di atas).
