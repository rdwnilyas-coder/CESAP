# CESAP — Code & Datasets

*Cost-Efficient Strategies for Production Indonesian Sentiment Analysis.*
🇮🇩 [Versi Bahasa Indonesia](README.id.md)

This repository contains the **code** and **standardized datasets** for a study comparing
open-source encoders, commercial LLMs, and hybrid pipelines for Indonesian sentiment analysis
on an accuracy-versus-cost basis. **The accompanying paper is forthcoming**; this repository is
referenced from it. Experimental results/figures are kept with the paper and are *not* included here.

## What's included
- `src/datasets/` — download + label standardization (to `negative/neutral/positive`) + locked golden test split + tests.
- `src/models/opensource/` — IndoBERT fine-tuning (full + LoRA) and prediction.
- `src/models/llm/` — LLM sentiment harness (prompting, structured-output parsing, cost accounting).
- `src/models/hybrid/` — confidence routing (C1), uncertainty cascade (C2), distillation (C3).
- `src/eval/`, `src/cost/` — metrics (Macro-F1), Pareto/routing/significance utilities, cost estimator.
- `data/` — the four datasets, label-standardized into a uniform 3-class scheme.

> Not in this repo: trained model weights (large; regenerate via the scripts) and the paper itself.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11+
pip install transformers datasets scikit-learn pandas pyarrow torch pydantic anthropic tiktoken
```

## Run (outline)
```bash
python src/datasets/download.py --datasets smsa nusax ugm finance
python src/datasets/standardize.py
python src/models/opensource/run_groupA.py        # open-source fine-tuning
# LLM arms (Group B) require an ANTHROPIC_API_KEY (read from env; never hard-code)
```

## Datasets & licenses (attribution)
| Key | Dataset | Source | License |
|-----|---------|--------|---------|
| `smsa` | IndoNLU **SmSA** | Wilie et al., 2020 (IndoNLU) | per IndoNLU |
| `nusax` | **NusaX** (Indonesian) | Winata et al., 2023 | CC BY-SA 4.0 |
| `ugm` | **Indonesian Sentiment Twitter** | Ferdiana et al., 2019 (`ridife/dataset-idsa`) | CC BY-NC 4.0 |
| `finance` | **ID-SMSA** (stock tweets) | Hartanto et al., 2025, *Data in Brief* (Mendeley `10.17632/tn4vzs8tdw.3`) | CC BY |

Please cite the original dataset authors. UGM data is **non-commercial (CC BY-NC)**; respect each source's terms.

## License
Code in `src/` is released for **research use**. Datasets in `data/` remain under their original licenses (above).
