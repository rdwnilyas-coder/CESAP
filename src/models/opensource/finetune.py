"""Group A: fine-tune model open-source ($0) per dataset, evaluasi di golden test.

Mendukung full fine-tune (default) & LoRA (untuk model besar yg tak muat full di 4GB).
Menyimpan: prediksi per-sampel + softmax confidence (untuk C1 routing nanti),
metrik lengkap (accuracy, macro/weighted-F1, per-class P/R/F1, confusion),
log training & inference, dan config -> results/groupA/<dataset>/<arm>/.

Contoh:
    python src/models/opensource/finetune.py --dataset smsa --arm indobert-base
    python src/models/opensource/finetune.py --dataset smsa --arm indobert-large   # auto-LoRA
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "groupA"

CANON = ["negative", "neutral", "positive"]
LBL2ID = {l: i for i, l in enumerate(CANON)}
ID2LBL = {i: l for l, i in LBL2ID.items()}

ARMS = {
    "indobert-base": "indobenchmark/indobert-base-p1",
    "indobert-large": "indobenchmark/indobert-large-p1",
    "indobertweet": "indolem/indobertweet-base-uncased",
    "lightweight": "indobenchmark/indobert-lite-base-p1",
}
LORA_ARMS = {"indobert-large"}              # full fine-tune tak muat 4GB -> LoRA
DEFAULT_BATCH = {"indobert-base": 16, "indobertweet": 16,
                 "lightweight": 32, "indobert-large": 16}


def make_ds(df: pd.DataFrame, tok, max_len: int) -> Dataset:
    df = df.copy()
    df["labels"] = df["label"].map(LBL2ID)
    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                batched=True)
    return ds.remove_columns(["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def full_metrics(y_true, y_pred) -> dict:
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CANON))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {CANON[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(sup[i])}
                      for i in range(len(CANON))},
        "confusion_matrix": {"labels": CANON,
                             "matrix": confusion_matrix(
                                 y_true, y_pred, labels=list(range(len(CANON)))).tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)  # nama dir di data/processed/ (termasuk *_distill)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=None)  # default per-metode (lihat bawah)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_save_model", action="store_true")
    args = ap.parse_args()

    # seed 42 = kanonik (dir asli); seed lain -> dir bersuffix agar tak menimpa
    arm_dir = args.arm if args.seed == 42 else f"{args.arm}-s{args.seed}"
    out_dir = RESULTS / args.dataset / arm_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log", mode="w"),
                                  logging.StreamHandler()])
    log = logging.getLogger("groupA")

    set_seed(args.seed)
    model_id = ARMS[args.arm]
    use_lora = args.arm in LORA_ARMS
    # Default hyperparameter per-metode: LoRA hanya melatih ~0.5% param -> butuh lr
    # lebih tinggi & epoch lebih banyak agar konvergen (lr full fine-tune underfit).
    lr = args.lr if args.lr is not None else (1e-4 if use_lora else 2e-5)
    epochs = args.epochs if args.epochs is not None else (5 if use_lora else 3)
    batch = args.batch or DEFAULT_BATCH[args.arm]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"=== {args.arm} ({model_id}) x {args.dataset} | method={'lora' if use_lora else 'full'} "
             f"| batch={batch} epochs={epochs} lr={lr} seed={args.seed} dev={dev} ===")

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        # indobert-lite: config bilang AlbertTokenizer (cari spiece.model) padahal
        # repo pakai vocab WordPiece BERT -> fallback ke BertTokenizerFast.
        from transformers import BertTokenizerFast
        log.info(f"AutoTokenizer gagal ({type(e).__name__}); fallback BertTokenizerFast")
        tok = BertTokenizerFast.from_pretrained(model_id)
    train_df = pd.read_parquet(PROC / args.dataset / "train.parquet")
    valid_df = pd.read_parquet(PROC / args.dataset / "valid.parquet")
    test_df = pd.read_parquet(PROC / args.dataset / "test.parquet")
    log.info(f"n: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

    train_ds = make_ds(train_df, tok, args.max_len)
    valid_ds = make_ds(valid_df, tok, args.max_len)
    test_ds = make_ds(test_df, tok, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(CANON), id2label=ID2LBL, label2id=LBL2ID,
        ignore_mismatched_sizes=True)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                          lora_dropout=0.1, target_modules=["query", "value"],
                          modules_to_save=["classifier"])
        model = get_peft_model(model, lcfg)
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        log.info(f"LoRA trainable params: {trn:,} / {tot:,} ({100*trn/tot:.2f}%)")

    targs = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"), num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=max(batch, 32),
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=(dev == "cuda"), seed=args.seed,
        logging_steps=50, report_to="none", disable_tqdm=False)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=valid_ds, compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tok), tokenizer=tok)

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    best_valid = trainer.evaluate()
    log.info(f"train_secs={train_secs:.1f} | best valid macro_f1={best_valid.get('eval_macro_f1'):.4f}")

    # --- inference di golden test ---
    t1 = time.time()
    pred = trainer.predict(test_ds)
    infer_secs = time.time() - t1
    logits = torch.tensor(pred.predictions)
    probs = torch.softmax(logits, dim=-1).numpy()
    y_pred = probs.argmax(-1)
    y_true = np.array([LBL2ID[l] for l in test_df["label"]])

    preds_out = pd.DataFrame({
        "id": test_df["id"].values, "text": test_df["text"].values,
        "true_label": test_df["label"].values,
        "pred_label": [ID2LBL[i] for i in y_pred],
        "confidence": probs.max(-1),
        "p_negative": probs[:, 0], "p_neutral": probs[:, 1], "p_positive": probs[:, 2],
    })
    preds_out.to_parquet(out_dir / "predictions.parquet", index=False)

    metrics = full_metrics(y_true, y_pred)
    summary = {
        "dataset": args.dataset, "arm": args.arm, "model_id": model_id,
        "method": "lora" if use_lora else "full",
        "config": {"epochs": epochs, "lr": lr, "max_len": args.max_len,
                   "batch": batch, "seed": args.seed},
        "n": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "test_metrics": metrics,
        "best_valid_macro_f1": float(best_valid.get("eval_macro_f1")),
        "timing_secs": {"train": round(train_secs, 1), "inference": round(infer_secs, 2)},
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2) if dev == "cuda" else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (out_dir / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    if not args.no_save_model:
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")

    # bersihkan checkpoint sementara (hemat disk)
    import shutil
    shutil.rmtree(out_dir / "_ckpt", ignore_errors=True)

    m = metrics
    log.info(f"\nTEST  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}")
    for c in CANON:
        pc = m["per_class"][c]
        log.info(f"  {c:8s} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    log.info(f"confusion (rows=true {CANON}):\n{np.array(m['confusion_matrix']['matrix'])}")
    log.info(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
