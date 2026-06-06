"""C3 distillation — assemble: merge label Opus, cek coverage + akurasi guru vs gold,
lalu bangun dataset distill (train=label Opus, valid/test=gold) utk fine-tune student.

    python src/models/hybrid/distill_assemble.py            # cek + build bila lengkap
    python src/models/hybrid/distill_assemble.py --missing  # tulis _missing.json saja
Output (bila lengkap): data/processed/<ds>_distill/{train,valid,test}.parquet
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
GC = ROOT / "results" / "groupC"
DATASETS = ["smsa", "nusax", "ugm", "finance"]
LABELS = ["negative", "neutral", "positive"]


def merge(d: Path) -> dict:
    m = {}
    for pat in ("_labels_b*.json", "_labels_rep*.json"):
        for f in sorted(d.glob(pat)):
            try:
                m.update(json.loads(f.read_text(encoding="utf-8-sig")))
            except Exception as e:
                print(f"  (skip {f.name}: {e})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true")
    args = ap.parse_args()

    all_ok = True
    for ds in DATASETS:
        d = GC / ds
        items = json.loads((d / "_items.json").read_text(encoding="utf-8"))
        gold = json.loads((d / "_gold.json").read_text(encoding="utf-8"))
        labels = merge(d)
        miss = [it for it in items if labels.get(it["id"]) not in LABELS]
        if miss:
            all_ok = False
            (d / "_missing.json").write_text(json.dumps(
                [{"id": it["id"], "text": it["text"]} for it in miss], ensure_ascii=False), encoding="utf-8")
            print(f"{ds}: missing={len(miss)} (tulis _missing.json)")
            continue
        # akurasi guru (Opus) vs gold pada subset train
        ids = [it["id"] for it in items]
        yt = [gold[i] for i in ids]; yp = [labels[i] for i in ids]
        tacc = accuracy_score(yt, yp)
        tf1 = f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
        print(f"{ds}: coverage OK ({len(items)}) | guru Opus vs gold: acc={tacc:.4f} macroF1={tf1:.4f}")
        if args.missing:
            continue
        # bangun dataset distill: train = label Opus, valid/test = gold
        dd = PROC / f"{ds}_distill"; dd.mkdir(parents=True, exist_ok=True)
        text_by_id = {it["id"]: it["text"] for it in items}
        tr = pd.DataFrame({"id": ids, "text": [text_by_id[i] for i in ids],
                           "label": [labels[i] for i in ids], "gold_label": yt})
        tr.to_parquet(dd / "train.parquet", index=False)
        for sp in ("valid", "test"):
            shutil.copy(PROC / ds / f"{sp}.parquet", dd / f"{sp}.parquet")
        print(f"    -> {dd} (train={len(tr)} label-Opus, valid/test=gold)")
    if not args.missing:
        print("\nSiap fine-tune student: finetune.py --dataset <ds>_distill --arm indobert-base" if all_ok
              else "\n(ada missing — repair dulu, lalu jalankan lagi)")


if __name__ == "__main__":
    main()
