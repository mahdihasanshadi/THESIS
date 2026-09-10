"""Datasets, prediction, metrics, and the sarcasm probes.

    python -m dmthd.evaluate --model_dir runs/tweets/bert-mini/dmthd/seed1 --csv data/tweets/test.csv \
        --scheme six --probe_neg data/probes/benign_sarcasm.csv --probe_pos data/probes/ironic_abuse.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

from .models import load_classifier, load_tokenizer
from .utils import get_device, label_names, map_labels, not_bullying_index, save_json


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, soft=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.soft = None if soft is None else np.asarray(soft, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"idx": i, "text": self.texts[i]}
        if self.labels is not None:
            item["label"] = int(self.labels[i])
        if self.soft is not None:
            item["soft"] = float(self.soft[i])
        return item


def make_collate(tokenizer, max_len: int):
    def collate(items):
        enc = tokenizer([it["text"] for it in items], padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt")
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                 "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long)}
        if "label" in items[0]:
            batch["labels"] = torch.tensor([it["label"] for it in items], dtype=torch.long)
        if "soft" in items[0]:
            batch["soft"] = torch.tensor([it["soft"] for it in items], dtype=torch.float32)
        return batch
    return collate


def make_loader(df, tokenizer, max_len, batch_size, shuffle, soft_col=None):
    soft = df[soft_col].values if (soft_col and soft_col in df.columns) else None
    ds = TextDataset(df["text"].values, df["label"].values if "label" in df.columns else None, soft)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(tokenizer, max_len))


@torch.no_grad()
def predict_probs(logits_fn, loader, device) -> np.ndarray:
    """logits_fn(input_ids, attention_mask) -> logits. Returns softmax probs [N, C] in loader order."""
    out = []
    for batch in loader:
        logits = logits_fn(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def expected_calibration_error(y, probs, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, probs, names) -> dict:
    y = np.asarray(y)
    pred = probs.argmax(1)
    res = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "per_class_f1": {n: float(v) for n, v in zip(names, f1_score(y, pred, average=None, labels=list(range(len(names)))))},
           "ece": expected_calibration_error(y, probs)}
    if len(names) == 2:
        res["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
        res["pr_auc"] = float(average_precision_score(y, probs[:, 1]))
    return res


def probe_rate(probs, nb_index: int) -> float:
    """Fraction predicted as any bullying class. On benign sarcasm this is the false-positive
    rate; on ironic abuse it is recall."""
    return float((probs.argmax(1) != nb_index).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--probe_neg", default=None, help="CSV with a text column of benign sarcasm")
    ap.add_argument("--probe_pos", default=None, help="CSV with a text column of ironic/implicit abuse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    tok = load_tokenizer(args.model_dir)
    names = label_names(args.scheme)
    model = load_classifier(args.model_dir, len(names)).to(device).eval()
    fn = lambda ii, am: model(input_ids=ii, attention_mask=am).logits

    df = map_labels(pd.read_csv(args.csv), args.label_col, args.scheme)
    probs = predict_probs(fn, make_loader(df, tok, args.max_len, args.batch, False), device)
    res = compute_metrics(df["label"].values, probs, names)
    nb = not_bullying_index(args.scheme)
    for key, path in [("benign_sarcasm_fpr", args.probe_neg), ("ironic_abuse_recall", args.probe_pos)]:
        if path:
            pdf = pd.read_csv(path)
            pp = predict_probs(fn, make_loader(pdf, tok, args.max_len, args.batch, False), device)
            res[key] = probe_rate(pp, nb)
            res[key + "_n"] = int(len(pdf))
    out = args.out or os.path.join(args.model_dir, "eval_" + os.path.basename(args.csv).replace(".csv", "") + ".json")
    save_json(res, out)
    print(res)


if __name__ == "__main__":
    main()
