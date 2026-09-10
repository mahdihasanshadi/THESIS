"""Deployment table: the same student in fp32 and after dynamic INT8 quantisation, on CPU.
Reports macro-F1, size on disk, and latency at batch 1 and 32 (median of repeats after warm-up).

    python -m dmthd.quantize_eval --model_dir runs/tweets/bert-mini/dmthd/seed1 --csv data/tweets/test.csv --scheme six
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluate import compute_metrics, make_loader, predict_probs
from .models import load_classifier, load_tokenizer
from .utils import label_names, map_labels, save_json


def state_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 2**20
    os.remove(path)
    return size


@torch.no_grad()
def latency(model, tok, texts, max_len, batches, warmup, repeats):
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    ii, am = enc["input_ids"], enc["attention_mask"]
    out = {}
    for bs in batches:
        for _ in range(warmup):
            model(input_ids=ii[:bs], attention_mask=am[:bs])
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for s in range(0, len(ii) - bs + 1, bs):
                model(input_ids=ii[s:s + bs], attention_mask=am[s:s + bs])
            times.append(time.perf_counter() - t0)
        n = (len(ii) // bs) * bs
        out[f"latency_ms_per_sample_b{bs}"] = 1000.0 * float(np.median(times)) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_time", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    names = label_names(args.scheme)
    tok = load_tokenizer(args.model_dir)
    df = map_labels(pd.read_csv(args.csv), args.label_col, args.scheme)
    loader = make_loader(df, tok, args.max_len, args.batch, False)
    texts = df["text"].astype(str).head(args.n_time).tolist()

    fp32 = load_classifier(args.model_dir, len(names)).to(device).eval()
    int8 = torch.ao.quantization.quantize_dynamic(load_classifier(args.model_dir, len(names)).eval(), {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    res = {"model_dir": args.model_dir}
    for name, m in (("fp32", fp32), ("int8", int8)):
        probs = predict_probs(lambda ii, am: m(input_ids=ii, attention_mask=am).logits, loader, device)
        r = compute_metrics(df["label"].values, probs, names)
        r = {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "ece": r["ece"], "size_MB": state_size_mb(m)}
        r.update(latency(m, tok, texts, args.max_len, [1, 32], args.warmup, args.repeats))
        res[name] = r
        print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
    res["macro_f1_drop_int8"] = res["fp32"]["macro_f1"] - res["int8"]["macro_f1"]
    res["speedup_b1"] = res["fp32"]["latency_ms_per_sample_b1"] / res["int8"]["latency_ms_per_sample_b1"]
    save_json(res, args.out or os.path.join(args.model_dir, "quantize_eval.json"))


if __name__ == "__main__":
    main()
