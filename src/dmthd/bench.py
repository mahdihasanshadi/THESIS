"""Efficiency benchmark that reviewers can trust: warm-up, repeats, median.

    python -m dmthd.bench --model_dirs runs/tweets/bert-mini/dmthd/seed1 runs/tweets/teachers/bert-large \
        --csv data/tweets/test.csv --device cpu --batches 1 32 --out runs/bench_cpu.csv

Reports parameters, size on disk, FLOPs per sequence (torch flop counter, if available),
latency in ms per sample (median of --repeats runs after --warmup batches), throughput,
and peak GPU memory. Run once with --device cuda on Kaggle and once with --device cpu here.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from .models import count_params, load_classifier, load_tokenizer


def dir_size_mb(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")) and "heads" not in f:
                total += os.path.getsize(os.path.join(root, f))
    return total / 2**20


def flops_per_sequence(model, ii, am):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(input_ids=ii[:1], attention_mask=am[:1])
        return float(fc.get_total_flops())
    except Exception:
        return float("nan")


@torch.no_grad()
def bench_one(model_dir, texts, device, batches, max_len, warmup, repeats, num_labels):
    tok = load_tokenizer(model_dir)
    model = load_classifier(model_dir, num_labels).to(device).eval()
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    ii_all, am_all = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    row = {"model": model_dir, "params_M": count_params(model) / 1e6, "disk_MB": dir_size_mb(model_dir),
           "flops_G_per_seq": flops_per_sequence(model, ii_all, am_all) / 1e9, "device": str(device)}
    for bs in batches:
        ii, am = ii_all[:bs], am_all[:bs]
        for _ in range(warmup):
            model(input_ids=ii, attention_mask=am)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for start in range(0, len(ii_all) - bs + 1, bs):
                model(input_ids=ii_all[start:start + bs], attention_mask=am_all[start:start + bs])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        n_samples = (len(ii_all) // bs) * bs
        med = float(np.median(times))
        row[f"latency_ms_per_sample_b{bs}"] = 1000.0 * med / n_samples
        row[f"throughput_per_s_b{bs}"] = n_samples / med
        if device.type == "cuda":
            row[f"peak_mem_MB_b{bs}"] = torch.cuda.max_memory_allocated() / 2**20
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=256, help="number of test texts to time")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--num_labels", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--out", default="bench.csv")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    texts = pd.read_csv(args.csv)["text"].astype(str).head(args.n).tolist()
    rows = [bench_one(d, texts, device, args.batches, args.max_len, args.warmup, args.repeats, args.num_labels)
            for d in args.model_dirs]
    df = pd.DataFrame(rows)
    if os.path.exists(args.out):
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
