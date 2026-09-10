"""Collect results.json files into one table (mean ± std over seeds) and compute paired
bootstrap confidence intervals for the difference between two configurations.

    python -m dmthd.aggregate --runs runs/tweets --out runs/tweets/summary.csv
    python -m dmthd.aggregate --compare runs/tweets/bert-mini/ft runs/tweets/bert-mini/dmthd

--compare expects seed sub-directories (seed1, seed2, ...) under each path; it pairs seeds,
resamples the test set 1,000 times and reports the 95% interval of the macro-F1 difference.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .utils import load_json


def collect(runs_dir):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "**", "results.json"), recursive=True):
        r = load_json(path)
        t = r.get("test", {})
        rows.append({"run": os.path.relpath(os.path.dirname(path), runs_dir),
                     "student": r.get("student", r.get("model_name")), "mode": r.get("mode", "teacher"),
                     "tag": r.get("tag", ""), "seed": r.get("seed"), "params": r.get("params"),
                     "macro_f1": t.get("macro_f1"), "accuracy": t.get("accuracy"), "ece": t.get("ece"),
                     "epochs_run": r.get("epochs_run"), "train_time_s": r.get("train_time_s")})
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["student", "mode", "tag"], dropna=False)
    out = g.agg(n_seeds=("seed", "count"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), ece_mean=("ece", "mean"),
                params=("params", "first")).reset_index()
    out["macro_f1"] = out.apply(lambda r: f"{r.macro_f1_mean:.4f} ± {0 if np.isnan(r.macro_f1_std) else r.macro_f1_std:.4f}", axis=1)
    return out


def paired_bootstrap(dir_a, dir_b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    deltas_all, per_seed = [], []
    seeds = sorted(set(os.listdir(dir_a)) & set(os.listdir(dir_b)))
    for s in seeds:
        pa, pb = os.path.join(dir_a, s, "test_probs.npy"), os.path.join(dir_b, s, "test_probs.npy")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        ya = np.load(os.path.join(dir_a, s, "test_labels.npy"))
        a, b = np.load(pa).argmax(1), np.load(pb).argmax(1)
        n = len(ya)
        base = f1_score(ya, b, average="macro") - f1_score(ya, a, average="macro")
        d = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            d.append(f1_score(ya[idx], b[idx], average="macro") - f1_score(ya[idx], a[idx], average="macro"))
        d = np.array(d)
        per_seed.append({"seed": s, "delta": base, "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
        deltas_all.append(d)
    if not per_seed:
        raise SystemExit("no matching seed directories with test_probs.npy")
    pooled = np.concatenate(deltas_all)
    return pd.DataFrame(per_seed), (float(np.mean([p["delta"] for p in per_seed])),
                                    float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE_DIR", "CANDIDATE_DIR"))
    args = ap.parse_args()
    if args.runs:
        df = collect(args.runs)
        if df.empty:
            raise SystemExit("no results.json found")
        summary = summarize(df)
        print(summary[["student", "mode", "tag", "n_seeds", "macro_f1", "acc_mean", "ece_mean", "params"]].to_string(index=False))
        if args.out:
            summary.to_csv(args.out, index=False)
            df.to_csv(args.out.replace(".csv", "_per_seed.csv"), index=False)
    if args.compare:
        per_seed, (mean_delta, lo, hi) = paired_bootstrap(*args.compare)
        print(per_seed.to_string(index=False))
        print(f"macro-F1 difference (candidate - baseline): {mean_delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print("SIGNIFICANT" if lo > 0 or hi < 0 else "NOT significant: the interval includes zero")


if __name__ == "__main__":
    main()
