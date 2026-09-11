"""How much does the per-instance teacher weighting actually differ from uniform?

The weights are w_k(i) = softmax_k(-CE(p_k(i), y_i) / tau) over frozen, cached teachers, so they are
a fixed function of the data: they vary per instance but not per epoch or seed. If tau is too large
for the spread of teacher errors, every weight collapses to 1/K and D-MTHD becomes uniform averaging
with extra steps. This script reads a teacher cache and reports, for a range of tau, how far the
weights are from uniform, so tau can be chosen before spending GPU time on a sweep.

    python -m dmthd.tune_tau --cache cache/tweets --data_dir data/tweets --scheme six

Reported per tau:
  mean max weight     1/K means uniform, 1.0 means hard selection of one teacher
  mean entropy ratio  H(w)/ln K; 1.0 is uniform, 0 is a one-hot choice
  share decisive      fraction of instances whose top weight exceeds 1.5/K
  winner share        how often each teacher takes the largest weight
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .losses import teacher_weights
from .utils import label_names, map_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--teachers", nargs="*", default=None, help="cache tags; default: all in meta.json")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--reliability", default="hard", choices=["hard", "soft"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, "meta.json"), encoding="utf-8"))
    tags = args.teachers or [t["tag"] for t in meta["teachers"]]
    df = map_labels(pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv")), args.label_col, args.scheme)
    y = torch.tensor(df["label"].values[: meta["n"]], dtype=torch.long)
    logits = torch.stack([torch.from_numpy(np.load(os.path.join(args.cache, f"{t}.npz"))["logits"].astype(np.float32))
                          for t in tags], 0)[:, : len(y)]
    soft = torch.tensor(df[meta.get("soft_column", "soft_label")].values[: len(y)], dtype=torch.float32) \
        if (args.reliability == "soft" and "soft_label" in df.columns) else None
    K = len(tags)
    print(f"{K} teachers {tags} over {len(y)} instances of {args.split}\n")
    rows = []
    for tau in args.taus:
        w = teacher_weights(logits, y, tau, per_instance=True, soft_targets=soft)
        ent = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(1) / np.log(K)
        winner = w.argmax(1)
        row = {"tau": tau, "mean_max_weight": float(w.max(1).values.mean()), "uniform_weight": 1.0 / K,
               "mean_entropy_ratio": float(ent.mean()),
               "share_decisive": float((w.max(1).values > 1.5 / K).float().mean())}
        for k, t in enumerate(tags):
            row[f"wins_{t}"] = float((winner == k).float().mean())
            row[f"mean_w_{t}"] = float(w[:, k].mean())
        rows.append(row)
        print(f"tau={tau:<5} max-w {row['mean_max_weight']:.3f} (uniform {1/K:.3f})  "
              f"entropy {row['mean_entropy_ratio']:.3f}  decisive {row['share_decisive']:.3f}  "
              + " ".join(f"{t}:{row['wins_' + t]:.2f}" for t in tags))
    out = args.out or os.path.join(args.cache, "tau_diagnostic.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n-> {out}")
    best = min(rows, key=lambda r: abs(r["mean_entropy_ratio"] - 0.8))
    print(f"tau near {best['tau']} gives weights that are neither uniform nor one-hot "
          f"(entropy ratio {best['mean_entropy_ratio']:.2f}); confirm on validation before adopting it.")


if __name__ == "__main__":
    main()
