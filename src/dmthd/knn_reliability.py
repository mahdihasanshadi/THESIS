"""Out-of-sample teacher reliability: how often is each teacher right on validation texts like this one?

    python -m dmthd.knn_reliability --cache cache/tweets --val_cache cache/tweets_val \
        --transfer_cache cache/tweets_transfer --data_dir data/tweets --teachers bert-large hatebert irony \
        --out cache/tweets/knn_homo.npz

The weighting of dmthd.losses scores a teacher's reliability on an instance by its cross-entropy
against the gold label, read on the split the teacher was fine-tuned on. There every teacher has
memorised the label, so the score is saturated and the weights are uniform whatever the temperature
(DECISIONS F28). This replaces that score with one the teacher cannot have memorised: its accuracy on
the validation split, which it never trained on, restricted to the k validation texts nearest the
instance in the teacher's own representation space. The estimate is gold-free at the instance, so it
is defined on unlabelled transfer text as well as on the training split, and it is out of sample on
both.

    r_k(i) = (number of the k nearest validation neighbours teacher k classifies correctly + 1) / (k + 2)
    w_k(i) = softmax_k( log r_k(i) / tau )

At tau = 1 the weight is proportional to the local accuracy. The report beside the output says how far
the weights sit from uniform and, on the training split, whether they route the class that holds
indirect abuse differently from the classes that name their target (the routing contrast of
paper/method_draft.md 3.7), so the paper can say whether a corrected signal changes anything before a
single student is trained on it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .utils import label_names, map_labels, save_json


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def local_reliability(val_pooled, val_correct, query_pooled, k=20, chunk=4096):
    """Laplace-smoothed accuracy of one teacher over the k validation neighbours (cosine, in the
    teacher's own pooled space) of every query row. Returns [n_query] in (0, 1)."""
    v = unit(val_pooled)
    c = np.asarray(val_correct, dtype=np.float32)
    k = min(k, len(v))
    out = np.empty(len(query_pooled), dtype=np.float32)
    for s in range(0, len(query_pooled), chunk):
        q = unit(query_pooled[s:s + chunk])
        sims = q @ v.T                                                    # [chunk, n_val]
        nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out[s:s + chunk] = (c[nn].sum(1) + 1.0) / (k + 2.0)
    return out


def weights_from_reliability(r, tau):
    """r [n, K] -> softmax over teachers of log r / tau."""
    z = np.log(np.clip(r, 1e-6, 1.0)) / tau
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def load_npz(cache, tag):
    z = np.load(os.path.join(cache, f"{tag}.npz"))
    return z["logits"].astype(np.float32), z["pooled"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="teacher cache on the training split")
    ap.add_argument("--val_cache", required=True, help="teacher cache on the validation split")
    ap.add_argument("--transfer_cache", default=None, help="teacher cache on the transfer set, if any")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teachers", nargs="+", required=True, help="cache tags, in committee order")
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128, help="accepted for driver symmetry; unused")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_val = json.load(open(os.path.join(args.val_cache, "meta.json"), encoding="utf-8"))["n"]
    va = map_labels(pd.read_csv(os.path.join(args.data_dir, "val.csv")), args.label_col, args.scheme).head(n_val)
    y_val = va["label"].values
    rel = {"train": [], "transfer": []}
    acc_val = {}
    for tag in args.teachers:
        z_val, h_val = load_npz(args.val_cache, tag)
        correct = (z_val.argmax(1) == y_val)
        acc_val[tag] = float(correct.mean())
        _, h_tr = load_npz(args.cache, tag)
        if args.limit:
            h_tr = h_tr[:args.limit]
        rel["train"].append(local_reliability(h_val, correct, h_tr, args.k))
        if args.transfer_cache:
            _, h_tf = load_npz(args.transfer_cache, tag)
            if args.limit:
                h_tf = h_tf[:args.limit]
            rel["transfer"].append(local_reliability(h_val, correct, h_tf, args.k))
    out = {"teachers": np.array(args.teachers), "k": args.k, "tau": args.tau}
    report = {"teachers": args.teachers, "k": args.k, "tau": args.tau, "val_accuracy": acc_val, "uniform": 1.0 / len(args.teachers)}
    for split in ("train", "transfer"):
        if not rel[split]:
            continue
        r = np.stack(rel[split], 1)                                        # [n, K]
        w = weights_from_reliability(r, args.tau)
        out[split], out[f"{split}_reliability"] = w, r.astype(np.float32)
        K = w.shape[1]
        ent = -(w * np.log(np.clip(w, 1e-9, 1))).sum(1).mean() / np.log(K)
        report[split] = {"n": int(len(w)), "mean_weight": {t: float(w[:, i].mean()) for i, t in enumerate(args.teachers)},
                         "mean_local_accuracy": {t: float(r[:, i].mean()) for i, t in enumerate(args.teachers)},
                         "entropy_ratio": float(ent), "mean_max_weight": float(w.max(1).mean()),
                         "share_led_by": {t: float((w.argmax(1) == i).mean()) for i, t in enumerate(args.teachers)}}
    # does the corrected signal route indirect abuse differently from named-target abuse?
    tr = map_labels(pd.read_csv(os.path.join(args.data_dir, "train.csv")), args.label_col, args.scheme)
    if args.limit:
        tr = tr.head(args.limit)
    names = label_names(args.scheme)
    n_tr = min(len(tr), len(out["train"]))
    lab = tr["label"].values[:n_tr]
    groups = None
    if "other_cyberbullying" in names:
        pos = lab == names.index("other_cyberbullying")
        neg = np.isin(lab, [names.index(c) for c in ("age", "ethnicity", "gender", "religion") if c in names])
        groups = ("other_cyberbullying", "targeted")
    elif "implicit_hate" in names:
        pos, neg = lab == names.index("implicit_hate"), lab == names.index("explicit_hate")
        groups = ("implicit_hate", "explicit_hate")
    if groups is not None and pos.any() and neg.any():
        w = out["train"][:n_tr]
        rng = np.random.default_rng(0)
        contrast = {}
        for i, t in enumerate(args.teachers):
            a, b = w[pos, i], w[neg, i]
            boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(1000)]
            contrast[t] = {"delta": float(a.mean() - b.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        report["routing_contrast"] = {"groups": groups, "by_teacher": contrast}
    np.savez_compressed(args.out, **out)
    save_json(report, os.path.splitext(args.out)[0] + ".json")
    print(json.dumps({k: v for k, v in report.items() if k != "routing_contrast"}, indent=1))
    if "routing_contrast" in report:
        print("routing contrast on the training split, mean weight on", report["routing_contrast"]["groups"][0],
              "minus on", report["routing_contrast"]["groups"][1], ":")
        for t, c in report["routing_contrast"]["by_teacher"].items():
            print(f"  {t:<16} {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
