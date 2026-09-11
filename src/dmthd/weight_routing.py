"""Does the per-instance weighting actually route to the right expert?

    python -m dmthd.weight_routing --cache cache/tweets --data_dir data/tweets --scheme six \
        --out runs/tweets/weight_routing

A committee of specialists only earns its name if the weights go somewhere sensible: the implicit
specialist should carry more of the load on abuse-by-implication than on abuse that says it outright.
That is a claim about the mechanism, not about the score, and it is the claim a reviewer will ask us
to show rather than assert. It is also cheap: the weights are a fixed function of the cached teacher
logits and the gold labels, so no training is involved.

For each tau the script reports, per teacher, the mean weight inside each group and the **routing
contrast**: mean weight on the implicit group minus mean weight on the explicit group, with a
percentile bootstrap interval. A contrast whose interval excludes zero is evidence of routing. A
contrast of zero says the committee treats both kinds of abuse identically, which would mean the
specialist is contributing knowledge but not selection, and the paper must say that instead.

Groups, by scheme:
  implicit3  the gold classes themselves (implicit_hate against explicit_hate)
  six        other_cyberbullying, the catch-all that holds most indirect abuse, against the four
             targeted classes
  binary     abusive against benign
Every scheme additionally gets a lexical split on the same coarse profanity proxy used by
implicit_analysis, because "contains a slur or an insult" is the crudest possible stand-in for
explicitness and a routing effect should survive it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .implicit_analysis import has_profanity
from .losses import teacher_weights
from .utils import ensure_dir, label_names, map_labels, not_bullying_index, save_json

# (implicit-like group, explicit-like group) per scheme: the contrast the paper reports.
CONTRAST = {"implicit3": ("implicit_hate", "explicit_hate"),
            "six": ("other_cyberbullying", "targeted"),
            "five": ("not_cyberbullying", "targeted"),
            "binary": ("cyberbullying", "not_cyberbullying")}


def groups_for(df, scheme):
    """A label per row naming the group it belongs to, and the lexical group."""
    names = label_names(scheme)
    gold = df["label"].map(dict(enumerate(names)))
    if scheme == "six":
        g = gold.where(gold.isin(["other_cyberbullying", "not_cyberbullying"]), "targeted")
    elif scheme == "five":
        g = gold.where(gold == "not_cyberbullying", "targeted")
    else:
        g = gold
    lex = np.where(df["text"].map(has_profanity), "with_profanity", "without_profanity")
    # the lexical split is only meaningful on the abusive rows: a benign post without profanity is
    # not an implicit attack, it is simply not an attack
    lex = pd.Series(np.where(df["label"].values == not_bullying_index(scheme), "benign", lex), index=df.index)
    return g, lex


def contrast(w_k, mask_a, mask_b, n_boot=2000, seed=0):
    """Mean weight on group A minus on group B, with a percentile bootstrap interval."""
    a, b = w_k[mask_a], w_k[mask_b]
    if len(a) < 5 or len(b) < 5:
        return None
    rng = np.random.default_rng(seed)
    d = float(a.mean() - b.mean())
    boot = [float(rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean())
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": d, "ci95": [float(lo), float(hi)], "n_a": int(len(a)), "n_b": int(len(b)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--teachers", nargs="*", default=None, help="cache tags; default: all in meta.json")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, "meta.json"), encoding="utf-8"))
    tags = args.teachers or [t["tag"] for t in meta["teachers"]]
    df = map_labels(pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv")), args.label_col, args.scheme)
    df = df.iloc[: meta["n"]].reset_index(drop=True)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    logits = torch.stack([torch.from_numpy(np.load(os.path.join(args.cache, f"{t}.npz"))["logits"].astype(np.float32))
                          for t in tags], 0)[:, : len(y)]
    grp, lex = groups_for(df, args.scheme)
    pos_name, neg_name = CONTRAST[args.scheme]
    ensure_dir(args.out)

    rows, res = [], {"cache": args.cache, "teachers": tags, "split": args.split,
                     "contrast_groups": {"implicit_like": pos_name, "explicit_like": neg_name},
                     "group_sizes": grp.value_counts().to_dict(), "by_tau": {}}
    for tau in args.taus:
        w = teacher_weights(logits, y, tau, per_instance=True).numpy()
        entry = {"mean_weight": {}, "routing_contrast": {}, "lexical_contrast": {}}
        for k, t in enumerate(tags):
            entry["mean_weight"][t] = {g: float(w[grp.values == g, k].mean()) for g in sorted(set(grp))}
            entry["routing_contrast"][t] = contrast(w[:, k], (grp.values == pos_name).nonzero()[0],
                                                    (grp.values == neg_name).nonzero()[0])
            entry["lexical_contrast"][t] = contrast(w[:, k], (lex.values == "without_profanity").nonzero()[0],
                                                    (lex.values == "with_profanity").nonzero()[0])
            r = {"tau": tau, "teacher": t}
            r.update({f"w_{g}": entry["mean_weight"][t][g] for g in entry["mean_weight"][t]})
            rc = entry["routing_contrast"][t]
            r["routing_delta"] = rc["delta"] if rc else float("nan")
            r["routing_ci_lo"] = rc["ci95"][0] if rc else float("nan")
            r["routing_ci_hi"] = rc["ci95"][1] if rc else float("nan")
            rows.append(r)
        res["by_tau"][str(tau)] = entry
        print(f"\ntau={tau}")
        for t in tags:
            rc = entry["routing_contrast"][t]
            tail = "" if not rc else (f"  {pos_name} minus {neg_name}: {rc['delta']:+.4f} "
                                      f"[{rc['ci95'][0]:+.4f}, {rc['ci95'][1]:+.4f}]"
                                      f"{'  <- excludes zero' if rc['excludes_zero'] else ''}")
            print(f"  {t:<16}" + " ".join(f"{g}={entry['mean_weight'][t][g]:.3f}"
                                          for g in sorted(entry["mean_weight"][t])) + tail)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "weight_routing.csv"), index=False)
    save_json(res, os.path.join(args.out, "weight_routing.json"))
    print(f"\n-> {args.out}")
    print("A contrast whose interval excludes zero is evidence that the weighting selects an expert "
          "rather than averaging; one that spans zero says the committee treats both kinds of abuse "
          "alike, and the paper reports that instead.")


if __name__ == "__main__":
    main()
