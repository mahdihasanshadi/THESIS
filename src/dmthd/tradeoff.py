"""Does any method in the grid discriminate better, or do they all sit on one trade-off curve?

    python -m dmthd.tradeoff runs/tweets

Every run reports two probe numbers: recall on ironically phrased abuse and the false-positive rate
on sarcasm that attacks nobody. A model that genuinely understands implication better should get
more of the first without paying in the second. A model that has merely become more willing to fire
moves along a single curve, trading one for the other.

Across the whole finished grid, which is it?
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

runs = sys.argv[1] if len(sys.argv) > 1 else "runs/tweets"
rows = []
for p in glob.glob(os.path.join(runs, "**", "eval_test.json"), recursive=True):
    d = json.load(open(p, encoding="utf-8"))
    if "benign_sarcasm_fpr" not in d or "ironic_abuse_recall" not in d:
        continue
    rel = os.path.relpath(os.path.dirname(p), runs).replace("\\", "/").split("/")
    rows.append({"run": "/".join(rel), "student": rel[0], "variant": rel[1] if len(rel) > 1 else "",
                 "macro_f1": d.get("macro_f1"), "fpr": d["benign_sarcasm_fpr"],
                 "recall": d["ironic_abuse_recall"]})
df = pd.DataFrame(rows)
print(f"{len(df)} evaluated models (teachers and students, every mode and seed)\n")

r = np.corrcoef(df["fpr"], df["recall"])[0, 1]
print(f"correlation between false-positive rate on benign sarcasm and recall on ironic abuse: {r:+.3f}")

# A model that only shifts its threshold keeps recall - fpr roughly constant. A model that
# discriminates better raises it.
df["margin"] = df["recall"] - df["fpr"]
print(f"\nrecall minus false-positive rate, across all {len(df)} models:")
print(f"  mean {df['margin'].mean():.3f}, standard deviation {df['margin'].std():.3f}, "
      f"range {df['margin'].min():.3f} to {df['margin'].max():.3f}")

print("\nby student (mean over its runs):")
g = df.groupby("student").agg(n=("margin", "size"), fpr=("fpr", "mean"), recall=("recall", "mean"),
                              margin=("margin", "mean"), macro_f1=("macro_f1", "mean"))
print(g.round(3).sort_values("margin", ascending=False).to_string())

print("\nthe five best and five worst single models by margin:")
d2 = df.sort_values("margin", ascending=False)
for _, x in pd.concat([d2.head(5), d2.tail(5)]).iterrows():
    print(f"  {x['margin']:+.3f}  recall {x['recall']:.3f}  fpr {x['fpr']:.3f}  {x['run']}")

print("\nIf the correlation is strongly positive and the margin barely varies, then no method in the\n"
      "grid discriminates better than any other: they differ in how readily they fire, not in what\n"
      "they can tell apart. That is the same conclusion the threshold-free AUC reaches, reached here\n"
      "across every model rather than one.")


out = os.path.join(runs, "sarcasm_tradeoff.csv")
df.to_csv(out, index=False)
print("\n->", out)
