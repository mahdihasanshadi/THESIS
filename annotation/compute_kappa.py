"""Agreement and gold labels from the filled annotation sheet.

    python annotation/compute_kappa.py --sheet annotation/sarcbully_sheet.csv --out probes

Prints Fleiss' kappa over the annotators who labelled each item (unsure and blank are treated as
missing), pairwise Cohen's kappa, the label distribution, and writes:
  probes/benign_sarcasm_human.csv   items whose majority label is benign_sarcasm
  probes/sarcastic_abuse_human.csv  items whose majority label is sarcastic_abuse
  annotation/adjudicate.csv         items without a majority, to settle in a meeting
"""
import argparse
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

LABELS = ["benign_sarcasm", "sarcastic_abuse", "not_sarcastic"]
ANNOTATORS = ["A1", "A2", "A3", "A4"]


def fleiss_kappa(rows):
    """rows: list of Counter(label -> count) per item, items with >= 2 ratings."""
    rows = [r for r in rows if sum(r.values()) >= 2]
    if not rows:
        return float("nan")
    n_max = max(sum(r.values()) for r in rows)
    # Fleiss' kappa assumes an equal number of raters per item; use only items with the modal count
    n = Counter(sum(r.values()) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if sum(r.values()) == n]
    N = len(rows)
    p_j = np.array([sum(r.get(l, 0) for r in rows) for l in LABELS], dtype=float) / (N * n)
    P_i = [(sum(c * c for c in r.values()) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = float(np.mean(P_i)), float((p_j ** 2).sum())
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan"), N, n


def cohen_kappa(a, b):
    m = [(x, y) for x, y in zip(a, b) if x in LABELS and y in LABELS]
    if len(m) < 2:
        return float("nan")
    xs, ys = zip(*m)
    po = np.mean([x == y for x, y in m])
    pe = sum((xs.count(l) / len(m)) * (ys.count(l) / len(m)) for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="annotation/sarcbully_sheet.csv")
    ap.add_argument("--out", default="probes")
    args = ap.parse_args()
    df = pd.read_csv(args.sheet, dtype=str).fillna("")
    for a in ANNOTATORS:
        df[a] = df[a].str.strip().str.lower()
    present = [a for a in ANNOTATORS if (df[a].isin(LABELS)).any()]
    print("annotators with labels:", present)

    counters, majority = [], []
    for _, r in df.iterrows():
        c = Counter(r[a] for a in present if r[a] in LABELS)
        counters.append(c)
        if c:
            top, cnt = c.most_common(1)[0]
            majority.append(top if cnt > len(present) / 2 or (cnt >= 2 and len(c) == 1) else "")
        else:
            majority.append("")
    df["gold"] = majority
    fk = fleiss_kappa(counters)
    if isinstance(fk, tuple):
        print(f"Fleiss' kappa = {fk[0]:.3f}  (N = {fk[1]} items with {fk[2]} ratings each)")
    for a, b in combinations(present, 2):
        print(f"Cohen's kappa {a}-{b} = {cohen_kappa(df[a].tolist(), df[b].tolist()):.3f}")
    print("gold distribution:", Counter(df["gold"]).most_common())

    os.makedirs(args.out, exist_ok=True)
    df[df["gold"] == "benign_sarcasm"][["id", "text"]].to_csv(os.path.join(args.out, "benign_sarcasm_human.csv"), index=False)
    df[df["gold"] == "sarcastic_abuse"][["id", "text"]].to_csv(os.path.join(args.out, "sarcastic_abuse_human.csv"), index=False)
    df[df["gold"] == ""][["id", "text"] + present + ["notes"]].to_csv(os.path.join(os.path.dirname(args.sheet) or ".", "adjudicate.csv"), index=False)
    print(f"wrote {int((df['gold']=='benign_sarcasm').sum())} benign, {int((df['gold']=='sarcastic_abuse').sum())} abusive, "
          f"{int((df['gold']=='').sum())} to adjudicate")


if __name__ == "__main__":
    main()
