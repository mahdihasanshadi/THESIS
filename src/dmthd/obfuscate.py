"""Obfuscated test-set variants for the robustness table. Bullying-class items get the kind of
edits users make to evade filters; benign items are left untouched so the false-positive side of
the metric is unchanged. Deterministic given --seed.

    python -m dmthd.obfuscate --csv data/tweets/test.csv --out_dir data/tweets --scheme six --seed 42

Writes test_obf_leet.csv, test_obf_swap.csv, test_obf_space.csv, test_obf_mixed.csv with the same
columns and labels as the input. Evaluate each with `dmthd.evaluate --csv <file>` and report the
drop relative to the clean test set.
"""
import argparse
import os
import random
import re

import pandas as pd

from .utils import label_names

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7", "l": "1"}


def _words(text):
    return re.findall(r"\S+", text)


def leet(text, rng, rate=0.5):
    out = []
    for w in _words(text):
        if w.startswith(("@", "#", "http")) or len(w) < 4 or rng.random() > rate:
            out.append(w)
            continue
        out.append("".join(LEET.get(c.lower(), c) if rng.random() < 0.6 else c for c in w))
    return " ".join(out)


def swap(text, rng, rate=0.5):
    out = []
    for w in _words(text):
        if w.startswith(("@", "#", "http")) or len(w) < 4 or rng.random() > rate:
            out.append(w)
            continue
        i = rng.randrange(1, len(w) - 2)
        out.append(w[:i] + w[i + 1] + w[i] + w[i + 2:])
    return " ".join(out)


def space(text, rng, rate=0.5):
    out = []
    for w in _words(text):
        if w.startswith(("@", "#", "http")) or len(w) < 4 or rng.random() > rate:
            out.append(w)
            continue
        i = rng.randrange(1, len(w) - 1)
        out.append(w[:i] + " " + w[i:])
    return " ".join(out)


def mixed(text, rng):
    return rng.choice([leet, swap, space])(text, rng, rate=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    benign = df[args.label_col].astype(str).isin(["not_cyberbullying", "0"]) if args.label_col != "label" else df["label"] == 0
    base = os.path.splitext(os.path.basename(args.csv))[0]
    for name, fn in (("leet", leet), ("swap", swap), ("space", space), ("mixed", mixed)):
        rng = random.Random(args.seed)
        d = df.copy()
        d.loc[~benign, "text"] = [fn(t, rng) if name != "mixed" else mixed(t, rng) for t in d.loc[~benign, "text"].astype(str)]
        changed = int((d["text"] != df["text"]).sum())
        out = os.path.join(args.out_dir, f"{base}_obf_{name}.csv")
        d.to_csv(out, index=False)
        print(f"{name:6s}: {changed} of {int((~benign).sum())} bullying items edited -> {out}")


if __name__ == "__main__":
    main()
