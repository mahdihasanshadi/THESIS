"""Day-1 data fix for the cyberbullying tweet corpus.

    python -m dmthd.prepare_tweets --raw data/raw/cyberbullying_tweets.csv --out data/tweets

Writes train.csv / val.csv / test.csv with columns text, label_name, label (six-class ids),
plus label_info.json and report.json with every removal count. The five-class and binary
schemes are derived at training time from label_name, so one split serves all three.
"""
import argparse
import os

import pandas as pd

from .data import dedup_and_split
from .utils import FIVE, SIX, ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="CSV with the raw corpus")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--text_col", default="tweet_text")
    ap.add_argument("--label_col", default="cyberbullying_type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_csv(args.raw, encoding="utf-8", encoding_errors="ignore")
    df = df[[args.text_col, args.label_col]].dropna()
    df = df[df[args.label_col].isin(SIX)]
    train, val, test, report = dedup_and_split(df, args.text_col, args.label_col, seed=args.seed,
                                               val_frac=args.val_frac, test_frac=args.test_frac)
    ensure_dir(args.out)
    six_map = {n: i for i, n in enumerate(SIX)}
    for name, d in [("train", train), ("val", val), ("test", test)]:
        out = pd.DataFrame({"text": d[args.text_col], "label_name": d[args.label_col]})
        out["label"] = out["label_name"].map(six_map).astype(int)
        out.to_csv(os.path.join(args.out, f"{name}.csv"), index=False, encoding="utf-8")
    save_json({"six": SIX, "five": FIVE, "text_column": "text", "label_column": "label", "max_len": 128},
              os.path.join(args.out, "label_info.json"))
    report["seed"] = args.seed
    report["source"] = os.path.abspath(args.raw)
    save_json(report, os.path.join(args.out, "report.json"))
    print("De-duplication and split report")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
