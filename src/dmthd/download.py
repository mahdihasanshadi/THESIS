"""Fetch a Hugging Face dataset split to CSV (no Kaggle login needed).

    python -m dmthd.download --hf <dataset id> --out data/raw/cyberbullying_tweets.csv \
        --text_col tweet_text --label_col cyberbullying_type

If the mirror uses other column names, pass --rename "text:tweet_text,label:cyberbullying_type".
"""
import argparse
import os

from datasets import load_dataset

from .utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rename", default="", help="old:new,old2:new2")
    args = ap.parse_args()

    ds = load_dataset(args.hf, args.config, split=args.split) if args.config else load_dataset(args.hf, split=args.split)
    df = ds.to_pandas()
    if args.rename:
        mapping = dict(pair.split(":") for pair in args.rename.split(","))
        df = df.rename(columns=mapping)
    ensure_dir(os.path.dirname(args.out) or ".")
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {args.out}: {len(df)} rows, columns {list(df.columns)}")


if __name__ == "__main__":
    main()
