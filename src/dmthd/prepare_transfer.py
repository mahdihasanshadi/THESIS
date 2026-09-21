"""Unlabelled in-domain text for out-of-sample distillation.

    python -m dmthd.prepare_transfer --data_dir data/tweets --out data/tweets/transfer.csv --probes probes \
        --raw data/raw/cyberbullying_tweets.csv --implicit_dir data/implicit

Why this exists. The teachers are fine-tuned on the training split and their outputs are cached on
the same split, where each of them assigns the gold label a probability near one (DECISIONS F28). A
student distilled there receives soft labels that are already the gold labels, and a per-instance
weighting reads a reliability that is saturated for every teacher. On text the teachers have not
fitted, their outputs are informative and their disagreements real, and a committee has a job a
single teacher does not: it is the better labeller (F29). This script collects that text.

Sources, used as text only; their labels are dropped before anything else reads them:
  olid      OLID / OffensEval 2019, via the TweetEval `offensive` configuration
  hateval   HatEval 2019, via the TweetEval `hate` configuration
  davidson  Davidson et al. 2017, hate_speech_offensive
  raw       the tweets our own preparation dropped for carrying more than one label. Their labels are
            unreliable; their text is in-domain and in no split.
  sentiment, emoji, emotion
            generic tweets from the other TweetEval configurations, for the size curve of DECISIONS
            D20: the same platform, but not abuse-focused. With --base they extend an existing set,
            whose rows are kept first and verbatim so that a prefix of the extended file is the base.
Not used: the TweetEval irony configuration (SemEval-2018 Task 3), which is the irony teacher's own
fine-tuning data and would put that teacher back in sample; and Founta et al. 2018, excluded under the
provenance rule in paper/setup_draft.md.

Every text is matched on the same key the splits use (utils.normalize_for_matching) against the
train, validation and test splits, every probe set, and the implicit benchmark's splits and
out-of-domain set when given, and dropped on any hit; exact duplicates within the transfer set are
dropped too. The result is unseen by every evaluation in the paper. transfer_report.json beside the
output records every count.
"""
import argparse
import html
import os

import pandas as pd

from .data import KEY, add_keys
from .utils import SIX, clean_text, ensure_dir, normalize_for_matching, save_json

SOURCES = {
    "olid": ("cardiffnlp/tweet_eval", "offensive", "text"),
    "hateval": ("cardiffnlp/tweet_eval", "hate", "text"),
    "davidson": ("tdavidson/hate_speech_offensive", None, "tweet"),
    "sentiment": ("cardiffnlp/tweet_eval", "sentiment", "text"),
    "emoji": ("cardiffnlp/tweet_eval", "emoji", "text"),
    "emotion": ("cardiffnlp/tweet_eval", "emotion", "text"),
}


def hub_texts(repo, config, column):
    import datasets
    d = datasets.load_dataset(repo, config) if config else datasets.load_dataset(repo)
    out = []
    for split in sorted(d.keys()):
        out.extend(d[split][column])
    return out


def conflicting_texts(raw_csv, text_col="tweet_text", label_col="cyberbullying_type", min_tokens=2):
    """One copy of every raw tweet that carried more than one label: dropped from the corpus by
    dmthd.prepare_tweets, so it is in no split, and in-domain by construction."""
    df = pd.read_csv(raw_csv, encoding="utf-8", encoding_errors="ignore")[[text_col, label_col]].dropna()
    df = df[df[label_col].isin(SIX)]
    df = add_keys(df, text_col)
    df = df[df[KEY].str.split().str.len() >= min_tokens]
    n_labels = df.groupby(KEY)[label_col].nunique()
    conflict = df[df[KEY].isin(set(n_labels[n_labels > 1].index))]
    return conflict.drop_duplicates(KEY, keep="first")[text_col].tolist()


def keys_of(csv_path, text_col="text"):
    df = pd.read_csv(csv_path, usecols=[text_col])
    return set(df[text_col].map(clean_text).map(normalize_for_matching))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="the prepared splits to keep the transfer set disjoint from")
    ap.add_argument("--out", required=True, help="output CSV (columns text, source); the report is written beside it")
    ap.add_argument("--probes", default=None, help="directory of probe CSVs to keep the transfer set disjoint from")
    ap.add_argument("--implicit_dir", default=None, help="the implicit benchmark's directory, likewise")
    ap.add_argument("--raw", default=None, help="raw corpus CSV, to recover the conflicting-label tweets")
    ap.add_argument("--base", default=None, help="an existing transfer CSV to extend: its rows come first, verbatim")
    ap.add_argument("--sources", default="olid,hateval,davidson")
    ap.add_argument("--min_tokens", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="keep a random subset of this size (smoke tests)")
    args = ap.parse_args()

    frames, report = [], {"sources": {}, "excluded_sources": ["tweet_eval/irony (the irony teacher's fine-tuning data)",
                                                              "founta2018 (provenance rule)"]}
    for name in [s for s in args.sources.split(",") if s]:
        repo, config, column = SOURCES[name]
        try:
            texts = hub_texts(repo, config, column)
        except Exception as e:                       # no internet, or the Hub is down: say so and go on
            report["sources"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"source {name} unavailable: {e}", flush=True)
            continue
        frames.append(pd.DataFrame({"text": [html.unescape(t) for t in texts], "source": name}))
        report["sources"][name] = {"rows_in": len(texts)}
    if args.raw and os.path.exists(args.raw):
        texts = conflicting_texts(args.raw, min_tokens=args.min_tokens)
        frames.append(pd.DataFrame({"text": texts, "source": "raw_conflicting"}))
        report["sources"]["raw_conflicting"] = {"rows_in": len(texts)}
    if not frames:
        raise SystemExit("no transfer text could be collected")
    df = pd.concat(frames, ignore_index=True)
    df = add_keys(df, "text")
    report["rows_in"] = int(len(df))
    df = df[df[KEY].str.split().str.len() >= args.min_tokens]
    report["rows_after_min_len"] = int(len(df))
    before = len(df)
    df = df.drop_duplicates(KEY, keep="first")
    report["rows_dropped_duplicates"] = int(before - len(df))

    # Disjoint from everything any model in the paper is evaluated on.
    seen, where = set(), {}
    for split in ("train", "val", "test"):
        p = os.path.join(args.data_dir, f"{split}.csv")
        if os.path.exists(p):
            where[f"tweets/{split}"] = keys_of(p)
    if args.probes and os.path.isdir(args.probes):
        for f in sorted(os.listdir(args.probes)):
            if f.endswith(".csv"):
                where[f"probes/{f}"] = keys_of(os.path.join(args.probes, f))
    if args.implicit_dir and os.path.isdir(args.implicit_dir):
        for f in sorted(os.listdir(args.implicit_dir)):
            if f.endswith(".csv"):
                where[f"implicit/{f}"] = keys_of(os.path.join(args.implicit_dir, f))
    base = None
    if args.base:
        base = pd.read_csv(args.base)
        where["base"] = set(base["text"].map(clean_text).map(normalize_for_matching))
        report["base"] = {"file": os.path.abspath(args.base), "rows": int(len(base))}
    report["rows_dropped_overlap"] = {}
    for name, ks in where.items():
        hit = df[KEY].isin(ks)
        report["rows_dropped_overlap"][name] = int(hit.sum())
        seen |= ks
    df = df[~df[KEY].isin(seen)]
    report["rows_out"] = int(len(df))
    report["rows_out_by_source"] = {k: int(v) for k, v in df["source"].value_counts().items()}

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    df = df[["text", "source"]]
    if base is not None:
        df = pd.concat([base[["text", "source"]], df], ignore_index=True)
    report["rows_written"] = int(len(df))
    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    df.to_csv(args.out, index=False, encoding="utf-8")
    report["seed"], report["output"] = args.seed, os.path.abspath(args.out)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    save_json(report, os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{stem}_report.json"))
    print("Transfer set report")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
