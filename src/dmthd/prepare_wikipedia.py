"""Wikipedia Talk personal attacks (Wulczyn, Thain and Dixon, 2017), the Phase-2 pipeline as a
script. Public data from Figshare; --download fetches it.

    python -m dmthd.prepare_wikipedia --raw E:/dmthd-work/data/raw/wikipedia --out E:/dmthd-work/data/wikipedia --download

Steps, in order (every count goes to report.json):
  1. aggregate the roughly ten crowd votes per comment into a soft label (mean attack vote);
     hard label = soft > 0.5, so a tie is not-attack;
  2. clean each comment: NEWLINE/TAB tokens, HTML tags and entities, URLs, IP addresses,
     timestamps, signatures and wiki markup removed, whitespace collapsed; case is kept;
  3. apply the corpus's official train/dev/test split, then drop comments shorter than three
     tokens, de-duplicate within each split, and remove from dev and test every comment whose
     text also appears in train (train is the reference);
  4. assert the three splits are disjoint on text and on rev_id.

Output columns: text, label_name, label (0 = not_cyberbullying, 1 = cyberbullying), soft_label, rev_id.
Use --scheme binary --label_col label --max_len 256 with the training scripts.
"""
import argparse
import html
import os
import re
import urllib.request

import pandas as pd

from .utils import ensure_dir, normalize_for_matching, save_json

URLS = {"attack_annotated_comments.tsv": "https://ndownloader.figshare.com/files/7554634",
        "attack_annotations.tsv": "https://ndownloader.figshare.com/files/7554637"}


def clean_comment(t: str) -> str:
    t = str(t).replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " ")
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", t)                                   # IP addresses
    t = re.sub(r"\d{1,2}:\d{2},? \d{1,2} \w+ \d{4} \(UTC\)", " ", t)                     # timestamps
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)                              # [[link|text]] -> text
    t = re.sub(r"\{\{[^}]*\}\}", " ", t)                                                # templates
    t = re.sub(r"'{2,}", "", t)                                                         # bold/italic quotes
    t = re.sub(r"={2,}", " ", t)                                                        # headings
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--min_tokens", type=int, default=3)
    args = ap.parse_args()
    ensure_dir(args.raw)
    if args.download:
        for name, url in URLS.items():
            p = os.path.join(args.raw, name)
            if not os.path.exists(p):
                print("downloading", name, flush=True)
                urllib.request.urlretrieve(url, p)

    comments = pd.read_csv(os.path.join(args.raw, "attack_annotated_comments.tsv"), sep="\t")
    ann = pd.read_csv(os.path.join(args.raw, "attack_annotations.tsv"), sep="\t")
    report = {"comments_in": int(len(comments)), "annotations_in": int(len(ann)),
              "annotators_per_comment_median": float(ann.groupby("rev_id").size().median())}

    soft = ann.groupby("rev_id")["attack"].mean().rename("soft_label")
    df = comments.merge(soft, on="rev_id", how="inner")
    df["label"] = (df["soft_label"] > 0.5).astype(int)
    df["label_name"] = df["label"].map({0: "not_cyberbullying", 1: "cyberbullying"})
    df["text"] = df["comment"].map(clean_comment)
    df["_key"] = df["text"].map(normalize_for_matching)
    df["split"] = df["split"].map({"train": "train", "dev": "val", "test": "test"})
    report["after_merge"] = int(len(df))

    df = df[df["_key"].str.split().str.len() >= args.min_tokens]
    report["after_min_len"] = int(len(df))

    splits = {}
    for name in ("train", "val", "test"):
        d = df[df["split"] == name]
        before = len(d)
        d = d.drop_duplicates("_key", keep="first")
        report[f"{name}_dropped_duplicates"] = int(before - len(d))
        splits[name] = d
    train_keys = set(splits["train"]["_key"])
    for name in ("val", "test"):
        d = splits[name]
        before = len(d)
        d = d[~d["_key"].isin(train_keys)]
        report[f"{name}_dropped_overlap_with_train"] = int(before - len(d))
        splits[name] = d
    val_keys = set(splits["val"]["_key"])
    before = len(splits["test"])
    splits["test"] = splits["test"][~splits["test"]["_key"].isin(val_keys)]
    report["test_dropped_overlap_with_val"] = int(before - len(splits["test"]))

    names = list(splits)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = splits[names[i]], splits[names[j]]
            assert not (set(a["_key"]) & set(b["_key"])), f"LEAK on text between {names[i]} and {names[j]}"
            assert not (set(a["rev_id"]) & set(b["rev_id"])), f"LEAK on rev_id between {names[i]} and {names[j]}"

    ensure_dir(args.out)
    for name, d in splits.items():
        d[["text", "label_name", "label", "soft_label", "rev_id"]].to_csv(os.path.join(args.out, f"{name}.csv"),
                                                                            index=False, encoding="utf-8")
        report[f"{name}_rows"] = int(len(d))
        report[f"{name}_attack_share"] = round(float(d["label"].mean()), 4)
    report["ambiguous_share_0.2_to_0.8"] = round(float(((df["soft_label"] >= 0.2) & (df["soft_label"] <= 0.8)).mean()), 4)
    save_json({"binary": ["not_cyberbullying", "cyberbullying"], "text_column": "text", "label_column": "label",
               "soft_column": "soft_label", "max_len": 256}, os.path.join(args.out, "label_info.json"))
    save_json(report, os.path.join(args.out, "report.json"))
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
