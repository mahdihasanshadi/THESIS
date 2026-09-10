"""Build the targeted test sets that measure the sarcasm claim. Inference only; never used
for training.

  benign_sarcasm.csv  sarcastic, non-abusive tweets from iSarcasmEval (SemEval-2022 Task 6).
                      Metric: false-positive rate = share predicted as any bullying class.
  ironic_abuse.csv    posts labelled `irony` in the Implicit Hate Corpus stage-2 data, plus
                      ISHate original rows labelled Implicit HS whose source is not ToxiGen or
                      the Implicit Hate Corpus (teacher provenance). Metric: recall.
  implicit_abuse.csv  all ISHate original Implicit HS rows under the same source rule.

    python -m dmthd.build_probes --raw E:/dmthd-work/data/raw --out E:/dmthd-work/data/probes

Expected raw layout (see README):
  raw/isarcasm/train.En.csv, raw/isarcasm/task_A_En_test.csv
  raw/implicit_hate/train.parquet            (SALT-NLP/ImplicitHate on Hugging Face)
  raw/ishate/{train,validation,test}.parquet (BenjaminOcampo/ISHate on Hugging Face)
"""
import argparse
import os

import pandas as pd

from .utils import clean_text, ensure_dir, normalize_for_matching, save_json

EXCLUDED_SOURCES = {"toxigen", "ihc"}


def load_isarcasm(raw):
    tr = pd.read_csv(os.path.join(raw, "isarcasm", "train.En.csv"))
    te = pd.read_csv(os.path.join(raw, "isarcasm", "task_A_En_test.csv"))
    tr = tr.rename(columns={"tweet": "text"})[["text", "sarcastic"]].assign(split="train")
    te = te.rename(columns={"tweet": "text"})[["text", "sarcastic"]].assign(split="test")
    df = pd.concat([tr, te], ignore_index=True).dropna(subset=["text"])
    return df[df["sarcastic"] == 1][["text", "split"]].assign(source="iSarcasmEval")


def load_ishate(raw):
    parts = []
    for split in ("train", "validation", "test"):
        p = os.path.join(raw, "ishate", f"{split}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            if "aug_method" in d.columns:
                d = d[d["aug_method"] == "orig"]
            parts.append(d.assign(split=split))
    d = pd.concat(parts, ignore_index=True)
    d = d[~d["source"].isin(EXCLUDED_SOURCES)]
    imp = d[d["implicit_layer"] == "Implicit HS"]
    return imp[["text", "source", "split"]]


def load_implicit_hate_irony(raw):
    p = os.path.join(raw, "implicit_hate", "train.parquet")
    d = pd.read_parquet(p)
    d = d[d["implicit_class"] == "irony"].rename(columns={"post": "text"})
    return d[["text"]].assign(source="ImplicitHate-stg2-irony", split="all")


def finalize(df):
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df["_key"] = df["text"].map(normalize_for_matching)
    df = df[df["_key"].str.split().str.len() >= 2].drop_duplicates("_key").drop(columns="_key")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ensure_dir(args.out)

    benign = finalize(load_isarcasm(args.raw))
    ishate_implicit = load_ishate(args.raw)
    ironic = finalize(pd.concat([load_implicit_hate_irony(args.raw), ishate_implicit], ignore_index=True))
    implicit = finalize(ishate_implicit)

    benign.to_csv(os.path.join(args.out, "benign_sarcasm.csv"), index=False)
    ironic.to_csv(os.path.join(args.out, "ironic_abuse.csv"), index=False)
    implicit.to_csv(os.path.join(args.out, "implicit_abuse.csv"), index=False)
    report = {"benign_sarcasm": int(len(benign)), "ironic_abuse": int(len(ironic)), "implicit_abuse": int(len(implicit)),
              "ironic_abuse_by_source": ironic["source"].value_counts().to_dict(),
              "excluded_sources": sorted(EXCLUDED_SOURCES),
              "note": "benign_sarcasm needs one manual screening pass to drop sarcastic tweets that are also abusive"}
    save_json(report, os.path.join(args.out, "report.json"))
    print(report)


if __name__ == "__main__":
    main()
