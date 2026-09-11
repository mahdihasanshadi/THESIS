"""Build the implicit-abuse benchmark: the corpus the paper's stated goal actually needs.

    python -m dmthd.prepare_implicit --raw E:/dmthd-work/data/raw --out E:/dmthd-work/data/implicit \
        --probes probes --download

Neither existing benchmark labels indirect abuse. The tweet corpus buries it inside
`other_cyberbullying`; the Wikipedia corpus is a binary attack flag. So a student trained on either
one never sees a single example marked as abuse-by-implication, and implicit detection has only ever
been measured at inference time. This builds a third benchmark where the distinction is the label:

    not_hate | explicit_hate | implicit_hate

from the two corpora that annotate exactly that distinction:
  ISHate (BenjaminOcampo/ISHate)          two columns, not one: `hateful_layer` separates HS from
                                          Non-HS, and `implicit_layer` splits the HS rows into
                                          Explicit and Implicit. Non-HS rows carry an empty
                                          `implicit_layer`, so reading that column alone silently
                                          throws away every benign example.
  Implicit Hate Corpus stage 1            `class` in {not_hate, explicit_hate, implicit_hate}.
  (tasksource/implicit-hate-stg1)         The SALT-NLP mirror ships stage 2 only, which has no
                                          benign class at all; stage 1 is the one with all three.

Three rules, each of which a reviewer will check.

1. Augmented rows are dropped. ISHate ships paraphrase and back-translation augmentations; keeping
   them inflates the corpus with near-copies of its own training data.
2. Teacher provenance. Any ISHate row whose `source` is ToxiGen is dropped: ToxiGen is
   machine-generated from GPT-3 and appears in the training data of several public abuse
   checkpoints, so a test set containing it cannot be called held out.
3. Probe holdout. The sarcasm probes were built from these same corpora. Training on them would
   silently contaminate every probe number in the paper, including the ones already reported. Every
   text occurring in any probe file is therefore removed from this benchmark entirely, in all three
   splits, and the count is reported. The probes stay genuinely unseen; that is worth more than the
   rows it costs.

Everything else follows the same pipeline as the other two benchmarks: minimum length, drop texts
carrying more than one label, drop duplicates, stratified split, assert the splits are disjoint.
"""
import argparse
import glob
import os

import pandas as pd

from .data import dedup_and_split
from .utils import IMPLICIT3, ensure_dir, normalize_for_matching, save_json

# Class names in the order used by label_names("implicit3").
NOT, EXPLICIT, IMPLICIT = IMPLICIT3

ISHATE_MAP = {"Explicit HS": EXPLICIT, "Implicit HS": IMPLICIT}
IHC_MAP = {"not_hate": NOT, "explicit_hate": EXPLICIT, "implicit_hate": IMPLICIT}
# `augmented` is how ISHate marks its generated rows in the validation and test files, which carry
# no aug_method column at all; excluding the source as well as the method catches both.
DEFAULT_EXCLUDED_SOURCES = ("toxigen", "augmented")

HF_ISHATE = "BenjaminOcampo/ISHate"
HF_IHC = "tasksource/implicit-hate-stg1"


def _col(df, *candidates):
    """Hugging Face mirrors rename columns between revisions; fail loudly rather than silently
    building a benchmark out of the wrong field."""
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"none of {candidates} in columns {list(df.columns)}")


def _download(raw):
    from datasets import load_dataset
    for name, hf, splits in (("ishate", HF_ISHATE, ("train", "validation", "test")),
                             ("implicit_hate_stg1", HF_IHC, ("train",))):
        d = ensure_dir(os.path.join(raw, name))
        for sp in splits:
            p = os.path.join(d, f"{sp}.parquet")
            if os.path.exists(p):
                continue
            print(f"downloading {hf} [{sp}] -> {p}", flush=True)
            load_dataset(hf, split=sp).to_pandas().to_parquet(p)


def load_ishate(raw, excluded):
    parts = []
    for sp in ("train", "validation", "test"):
        p = os.path.join(raw, "ishate", f"{sp}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame(columns=["text", "label_name", "corpus", "origin", "subtle"]), {"available": False}
    d = pd.concat(parts, ignore_index=True)
    stats = {"rows_raw": int(len(d))}
    if "aug_method" in d.columns:
        d = d[d["aug_method"].fillna("orig") == "orig"]
        stats["rows_after_orig_only"] = int(len(d))
    src = _col(d, "source", "dataset", "origin")
    d = d.assign(_origin=d[src].astype(str).str.lower())
    stats["by_source_before_exclusion"] = d["_origin"].value_counts().to_dict()
    d = d[~d["_origin"].isin(excluded)]
    stats["rows_after_source_exclusion"] = int(len(d))
    txt = _col(d, "text", "post", "tweet")
    hate, imp = _col(d, "hateful_layer"), _col(d, "implicit_layer")
    # Two columns, in this order: Non-HS is only recorded in `hateful_layer`, and those rows have an
    # empty `implicit_layer`. Mapping `implicit_layer` alone would drop every benign example.
    lab = d[imp].map(ISHATE_MAP).where(d[hate].astype(str) != "Non-HS", NOT)
    keep = lab.notna()
    stats["rows_without_a_usable_label"] = int((~keep).sum())
    out = pd.DataFrame({"text": d[txt][keep].astype(str), "label_name": lab[keep],
                        "corpus": "ISHate", "origin": d["_origin"][keep],
                        "subtle": d["subtlety_layer"][keep].astype(str) if "subtlety_layer" in d.columns else ""})
    stats["rows_kept"] = int(len(out))
    stats["by_class"] = out["label_name"].value_counts().to_dict()
    return out.reset_index(drop=True), stats


def load_ihc(raw):
    paths = sorted(glob.glob(os.path.join(raw, "implicit_hate_stg1", "*.parquet")))
    if not paths:
        return pd.DataFrame(columns=["text", "label_name", "corpus", "origin", "subtle"]), {"available": False}
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    stats = {"rows_raw": int(len(d))}
    lab = _col(d, "class", "label", "stage1")
    txt = _col(d, "post", "text", "tweet")
    keep = d[lab].isin(IHC_MAP)
    out = pd.DataFrame({"text": d[txt][keep].astype(str), "label_name": d[lab][keep].map(IHC_MAP),
                        "corpus": "ImplicitHate", "origin": "ihc", "subtle": ""})
    stats["rows_kept"] = int(len(out))
    stats["by_class"] = out["label_name"].value_counts().to_dict()
    return out.reset_index(drop=True), stats


def probe_keys(probe_dir):
    """Every text used by any probe set, as matching keys."""
    keys, files = set(), {}
    for p in sorted(glob.glob(os.path.join(probe_dir, "*.csv"))):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "text" not in d.columns:
            continue
        k = set(d["text"].map(normalize_for_matching))
        files[os.path.basename(p)] = len(k)
        keys |= k
    return keys, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding ishate/ and implicit_hate/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default="probes", help="probe directory to hold out; empty disables")
    ap.add_argument("--exclude_sources", default=",".join(DEFAULT_EXCLUDED_SOURCES))
    ap.add_argument("--corpora", default="ImplicitHate",
                    help="which source corpora form the train/val/test splits. Default is the "
                         "Implicit Hate Corpus alone, and the reason is measured rather than "
                         "assumed: 95 per cent of implicit examples come from it and 89 per cent of "
                         "explicit examples come from ISHate, while a lexical model tells the two "
                         "corpora apart at 0.91 macro-F1. A model trained on both can therefore earn "
                         "an implicit-versus-explicit score by recognising the source. It also does "
                         "worse where it matters: on identical Implicit Hate test rows, training on "
                         "both gives implicit-hate F1 0.473 against 0.548 for training on that "
                         "corpus alone. Pass `ImplicitHate,ISHate` to rebuild the combined version.")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    if args.download:
        _download(args.raw)
    excluded = {s.strip().lower() for s in args.exclude_sources.split(",") if s.strip()}

    ish, ish_stats = load_ishate(args.raw, excluded)
    ihc, ihc_stats = load_ihc(args.raw)
    if not len(ish) and not len(ihc):
        raise SystemExit(f"no source corpora under {args.raw}; pass --download or fetch them first")
    df = pd.concat([ish, ihc], ignore_index=True)
    report = {"excluded_sources": sorted(excluded), "ishate": ish_stats, "implicit_hate": ihc_stats,
              "rows_combined": int(len(df))}

    if args.probes and os.path.isdir(args.probes):
        keys, files = probe_keys(args.probes)
        hit = df["text"].map(normalize_for_matching).isin(keys)
        report["probe_holdout"] = {"probe_files": files, "probe_texts": len(keys),
                                   "rows_removed": int(hit.sum()),
                                   "removed_by_class": df[hit]["label_name"].value_counts().to_dict(),
                                   "policy": "probe texts are removed from all splits so every probe "
                                             "number in the paper stays measured on unseen text"}
        df = df[~hit].reset_index(drop=True)
    else:
        report["probe_holdout"] = {"policy": "disabled", "warning":
                                   "the probe sets share sources with this corpus; probe metrics "
                                   "from a model trained on it would be contaminated"}

    # The splits come from the chosen corpora only. Whatever is left over becomes an out-of-domain
    # test set instead of being thrown away: a model that reads implication should still read it in
    # a corpus collected by other people from another platform, and that is a stronger claim than
    # any in-domain score.
    chosen = [c.strip() for c in args.corpora.split(",") if c.strip()]
    unknown = set(chosen) - set(df["corpus"].unique())
    if unknown:
        raise SystemExit(f"--corpora names {sorted(unknown)}, but the data holds "
                         f"{sorted(df['corpus'].unique())}")
    primary, held = df[df["corpus"].isin(chosen)].copy(), df[~df["corpus"].isin(chosen)].copy()
    report["corpora_in_splits"] = chosen
    report["rows_primary"], report["rows_held_out"] = int(len(primary)), int(len(held))

    train, val, test, dd = dedup_and_split(primary, "text", "label_name", seed=args.seed,
                                           val_frac=args.val_frac, test_frac=args.test_frac)
    report["dedup"] = dd
    ensure_dir(args.out)
    ids = {n: i for i, n in enumerate(IMPLICIT3)}
    cols = ["text", "label_name", "corpus", "origin", "subtle"]
    for name, d in (("train", train), ("val", val), ("test", test)):
        out = d[cols].copy()
        out["label"] = out["label_name"].map(ids).astype(int)
        out.to_csv(os.path.join(args.out, f"{name}.csv"), index=False, encoding="utf-8")
    report["corpus_mix"] = {n: d["corpus"].value_counts().to_dict()
                            for n, d in (("train", train), ("val", val), ("test", test))}

    # Out-of-domain sets, one per held-out corpus, with every text that occurs anywhere in the
    # splits removed first: the two corpora share rows, and an "out-of-domain" set containing
    # training text would measure memorisation.
    seen = set(pd.concat([train, val, test])["text"].map(normalize_for_matching))
    report["out_of_domain"] = {}
    for corpus in sorted(held["corpus"].unique()):
        d = held[held["corpus"] == corpus].copy()
        d["_key"] = d["text"].map(normalize_for_matching)
        overlap = int(d["_key"].isin(seen).sum())
        d = d[~d["_key"].isin(seen)]
        # conflicts must be found before duplicates are dropped, or every key is unique by
        # construction and the check silently passes
        n_labels = d.groupby("_key")["label_name"].nunique()
        conflicting = set(n_labels[n_labels > 1].index)
        d = d[~d["_key"].isin(conflicting)].drop_duplicates("_key")
        out = d[cols].copy()
        out["label"] = out["label_name"].map(ids).astype(int)
        path = os.path.join(args.out, f"test_ood_{corpus.lower()}.csv")
        out.to_csv(path, index=False, encoding="utf-8")
        report["out_of_domain"][corpus] = {
            "file": os.path.basename(path), "rows": int(len(out)),
            "rows_dropped_overlapping_the_splits": overlap,
            "texts_dropped_with_conflicting_labels": int(len(conflicting)),
            "by_class": out["label_name"].value_counts().to_dict()}
    save_json({"implicit3": IMPLICIT3, "text_column": "text", "label_column": "label", "max_len": 128},
              os.path.join(args.out, "label_info.json"))
    report["seed"] = args.seed
    save_json(report, os.path.join(args.out, "report.json"))
    print("Implicit benchmark")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
