"""A verifiable fingerprint of a prepared corpus, for release alongside the paper.

    python -m dmthd.manifest --data_dir data/implicit --out paper/manifests/implicit_manifest.csv

None of the three corpora may be re-hosted: each carries its own licence and none of them is ours to
redistribute. That normally leaves a paper claiming "reproducible" on the strength of a script alone,
which nobody can check without first rebuilding everything and hoping the result matches.

This writes one row per example with the split it landed in, its label, its source corpus, and a
SHA-1 of the normalised text. It contains no text, so it redistributes nothing; it is enough for a
reader who has the source corpora to confirm, row by row, that their rebuild is byte-identical to
ours, and to find exactly which rows differ if it is not. `--verify` does that comparison.

The file also records the split fingerprints, so a teacher cache built here can be shown to belong to
the same split a reader reconstructs.
"""
import argparse
import hashlib
import os

import pandas as pd

from .utils import normalize_for_matching, save_json, split_fingerprint

SPLITS = ("train", "val", "test")


def key_hash(text):
    return hashlib.sha1(normalize_for_matching(text).encode("utf-8")).hexdigest()[:16]


def build(data_dir):
    rows, fp = [], {}
    for sp in SPLITS:
        p = os.path.join(data_dir, f"{sp}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        fp[sp] = split_fingerprint(d["text"].tolist(), d["label"].tolist() if "label" in d else None)
        out = pd.DataFrame({"split": sp, "row": range(len(d)), "sha1": d["text"].map(key_hash)})
        for col in ("label_name", "label", "corpus", "origin"):
            if col in d.columns:
                out[col] = d[col].values
        rows.append(out)
    if not rows:
        raise SystemExit(f"no train/val/test CSVs under {data_dir}")
    return pd.concat(rows, ignore_index=True), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None, help="an existing manifest to compare this corpus against")
    args = ap.parse_args()

    man, fp = build(args.data_dir)
    if args.verify:
        ref = pd.read_csv(args.verify)
        same = len(ref) == len(man) and (ref["sha1"].values == man["sha1"].values).all() \
            and (ref["split"].values == man["split"].values).all()
        if same:
            print(f"MATCH: {len(man)} rows identical to {args.verify}, split for split")
            return
        a, b = set(zip(ref["split"], ref["sha1"])), set(zip(man["split"], man["sha1"]))
        print(f"DIFFERENT: {len(ref)} rows in the reference, {len(man)} here; "
              f"{len(a - b)} only in the reference, {len(b - a)} only here")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    man.to_csv(args.out, index=False)
    meta = {"corpus": os.path.basename(os.path.abspath(args.data_dir)), "rows": int(len(man)),
            "split_fingerprints": fp, "hash": "sha1 of the normalised text, first 16 hex characters",
            "normalisation": "lower-case, URLs and @mentions stripped, punctuation to spaces, "
                             "whitespace collapsed (dmthd.utils.normalize_for_matching)",
            "note": "no text is included; this file exists so a reader who holds the source corpora "
                    "can verify their rebuild matches ours row for row"}
    stem = args.out[: -len(".gz")] if args.out.endswith(".gz") else args.out
    save_json(meta, os.path.splitext(stem)[0] + "_meta.json")
    print(f"{len(man)} rows -> {args.out}")
    for sp, h in fp.items():
        print(f"  {sp}: {int((man['split'] == sp).sum())} rows, split fingerprint {h}")


if __name__ == "__main__":
    main()
