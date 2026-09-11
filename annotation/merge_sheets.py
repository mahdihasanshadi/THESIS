"""Merge the per-annotator files back into one sheet, checking that nobody reordered or edited rows.

    python annotation/merge_sheets.py --sheets annotation/sheets --sheet annotation/sarcbully_sheet.csv

Reads every `annotator_*.xlsx` or `annotator_*.csv` in the folder, matches rows by id (order does not
matter), and writes `<sheet>_filled.csv` with one column per annotator, ready for compute_kappa.py.
Reports, per person, how many items are labelled and whether any label is outside the allowed set.
"""
import argparse
import glob
import os
import re

import pandas as pd

LABELS = {"benign_sarcasm", "sarcastic_abuse", "not_sarcastic", "unsure"}


def read_one(path):
    df = pd.read_excel(path, sheet_name="annotation", dtype=str) if path.endswith(".xlsx") else pd.read_csv(path, dtype=str)
    df = df.fillna("")
    if "id" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"{path}: needs `id` and `label` columns, found {list(df.columns)}")
    df["label"] = df["label"].str.strip().str.lower()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="annotation/sheets")
    ap.add_argument("--sheet", default="annotation/sarcbully_sheet.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = pd.read_csv(args.sheet, dtype=str).fillna("")
    base = base[[c for c in ("id", "text") if c in base.columns]].copy()
    files = sorted(glob.glob(os.path.join(args.sheets, "annotator_*.xlsx")) + glob.glob(os.path.join(args.sheets, "annotator_*.csv")))
    seen = {}
    for p in files:
        who = re.sub(r"^annotator_", "", os.path.splitext(os.path.basename(p))[0])
        if who in seen:            # prefer the workbook when both formats are present
            if p.endswith(".csv"):
                continue
        df = read_one(p)
        labelled = df[df["label"] != ""]
        bad = sorted(set(labelled["label"]) - LABELS)
        missing_ids = set(base["id"]) - set(df["id"])
        extra_ids = set(df["id"]) - set(base["id"])
        print(f"{who}: {len(labelled)}/{len(base)} labelled"
              + (f", INVALID LABELS {bad}" if bad else "")
              + (f", {len(missing_ids)} ids missing" if missing_ids else "")
              + (f", {len(extra_ids)} unknown ids" if extra_ids else ""))
        m = df[["id", "label"]].drop_duplicates("id").set_index("id")["label"]
        base[who] = base["id"].map(m).fillna("")
        if "notes" in df.columns and df["notes"].str.strip().any():
            n = df[["id", "notes"]].drop_duplicates("id").set_index("id")["notes"]
            base[f"notes_{who}"] = base["id"].map(n).fillna("")
        seen[who] = p
    if not seen:
        raise SystemExit(f"no annotator_* files found in {args.sheets}")
    note_cols = [c for c in base.columns if c.startswith("notes_")]
    base["notes"] = base[note_cols].apply(lambda r: " | ".join(x for x in r if str(x).strip()), axis=1) if note_cols else ""
    base = base.drop(columns=note_cols)
    out = args.out or args.sheet.replace(".csv", "_filled.csv")
    base.to_csv(out, index=False, encoding="utf-8")
    print(f"\nmerged {len(seen)} annotators -> {out}")
    print("next: python annotation/compute_kappa.py --sheet " + out + " --out probes")


if __name__ == "__main__":
    main()
