"""Build the 300-tweet annotation sheet: every row the classical classifier flagged as abusive
(probes/benign_sarcasm_review.csv) plus random unflagged rows to reach 300. Model scores are kept
in a separate key file so annotators never see them.

    python annotation/make_sheet.py --probes probes --out annotation --n 300 --seed 42
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--out", default="annotation")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    flagged = pd.read_csv(os.path.join(args.probes, "benign_sarcasm_review.csv"))
    flagged["flagged"] = True
    kept = pd.read_csv(os.path.join(args.probes, "benign_sarcasm_screened.csv"))
    kept["flagged"] = False
    kept["p_bullying"], kept["pred_class"] = float("nan"), ""
    n_random = max(0, args.n - len(flagged))
    sample = kept.sample(n=min(n_random, len(kept)), random_state=args.seed)
    sheet = pd.concat([flagged, sample], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sheet.insert(0, "id", [f"S{i:03d}" for i in range(1, len(sheet) + 1)])

    os.makedirs(args.out, exist_ok=True)
    public = sheet[["id", "text"]].copy()
    for a in ("A1", "A2", "A3", "A4"):
        public[a] = ""
    public["notes"] = ""
    public.to_csv(os.path.join(args.out, "sarcbully_sheet.csv"), index=False, encoding="utf-8")
    sheet[["id", "source", "split", "flagged", "p_bullying", "pred_class"]].to_csv(os.path.join(args.out, "sarcbully_key.csv"), index=False)
    print(f"sheet: {len(public)} rows ({int(sheet['flagged'].sum())} flagged, {int((~sheet['flagged']).sum())} random) -> {args.out}/sarcbully_sheet.csv")


if __name__ == "__main__":
    main()
