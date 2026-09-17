"""Score the five predictions of DECISIONS D19 against a finished run, so the verdict is read off the
tables rather than argued from them.

    python scripts/transfer_verdict.py --tables paper/tables

Reads significance.csv (dmthd.significance) and implicit.csv (dmthd.tables). Prints each prediction,
the number that decides it, and PASS / FAIL / not run.
"""
import argparse
import os
import re

import pandas as pd


def interval(s):
    lo, hi = re.findall(r"[-+]?\d*\.\d+", str(s))[:2]
    return float(lo), float(hi)


def row(sig, base, cand):
    r = sig[(sig["Baseline"] == base) & (sig["Candidate"] == cand)]
    return None if r.empty else r.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="paper/tables")
    a = ap.parse_args()
    sig = pd.read_csv(os.path.join(a.tables, "significance.csv"), dtype=str)
    sig = sig[sig["Student"] == "BERT-mini"]
    verdicts = []

    def judge(name, r, test, detail):
        if r is None:
            verdicts.append((name, "not run", ""))
            return
        d, (lo, hi) = float(r["Difference"]), interval(r["95 per cent interval"])
        verdicts.append((name, "PASS" if test(d, lo, hi) else "FAIL", detail(d, lo, hi)))

    judge("1. the committee with the transfer set beats fine-tuning by >= 0.010, interval excluding zero",
          row(sig, "Fine-tune only", "Uniform multi-teacher + transfer set"),
          lambda d, lo, hi: d >= 0.010 and lo > 0, lambda d, lo, hi: f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    judge("2. the committee is the better labeller (transfer: committee minus one teacher > 0)",
          row(sig, "Single-teacher KD + transfer set", "Uniform multi-teacher + transfer set"),
          lambda d, lo, hi: d > 0, lambda d, lo, hi: f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] (interval may include zero; F29 carries the claim)")
    judge("3. out-of-sample reliability adds at most 0.003 over averaging",
          row(sig, "Uniform multi-teacher + transfer set", "Out-of-sample reliability weighting + transfer set"),
          lambda d, lo, hi: d <= 0.003, lambda d, lo, hi: f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    judge("4. hard pseudo-labels gain less than soft labels (hard minus soft < 0)",
          row(sig, "Uniform multi-teacher + transfer set", "Committee hard pseudo-labels + transfer set"),
          lambda d, lo, hi: d < 0, lambda d, lo, hi: f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    imp_path = os.path.join(a.tables, "implicit.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path, dtype=str)
        col = "Sarcasm-discrimination AUC"
        arms = imp[imp["Method"].str.contains("transfer set")]
        if not arms.empty and col in arms.columns:
            vals = [float(v) for v in arms[col] if v not in ("--", "nan")]
            ok = all(0.756 <= v <= 0.777 for v in vals)
            verdicts.append(("5. sarcasm-discrimination AUC on the transfer arms stays within 0.756 to 0.777",
                             "PASS" if ok else "FAIL", ", ".join(f"{m}: {v}" for m, v in zip(arms["Method"], arms[col]))))
        else:
            verdicts.append(("5. sarcasm-discrimination AUC on the transfer arms stays within 0.756 to 0.777", "not run", ""))
    for name, verdict, detail in verdicts:
        print(f"{verdict:<8} {name}\n         {detail}")
    print("\nThe route stands on prediction 1. Predictions 2 to 5 shape the wording, not the decision.")


if __name__ == "__main__":
    main()
