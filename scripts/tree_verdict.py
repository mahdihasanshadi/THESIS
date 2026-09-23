"""Score the predictions of DECISIONS D22 against a finished run, so the verdict on the non-neural
student, and what the thesis then says about the papers that use one, is read off the tables.

    python scripts/tree_verdict.py --tables paper/tables

Reads trees.csv (dmthd.tables) and significance.csv (dmthd.significance). Prints each prediction, the
numbers that decide it, and PASS / FAIL / not run, then the decision rule's answer.
"""
import argparse
import os
import re

import pandas as pd

GOLD = "Gold labels only (no teacher)"
ONE = "Gold labels + one teacher's soft labels"
COMMITTEE = "Gold labels + the committee's soft labels"
SOFT = "The committee's soft labels alone"
PSEUDO = "The committee's hard pseudo-labels"
DISTILLED = (ONE, COMMITTEE, SOFT, PSEUDO)


def mean_of(s):
    return float(str(s).split("$")[0].strip())


def interval(s):
    lo, hi = re.findall(r"[-+]?\d*\.\d+", str(s))[:2]
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="paper/tables")
    ap.add_argument("--neural_ft", type=float, default=0.8393, help="BERT-mini fine-tuned alone")
    ap.add_argument("--floor", type=float, default=0.8798, help="TF-IDF + logistic regression")
    a = ap.parse_args()
    path = os.path.join(a.tables, "trees.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: the tree student (D22) has not run yet")
    t = pd.read_csv(path, dtype=str)
    f1 = {r["Target the trees fit"]: mean_of(r["Macro-F1"]) for _, r in t.iterrows()}
    sig_path = os.path.join(a.tables, "significance.csv")
    sig = pd.read_csv(sig_path, dtype=str) if os.path.exists(sig_path) else None
    rows = sig[sig["Student"] == "Gradient-boosted trees"] if sig is not None else pd.DataFrame()
    against_gold = {r["Candidate"]: r for _, r in rows.iterrows() if r["Baseline"] == GOLD}
    verdicts = []

    # 1. no distilled target beats the gold labels by more than 0.010, and no interval excludes zero
    gains = {k: f1[k] - f1[GOLD] for k in DISTILLED if k in f1 and GOLD in f1}
    clear = [k for k, r in against_gold.items() if r["Interval excludes zero"] == "yes" and float(r["Difference"]) > 0]
    ok1 = bool(gains) and max(gains.values()) < 0.010 and not clear
    verdicts.append(("1. distillation adds less than 0.010 to a tree student, no interval clear of zero",
                     "PASS" if ok1 else "FAIL" if gains else "not run",
                     ", ".join(f"{k.split('+')[-1].strip()} {v:+.4f}" for k, v in gains.items())
                     + (f"; clear of zero: {', '.join(clear)}" if clear else "; none clear of zero")))

    # 2. the committee does not beat one teacher by more than 0.005
    if ONE in f1 and COMMITTEE in f1:
        d = f1[COMMITTEE] - f1[ONE]
        verdicts.append(("2. the committee beats one teacher by at most 0.005", "PASS" if d <= 0.005 else "FAIL",
                         f"committee {f1[COMMITTEE]:.4f} against one teacher {f1[ONE]:.4f}, {d:+.4f}"))

    # 3. frozen features cost more than any label can return
    if f1:
        best = max(f1.values())
        verdicts.append(("3. every tree arm lands below the fine-tuned neural student and the classical floor",
                         "PASS" if best < min(a.neural_ft, a.floor) else "FAIL",
                         f"best tree arm {best:.4f}; BERT-mini fine-tuned {a.neural_ft:.4f}; floor {a.floor:.4f}"))

    width = max(len(v[0]) for v in verdicts)
    for text, mark, detail in verdicts:
        print(f"{text:<{width}}  {mark:<8} {detail}")
    if not gains:
        return
    exception = bool(clear) and max(gains.values()) >= 0.010
    print("\nDecision rule (D22):")
    if exception:
        print("  A distilled target beats the gold labels by at least 0.010 with an interval clear of zero.")
        print("  The thesis reports the tree student as the exception, states that the audit's null holds for")
        print("  students that fine-tune their own representation, and softens the related-work sentence.")
    else:
        print("  No distilled target clears the bar. The audit extends to the non-neural student: on frozen")
        print("  features, in sample, the teachers' soft labels add nothing a tree model can use either, and")
        print("  Chapter 2 names the confound in the papers that report otherwise.")


if __name__ == "__main__":
    main()
