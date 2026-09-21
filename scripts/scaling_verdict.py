"""Score the four predictions of DECISIONS D20 against a finished run, so the verdict on the size
curve, and the title decision that hangs on it, is read off the tables rather than argued from them.

    python scripts/scaling_verdict.py --tables paper/tables

Reads scaling.csv (dmthd.tables) and significance.csv (dmthd.significance). Prints each prediction,
the numbers that decide it, and PASS / FAIL / not run, then the decision rule's answer.
"""
import argparse
import os
import re

import pandas as pd


def interval(s):
    lo, hi = re.findall(r"[-+]?\d*\.\d+", str(s))[:2]
    return float(lo), float(hi)


def mean_of(s):
    return float(str(s).split("$")[0].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="paper/tables")
    a = ap.parse_args()
    sc = pd.read_csv(os.path.join(a.tables, "scaling.csv"), dtype=str)
    sc["f1"] = sc["Macro-F1"].map(mean_of)
    sc["rows"] = sc["Transfer rows"].astype(int)
    by_arm = {r["Arm"]: r for _, r in sc.iterrows()}
    ft = by_arm["Fine-tune only"]["f1"]
    abuse = sc[sc["Composition"] == "abuse-domain tweets"].sort_values("rows")
    mixed = sc[sc["Composition"] == "abuse-domain + generic"].sort_values("rows")
    generic = sc[sc["Composition"] == "generic tweets"]
    matched = sc[sc["Arm"].str.contains("matched")]
    sig_path = os.path.join(a.tables, "significance.csv")
    sig = pd.read_csv(sig_path, dtype=str) if os.path.exists(sig_path) else None
    verdicts = []

    # 1. ordered gains at fixed composition, and 42k reproduces F31
    gains = [(int(r.rows), r.f1 - ft) for r in abuse.itertuples()]
    inversions = [(gains[i - 1][1] - gains[i][1]) for i in range(1, len(gains)) if gains[i][1] < gains[i - 1][1]]
    base = next((g for n, g in gains if n == 42013), None)
    ok1 = len(inversions) <= 1 and all(v < 0.002 for v in inversions) and base is not None and 0.006 <= base <= 0.009
    verdicts.append(("1. the gain rises with size at fixed composition, and 42k reproduces F31 (+0.006 to +0.009)",
                     "PASS" if ok1 else "FAIL",
                     ", ".join(f"{n // 1000}k {g:+.4f}" for n, g in gains) + f"; inversions {len(inversions)}"))

    # 2. composition: generic 42k gains at least 0.003 less than abuse 42k; generic rows beyond 42k add less per row
    if base is not None and not generic.empty:
        g42 = float(generic.iloc[0].f1) - ft
        per_row_abuse = base / 42013
        beyond = [(int(r.rows), (r.f1 - ft - base) / (int(r.rows) - 42013)) for r in mixed.itertuples()]
        ok2a = base - g42 >= 0.003
        ok2b = all(v < per_row_abuse for _, v in beyond) if beyond else False
        verdicts.append(("2a. 42k generic tweets gain at least 0.003 less than 42k abuse-domain tweets",
                         "PASS" if ok2a else "FAIL", f"abuse {base:+.4f}, generic {g42:+.4f}, difference {base - g42:+.4f}"))
        verdicts.append(("2b. each generic row added beyond 42k adds less than the abuse-domain rows did",
                         "PASS" if ok2b else "FAIL",
                         f"abuse {per_row_abuse * 10000:+.4f} per 10k rows; beyond 42k: "
                         + ", ".join(f"to {n // 1000}k {v * 10000:+.4f} per 10k" for n, v in beyond)))
    else:
        verdicts.append(("2. composition control", "not run", ""))

    # 3. steps: matched fine-tuning gains at most +0.002; the 42k arm beats it by at least 0.004
    if base is not None and not matched.empty:
        m = float(matched.iloc[0].f1) - ft
        verdicts.append(("3a. fine-tuning for as many updates as the 42k arm gains at most +0.002",
                         "PASS" if m <= 0.002 else "FAIL", f"matched steps {m:+.4f}"))
        verdicts.append(("3b. the 42k arm beats the matched-steps control by at least 0.004",
                         "PASS" if base - m >= 0.004 else "FAIL", f"{base - m:+.4f}"))
    else:
        verdicts.append(("3. steps control", "not run", ""))

    # 4. the decision rule
    if not mixed.empty:
        top = mixed.iloc[-1]
        top_gain = float(top.f1) - ft
        row = None
        if sig is not None:
            hit = sig[(sig["Student"] == "BERT-mini") & (sig["Baseline"] == "Fine-tune only") & (sig["Candidate"] == top["Arm"])]
            row = None if hit.empty else hit.iloc[0]
        lo, hi = interval(row["95 per cent interval"]) if row is not None else (float("nan"), float("nan"))
        prev = float(mixed.iloc[-2].f1) if len(mixed) > 1 else float(abuse.iloc[-1].f1)
        rising = float(top.f1) - prev > 0
        flat = base is not None and abs(float(top.f1) - (ft + base)) <= 0.003
        lead = top_gain >= 0.012 and (row is not None and lo > 0) and rising
        verdicts.append((f"4. decision rule: {top['Arm']} gains >= +0.012, interval excludes zero, still rising past the previous size",
                         "PASS" if lead else "FAIL",
                         f"gain {top_gain:+.4f}, interval [{lo:+.4f}, {hi:+.4f}], rise over previous size {float(top.f1) - prev:+.4f}, "
                         f"within 0.003 of 42k: {'yes' if flat else 'no'}"))
        decision = ("the constructive result leads the paper and the title changes to say so" if lead else
                    "the curve is flat beyond 42k: the recipe is the audit's coda and B2 stands" if flat else
                    "neither branch of the rule fires as written: the curve rises but not far enough, or the interval includes zero; "
                    "report it as a partial result and decide the framing with the slope and the caveats in view")
    else:
        decision = "the larger arms did not run"
    for name, verdict, detail in verdicts:
        print(f"{verdict:<8} {name}\n         {detail}")
    print(f"\nDecision rule (D20): {decision}.")

    # D21: the two controls the constructive framing still lacked after v6
    if sig is not None and not mixed.empty:
        top_arm = mixed.iloc[-1]["Arm"]
        more = []
        hit = sig[(sig["Student"] == "BERT-mini") & sig["Baseline"].str.startswith("Fine-tune only, matched to the")
                  & (sig["Candidate"] == top_arm)]
        if hit.empty:
            more.append(("5. the largest arm keeps >= +0.006 over fine-tuning at its own number of updates, interval excluding zero",
                         "not run", ""))
        else:
            d, (lo, hi) = float(hit.iloc[0]["Difference"]), interval(hit.iloc[0]["95 per cent interval"])
            more.append(("5. the largest arm keeps >= +0.006 over fine-tuning at its own number of updates, interval excluding zero",
                         "PASS" if d >= 0.006 and lo > 0 else "FAIL", f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"))
        others = sig[(sig["Student"] != "BERT-mini") & (sig["Baseline"] == "Fine-tune only") & sig["Candidate"].str.contains("transfer rows")]
        if others.empty:
            more.append(("6. on each other student the largest arm gains >= +0.005 over fine-tuning", "not run", ""))
        for _, r in others.iterrows():
            d, (lo, hi) = float(r["Difference"]), interval(r["95 per cent interval"])
            more.append((f"6. {r['Student']}: the largest arm gains >= +0.005 over fine-tuning", "PASS" if d >= 0.005 else "FAIL",
                         f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" (interval excludes zero)" if lo > 0 else "")))
        print()
        for name, verdict, detail in more:
            print(f"{verdict:<8} {name}\n         {detail}")
        ran = [v for _, v, _ in more if v != "not run"]
        if ran:
            print("\nD21: " + ("both controls hold; the constructive sentence may be stated for compact students in the plural"
                               if all(v == "PASS" for v in ran) else
                               "at least one control fails; state the result for the student and compute where it holds, and say where it does not"))


if __name__ == "__main__":
    main()
