"""Every paired comparison the paper states, in one table, with its bootstrap interval.

    python -m dmthd.significance --runs runs/tweets --out paper/tables

The results draft used to quote intervals copied out of the Kaggle console, where the log capture
prints each comparison's output under the next command's line, so a copied number can land on the
wrong row. This computes them again from the saved test predictions with the same test as
`dmthd.aggregate --compare` (seeds paired by directory name, 1,000 test-set resamples each, the
pooled 2.5 and 97.5 percentiles), which is deterministic, so the table is the same wherever it runs.
A comparison whose runs are missing is skipped, and the seed count is shown because the ablations,
controls and sweeps have one seed only.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .aggregate import paired_bootstrap
from .tables import ABLATION_LABEL, MODE_LABEL, STUDENT_LABEL, write

# baseline, candidate, and the question the row answers
EVERY_STUDENT = [("ft", "skd", "does one teacher help"),
                 ("ft", "uniform", "does a committee help"),
                 ("ft", "dmthd", "does the method help"),
                 ("ft", "uniform_hetero", "does a committee with DeBERTa help"),
                 ("ft", "dmthd_hetero", "does the method with DeBERTa help"),
                 ("skd", "uniform", "does a committee beat one teacher"),
                 ("skd", "dmthd", "does the weighted committee beat one teacher"),
                 ("uniform", "uniform_hetero", "does a DeBERTa teacher change averaging"),
                 ("uniform", "dmthd", "does per-instance weighting beat averaging"),
                 ("dmthd", "dmthd_hetero", "does a DeBERTa teacher change the method"),
                 ("uniform_hetero", "dmthd_hetero", "weighting against averaging, with DeBERTa")]
HEADLINE_ONLY = [("ft", "uniform_spec", "does a committee with the specialist help"),
                 ("ft", "dmthd_spec", "does the method with the specialist help"),
                 ("uniform", "uniform_spec", "does adding the specialist change averaging"),
                 ("dmthd", "dmthd_spec", "does adding the specialist change the method"),
                 ("uniform_spec", "dmthd_spec", "weighting against averaging, with the specialist"),
                 ("dmthd_spec", "ablation_spec_only", "the specialist alone against the full committee"),
                 ("ft", "ablation_implicit_pretrain", "pre-training on the implicit corpus instead"),
                 ("dmthd_spec", "ablation_implicit_pretrain", "the same knowledge by pre-training, not distillation"),
                 ("uniform", "sweep_tau_0.05", "the sharpest weighting against averaging"),
                 ("dmthd", "sweep_tau_0.05", "the sharpest weighting against the default"),
                 ("ft", "ablation_from_scratch", "what pre-training is worth")]


def label(mode):
    return MODE_LABEL.get(mode, ABLATION_LABEL.get(mode, mode.replace("sweep_tau_", "D-MTHD, tau = ")))


def one(job):
    student, base, cand, question, runs = job
    a, b = os.path.join(runs, student, base), os.path.join(runs, student, cand)
    try:
        per_seed, (delta, lo, hi) = paired_bootstrap(a, b)
    except SystemExit:
        return None
    return {"Student": STUDENT_LABEL.get(student, student), "Question": question, "Baseline": label(base),
            "Candidate": label(cand), "Seeds": len(per_seed), "Difference": f"{delta:+.4f}",
            "95 per cent interval": f"[{lo:+.4f}, {hi:+.4f}]",
            "Interval excludes zero": "yes" if lo > 0 or hi < 0 else "no"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline", default="bert-mini")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    students = sorted(d for d in os.listdir(args.runs)
                      if os.path.isdir(os.path.join(args.runs, d)) and d not in ("teachers", "weight_routing"))
    students.sort(key=lambda s: s != args.headline)
    jobs = [(s, b, c, q, args.runs) for s in students for b, c, q in EVERY_STUDENT]
    jobs += [(args.headline, b, c, q, args.runs) for b, c, q in HEADLINE_ONLY]
    jobs = [j for j in jobs if os.path.isdir(os.path.join(args.runs, j[0], j[1])) and os.path.isdir(os.path.join(args.runs, j[0], j[2]))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for r in pool.map(one, jobs) if r]
    write(pd.DataFrame(rows), args.out, "significance",
          "Paired bootstrap on the test set: candidate minus baseline macro-F1, seeds paired by index, "
          "1,000 resamples per seed, 95 per cent interval of the pooled differences. Rows with one seed "
          "are indicative only.", "significance")
    for r in rows:
        print(f"  {r['Student']:<18} {r['Candidate']:<48} - {r['Baseline']:<44} {r['Difference']} {r['95 per cent interval']}"
              f"  n={r['Seeds']}  {r['Interval excludes zero']}")


if __name__ == "__main__":
    main()
