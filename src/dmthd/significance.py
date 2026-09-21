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
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .aggregate import paired_bootstrap
from .tables import ABLATION_LABEL, MATCH_RE, SCALE_RE, STUDENT_LABEL, mode_label, write

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
                 ("ft", "ablation_from_scratch", "what pre-training is worth"),
                 # out-of-sample distillation (D19): the transfer set, the committee as labeller, and the corrected weights
                 ("ft", "skd_transfer", "one teacher with a transfer set against fine-tuning"),
                 ("ft", "uniform_transfer", "the committee with a transfer set against fine-tuning"),
                 ("ft", "uniform_hetero_transfer", "the DeBERTa committee with a transfer set against fine-tuning"),
                 ("ft", "dmthd_knn_transfer", "corrected weighting with a transfer set against fine-tuning"),
                 ("ft", "pseudo_transfer", "hard pseudo-labels with a transfer set against fine-tuning"),
                 ("skd", "skd_transfer", "what the transfer set adds to one teacher"),
                 ("uniform", "uniform_transfer", "what the transfer set adds to the committee"),
                 ("skd_transfer", "uniform_transfer", "is the committee the better labeller"),
                 ("uniform_transfer", "uniform_hetero_transfer", "does the DeBERTa teacher help as a labeller"),
                 ("uniform_transfer", "dmthd_knn_transfer", "does out-of-sample reliability beat averaging"),
                 ("uniform_transfer", "pseudo_transfer", "soft labels against hard pseudo-labels")]


def label(mode):
    return mode_label(mode, ABLATION_LABEL.get(mode, mode.replace("sweep_tau_", "D-MTHD, tau = ")))


def _rows_of(home, arm):
    """The transfer rows an arm trained on, from any of its seeds."""
    for seed in sorted(os.listdir(os.path.join(home, arm))):
        try:
            with open(os.path.join(home, arm, seed, "results.json"), encoding="utf-8") as f:
                return int(json.load(f).get("transfer_rows") or 0)
        except (OSError, ValueError):
            continue
    return 0


def scaling_jobs(runs, headline):
    """The size-curve comparisons (D20, D21), built from whatever arms exist. On the headline student:
    each arm against fine-tuning, each size against the next smaller one, the composition control
    against the abuse-domain arm of the same size, the base-size arm against fine-tuning at matched
    steps, and the largest arm against fine-tuning at its own number of steps. On any other student
    that has a curve arm: that arm against fine-tuning and against the same teacher in sample.
    Returns (student, baseline, candidate, question) tuples."""
    jobs = []
    students = sorted(d for d in os.listdir(runs)
                      if os.path.isdir(os.path.join(runs, d)) and d not in ("teachers", "weight_routing"))
    for student in students:
        home = os.path.join(runs, student)
        arms = sorted(d for d in os.listdir(home) if SCALE_RE.match(d))
        if not arms:
            continue
        if student != headline:
            for a in arms:
                jobs.append((student, "ft", a, f"one teacher with {a.split('_')[-1]} transfer rows against fine-tuning"))
                jobs.append((student, "skd", a, "the transfer set against the same teacher in sample"))
            continue
        sized = sorted((_rows_of(home, a), a) for a in arms if not SCALE_RE.match(a).group(2))
        jobs += [(student, "ft", a, f"one teacher with {a.split('_')[-1]} transfer rows against fine-tuning") for _, a in sized]
        jobs += [(student, sized[i - 1][1], sized[i][1], "does the gain grow from the smaller set") for i in range(1, len(sized))]
        for a in arms:
            m = SCALE_RE.match(a)
            if m.group(2):
                twin = f"{m.group(1)}_transfer_{m.group(3)}"
                jobs.append((student, "ft", a, "generic tweets of the same size against fine-tuning"))
                if twin in arms:
                    jobs.append((student, a, twin, "abuse-domain text against generic text of the same size"))
        if sized:
            jobs.append((student, "ft", "ft_matched", "more fine-tuning steps alone"))
            # the matched-steps control follows the base-size arm, which is the composition control's twin
            generic = [_rows_of(home, a) for a in arms if SCALE_RE.match(a).group(2)]
            base = next((a for n, a in sized if generic and n == generic[0]), sized[-1][1])
            largest = sized[-1][1]
            jobs.append((student, "ft_matched", base, "the transfer set against fine-tuning at matched steps"))
            if largest != base:
                jobs.append((student, "ft_matched", largest, "the largest set against the matched-steps control"))
            matched_l = next((d for d in sorted(os.listdir(home)) if MATCH_RE.match(d) and MATCH_RE.match(d).group(1)), None)
            if matched_l:
                jobs.append((student, "ft", matched_l, "fine-tuning for the largest arm's number of updates"))
                jobs.append((student, matched_l, largest, "the largest set against fine-tuning at its own number of updates"))
    return jobs


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
    jobs += [(s, b, c, q, args.runs) for s, b, c, q in scaling_jobs(args.runs, args.headline)]
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
