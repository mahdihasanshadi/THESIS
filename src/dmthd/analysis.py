"""Analyses the paper needs beyond the main table. Each sub-command reads finished run
directories and writes CSV/JSON next to them; figures are drawn later from these files.

  complementarity  pairwise teacher error overlap, Cohen's kappa between teachers, oracle-ensemble
                   upper bound (a test item is "solved by the oracle" if any teacher is right)
      python -m dmthd.analysis complementarity --dirs runs/tweets/teachers/bert-large runs/tweets/teachers/hatebert ... --out runs/tweets/complementarity.json

  agreement_bands  macro-F1 by annotator-agreement band on Wikipedia for one or more run dirs
      python -m dmthd.analysis agreement_bands --dirs runs/wikipedia/bert-mini/ft/seed1 runs/wikipedia/bert-mini/dmthd/seed1 --test data/wikipedia/test.csv --out runs/wikipedia/agreement_bands.csv

  weights          mean dynamic teacher weight per epoch from history.csv of D-MTHD runs
      python -m dmthd.analysis weights --dirs runs/tweets/bert-mini/dmthd/seed1 ... --out runs/tweets/weights.csv

  pareto           join summary.csv with bench_*.csv into one table of macro-F1 vs latency vs params
      python -m dmthd.analysis pareto --summary runs/tweets/summary.csv --bench runs/tweets/bench_cuda.csv --out runs/tweets/pareto.csv
"""
import argparse
import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from .utils import load_json, save_json


def _load_preds(d):
    probs = np.load(os.path.join(d, "test_probs.npy"))
    labels = np.load(os.path.join(d, "test_labels.npy"))
    return probs.argmax(1), labels, probs


def complementarity(dirs, out):
    preds, labels = {}, None
    for d in dirs:
        p, y, _ = _load_preds(d)
        preds[os.path.basename(os.path.normpath(d))] = p
        labels = y if labels is None else labels
    names = list(preds)
    correct = {n: preds[n] == labels for n in names}
    res = {"teachers": names, "n_test": int(len(labels)),
           "accuracy": {n: float(correct[n].mean()) for n in names},
           "macro_f1": {n: float(f1_score(labels, preds[n], average="macro")) for n in names},
           "pairwise": {}, "oracle_accuracy": float(np.any(np.stack([correct[n] for n in names]), 0).mean()),
           "all_wrong_share": float((~np.any(np.stack([correct[n] for n in names]), 0)).mean()),
           "all_right_share": float(np.all(np.stack([correct[n] for n in names]), 0).mean())}
    for a, b in combinations(names, 2):
        both_wrong = (~correct[a] & ~correct[b]).mean()
        res["pairwise"][f"{a}|{b}"] = {"kappa_predictions": float(cohen_kappa_score(preds[a], preds[b])),
                                      "error_overlap": float(both_wrong / max((~correct[a]).mean(), 1e-9)),
                                      "both_wrong_share": float(both_wrong),
                                      "disagreement_rate": float((preds[a] != preds[b]).mean())}
    # oracle macro-F1: pick, per item, a teacher that is right if any is (upper bound on selection)
    oracle = preds[names[0]].copy()
    for n in names[1:]:
        oracle = np.where(correct[names[0]], oracle, np.where(correct[n], preds[n], oracle))
    res["oracle_macro_f1_upper_bound"] = float(f1_score(labels, oracle, average="macro"))
    save_json(res, out)
    print({k: v for k, v in res.items() if k not in ("pairwise",)})
    print(pd.DataFrame(res["pairwise"]).T.round(3).to_string())


def agreement_bands(dirs, test_csv, out, soft_col="soft_label"):
    df = pd.read_csv(test_csv)
    s = df[soft_col].values
    bands = [("unanimous_0.0-0.1", s <= 0.1), ("clear_0.1-0.2", (s > 0.1) & (s < 0.2)),
             ("ambiguous_0.2-0.8", (s >= 0.2) & (s <= 0.8)), ("clear_0.8-0.9", (s > 0.8) & (s < 0.9)),
             ("unanimous_0.9-1.0", s >= 0.9)]
    rows = []
    for d in dirs:
        p, y, _ = _load_preds(d)
        assert len(p) == len(df), f"{d}: predictions ({len(p)}) do not match test rows ({len(df)})"
        for name, m in bands:
            if m.sum() == 0:
                continue
            rows.append({"run": d, "band": name, "n": int(m.sum()), "macro_f1": float(f1_score(y[m], p[m], average="macro")),
                         "accuracy": float((p[m] == y[m]).mean())})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).pivot(index="band", columns="run", values="macro_f1").round(4).to_string())


def weights(dirs, out):
    rows = []
    for d in dirs:
        h = pd.read_csv(os.path.join(d, "history.csv"))
        wcols = [c for c in h.columns if c.startswith("w_")]
        for _, r in h.iterrows():
            row = {"run": d, "epoch": int(r["epoch"])}
            row.update({c: float(r[c]) for c in wcols})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).groupby("epoch").mean(numeric_only=True).round(3).to_string())


def _dir_tag(path):
    parts = [p for p in path.replace("\\", "/").rstrip("/").split("/") if p]
    return parts[-3] if parts and parts[-1].startswith("seed") else parts[-1]


def pareto(summary_csv, bench_csv, out):
    """Macro-F1 (mean over seeds) against parameters and batch-1 latency, keyed by the run
    directory tag (bert-mini, distilbert, teachers/<tag>) so the summary and the benchmark join."""
    per_seed = pd.read_csv(summary_csv.replace(".csv", "_per_seed.csv"))
    per_seed["dir_tag"] = per_seed["run"].apply(lambda r: r.replace("\\", "/").split("/")[1] if r.replace("\\", "/").startswith("teachers/") else r.replace("\\", "/").split("/")[0])
    s = per_seed.groupby(["dir_tag", "mode", "tag"], dropna=False).agg(macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                                                                         n_seeds=("seed", "count"), params=("params", "first")).reset_index()
    b = pd.read_csv(bench_csv)
    b["dir_tag"] = b["model"].apply(_dir_tag)
    lat = [c for c in b.columns if c.startswith("latency_ms_per_sample_b1")]
    b = b[["dir_tag", "params_M", "flops_G_per_seq"] + lat].drop_duplicates("dir_tag")
    m = s.merge(b, on="dir_tag", how="left")
    m.to_csv(out, index=False)
    print(m[["dir_tag", "mode", "tag", "n_seeds", "macro_f1", "params_M"] + lat].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("complementarity"); c.add_argument("--dirs", nargs="+", required=True); c.add_argument("--out", required=True)
    a = sub.add_parser("agreement_bands"); a.add_argument("--dirs", nargs="+", required=True); a.add_argument("--test", required=True); a.add_argument("--out", required=True)
    w = sub.add_parser("weights"); w.add_argument("--dirs", nargs="+", required=True); w.add_argument("--out", required=True)
    p = sub.add_parser("pareto"); p.add_argument("--summary", required=True); p.add_argument("--bench", required=True); p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "complementarity":
        complementarity(args.dirs, args.out)
    elif args.cmd == "agreement_bands":
        agreement_bands(args.dirs, args.test, args.out)
    elif args.cmd == "weights":
        weights(args.dirs, args.out)
    else:
        pareto(args.summary, args.bench, args.out)


if __name__ == "__main__":
    main()
