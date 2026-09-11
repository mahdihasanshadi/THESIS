"""Test fixture for the table generator: builds a synthetic runs/ tree containing every artefact the
real grid produces (teachers, four modes x three seeds x several students, heterogeneous committee,
ablations, sweeps, probe evaluations, obfuscation, transfer, quantisation, bench) and asserts that
`dmthd.tables` emits every table. The numbers are arbitrary fixtures, written to a temp directory
that is never part of the paper; the point is to exercise the formatting code paths.

    python scripts/smoke_tables.py --root E:/dmthd-work/tables_fixture
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
CLASSES = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]
rng = random.Random(0)


def w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def result(macro, params=11_300_000, seed=1, **extra):
    return {"seed": seed, "params": params, "train_time_s": 100.0,
            "test": {"macro_f1": macro, "accuracy": macro + 0.01, "ece": 0.03,
                     "per_class_f1": {c: macro for c in CLASSES}}, **extra}


def build(root):
    runs = os.path.join(root, "runs", "tweets")
    shutil.rmtree(root, ignore_errors=True)
    for tag, f1 in (("bert-large", 0.897), ("hatebert", 0.889), ("irony", 0.893), ("deberta-base", 0.885)):
        w(os.path.join(runs, "teachers", tag, "results.json"), result(f1, params=335_000_000))
        w(os.path.join(runs, "teachers", tag, "eval_test.json"),
          {"macro_f1": f1, "benign_sarcasm_fpr": 0.30, "ironic_abuse_recall": 0.66})
    w(os.path.join(runs, "tfidf_lr", "results.json"), result(0.8798, params=None))

    students = {"bert-mini": 11_300_000, "bert-small": 29_100_000, "distilbert": 66_000_000,
                "deberta-xsmall": 70_000_000, "bilstm": 10_400_000}
    modes = ["ft", "skd", "uniform", "dmthd", "dmthd_hetero"]
    for st, params in students.items():
        for mi, mode in enumerate(modes):
            for seed in (1, 2, 3):
                base = 0.877 + 0.004 * mi + rng.uniform(-0.001, 0.001)
                d = os.path.join(runs, st, mode, f"seed{seed}")
                w(os.path.join(d, "results.json"), result(round(base, 4), params=params, seed=seed))
                w(os.path.join(d, "eval_test.json"),
                  {"macro_f1": base, "benign_sarcasm_fpr": 0.30 - 0.02 * mi, "ironic_abuse_recall": 0.65 + 0.02 * mi})
                if mode in ("ft", "dmthd") and seed == 1:
                    for v, drop in (("leet", 0.10), ("swap", 0.12), ("space", 0.14), ("mixed", 0.15)):
                        w(os.path.join(d, f"eval_test_obf_{v}.json"), {"macro_f1": round(base - drop, 4)})
                    w(os.path.join(d, "transfer_wikipedia.json"), {"binary_macro_f1": 0.31, "roc_auc": 0.55})
                if mode == "dmthd" and seed == 1:
                    w(os.path.join(d, "quantize_eval.json"),
                      {"fp32": {"macro_f1": base, "size_MB": params / 2.5e5, "latency_ms_per_sample_b1": 4.0},
                       "int8": {"macro_f1": base - 0.002, "size_MB": params / 9e5, "latency_ms_per_sample_b1": 2.2},
                       "macro_f1_drop_int8": 0.002, "speedup_b1": 1.8})
    for tag in ("no_dynamic", "no_hidden", "no_aux", "per_batch", "from_scratch"):
        for seed in (1, 2, 3):
            w(os.path.join(runs, "bert-mini", f"ablation_{tag}", f"seed{seed}", "results.json"),
              result(round(0.889 - rng.uniform(0.002, 0.01), 4), seed=seed))
    for st in ("deberta-xsmall", "bilstm"):
        for seed in (1, 2, 3):
            w(os.path.join(runs, st, "ablation_no_hidden", f"seed{seed}", "results.json"),
              result(round(0.880 + rng.uniform(0, 0.003), 4), params=students[st], seed=seed))
    for param, values in (("tau", (0.5, 2.0, 5.0)), ("T", (1.0, 2.0, 8.0)), ("alpha", (0.2, 0.6)), ("delta", (0.1, 0.5))):
        for v in values:
            w(os.path.join(runs, "bert-mini", f"sweep_{param}_{v}", "seed1", "results.json"),
              result(round(0.885 + rng.uniform(-0.004, 0.004), 4), tag=f"sweep_{param}_{v}"))
    rows = []
    for st, params in students.items():
        rows.append({"model": f"{runs}/{st}/dmthd/seed1", "params_M": params / 1e6, "flops_G_per_seq": params / 1e9 * 2,
                     "latency_ms_per_sample_b1": 4.0, "latency_ms_per_sample_b32": 0.4, "device": "cuda"})
    pd.DataFrame(rows).to_csv(os.path.join(runs, "bench_cuda.csv"), index=False)
    data = os.path.join(root, "data", "tweets")
    w(os.path.join(data, "report.json"), {"rows_in": 47692, "rows_out": 43259, "rows_dropped_duplicates": 507,
                                          "split": {"train": 34607, "val": 4326, "test": 4326}, "seed": 42})
    return runs, data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tables_fixture")
    a = ap.parse_args()
    runs, data = build(a.root)
    out = os.path.join(a.root, "tables")
    r = subprocess.run([sys.executable, "-m", "dmthd.tables", "--runs", runs, "--data", data, "--out", out],
                       env={**os.environ, "PYTHONPATH": SRC})
    if r.returncode != 0:
        sys.exit("tables failed on the fixture")
    expected = ["teachers", "main", "ablations", "homogeneity", "committee", "efficiency", "robustness", "sweeps", "dataset"]
    missing = [n for n in expected if not os.path.exists(os.path.join(out, f"{n}.tex"))]
    if missing:
        sys.exit(f"TABLES SMOKE FAILED: missing {missing}")
    for n in expected:
        tex = open(os.path.join(out, f"{n}.tex"), encoding="utf-8").read()
        assert "\\toprule" in tex and "\\bottomrule" in tex and tex.count("\\\\") >= 2, f"{n}.tex looks malformed"
    print("\nTABLES SMOKE PASSED:", ", ".join(expected))
