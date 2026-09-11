"""End-to-end smoke test of the Kaggle driver itself, on CPU with tiny models and a few hundred
rows. Catches driver bugs (stage wiring, attribute clashes, committee logic) before they reach a GPU
session, which is how the `self.students` clash and the redundant `skd_hetero` grid were found.
Four passes:

  1. full grid on tweets: two tiny homogeneous teachers + DeBERTa-v3-xsmall as the heterogeneous
     teacher, BERT-tiny + BiLSTM students, modes ft and dmthd, every later stage (probes, sweep,
     robustness, quant, bench, aggregate). MIN_TEACHER_F1=0 so nothing is dropped, SPECIALIST=0 so
     the pass stays independent of the implicit corpus.
  2. collapse handling: same teachers with MIN_TEACHER_F1=0.99, teachers stage only, so every
     teacher is "collapsed", retrained once and dropped; the students stage must then run the
     control and skip the committee modes without crashing.
  3. the implicit benchmark end to end, which exercises the implicit3 label scheme through every
     stage that has a hard-coded class name.
  4. the implicit specialist: trained on the implicit benchmark, then task-adapted into the tweets
     committee. This is the path where a 3-class checkpoint is re-headed to 6 classes, so it is the
     one most likely to break quietly.

    python scripts/smoke_driver.py --raw E:/dmthd-work/data/raw/cyberbullying_tweets.csv \
        --implicit_raw E:/dmthd-work/data/raw --root E:/dmthd-work/smoke_driver
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle", "run_benchmark.py")
TINY = "google/bert_uncased_L-2_H-128_A-2"


def run(env, stage, root, dataset="tweets", raw=None):
    e = {**os.environ, "PYTHONPATH": SRC, "ROOT": root, "GPU": "0", "LIMIT": "200",
         "TEACHER_EPOCHS": "1", "STUDENT_EPOCHS": "1", "SEEDS": "1", "MODES": "ft,dmthd",
         "COMMITTEES": "homo,hetero", "SPECIALIST": "0",
         "TEACHERS": f"{TINY}:tiny-a,google/bert_uncased_L-4_H-256_A-4:mini-b",
         "HETERO_TEACHER": "microsoft/deberta-v3-xsmall:deberta-xsmall",
         "STUDENTS": f"{TINY}:tiny", "HETERO_STUDENTS": "bilstm:bilstm", **env}
    cmd = [sys.executable, DRIVER, "--dataset", dataset, "--stage", stage] + (["--raw", raw] if raw else [])
    print("\n>>>", " ".join(cmd), {k: v for k, v in env.items()}, flush=True)
    if subprocess.run(cmd, env=e).returncode != 0:
        sys.exit(f"DRIVER SMOKE FAILED at {dataset}/{stage} with env {env}")


ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="the tweet corpus CSV")
ap.add_argument("--implicit_raw", default=None, help="directory holding ishate/ and implicit_hate_stg1/")
ap.add_argument("--root", default="smoke_driver")
a = ap.parse_args()
shutil.rmtree(a.root, ignore_errors=True)

# 1. the full grid
run({"MIN_TEACHER_F1": "0"}, "all", a.root, raw=a.raw)

# 2. every teacher collapses
for d in ("runs/tweets/teachers", "cache/tweets"):
    shutil.rmtree(os.path.join(a.root, d), ignore_errors=True)
for d in os.listdir(os.path.join(a.root, "runs", "tweets")):
    p = os.path.join(a.root, "runs", "tweets", d)
    if os.path.isdir(p) and d != "teachers":
        for sub in os.listdir(p):
            if sub != "ft":
                shutil.rmtree(os.path.join(p, sub), ignore_errors=True)
run({"MIN_TEACHER_F1": "0.99"}, "teachers", a.root, raw=a.raw)
run({"MIN_TEACHER_F1": "0.99"}, "students", a.root, raw=a.raw)
dropped = os.path.join(a.root, "runs", "tweets", "dropped_teachers.json")
assert os.path.exists(dropped), "dropped_teachers.json missing"

if a.implicit_raw:
    # 3. the implicit benchmark, which has its own label scheme
    run({"MIN_TEACHER_F1": "0", "IMPLICIT_RAW": a.implicit_raw}, "all", a.root, dataset="implicit")
    res = os.path.join(a.root, "runs", "implicit", "tiny", "dmthd", "seed1", "results.json")
    assert os.path.exists(res), f"implicit student produced no results: {res}"
    names = list(json.load(open(res))["test"]["per_class_f1"])
    assert names == ["not_hate", "explicit_hate", "implicit_hate"], f"wrong class order: {names}"

    # 4. the specialist: trained on implicit, re-headed for tweets
    shutil.rmtree(os.path.join(a.root, "runs", "tweets"), ignore_errors=True)
    shutil.rmtree(os.path.join(a.root, "cache", "tweets"), ignore_errors=True)
    run({"MIN_TEACHER_F1": "0", "IMPLICIT_RAW": a.implicit_raw, "SPECIALIST": "1",
         "SPECIALIST_BASE": TINY, "COMMITTEES": "homo"}, "specialist", a.root, raw=a.raw)
    run({"MIN_TEACHER_F1": "0", "IMPLICIT_RAW": a.implicit_raw, "SPECIALIST": "1",
         "SPECIALIST_BASE": TINY, "COMMITTEES": "homo"}, "teachers", a.root, raw=a.raw)
    run({"MIN_TEACHER_F1": "0", "IMPLICIT_RAW": a.implicit_raw, "SPECIALIST": "1",
         "SPECIALIST_BASE": TINY, "COMMITTEES": "homo"}, "cache", a.root, raw=a.raw)
    adapted = os.path.join(a.root, "runs", "tweets", "teachers", "implicit-spec", "results.json")
    assert os.path.exists(adapted), "specialist never reached the tweets committee"
    n = len(json.load(open(adapted))["test"]["per_class_f1"])
    assert n == 6, f"specialist was not re-headed for the tweet task: {n} classes"
    meta = json.load(open(os.path.join(a.root, "cache", "tweets", "meta.json")))
    tags = [t["tag"] for t in meta["teachers"]]
    assert "implicit-spec" in tags, f"specialist missing from the cache: {tags}"
    print(f"\nspecialist in committee, cache tags {tags}")

print("\nDRIVER SMOKE PASSED:", open(dropped).read())
