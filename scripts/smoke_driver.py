"""End-to-end smoke test of the Kaggle driver itself, on CPU with tiny models and 200 rows.
Catches driver bugs (stage wiring, attribute clashes, committee logic) before they reach a GPU
session. Two passes:

  1. full grid: two tiny homogeneous teachers + DeBERTa-v3-xsmall as the heterogeneous teacher,
     BERT-tiny + BiLSTM students, modes ft and dmthd, every later stage (probes, sweep, robustness,
     quant, bench, aggregate). MIN_TEACHER_F1=0 so nothing is dropped.
  2. collapse handling: same teachers with MIN_TEACHER_F1=0.99, teachers stage only, so every
     teacher is "collapsed", retrained once and dropped; the students stage must then run the
     control and skip the committee modes without crashing.

    python scripts/smoke_driver.py --raw E:/dmthd-work/data/raw/cyberbullying_tweets.csv --root E:/dmthd-work/smoke_driver
"""
import argparse
import os
import shutil
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle", "run_benchmark.py")


def run(env, stage, raw, root):
    e = {**os.environ, "PYTHONPATH": SRC, "ROOT": root, "GPU": "0", "LIMIT": "200", "TEACHER_EPOCHS": "1", "STUDENT_EPOCHS": "1",
         "SEEDS": "1", "MODES": "ft,dmthd", "COMMITTEES": "homo,hetero",
         "TEACHERS": "google/bert_uncased_L-2_H-128_A-2:tiny-a,google/bert_uncased_L-4_H-256_A-4:mini-b",
         "HETERO_TEACHER": "microsoft/deberta-v3-xsmall:deberta-xsmall",
         "STUDENTS": "google/bert_uncased_L-2_H-128_A-2:tiny", "HETERO_STUDENTS": "bilstm:bilstm", **env}
    cmd = [sys.executable, DRIVER, "--dataset", "tweets", "--stage", stage, "--raw", raw]
    print("\n>>>", " ".join(cmd), {k: v for k, v in env.items()}, flush=True)
    r = subprocess.run(cmd, env=e)
    if r.returncode != 0:
        sys.exit(f"DRIVER SMOKE FAILED at stage {stage} with env {env}")


ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--root", default="smoke_driver")
a = ap.parse_args()
shutil.rmtree(a.root, ignore_errors=True)
run({"MIN_TEACHER_F1": "0"}, "all", a.raw, a.root)
for d in ("runs/tweets/teachers", "cache/tweets"):
    shutil.rmtree(os.path.join(a.root, d), ignore_errors=True)
for d in os.listdir(os.path.join(a.root, "runs", "tweets")):
    p = os.path.join(a.root, "runs", "tweets", d)
    if os.path.isdir(p) and d != "teachers":
        for sub in os.listdir(p):
            if sub != "ft":
                shutil.rmtree(os.path.join(p, sub), ignore_errors=True)
run({"MIN_TEACHER_F1": "0.99"}, "teachers", a.raw, a.root)
run({"MIN_TEACHER_F1": "0.99"}, "students", a.raw, a.root)
dropped = os.path.join(a.root, "runs", "tweets", "dropped_teachers.json")
assert os.path.exists(dropped), "dropped_teachers.json missing"
print("\nDRIVER SMOKE PASSED:", open(dropped).read())
