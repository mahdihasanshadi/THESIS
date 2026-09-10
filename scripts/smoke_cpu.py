"""End-to-end smoke test on CPU with tiny models and a few hundred rows. Exercises every
stage (prepare, teacher, cache, student in all four modes, evaluate, bench, aggregate) in a
few minutes so that bugs are found here, not on Kaggle.

    python scripts/smoke_cpu.py --raw <any csv with tweet_text,cyberbullying_type>
"""
import argparse
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def sh(cmd):
    print("\n$", cmd, flush=True)
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run(cmd, shell=True, env=env)
    if r.returncode != 0:
        sys.exit(f"FAILED: {cmd}")


ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--root", default="smoke", help="where data/cache/runs go (use a drive with space, not OneDrive)")
ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--wiki_dir", default=None, help="prepared Wikipedia split dir; adds a disagreement-aware run")
a = ap.parse_args()

ROOT = a.root
data, cache, runs = f"{ROOT}/data", f"{ROOT}/cache", f"{ROOT}/runs"
sh(f"python -m dmthd.prepare_tweets --raw {a.raw} --out {data}")
sh(f"python -m dmthd.train_teacher --model_name google/bert_uncased_L-2_H-128_A-2 --data_dir {data} --out_dir {runs}/teachers/tiny-a --epochs 1 --batch 16 --limit {a.limit} --seed 1")
sh(f"python -m dmthd.train_teacher --model_name google/bert_uncased_L-4_H-256_A-4 --data_dir {data} --out_dir {runs}/teachers/mini-b --epochs 1 --batch 16 --limit {a.limit} --seed 2")
sh(f"python -m dmthd.cache_teachers --data_dir {data} --out {cache} --teachers {runs}/teachers/tiny-a {runs}/teachers/mini-b --aux_model cardiffnlp/twitter-roberta-base-irony --limit {a.limit} --batch 32")
S = "google/bert_uncased_L-2_H-128_A-2"
common = f"--student {S} --data_dir {data} --limit {a.limit} --epochs 1 --batch 16"
sh(f"python -m dmthd.train_student {common} --mode ft --out_dir {runs}/tiny/ft/seed1 --seed 1")
sh(f"python -m dmthd.train_student {common} --mode skd --cache {cache} --teachers tiny-a --out_dir {runs}/tiny/skd/seed1 --seed 1")
sh(f"python -m dmthd.train_student {common} --mode uniform --cache {cache} --teachers tiny-a mini-b --out_dir {runs}/tiny/uniform/seed1 --seed 1")
sh(f"python -m dmthd.train_student {common} --mode dmthd --cache {cache} --teachers tiny-a mini-b --aux --delta 0.3 --out_dir {runs}/tiny/dmthd/seed1 --seed 1")
sh(f"python -m dmthd.train_student {common} --mode dmthd --cache {cache} --teachers tiny-a mini-b --per_batch --no_hidden --out_dir {runs}/tiny/ablation/seed1 --seed 1 --tag per_batch_no_hidden")
sh(f"python -m dmthd.train_student --student bilstm --data_dir {data} --limit {a.limit} --epochs 1 --batch 16 --mode dmthd --cache {cache} --teachers tiny-a mini-b --aux --delta 0.3 --out_dir {runs}/bilstm/dmthd/seed1 --seed 1")
sh(f"python -m dmthd.evaluate --model_dir {runs}/tiny/dmthd/seed1 --csv {data}/test.csv --scheme six --probe_neg {data}/val.csv --probe_pos {data}/val.csv")
sh(f"python -m dmthd.quantize_eval --model_dir {runs}/tiny/dmthd/seed1 --csv {data}/test.csv --scheme six --n_time 32 --warmup 2 --repeats 2")
if a.wiki_dir:
    wd, wc, wr = a.wiki_dir, f"{ROOT}/cache_wiki", f"{ROOT}/runs_wiki"
    W = "--scheme binary --label_col label --max_len 64"
    sh(f"python -m dmthd.train_teacher --model_name google/bert_uncased_L-2_H-128_A-2 --data_dir {wd} --out_dir {wr}/teachers/tiny-w --epochs 1 --batch 16 --limit {a.limit} --seed 1 {W}")
    sh(f"python -m dmthd.cache_teachers --data_dir {wd} --out {wc} --teachers {wr}/teachers/tiny-w --aux_model cardiffnlp/twitter-roberta-base-irony --limit {a.limit} --batch 32 {W}")
    sh(f"python -m dmthd.train_student --student {S} --data_dir {wd} --limit {a.limit} --epochs 1 --batch 16 --mode dmthd --cache {wc} --teachers tiny-w --aux --delta 0.3 --disagreement --kappa 1.0 --reliability soft --out_dir {wr}/tiny/dmthd_dis/seed1 --seed 1 {W}")
sh(f"python -m dmthd.bench --model_dirs {runs}/tiny/dmthd/seed1 {runs}/teachers/tiny-a --csv {data}/test.csv --device cpu --n 64 --warmup 3 --repeats 2 --out {runs}/bench_cpu.csv")
sh(f"python -m dmthd.aggregate --runs {runs} --out {runs}/summary.csv")
sh(f"python -m dmthd.aggregate --compare {runs}/tiny/ft {runs}/tiny/dmthd")
print("\nSMOKE TEST PASSED")
