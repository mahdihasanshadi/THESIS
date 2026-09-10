"""One-command driver for the tweet benchmark. Paste into a Kaggle or Colab cell:

    !git clone <your repo> dmthd-p3 && cd dmthd-p3 && pip install -q -r requirements.txt
    !cd dmthd-p3 && PYTHONPATH=src python kaggle/run_tweets.py --stage all --raw /kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv

Stages: prepare | teachers | cache | students | bench | aggregate | all. Every stage skips work
whose results.json already exists, so a killed session resumes where it stopped.
Environment overrides: TEACHERS, STUDENTS, SEEDS, MODES, ROOT (see defaults below).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.environ.get("ROOT", ".")
DATA = f"{ROOT}/data/tweets"
CACHE = f"{ROOT}/cache/tweets"
RUNS = f"{ROOT}/runs/tweets"
TEACHERS = os.environ.get("TEACHERS", "bert-large-uncased:bert-large,GroNLP/hateBERT:hatebert,cardiffnlp/twitter-roberta-base-irony:irony")
STUDENTS = os.environ.get("STUDENTS", "google/bert_uncased_L-4_H-256_A-4:bert-mini,google/bert_uncased_L-4_H-512_A-8:bert-small,distilbert-base-uncased:distilbert")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MODES = os.environ.get("MODES", "ft,skd,uniform,dmthd").split(",")
AUX_MODEL = "cardiffnlp/twitter-roberta-base-irony"
GPU = os.environ.get("GPU", "1") == "1"


def sh(cmd):
    print("\n$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {cmd}")


def pairs(spec):
    return [tuple(x.split(":")) for x in spec.split(",")]


def done(d):
    return os.path.exists(os.path.join(d, "results.json"))


def stage_prepare(raw):
    if os.path.exists(f"{DATA}/test.csv"):
        return
    sh(f"python -m dmthd.prepare_tweets --raw {raw} --out {DATA}")


def stage_teachers():
    fp = "--fp16" if GPU else ""
    for name, tag in pairs(TEACHERS):
        out = f"{RUNS}/teachers/{tag}"
        if done(out):
            continue
        ck = "--grad_ckpt" if "large" in name else ""
        sh(f"python -m dmthd.train_teacher --model_name {name} --data_dir {DATA} --out_dir {out} --epochs 5 --lr 2e-5 --batch 32 {fp} {ck}")


def stage_cache():
    if os.path.exists(f"{CACHE}/meta.json"):
        return
    dirs = " ".join(f"{RUNS}/teachers/{tag}" for _, tag in pairs(TEACHERS))
    sh(f"python -m dmthd.cache_teachers --data_dir {DATA} --out {CACHE} --teachers {dirs} --aux_model {AUX_MODEL}")


def stage_students():
    fp = "--fp16" if GPU else ""
    tags = [tag for _, tag in pairs(TEACHERS)]
    best_single = tags[0]
    for name, stag in pairs(STUDENTS):
        for mode in MODES:
            for seed in SEEDS:
                out = f"{RUNS}/{stag}/{mode}/seed{seed}"
                if done(out):
                    continue
                common = f"--student {name} --data_dir {DATA} --out_dir {out} --seed {seed} {fp}"
                if mode == "ft":
                    sh(f"python -m dmthd.train_student {common} --mode ft")
                elif mode == "skd":
                    sh(f"python -m dmthd.train_student {common} --mode skd --cache {CACHE} --teachers {best_single}")
                elif mode == "uniform":
                    sh(f"python -m dmthd.train_student {common} --mode uniform --cache {CACHE} --teachers {' '.join(tags)}")
                else:
                    sh(f"python -m dmthd.train_student {common} --mode dmthd --cache {CACHE} --teachers {' '.join(tags)} --aux --delta 0.3")
    # ablations on the first (headline) student, dmthd mode
    name, stag = pairs(STUDENTS)[0]
    abl = {"no_dynamic": "--mode uniform --aux --delta 0.3", "no_hidden": "--mode dmthd --no_hidden --aux --delta 0.3",
           "no_aux": "--mode dmthd", "per_batch": "--mode dmthd --per_batch --aux --delta 0.3"}
    for tag, flags in abl.items():
        for seed in SEEDS:
            out = f"{RUNS}/{stag}/ablation_{tag}/seed{seed}"
            if done(out):
                continue
            sh(f"python -m dmthd.train_student --student {name} --data_dir {DATA} --out_dir {out} --seed {seed} {fp} "
               f"--cache {CACHE} --teachers {' '.join(tags)} {flags} --tag {tag}")


def stage_bench():
    dirs = [f"{RUNS}/teachers/{tag}" for _, tag in pairs(TEACHERS)] + [f"{RUNS}/{stag}/dmthd/seed{SEEDS[0]}" for _, stag in pairs(STUDENTS)]
    dev = "cuda" if GPU else "cpu"
    sh(f"python -m dmthd.bench --model_dirs {' '.join(dirs)} --csv {DATA}/test.csv --device {dev} --out {RUNS}/bench_{dev}.csv")


def stage_aggregate():
    sh(f"python -m dmthd.aggregate --runs {RUNS} --out {RUNS}/summary.csv")
    _, stag = pairs(STUDENTS)[0]
    for cand in ("dmthd", "uniform", "skd"):
        sh(f"python -m dmthd.aggregate --compare {RUNS}/{stag}/ft {RUNS}/{stag}/{cand}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--raw", default="/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv")
    a = ap.parse_args()
    os.environ.setdefault("PYTHONPATH", "src")
    order = ["prepare", "teachers", "cache", "students", "bench", "aggregate"]
    todo = order if a.stage == "all" else [a.stage]
    for st in todo:
        {"prepare": lambda: stage_prepare(a.raw), "teachers": stage_teachers, "cache": stage_cache,
         "students": stage_students, "bench": stage_bench, "aggregate": stage_aggregate}[st]()
