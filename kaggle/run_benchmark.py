"""One-command driver for a whole benchmark on Kaggle or Colab.

    PYTHONPATH=src python kaggle/run_benchmark.py --dataset tweets    --raw /kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv
    PYTHONPATH=src python kaggle/run_benchmark.py --dataset wikipedia --raw /kaggle/working/raw_wikipedia   (downloads from Figshare)

Stages: prepare | teachers | cache | students | probes | bench | aggregate | all. Every stage skips work
whose results.json already exists, so a killed session resumes where it stopped.

Resuming across Kaggle sessions: attach the previous notebook's output as an input and set
RESUME_FROM=/kaggle/input/<that-output>; its runs/ and cache/ trees are copied in before anything runs.

Environment overrides: ROOT, TEACHERS, STUDENTS, SEEDS, MODES, GPU, RESUME_FROM, TEACHER_EPOCHS.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

DATASETS = {
    "tweets": {"scheme": "six", "label_col": "label_name", "max_len": 128, "num_labels": 6,
               "teacher_epochs": 5, "teacher_batch": 32, "student_batch": 32},
    "wikipedia": {"scheme": "binary", "label_col": "label", "max_len": 256, "num_labels": 2,
                  "teacher_epochs": 3, "teacher_batch": 16, "student_batch": 32},
}
ROOT = os.environ.get("ROOT", ".")
TEACHERS = os.environ.get("TEACHERS", "bert-large-uncased:bert-large,GroNLP/hateBERT:hatebert,cardiffnlp/twitter-roberta-base-irony:irony")
STUDENTS = os.environ.get("STUDENTS", "google/bert_uncased_L-4_H-256_A-4:bert-mini,google/bert_uncased_L-4_H-512_A-8:bert-small,distilbert-base-uncased:distilbert")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MODES = os.environ.get("MODES", "ft,skd,uniform,dmthd").split(",")
AUX_MODEL = "cardiffnlp/twitter-roberta-base-irony"
GPU = os.environ.get("GPU", "1") == "1"
RESUME_FROM = os.environ.get("RESUME_FROM", "")
PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes")


def sh(cmd):
    print("\n$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {cmd}")


def pairs(spec):
    return [tuple(x.split(":")) for x in spec.split(",")]


def done(d):
    return os.path.exists(os.path.join(d, "results.json"))


class Bench:
    def __init__(self, name, raw):
        self.name, self.raw, self.cfg = name, raw, DATASETS[name]
        self.data = f"{ROOT}/data/{name}"
        self.cache = f"{ROOT}/cache/{name}"
        self.runs = f"{ROOT}/runs/{name}"
        self.common = f"--scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']}"
        self.fp = "--fp16" if GPU else ""
        self.tags = [tag for _, tag in pairs(TEACHERS)]

    def resume(self):
        if not RESUME_FROM:
            return
        for sub in ("runs", "cache"):
            for src in glob.glob(os.path.join(RESUME_FROM, "**", sub, self.name), recursive=True):
                dst = f"{ROOT}/{sub}/{self.name}"
                print(f"resuming: copying {src} -> {dst}", flush=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)

    def prepare(self):
        if os.path.exists(f"{self.data}/test.csv"):
            return
        if self.name == "tweets":
            sh(f"python -m dmthd.prepare_tweets --raw {self.raw} --out {self.data}")
        else:
            sh(f"python -m dmthd.prepare_wikipedia --raw {self.raw} --out {self.data} --download")

    def teachers(self):
        ep = os.environ.get("TEACHER_EPOCHS", self.cfg["teacher_epochs"])
        for name, tag in pairs(TEACHERS):
            out = f"{self.runs}/teachers/{tag}"
            if done(out):
                continue
            ck = "--grad_ckpt" if "large" in name else ""
            sh(f"python -m dmthd.train_teacher --model_name {name} --data_dir {self.data} --out_dir {out} "
               f"--epochs {ep} --lr 2e-5 --batch {self.cfg['teacher_batch']} {self.fp} {ck} {self.common}")

    def cache_teachers(self):
        if os.path.exists(f"{self.cache}/meta.json"):
            return
        dirs = " ".join(f"{self.runs}/teachers/{tag}" for tag in self.tags)
        sh(f"python -m dmthd.cache_teachers --data_dir {self.data} --out {self.cache} --teachers {dirs} "
           f"--aux_model {AUX_MODEL} {self.common}")

    def students(self):
        best_single = self.tags[0]
        bs = self.cfg["student_batch"]
        for name, stag in pairs(STUDENTS):
            for mode in MODES:
                for seed in SEEDS:
                    out = f"{self.runs}/{stag}/{mode}/seed{seed}"
                    if done(out):
                        continue
                    base = f"--student {name} --data_dir {self.data} --out_dir {out} --seed {seed} --batch {bs} {self.fp} {self.common}"
                    if mode == "ft":
                        sh(f"python -m dmthd.train_student {base} --mode ft")
                    elif mode == "skd":
                        sh(f"python -m dmthd.train_student {base} --mode skd --cache {self.cache} --teachers {best_single}")
                    elif mode == "uniform":
                        sh(f"python -m dmthd.train_student {base} --mode uniform --cache {self.cache} --teachers {' '.join(self.tags)}")
                    else:
                        sh(f"python -m dmthd.train_student {base} --mode dmthd --cache {self.cache} --teachers {' '.join(self.tags)} --aux --delta 0.3")
        name, stag = pairs(STUDENTS)[0]
        abl = {"no_dynamic": "--mode uniform --aux --delta 0.3", "no_hidden": "--mode dmthd --no_hidden --aux --delta 0.3",
               "no_aux": "--mode dmthd", "per_batch": "--mode dmthd --per_batch --aux --delta 0.3"}
        for tag, flags in abl.items():
            for seed in SEEDS:
                out = f"{self.runs}/{stag}/ablation_{tag}/seed{seed}"
                if done(out):
                    continue
                sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {seed} "
                   f"--batch {bs} {self.fp} {self.common} --cache {self.cache} --teachers {' '.join(self.tags)} {flags} --tag {tag}")

    def probes(self):
        neg, pos = os.path.join(PROBES, "benign_sarcasm.csv"), os.path.join(PROBES, "ironic_abuse.csv")
        if not (os.path.exists(neg) and os.path.exists(pos)):
            print("no probe sets found, skipping", flush=True)
            return
        targets = [f"{self.runs}/teachers/{tag}" for tag in self.tags]
        for _, stag in pairs(STUDENTS):
            targets += [f"{self.runs}/{stag}/{mode}/seed{s}" for mode in MODES for s in SEEDS]
        for d in targets:
            if done(d) and not os.path.exists(os.path.join(d, "eval_test.json")):
                sh(f"python -m dmthd.evaluate --model_dir {d} --csv {self.data}/test.csv --scheme {self.cfg['scheme']} "
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --probe_neg {neg} --probe_pos {pos}")

    def bench(self):
        dirs = [f"{self.runs}/teachers/{tag}" for tag in self.tags] + [f"{self.runs}/{stag}/dmthd/seed{SEEDS[0]}" for _, stag in pairs(STUDENTS)]
        dirs = [d for d in dirs if os.path.exists(d)]
        dev = "cuda" if GPU else "cpu"
        sh(f"python -m dmthd.bench --model_dirs {' '.join(dirs)} --csv {self.data}/test.csv --device {dev} "
           f"--max_len {self.cfg['max_len']} --num_labels {self.cfg['num_labels']} --out {self.runs}/bench_{dev}.csv")

    def aggregate(self):
        sh(f"python -m dmthd.aggregate --runs {self.runs} --out {self.runs}/summary.csv")
        _, stag = pairs(STUDENTS)[0]
        for cand in ("dmthd", "uniform", "skd"):
            if os.path.isdir(f"{self.runs}/{stag}/{cand}"):
                sh(f"python -m dmthd.aggregate --compare {self.runs}/{stag}/ft {self.runs}/{stag}/{cand}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tweets", choices=list(DATASETS))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONPATH", "src")
    raw = a.raw or {"tweets": "/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv",
                    "wikipedia": f"{ROOT}/raw_wikipedia"}[a.dataset]
    b = Bench(a.dataset, raw)
    b.resume()
    order = ["prepare", "teachers", "cache", "students", "probes", "bench", "aggregate"]
    for st in (order if a.stage == "all" else [a.stage]):
        getattr(b, "cache_teachers" if st == "cache" else st)()
