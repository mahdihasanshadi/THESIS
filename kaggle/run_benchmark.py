"""One-command driver for a whole benchmark on Kaggle or Colab.

    PYTHONPATH=src python kaggle/run_benchmark.py --dataset tweets    --raw /kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv
    PYTHONPATH=src python kaggle/run_benchmark.py --dataset wikipedia --raw /kaggle/working/raw_wikipedia   (downloads from Figshare)

Stages: prepare | teachers | cache | students | probes | quant | bench | aggregate | all. Every stage
skips work whose results.json already exists, so a killed session resumes where it stopped.

Design (the comparison grid): two teacher committees x two student families.
  COMMITTEES=homo,hetero   homo = BERT-large + HateBERT + RoBERTa-irony; hetero = homo + DeBERTa-v3-base
  STUDENTS (homogeneous)   BERT-mini, BERT-small, DistilBERT
  HETERO_STUDENTS          DeBERTa-v3-xsmall (another transformer family), bilstm (not a transformer)
Run directories: runs/<dataset>/<student>/<mode>[_hetero]/seed<k>. Ablations run on the first
student with the homogeneous committee. On binary datasets with annotator fractions the extra
mode `dmthd_dis` (disagreement-aware, soft reliability) is added when DISAGREEMENT=1.

Resuming across Kaggle sessions: attach the previous notebook's output as an input and set
RESUME_FROM=/kaggle/input/<that-output>; its runs/ and cache/ trees are copied in first.

Environment overrides: ROOT, TEACHERS, HETERO_TEACHER, COMMITTEES, STUDENTS, HETERO_STUDENTS, SEEDS,
MODES, GPU, RESUME_FROM, TEACHER_EPOCHS, DISAGREEMENT, KAPPA.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

DATASETS = {
    "tweets": {"scheme": "six", "label_col": "label_name", "max_len": 128, "num_labels": 6,
               "teacher_epochs": 5, "teacher_batch": 32, "student_batch": 32, "soft": False},
    "wikipedia": {"scheme": "binary", "label_col": "label", "max_len": 256, "num_labels": 2,
                  "teacher_epochs": 3, "teacher_batch": 16, "student_batch": 32, "soft": True},
}
ROOT = os.environ.get("ROOT", ".")
TEACHERS = os.environ.get("TEACHERS", "bert-large-uncased:bert-large,GroNLP/hateBERT:hatebert,cardiffnlp/twitter-roberta-base-irony:irony")
HETERO_TEACHER = os.environ.get("HETERO_TEACHER", "microsoft/deberta-v3-base:deberta-base")
COMMITTEES = os.environ.get("COMMITTEES", "homo,hetero").split(",")
STUDENTS = os.environ.get("STUDENTS", "google/bert_uncased_L-4_H-256_A-4:bert-mini,google/bert_uncased_L-4_H-512_A-8:bert-small,distilbert-base-uncased:distilbert")
HETERO_STUDENTS = os.environ.get("HETERO_STUDENTS", "microsoft/deberta-v3-xsmall:deberta-xsmall,bilstm:bilstm")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MODES = os.environ.get("MODES", "ft,skd,uniform,dmthd").split(",")
AUX_MODEL = "cardiffnlp/twitter-roberta-base-irony"
GPU = os.environ.get("GPU", "1") == "1"
RESUME_FROM = os.environ.get("RESUME_FROM", "")
DISAGREEMENT = os.environ.get("DISAGREEMENT", "1") == "1"
KAPPA = os.environ.get("KAPPA", "1.0")
PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes")


QUIET = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}


def sh(cmd):
    print("\n$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env={**os.environ, **QUIET})
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {cmd}")


def pairs(spec):
    return [tuple(x.split(":")) for x in spec.split(",") if x]


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
        self.all_teachers = pairs(TEACHERS) + (pairs(HETERO_TEACHER) if "hetero" in COMMITTEES else [])
        self.committee = {"homo": [t for _, t in pairs(TEACHERS)],
                          "hetero": [t for _, t in pairs(TEACHERS)] + [t for _, t in pairs(HETERO_TEACHER)]}
        self.students = pairs(STUDENTS) + pairs(HETERO_STUDENTS)

    # ---- stages ----
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
        for name, tag in self.all_teachers:
            out = f"{self.runs}/teachers/{tag}"
            if done(out):
                continue
            ck = "--grad_ckpt" if "large" in name else ""
            sh(f"python -m dmthd.train_teacher --model_name {name} --data_dir {self.data} --out_dir {out} "
               f"--epochs {ep} --lr 2e-5 --batch {self.cfg['teacher_batch']} {self.fp} {ck} {self.common}")

    def cache_teachers(self):
        tags = [t for _, t in self.all_teachers]
        if os.path.exists(f"{self.cache}/meta.json"):
            import json
            have = {t["tag"] for t in json.load(open(f"{self.cache}/meta.json"))["teachers"]}
            if set(tags) <= have:
                return
        dirs = " ".join(f"{self.runs}/teachers/{tag}" for tag in tags)
        sh(f"python -m dmthd.cache_teachers --data_dir {self.data} --out {self.cache} --teachers {dirs} "
           f"--aux_model {AUX_MODEL} {self.common}")

    def _student_cmd(self, name, out, seed, mode, tags, extra=""):
        bs = self.cfg["student_batch"]
        base = f"--student {name} --data_dir {self.data} --out_dir {out} --seed {seed} --batch {bs} {self.fp} {self.common}"
        if mode == "ft":
            return f"python -m dmthd.train_student {base} --mode ft {extra}"
        if mode == "skd":
            return f"python -m dmthd.train_student {base} --mode skd --cache {self.cache} --teachers {tags[0]} {extra}"
        if mode == "uniform":
            return f"python -m dmthd.train_student {base} --mode uniform --cache {self.cache} --teachers {' '.join(tags)} {extra}"
        if mode == "dmthd_dis":
            return (f"python -m dmthd.train_student {base} --mode dmthd --cache {self.cache} --teachers {' '.join(tags)} "
                    f"--aux --delta 0.3 --disagreement --kappa {KAPPA} --reliability soft --tag disagreement {extra}")
        return f"python -m dmthd.train_student {base} --mode dmthd --cache {self.cache} --teachers {' '.join(tags)} --aux --delta 0.3 {extra}"

    def students(self):
        modes = list(MODES) + (["dmthd_dis"] if (DISAGREEMENT and self.cfg["soft"]) else [])
        for comm in COMMITTEES:
            tags = self.committee[comm]
            suffix = "" if comm == "homo" else "_hetero"
            for name, stag in self.students:
                for mode in modes:
                    if mode == "ft" and comm != "homo":
                        continue  # the control does not depend on the committee
                    for seed in SEEDS:
                        out = f"{self.runs}/{stag}/{mode}{suffix}/seed{seed}"
                        if done(out):
                            continue
                        sh(self._student_cmd(name, out, seed, mode, tags))
        # ablations: first student, homogeneous committee
        name, stag = self.students[0]
        tags = self.committee["homo"]
        abl = {"no_dynamic": "--mode uniform --aux --delta 0.3", "no_hidden": "--mode dmthd --no_hidden --aux --delta 0.3",
               "no_aux": "--mode dmthd", "per_batch": "--mode dmthd --per_batch --aux --delta 0.3",
               "from_scratch": "--mode dmthd --aux --delta 0.3 --from_scratch"}
        for tag, flags in abl.items():
            for seed in SEEDS:
                out = f"{self.runs}/{stag}/ablation_{tag}/seed{seed}"
                if done(out):
                    continue
                sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {seed} "
                   f"--batch {self.cfg['student_batch']} {self.fp} {self.common} --cache {self.cache} --teachers {' '.join(tags)} {flags} --tag {tag}")

    def _run_dirs(self):
        dirs = [f"{self.runs}/teachers/{tag}" for _, tag in self.all_teachers]
        for _, stag in self.students:
            dirs += glob.glob(f"{self.runs}/{stag}/*/seed*")
        return [d for d in dirs if done(d)]

    def probes(self):
        neg, pos = os.path.join(PROBES, "benign_sarcasm_screened.csv"), os.path.join(PROBES, "ironic_abuse.csv")
        if not os.path.exists(neg):
            neg = os.path.join(PROBES, "benign_sarcasm.csv")
        if not (os.path.exists(neg) and os.path.exists(pos)):
            print("no probe sets found, skipping", flush=True)
            return
        for d in self._run_dirs():
            if not os.path.exists(os.path.join(d, "eval_test.json")):
                sh(f"python -m dmthd.evaluate --model_dir {d} --csv {self.data}/test.csv --scheme {self.cfg['scheme']} "
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --probe_neg {neg} --probe_pos {pos}")

    def sweep(self):
        """Validation-only hyper-parameter sweeps on the headline student, one seed, homogeneous
        committee. Test numbers are written too but the paper reports the validation column."""
        name, stag = self.students[0]
        tags = self.committee["homo"]
        grid = [("tau", v, f"--tau {v}") for v in (0.5, 2.0, 5.0)] + [("T", v, f"--T {v}") for v in (1.0, 2.0, 8.0)] + \
               [("alpha", v, f"--alpha {v} --beta {round(0.8 - v, 2)}") for v in (0.2, 0.6)] + [("delta", v, f"--delta {v}") for v in (0.1, 0.5)]
        for param, value, flags in grid:
            out = f"{self.runs}/{stag}/sweep_{param}_{value}/seed{SEEDS[0]}"
            if done(out):
                continue
            sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {SEEDS[0]} "
               f"--batch {self.cfg['student_batch']} {self.fp} {self.common} --mode dmthd --cache {self.cache} "
               f"--teachers {' '.join(tags)} --aux --delta 0.3 {flags} --tag sweep_{param}_{value}")

    def robustness(self):
        """Obfuscated test variants and cross-dataset transfer for the headline D-MTHD and
        fine-tune-only students (first seed)."""
        obf_dir = self.data
        if not os.path.exists(os.path.join(obf_dir, "test_obf_mixed.csv")):
            sh(f"python -m dmthd.obfuscate --csv {self.data}/test.csv --out_dir {obf_dir} --scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']}")
        _, stag = self.students[0]
        for mode in ("ft", "dmthd"):
            d = f"{self.runs}/{stag}/{mode}/seed{SEEDS[0]}"
            if not done(d):
                continue
            for variant in ("leet", "swap", "space", "mixed"):
                out = os.path.join(d, f"eval_test_obf_{variant}.json")
                if not os.path.exists(out):
                    sh(f"python -m dmthd.evaluate --model_dir {d} --csv {obf_dir}/test_obf_{variant}.csv --scheme {self.cfg['scheme']} "
                       f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --out {out}")
            other = "wikipedia" if self.name == "tweets" else "tweets"
            other_cfg = DATASETS[other]
            other_test = f"{ROOT}/data/{other}/test.csv"
            if os.path.exists(other_test) and not os.path.exists(os.path.join(d, f"transfer_{other}.json")):
                sh(f"python -m dmthd.transfer_eval --model_dir {d} --model_scheme {self.cfg['scheme']} --csv {other_test} "
                   f"--csv_scheme {other_cfg['scheme']} --csv_label_col {other_cfg['label_col']} --max_len {other_cfg['max_len']}")

    def quant(self):
        for _, stag in self.students:
            d = f"{self.runs}/{stag}/dmthd/seed{SEEDS[0]}"
            if done(d) and not os.path.exists(os.path.join(d, "quantize_eval.json")):
                sh(f"python -m dmthd.quantize_eval --model_dir {d} --csv {self.data}/test.csv --scheme {self.cfg['scheme']} "
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']}")

    def bench(self):
        dirs = [f"{self.runs}/teachers/{tag}" for _, tag in self.all_teachers] + [f"{self.runs}/{stag}/dmthd/seed{SEEDS[0]}" for _, stag in self.students]
        dirs = [d for d in dirs if os.path.exists(d)]
        dev = "cuda" if GPU else "cpu"
        sh(f"python -m dmthd.bench --model_dirs {' '.join(dirs)} --csv {self.data}/test.csv --device {dev} "
           f"--max_len {self.cfg['max_len']} --num_labels {self.cfg['num_labels']} --out {self.runs}/bench_{dev}.csv")

    def aggregate(self):
        sh(f"python -m dmthd.aggregate --runs {self.runs} --out {self.runs}/summary.csv")
        _, stag = self.students[0]
        for cand in ("dmthd", "uniform", "skd", "dmthd_hetero", "dmthd_dis"):
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
    order = ["prepare", "teachers", "cache", "students", "probes", "sweep", "robustness", "quant", "bench", "aggregate"]
    for st in (order if a.stage == "all" else [a.stage]):
        getattr(b, "cache_teachers" if st == "cache" else st)()
