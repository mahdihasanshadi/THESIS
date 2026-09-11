"""One-command driver for a whole benchmark on Kaggle or Colab.

    PYTHONPATH=src python kaggle/run_benchmark.py --dataset tweets    --raw /kaggle/working/raw/cyberbullying_tweets.csv
    PYTHONPATH=src python kaggle/run_benchmark.py --dataset wikipedia                       (downloads from Figshare)

Stages: prepare | teachers | cache | students | probes | sweep | robustness | quant | bench | aggregate | all.
Every stage skips work whose results.json already exists, so a killed session resumes where it stopped.

Design (the comparison grid): two teacher committees x two student families.
  COMMITTEES=homo,hetero   homo = BERT-large + HateBERT + RoBERTa-irony; hetero = homo + DeBERTa-v3-base
  STUDENTS (homogeneous)   BERT-mini, BERT-small, DistilBERT
  HETERO_STUDENTS          DeBERTa-v3-xsmall (another transformer family), bilstm (not a transformer)
Run directories: runs/<dataset>/<student>/<mode>[_hetero]/seed<k>. Ablations run on the first
student with the homogeneous committee. On binary datasets with annotator fractions the extra
mode `dmthd_dis` (disagreement-aware, soft reliability) is added when DISAGREEMENT=1.

Teacher safety: a teacher whose test macro-F1 is below MIN_TEACHER_F1 (default 0.5) is treated as
collapsed (e.g. a NaN loss), retrained once with conservative settings (lr 1e-5, fp32), and
dropped from every committee if it is still collapsed. Dropped teachers are listed in
runs/<dataset>/dropped_teachers.json and the cache is rebuilt.

Resuming across Kaggle sessions: attach the previous notebook's output as an input and set
RESUME_FROM=/kaggle/input/<that-output>; its runs/ and cache/ trees are copied in first.

Environment overrides: ROOT, TEACHERS, HETERO_TEACHER, COMMITTEES, STUDENTS, HETERO_STUDENTS, SEEDS,
MODES, GPU, RESUME_FROM, TEACHER_EPOCHS, STUDENT_EPOCHS, DISAGREEMENT, KAPPA, MIN_TEACHER_F1, LIMIT (debug).
"""
import argparse
import glob
import json
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
COMMITTEES = [c for c in os.environ.get("COMMITTEES", "homo,hetero").split(",") if c]
STUDENTS = os.environ.get("STUDENTS", "google/bert_uncased_L-4_H-256_A-4:bert-mini,google/bert_uncased_L-4_H-512_A-8:bert-small,distilbert-base-uncased:distilbert")
HETERO_STUDENTS = os.environ.get("HETERO_STUDENTS", "microsoft/deberta-v3-xsmall:deberta-xsmall,bilstm:bilstm")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MODES = [m for m in os.environ.get("MODES", "ft,skd,uniform,dmthd").split(",") if m]
AUX_MODEL = "cardiffnlp/twitter-roberta-base-irony"
GPU = os.environ.get("GPU", "1") == "1"
RESUME_FROM = os.environ.get("RESUME_FROM", "")
DISAGREEMENT = os.environ.get("DISAGREEMENT", "1") == "1"
KAPPA = os.environ.get("KAPPA", "1.0")
MIN_TEACHER_F1 = float(os.environ.get("MIN_TEACHER_F1", "0.5"))
LIMIT = os.environ.get("LIMIT", "")
STUDENT_EPOCHS = os.environ.get("STUDENT_EPOCHS", "6")
PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes")
QUIET = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}


def sh(cmd, check=True):
    print("\n$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env={**os.environ, **QUIET})
    if r.returncode != 0 and check:
        sys.exit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def pairs(spec):
    return [tuple(x.split(":")) for x in spec.split(",") if x]


def done(d):
    return os.path.exists(os.path.join(d, "results.json"))


def test_f1(d):
    try:
        return float(json.load(open(os.path.join(d, "results.json")))["test"]["macro_f1"])
    except Exception:
        return float("nan")


class Bench:
    def __init__(self, name, raw):
        self.name, self.raw, self.cfg = name, raw, DATASETS[name]
        self.data = f"{ROOT}/data/{name}"
        self.cache = f"{ROOT}/cache/{name}"
        self.runs = f"{ROOT}/runs/{name}"
        self.common = f"--scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']}"
        self.limit = f"--limit {LIMIT}" if LIMIT else ""
        self.fp = "--fp16" if GPU else ""
        self.teacher_list = pairs(TEACHERS) + (pairs(HETERO_TEACHER) if "hetero" in COMMITTEES else [])
        self.student_list = pairs(STUDENTS) + pairs(HETERO_STUDENTS)
        self.dropped = []
        self.cache_dirty = False
        self._load_dropped()

    # ---- committees (recomputed so dropped teachers disappear everywhere) ----
    def committee(self, comm):
        homo = [t for _, t in pairs(TEACHERS) if t not in self.dropped]
        if comm == "homo":
            return homo
        return homo + [t for _, t in pairs(HETERO_TEACHER) if t not in self.dropped]

    def active_committees(self):
        out = []
        for comm in COMMITTEES:
            tags = self.committee(comm)
            if not tags:
                print(f"committee {comm}: no teachers left, skipped", flush=True)
                continue
            if comm == "hetero" and tags == self.committee("homo"):
                print("committee hetero: identical to homo after drops, skipped", flush=True)
                continue
            out.append((comm, tags))
        return out

    def _load_dropped(self):
        p = os.path.join(self.runs, "dropped_teachers.json")
        if os.path.exists(p):
            self.dropped = json.load(open(p)).get("dropped", [])

    def _save_dropped(self):
        os.makedirs(self.runs, exist_ok=True)
        json.dump({"dropped": self.dropped, "min_teacher_f1": MIN_TEACHER_F1}, open(os.path.join(self.runs, "dropped_teachers.json"), "w"))

    # ---- stages ----
    def resume(self):
        """Copy a previous session's runs/ and cache/ in. Fails fast on a bad RESUME_FROM: silently
        ignoring it would retrain every teacher, wasting hours of GPU time."""
        if not RESUME_FROM:
            return
        if not os.path.isdir(RESUME_FROM):
            sys.exit(f"RESUME_FROM={RESUME_FROM} does not exist. Attach the previous notebook's output as an "
                     f"input and use its path (right panel, Input), or clear RESUME_FROM to start fresh. "
                     f"Available inputs: {glob.glob('/kaggle/input/*')}")
        copied = []
        for sub in ("runs", "cache"):
            for src in glob.glob(os.path.join(RESUME_FROM, "**", sub, self.name), recursive=True):
                dst = f"{ROOT}/{sub}/{self.name}"
                print(f"resuming: copying {src} -> {dst}", flush=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied.append(src)
        if not copied:
            sys.exit(f"RESUME_FROM={RESUME_FROM} contains no runs/{self.name} or cache/{self.name} to resume from. "
                     f"Its top level holds: {sorted(os.listdir(RESUME_FROM))[:20]}. Point it at the right input, "
                     f"or clear RESUME_FROM to start fresh.")
        n_teachers = len(glob.glob(f"{self.runs}/teachers/*/results.json"))
        n_runs = len(glob.glob(f"{self.runs}/*/*/seed*/results.json"))
        print(f"resumed: {n_teachers} finished teachers, {n_runs} finished student runs", flush=True)
        self._load_dropped()

    def prepare(self):
        if os.path.exists(f"{self.data}/test.csv"):
            return
        if self.name == "tweets":
            sh(f"python -m dmthd.prepare_tweets --raw {self.raw} --out {self.data}")
        else:
            sh(f"python -m dmthd.prepare_wikipedia --raw {self.raw} --out {self.data} --download")

    def _train_teacher(self, name, out, lr, fp16, epochs):
        ck = "--grad_ckpt" if "large" in name else ""
        sh(f"python -m dmthd.train_teacher --model_name {name} --data_dir {self.data} --out_dir {out} "
           f"--epochs {epochs} --lr {lr} --batch {self.cfg['teacher_batch']} {fp16} {ck} {self.common} {self.limit} --no_resume", check=False)

    def teachers(self):
        ep = os.environ.get("TEACHER_EPOCHS", self.cfg["teacher_epochs"])
        for name, tag in self.teacher_list:
            if tag in self.dropped:
                continue
            out = f"{self.runs}/teachers/{tag}"
            if done(out) and test_f1(out) >= MIN_TEACHER_F1:
                continue
            if done(out):
                print(f"teacher {tag} collapsed (test macro-F1 {test_f1(out):.3f} < {MIN_TEACHER_F1}); retraining with lr 1e-5 in fp32", flush=True)
                shutil.rmtree(out, ignore_errors=True)
                self._train_teacher(name, out, "1e-5", "", ep)
            else:
                self._train_teacher(name, out, "2e-5", self.fp, ep)
                if not (done(out) and test_f1(out) >= MIN_TEACHER_F1):
                    print(f"teacher {tag} collapsed on the first attempt; retraining with lr 1e-5 in fp32", flush=True)
                    shutil.rmtree(out, ignore_errors=True)
                    self._train_teacher(name, out, "1e-5", "", ep)
            self.cache_dirty = True
            if not (done(out) and test_f1(out) >= MIN_TEACHER_F1):
                print(f"teacher {tag} still collapsed after retraining: DROPPED from every committee", flush=True)
                self.dropped.append(tag)
                self._save_dropped()
                shutil.rmtree(out, ignore_errors=True)

    def cache_teachers(self):
        tags = [t for _, t in self.teacher_list if t not in self.dropped]
        meta = f"{self.cache}/meta.json"
        if os.path.exists(meta) and not self.cache_dirty:
            have = {t["tag"] for t in json.load(open(meta))["teachers"]}
            if set(tags) <= have:
                return
        if os.path.exists(self.cache):
            shutil.rmtree(self.cache, ignore_errors=True)
        dirs = " ".join(f"{self.runs}/teachers/{tag}" for tag in tags)
        sh(f"python -m dmthd.cache_teachers --data_dir {self.data} --out {self.cache} --teachers {dirs} "
           f"--aux_model {AUX_MODEL} {self.common} {self.limit}")

    def _student_cmd(self, name, out, seed, mode, tags, extra=""):
        base = (f"--student {name} --data_dir {self.data} --out_dir {out} --seed {seed} --batch {self.cfg['student_batch']} "
                f"--epochs {STUDENT_EPOCHS} {self.fp} {self.common} {self.limit}")
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
        committees = self.active_committees()
        # the fine-tune-only control does not depend on any committee
        if "ft" in modes:
            for name, stag in self.student_list:
                for seed in SEEDS:
                    out = f"{self.runs}/{stag}/ft/seed{seed}"
                    if not done(out):
                        sh(self._student_cmd(name, out, seed, "ft", []))
        for comm, tags in committees:
            suffix = "" if comm == "homo" else "_hetero"
            for name, stag in self.student_list:
                for mode in [m for m in modes if m != "ft"]:
                    for seed in SEEDS:
                        out = f"{self.runs}/{stag}/{mode}{suffix}/seed{seed}"
                        if not done(out):
                            sh(self._student_cmd(name, out, seed, mode, tags))
        homo = self.committee("homo")
        if not homo:
            print("no homogeneous teachers left: ablations skipped", flush=True)
            return
        name, stag = self.student_list[0]
        abl = {"no_dynamic": "--mode uniform --aux --delta 0.3", "no_hidden": "--mode dmthd --no_hidden --aux --delta 0.3",
               "no_aux": "--mode dmthd", "per_batch": "--mode dmthd --per_batch --aux --delta 0.3",
               "from_scratch": "--mode dmthd --aux --delta 0.3 --from_scratch"}
        for tag, flags in abl.items():
            if tag == "from_scratch" and name == "bilstm":
                continue
            for seed in SEEDS:
                out = f"{self.runs}/{stag}/ablation_{tag}/seed{seed}"
                if not done(out):
                    sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {seed} "
                       f"--batch {self.cfg['student_batch']} --epochs {STUDENT_EPOCHS} {self.fp} {self.common} {self.limit} "
                       f"--cache {self.cache} --teachers {' '.join(homo)} {flags} --tag {tag}")

    def _run_dirs(self):
        dirs = [f"{self.runs}/teachers/{tag}" for _, tag in self.teacher_list if tag not in self.dropped]
        for _, stag in self.student_list:
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
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --probe_neg {neg} --probe_pos {pos}", check=False)

    def sweep(self):
        homo = self.committee("homo")
        if not homo:
            return
        name, stag = self.student_list[0]
        grid = [("tau", v, f"--tau {v}") for v in (0.5, 2.0, 5.0)] + [("T", v, f"--T {v}") for v in (1.0, 2.0, 8.0)] + \
               [("alpha", v, f"--alpha {v} --beta {round(0.8 - v, 2)}") for v in (0.2, 0.6)] + [("delta", v, f"--delta {v}") for v in (0.1, 0.5)]
        for param, value, flags in grid:
            out = f"{self.runs}/{stag}/sweep_{param}_{value}/seed{SEEDS[0]}"
            if not done(out):
                sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {SEEDS[0]} "
                   f"--batch {self.cfg['student_batch']} --epochs {STUDENT_EPOCHS} {self.fp} {self.common} {self.limit} --mode dmthd "
                   f"--cache {self.cache} --teachers {' '.join(homo)} --aux --delta 0.3 {flags} --tag sweep_{param}_{value}")

    def robustness(self):
        if not os.path.exists(os.path.join(self.data, "test_obf_mixed.csv")):
            sh(f"python -m dmthd.obfuscate --csv {self.data}/test.csv --out_dir {self.data} --scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']}")
        _, stag = self.student_list[0]
        for mode in ("ft", "dmthd"):
            d = f"{self.runs}/{stag}/{mode}/seed{SEEDS[0]}"
            if not done(d):
                continue
            for variant in ("leet", "swap", "space", "mixed"):
                out = os.path.join(d, f"eval_test_obf_{variant}.json")
                if not os.path.exists(out):
                    sh(f"python -m dmthd.evaluate --model_dir {d} --csv {self.data}/test_obf_{variant}.csv --scheme {self.cfg['scheme']} "
                       f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --out {out}", check=False)
            other = "wikipedia" if self.name == "tweets" else "tweets"
            other_cfg, other_test = DATASETS[other], f"{ROOT}/data/{other}/test.csv"
            if os.path.exists(other_test) and not os.path.exists(os.path.join(d, f"transfer_{other}.json")):
                sh(f"python -m dmthd.transfer_eval --model_dir {d} --model_scheme {self.cfg['scheme']} --csv {other_test} "
                   f"--csv_scheme {other_cfg['scheme']} --csv_label_col {other_cfg['label_col']} --max_len {other_cfg['max_len']}", check=False)

    def quant(self):
        for _, stag in self.student_list:
            d = f"{self.runs}/{stag}/dmthd/seed{SEEDS[0]}"
            if done(d) and not os.path.exists(os.path.join(d, "quantize_eval.json")):
                sh(f"python -m dmthd.quantize_eval --model_dir {d} --csv {self.data}/test.csv --scheme {self.cfg['scheme']} "
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']}", check=False)

    def bench(self):
        dirs = [f"{self.runs}/teachers/{tag}" for _, tag in self.teacher_list if tag not in self.dropped]
        dirs += [f"{self.runs}/{stag}/dmthd/seed{SEEDS[0]}" for _, stag in self.student_list]
        dirs = [d for d in dirs if os.path.exists(d)]
        dev = "cuda" if GPU else "cpu"
        sh(f"python -m dmthd.bench --model_dirs {' '.join(dirs)} --csv {self.data}/test.csv --device {dev} "
           f"--max_len {self.cfg['max_len']} --num_labels {self.cfg['num_labels']} --out {self.runs}/bench_{dev}.csv", check=False)

    def aggregate(self):
        sh(f"python -m dmthd.aggregate --runs {self.runs} --out {self.runs}/summary.csv", check=False)
        _, stag = self.student_list[0]
        for cand in ("dmthd", "uniform", "skd", "dmthd_hetero", "dmthd_dis"):
            if os.path.isdir(f"{self.runs}/{stag}/{cand}") and os.path.isdir(f"{self.runs}/{stag}/ft"):
                sh(f"python -m dmthd.aggregate --compare {self.runs}/{stag}/ft {self.runs}/{stag}/{cand}", check=False)
        if self.dropped:
            print(f"NOTE: teachers dropped as collapsed: {self.dropped} (see {self.runs}/dropped_teachers.json)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tweets", choices=list(DATASETS))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONPATH", "src")
    raw = a.raw or {"tweets": "/kaggle/working/raw/cyberbullying_tweets.csv", "wikipedia": f"{ROOT}/raw_wikipedia"}[a.dataset]
    b = Bench(a.dataset, raw)
    b.resume()
    order = ["prepare", "teachers", "cache", "students", "probes", "sweep", "robustness", "quant", "bench", "aggregate"]
    for st in (order if a.stage == "all" else [a.stage]):
        getattr(b, "cache_teachers" if st == "cache" else st)()
