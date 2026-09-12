"""One-command driver for a whole benchmark on Kaggle or Colab.

    PYTHONPATH=src python kaggle/run_benchmark.py --dataset tweets    --raw /kaggle/working/raw/cyberbullying_tweets.csv
    PYTHONPATH=src python kaggle/run_benchmark.py --dataset wikipedia                       (downloads from Figshare)
    PYTHONPATH=src python kaggle/run_benchmark.py --dataset implicit                        (downloads from the Hub)

Stages: prepare | specialist | teachers | cache | students | probes | sweep | robustness | quant |
bench | aggregate | all.
Every stage skips work whose results.json already exists, so a killed session resumes where it stopped.

Design (the comparison grid): three teacher committees x two student families.
  COMMITTEES               homo = BERT-large + HateBERT + RoBERTa-irony;
                           spec = homo + the implicit-abuse specialist (headline student only);
                           hetero = homo + DeBERTa-v3-base
  STUDENTS (homogeneous)   BERT-mini, BERT-small, DistilBERT
  HETERO_STUDENTS          DeBERTa-v3-xsmall (another transformer family), bilstm (not a transformer)
The implicit-abuse specialist is HateBERT trained on the implicit benchmark, then task-adapted like
any other teacher. It is the only committee member that has seen abuse-by-implication labelled as
such, and it forms the `spec` committee. Keeping it out of `homo` means every run already finished
with the three-teacher committee stays valid, and "does the specialist help?" becomes a controlled
comparison of two full committees over three seeds rather than a single-seed ablation.
Run directories: runs/<dataset>/<student>/<mode>[_spec|_hetero]/seed<k>. Ablations run on the
first student with the homogeneous committee. On binary datasets with annotator fractions the extra
mode `dmthd_dis` (disagreement-aware, soft reliability) is added when DISAGREEMENT=1.

Teacher safety: a teacher whose test macro-F1 is below MIN_TEACHER_F1 (default 0.5) is treated as
collapsed (e.g. a NaN loss), retrained once with conservative settings (lr 1e-5, fp32), and
dropped from every committee if it is still collapsed. Dropped teachers are listed in
runs/<dataset>/dropped_teachers.json and the cache is rebuilt.

Resuming across Kaggle sessions: attach the previous notebook's output as an input and set
RESUME_FROM=/kaggle/input/<that-output>; its runs/ and cache/ trees are copied in first.

Environment overrides: ROOT, TEACHERS, HETERO_TEACHER, COMMITTEES, STUDENTS, HETERO_STUDENTS, SEEDS,
MODES, GPU, RESUME_FROM, TEACHER_EPOCHS, STUDENT_EPOCHS, DISAGREEMENT, KAPPA, MIN_TEACHER_F1,
SPECIALIST, SPECIALIST_BASE, SPEC_STUDENTS, IMPLICIT_RAW, LIMIT (debug).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

DATASETS = {
    "tweets": {"scheme": "six", "label_col": "label_name", "max_len": 128, "num_labels": 6,
               "teacher_epochs": 5, "teacher_batch": 32, "student_batch": 32, "soft": False},
    "wikipedia": {"scheme": "binary", "label_col": "label", "max_len": 256, "num_labels": 2,
                  "teacher_epochs": 3, "teacher_batch": 16, "student_batch": 32, "soft": True},
    # The benchmark the paper's goal needs: abuse-by-implication is a label here, not a hidden
    # subset of a catch-all class.
    "implicit": {"scheme": "implicit3", "label_col": "label_name", "max_len": 128, "num_labels": 3,
                 "teacher_epochs": 4, "teacher_batch": 32, "student_batch": 32, "soft": False,
                 # Its classes come unevenly from two sources that a lexical model separates at
                 # 0.91 macro-F1, so every score is also reported within each source, where
                 # recognising the source cannot help.
                 "group_col": "corpus"},
}
ROOT = os.environ.get("ROOT", ".")
TEACHERS = os.environ.get("TEACHERS", "bert-large-uncased:bert-large,GroNLP/hateBERT:hatebert,cardiffnlp/twitter-roberta-base-irony:irony")
HETERO_TEACHER = os.environ.get("HETERO_TEACHER", "microsoft/deberta-v3-base:deberta-base")
COMMITTEES = [c for c in os.environ.get("COMMITTEES", "homo,spec,hetero").split(",") if c]
# `spec` is the homogeneous committee plus the implicit specialist. It is a separate committee rather
# than an extra member of `homo` for two reasons. It leaves every run already finished with the
# three-teacher committee valid, which is worth about ten GPU-hours; and it turns "does the specialist
# help?" into a controlled multi-seed comparison of two full committees instead of a one-seed
# ablation. Restricted by default to the headline student, because the question is about the
# committee and not about the student.
SPEC_STUDENTS = [s for s in os.environ.get("SPEC_STUDENTS", "bert-mini").split(",") if s]
STUDENTS = os.environ.get("STUDENTS", "google/bert_uncased_L-4_H-256_A-4:bert-mini,google/bert_uncased_L-4_H-512_A-8:bert-small,distilbert-base-uncased:distilbert")
HETERO_STUDENTS = os.environ.get("HETERO_STUDENTS", "microsoft/deberta-v3-xsmall:deberta-xsmall,bilstm:bilstm")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
MODES = [m for m in os.environ.get("MODES", "ft,skd,uniform,dmthd").split(",") if m]
AUX_MODEL = "cardiffnlp/twitter-roberta-base-irony"
# An implicit-abuse specialist for the other two benchmarks. Neither the tweet corpus nor the
# Wikipedia corpus labels abuse-by-implication, so no teacher fine-tuned on them can hold that
# knowledge, and the committee has no member able to recognise it. This teacher is HateBERT trained
# first on the implicit benchmark; the usual task adaptation then re-heads it for the target
# benchmark, so it enters the committee in the target label space while keeping what it learned.
# It forms the `spec` committee rather than joining `homo`; see COMMITTEES above.
# It is the mechanism the paper's implicit claim rests on. SPECIALIST=0 turns it off.
SPECIALIST = os.environ.get("SPECIALIST", "1") == "1"
SPECIALIST_BASE = os.environ.get("SPECIALIST_BASE", "GroNLP/hateBERT")
SPECIALIST_TAG = os.environ.get("SPECIALIST_TAG", "implicit-spec")
SPECIALIST_SRC = f"{ROOT}/runs/implicit/specialist/{SPECIALIST_TAG}"
IMPLICIT_RAW = os.environ.get("IMPLICIT_RAW", f"{ROOT}/raw")
GPU = os.environ.get("GPU", "1") == "1"
RESUME_FROM = os.environ.get("RESUME_FROM", "")
DISAGREEMENT = os.environ.get("DISAGREEMENT", "1") == "1"
KAPPA = os.environ.get("KAPPA", "1.0")
MIN_TEACHER_F1 = float(os.environ.get("MIN_TEACHER_F1", "0.5"))
LIMIT = os.environ.get("LIMIT", "")
# "auto" copies model weights back only for runs whose probe evaluation is still pending;
# "all" copies every checkpoint (needs ~15 GB of the 20 GB Kaggle gives); "none" copies none.
RESUME_WEIGHTS = os.environ.get("RESUME_WEIGHTS", "auto")
STUDENT_EPOCHS = os.environ.get("STUDENT_EPOCHS", "6")
ABLATION_SEEDS = [int(x) for x in os.environ.get("ABLATION_SEEDS", ",".join(str(s) for s in SEEDS)).split(",")]
# Kaggle kills a session at 12 h and the packaging cell never runs. Stop launching new work
# before that so the notebook finishes cleanly with a downloadable output.
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "39600"))
START_T = time.time()
PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes")
QUIET = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}


class OutOfTime(Exception):
    """Raised when the session's time budget is spent, so the driver stops launching new work and
    the notebook still reaches its packaging cell."""


def budget_left():
    return TIME_BUDGET_S - (time.time() - START_T)


def check_budget(what):
    if budget_left() <= 0:
        raise OutOfTime(what)


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


def is_local_path(name):
    """A checkpoint directory rather than a Hugging Face id. Hub ids are `owner/model`: exactly one
    slash, no leading dot, slash or drive letter."""
    return (name.startswith(("./", "../", "/", "\\")) or "\\" in name
            or name.count("/") > 1 or (len(name) > 2 and name[1] == ":"))


def _prepare_implicit(raw, out):
    """The implicit benchmark holds out every probe text, so probe metrics stay measured on
    text no model has trained on. See dmthd/prepare_implicit.py."""
    sh(f"python -m dmthd.prepare_implicit --raw {raw} --out {out} --probes {PROBES} --download", check=False)


class Bench:
    def __init__(self, name, raw):
        self.name, self.raw, self.cfg = name, raw, DATASETS[name]
        self.data = f"{ROOT}/data/{name}"
        self.cache = f"{ROOT}/cache/{name}"
        self.runs = f"{ROOT}/runs/{name}"
        self.common = f"--scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']}"
        self.limit = f"--limit {LIMIT}" if LIMIT else ""
        self.fp = "--fp16" if GPU else ""
        self.group = f"--group_col {self.cfg['group_col']}" if self.cfg.get("group_col") else ""
        # The specialist joins the homogeneous committee of every benchmark except the one it was
        # trained on, where it would be a second copy of the task teacher.
        self.use_specialist = SPECIALIST and name != "implicit"
        self.spec = [(SPECIALIST_SRC, SPECIALIST_TAG)] if self.use_specialist else []
        self.teacher_list = pairs(TEACHERS) + self.spec + (pairs(HETERO_TEACHER) if "hetero" in COMMITTEES else [])
        self.student_list = pairs(STUDENTS) + pairs(HETERO_STUDENTS)
        self.dropped = []
        self.cache_dirty = False
        self._load_dropped()

    # ---- committees (recomputed so dropped teachers disappear everywhere) ----
    def committee(self, comm):
        homo = [t for _, t in pairs(TEACHERS) if t not in self.dropped]
        if comm == "homo":
            return homo
        if comm == "spec":
            return homo + [t for _, t in self.spec if t not in self.dropped]
        return homo + [t for _, t in pairs(HETERO_TEACHER) if t not in self.dropped]

    def active_committees(self):
        out = []
        for comm in COMMITTEES:
            tags = self.committee(comm)
            if not tags:
                print(f"committee {comm}: no teachers left, skipped", flush=True)
                continue
            if comm != "homo" and tags == self.committee("homo"):
                print(f"committee {comm}: identical to homo after drops, skipped", flush=True)
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
    WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".h5", ".msgpack", ".ckpt")

    def _copy_run_tree(self, src, dst):
        """Copy a previous session's runs/ tree without the model weights, except where a run still
        needs them (no eval_test.json yet, so its probe evaluation has not been done). Kaggle gives
        20 GB of working space and a finished grid of weights is most of that, so copying them all
        back in every session is both slow and a real risk of running out of disk.

        Teachers are the exception and always keep their weights. A student checkpoint is needed only
        until its own probe evaluation is written, but a teacher is the *source* of the logit cache,
        and any session that changes the committee deletes that cache and rebuilds it from these
        directories. Dropping teacher weights therefore looks harmless for a whole session and then
        fails the moment a new teacher is added, which is exactly what happened when the implicit
        specialist joined: four resumed teachers came back as JSON with no weights and caching died
        on the first one."""
        kept = skipped = 0
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            out = os.path.join(dst, rel) if rel != "." else dst
            is_teacher = rel.split(os.sep)[0] == "teachers"
            needs_weights = RESUME_WEIGHTS == "all" or is_teacher or (
                RESUME_WEIGHTS == "auto" and os.path.exists(os.path.join(root, "results.json"))
                and not os.path.exists(os.path.join(root, "eval_test.json")))
            os.makedirs(out, exist_ok=True)
            for f in files:
                if f.endswith(self.WEIGHT_SUFFIXES) and not needs_weights:
                    skipped += 1
                    continue
                shutil.copy2(os.path.join(root, f), os.path.join(out, f))
                kept += 1
        return kept, skipped

    def resume(self):
        """Copy previous sessions' runs/ and cache/ in. RESUME_FROM may name several sources,
        comma-separated; they are merged richest-last so the most complete session wins. Fails fast
        on a bad path: silently ignoring it would retrain every teacher, wasting hours of GPU time."""
        if not RESUME_FROM:
            return
        sources = [x.strip() for x in RESUME_FROM.split(",") if x.strip()]
        missing = [x for x in sources if not os.path.isdir(x)]
        if missing:
            sys.exit(f"RESUME_FROM path(s) do not exist: {missing}. Attach the previous notebook's output as an "
                     f"input and use its path (right panel, Input), or clear RESUME_FROM to start fresh. "
                     f"Available inputs: {glob.glob('/kaggle/input/*')}")

        def found(src_root, sub):
            # sorted+unique: `**` can yield the same directory twice, which would copy gigabytes twice
            return sorted(set(glob.glob(os.path.join(src_root, "**", sub, self.name), recursive=True)))

        def richness(src):
            return len(glob.glob(os.path.join(src, "**", "runs", self.name, "*", "*", "seed*", "results.json"), recursive=True))

        copied = []
        # The implicit specialist is a teacher *source*, not a run of this benchmark: it lives under
        # runs/implicit/specialist and would be missed by the loop below, so the tweet run would
        # retrain it and spend the same GPU-hours twice. Its weights are needed, so no filtering.
        # It holds the specialist teacher and the implicit-pretrained student, so the whole directory
        # is copied under its own names rather than one checkpoint.
        dst = f"{ROOT}/runs/implicit/specialist"
        if self.use_specialist and not glob.glob(os.path.join(dst, "*", "results.json")):
            for src_root in sources:
                hits = sorted(set(glob.glob(os.path.join(src_root, "**", "runs", "implicit", "specialist"),
                                            recursive=True)))
                hit = next((h for h in hits if glob.glob(os.path.join(h, "*", "results.json"))), None)
                if hit:
                    shutil.copytree(hit, dst, dirs_exist_ok=True)
                    have = sorted(os.path.basename(os.path.dirname(r))
                                  for r in glob.glob(os.path.join(dst, "*", "results.json")))
                    print(f"resuming implicit specialist: {hit} -> {dst} ({', '.join(have)})", flush=True)
                    copied.append(hit)
                    break
        for src_root in sorted(sources, key=richness):
            for src in found(src_root, "runs"):
                kept, skipped = self._copy_run_tree(src, f"{ROOT}/runs/{self.name}")
                print(f"resuming: {src} -> {ROOT}/runs/{self.name} ({kept} files, {skipped} weight files left behind)", flush=True)
                copied.append(src)
            for src in found(src_root, "cache"):          # the caches are what student training needs
                print(f"resuming: {src} -> {ROOT}/cache/{self.name}", flush=True)
                shutil.copytree(src, f"{ROOT}/cache/{self.name}", dirs_exist_ok=True)
                copied.append(src)
        if not copied:
            sys.exit(f"RESUME_FROM={RESUME_FROM} contains no runs/{self.name} or cache/{self.name} to resume from. "
                     f"Top level of the first source: {sorted(os.listdir(sources[0]))[:20]}. Point it at the right "
                     f"input, or clear RESUME_FROM to start fresh.")
        n_teachers = len(glob.glob(f"{self.runs}/teachers/*/results.json"))
        n_runs = len(glob.glob(f"{self.runs}/*/*/seed*/results.json"))
        try:
            free = shutil.disk_usage(ROOT).free / 2**30
            print(f"resumed: {n_teachers} finished teachers, {n_runs} finished student runs; {free:.1f} GB free", flush=True)
        except Exception:
            print(f"resumed: {n_teachers} finished teachers, {n_runs} finished student runs", flush=True)
        self._load_dropped()

    def prepare(self):
        if os.path.exists(f"{self.data}/test.csv"):
            return
        if self.name == "tweets":
            sh(f"python -m dmthd.prepare_tweets --raw {self.raw} --out {self.data}")
        elif self.name == "implicit":
            _prepare_implicit(self.raw, self.data)
        else:
            sh(f"python -m dmthd.prepare_wikipedia --raw {self.raw} --out {self.data} --download")

    def specialist(self):
        """Train the implicit-abuse specialist once, on the implicit benchmark, before the teachers
        of this benchmark are task-adapted. Failure is not fatal: the specialist is dropped and the
        run continues with the committee it can build, which is reported."""
        name, stag = self.student_list[0]
        pre = self._pretrained_student(stag)
        want_teacher = SPECIALIST_TAG not in self.dropped and not done(SPECIALIST_SRC)
        want_control = name != "bilstm" and not done(pre)
        # The two are independent: a session that resumes a finished specialist may still owe the
        # control, and vice versa. Checking them separately is what stops a resume from silently
        # skipping one of them.
        if not self.use_specialist or not (want_teacher or want_control):
            return
        check_budget("implicit specialist")
        cfg, data = DATASETS["implicit"], f"{ROOT}/data/implicit"
        if not os.path.exists(f"{data}/test.csv"):
            _prepare_implicit(IMPLICIT_RAW, data)
        if not os.path.exists(f"{data}/test.csv"):
            print("implicit corpus unavailable: specialist dropped from every committee", flush=True)
            self.dropped.append(SPECIALIST_TAG)
            self._save_dropped()
            return
        common = (f"--scheme {cfg['scheme']} --label_col {cfg['label_col']} --max_len {cfg['max_len']} "
                  f"{self.fp} {self.limit}")
        if want_teacher:
            sh(f"python -m dmthd.train_teacher --model_name {SPECIALIST_BASE} --data_dir {data} "
               f"--out_dir {SPECIALIST_SRC} --epochs {cfg['teacher_epochs']} --lr 2e-5 "
               f"--batch {cfg['teacher_batch']} {common} --no_resume", check=False)
            if not done(SPECIALIST_SRC):
                print("implicit specialist failed to train: dropped from every committee", flush=True)
                self.dropped.append(SPECIALIST_TAG)
                self._save_dropped()
            else:
                print(f"implicit specialist ready, test macro-F1 {test_f1(SPECIALIST_SRC):.4f}", flush=True)
        # The control the whole implicit claim has to survive: instead of distilling the specialist's
        # knowledge, simply train the student on the implicit corpus and then on the task. That is
        # what any practitioner would try first, it costs one small run, and if it matches the spec
        # committee then the contribution is the data and not the distillation. Better to find that
        # out here than from a reviewer.
        if want_control:
            sh(f"python -m dmthd.train_student --student {name} --data_dir {data} --out_dir {pre} "
               f"--seed {SEEDS[0]} --batch {cfg['student_batch']} --epochs {STUDENT_EPOCHS} {common} "
               f"--mode ft --tag implicit_pretrain", check=False)

    def _pretrained_student(self, stag):
        return f"{ROOT}/runs/implicit/specialist/student-{stag}"

    def _train_teacher(self, name, out, lr, fp16, epochs):
        ck = "--grad_ckpt" if "large" in name else ""
        sh(f"python -m dmthd.train_teacher --model_name {name} --data_dir {self.data} --out_dir {out} "
           f"--epochs {epochs} --lr {lr} --batch {self.cfg['teacher_batch']} {fp16} {ck} {self.common} {self.limit} --no_resume", check=False)

    def teachers(self):
        ep = os.environ.get("TEACHER_EPOCHS", self.cfg["teacher_epochs"])
        for name, tag in self.teacher_list:
            if tag in self.dropped:
                continue
            # a locally trained source checkpoint (the implicit specialist); if its own training
            # never finished there is nothing to adapt, so drop it rather than crash the grid
            if is_local_path(name) and not os.path.isdir(name):
                print(f"teacher source {name} missing: {tag} dropped from every committee", flush=True)
                self.dropped.append(tag)
                self._save_dropped()
                continue
            out = f"{self.runs}/teachers/{tag}"
            if done(out) and test_f1(out) >= MIN_TEACHER_F1:
                continue
            check_budget(f"teacher {tag}")
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
        # `ft` uses no teacher and `skd` uses only the first one, so neither depends on the committee:
        # running them per committee would repeat identical work (15 runs on the full grid)
        committee_free = [m for m in modes if m in ("ft", "skd")]
        for mode in committee_free:
            for name, stag in self.student_list:
                for seed in SEEDS:
                    out = f"{self.runs}/{stag}/{mode}/seed{seed}"
                    if not done(out):
                        check_budget(f"{stag}/{mode}/seed{seed}")
                        sh(self._student_cmd(name, out, seed, mode, self.committee("homo")))
        for comm, tags in committees:
            suffix = "" if comm == "homo" else f"_{comm}"
            for name, stag in self.student_list:
                if comm == "spec" and SPEC_STUDENTS and stag not in SPEC_STUDENTS:
                    continue
                for mode in [m for m in modes if m not in committee_free]:
                    for seed in SEEDS:
                        out = f"{self.runs}/{stag}/{mode}{suffix}/seed{seed}"
                        if not done(out):
                            check_budget(f"{stag}/{mode}{suffix}/seed{seed}")
                            sh(self._student_cmd(name, out, seed, mode, tags))
        homo = self.committee("homo")
        if not homo:
            print("no homogeneous teachers left: ablations skipped", flush=True)
            return
        name, stag = self.student_list[0]
        # tag -> (committee for this ablation, flags)
        abl = {"no_dynamic": (homo, "--mode uniform --aux --delta 0.3"),
               "no_hidden": (homo, "--mode dmthd --no_hidden --aux --delta 0.3"),
               "no_aux": (homo, "--mode dmthd"),
               "per_batch": (homo, "--mode dmthd --per_batch --aux --delta 0.3"),
               "from_scratch": (homo, "--mode dmthd --aux --delta 0.3 --from_scratch")}
        # Distil from the implicit specialist and nobody else. If this matches the `spec` committee,
        # the contribution is a teacher choice and not a committee, and the paper must say so. The
        # other half of the question, the committee without the specialist, is the `homo` committee
        # itself, measured over all three seeds rather than as a one-seed ablation.
        if self.use_specialist and SPECIALIST_TAG not in self.dropped and done(SPECIALIST_SRC):
            abl["spec_only"] = ([SPECIALIST_TAG], "--mode skd --aux --delta 0.3")
        for tag, (tags, flags) in abl.items():
            if tag == "from_scratch" and name == "bilstm":
                continue
            for seed in ABLATION_SEEDS:
                out = f"{self.runs}/{stag}/ablation_{tag}/seed{seed}"
                if not done(out):
                    check_budget(f"{stag}/ablation_{tag}/seed{seed}")
                    sh(f"python -m dmthd.train_student --student {name} --data_dir {self.data} --out_dir {out} --seed {seed} "
                       f"--batch {self.cfg['student_batch']} --epochs {STUDENT_EPOCHS} {self.fp} {self.common} {self.limit} "
                       f"--cache {self.cache} --teachers {' '.join(tags)} {flags} --tag {tag}")
        # The data-versus-distillation control: same student, initialised from its own fine-tune on
        # the implicit corpus, then fine-tuned on this task with no teachers at all. If this matches
        # D-MTHD, the implicit knowledge came from the data and the committee is decoration.
        src = self._pretrained_student(stag)
        if self.use_specialist and done(src):
            for seed in ABLATION_SEEDS:
                out = f"{self.runs}/{stag}/ablation_implicit_pretrain/seed{seed}"
                if not done(out):
                    check_budget(f"{stag}/ablation_implicit_pretrain/seed{seed}")
                    sh(f"python -m dmthd.train_student --student {src} --data_dir {self.data} --out_dir {out} --seed {seed} "
                       f"--batch {self.cfg['student_batch']} --epochs {STUDENT_EPOCHS} {self.fp} {self.common} "
                       f"{self.limit} --mode ft --tag implicit_pretrain")

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
                   f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --probe_neg {neg} "
                   f"--probe_pos {pos} {self.group}", check=False)
        # Sarcasm-discrimination AUC for every mode of the headline student. Recall at a fixed
        # threshold cannot tell "misses indirect abuse" from "sees it but cannot separate it from
        # harmless sarcasm"; this is threshold-free, so it is the number the implicit claim rests on.
        _, stag = self.student_list[0]
        for d in sorted(glob.glob(f"{self.runs}/{stag}/*/seed{SEEDS[0]}")):
            if done(d) and not os.path.exists(os.path.join(d, "implicit_analysis", "implicit_analysis.json")):
                sh(f"python -m dmthd.implicit_analysis --model_dir {d} --test {self.data}/test.csv "
                   f"--probes {PROBES} --scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']} "
                   f"--max_len {self.cfg['max_len']} --out {d}/implicit_analysis", check=False)

    def sweep(self):
        homo = self.committee("homo")
        if not homo:
            return
        # cheap diagnostic first: how far from uniform are the per-instance weights at each tau?
        if not os.path.exists(f"{self.cache}/tau_diagnostic.csv"):
            sh(f"python -m dmthd.tune_tau --cache {self.cache} --data_dir {self.data} --scheme {self.cfg['scheme']} "
               f"--label_col {self.cfg['label_col']}", check=False)
        # and the mechanism question the committee exists to answer: does the weighting send
        # implication to the specialist, or does it average over everyone?
        if not os.path.exists(f"{self.runs}/weight_routing/weight_routing.json"):
            rel = "soft" if self.cfg["soft"] else "hard"   # must match what the students trained with
            sh(f"python -m dmthd.weight_routing --cache {self.cache} --data_dir {self.data} "
               f"--scheme {self.cfg['scheme']} --label_col {self.cfg['label_col']} "
               f"--reliability {rel} --out {self.runs}/weight_routing", check=False)
        name, stag = self.student_list[0]
        # tau: with frozen teachers the weights are a fixed function of the data, and at tau = 1 the
        # observed means sit within 0.03 of uniform, so the sweep must reach much sharper values or
        # the dynamic weighting cannot be told apart from uniform averaging
        grid = [("tau", v, f"--tau {v}") for v in (0.05, 0.1, 0.2, 0.5)] + [("T", v, f"--T {v}") for v in (1.0, 2.0, 8.0)] + \
               [("alpha", v, f"--alpha {v} --beta {round(0.8 - v, 2)}") for v in (0.2, 0.6)] + [("delta", v, f"--delta {v}") for v in (0.1, 0.5)]
        for param, value, flags in grid:
            out = f"{self.runs}/{stag}/sweep_{param}_{value}/seed{SEEDS[0]}"
            if not done(out):
                check_budget(f"sweep {param}={value}")
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
            # out-of-domain sets written by the data preparation: the same task and the same label
            # space, annotated by other people from another platform. Same scheme, so this is a
            # straight evaluation rather than a transfer with a label mapping.
            for ood in sorted(glob.glob(f"{self.data}/test_ood_*.csv")):
                tag = os.path.basename(ood)[:-4]
                out = os.path.join(d, f"eval_{tag}.json")
                if not os.path.exists(out):
                    sh(f"python -m dmthd.evaluate --model_dir {d} --csv {ood} --scheme {self.cfg['scheme']} "
                       f"--label_col {self.cfg['label_col']} --max_len {self.cfg['max_len']} --out {out}", check=False)
            # transfer to every other benchmark that has been prepared, not just one. The pair that
            # matters most is tweets -> implicit: it asks whether a model trained on a corpus that
            # never labels implication detects it at all.
            for other in (k for k in DATASETS if k != self.name):
                other_cfg, other_test = DATASETS[other], f"{ROOT}/data/{other}/test.csv"
                if not os.path.exists(other_test) or os.path.exists(os.path.join(d, f"transfer_{other}.json")):
                    continue
                focus = "--focus_class implicit_hate" if other_cfg["scheme"] == "implicit3" else ""
                sh(f"python -m dmthd.transfer_eval --model_dir {d} --model_scheme {self.cfg['scheme']} --csv {other_test} "
                   f"--csv_scheme {other_cfg['scheme']} --csv_label_col {other_cfg['label_col']} "
                   f"--max_len {other_cfg['max_len']} {focus}", check=False)

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
        out = f"{self.runs}/bench_{dev}.csv"
        # A finished bench is not recomputed. The directories it measures are the ones whose weights
        # a resume is entitled to drop, so re-running here on a resumed session would time a subset
        # and overwrite a complete table with a shorter one. Delete the csv to force a rebuild.
        if os.path.exists(out):
            print(f"bench: {out} already exists, keeping it", flush=True)
            return
        sh(f"python -m dmthd.bench --model_dirs {' '.join(dirs)} --csv {self.data}/test.csv --device {dev} "
           f"--max_len {self.cfg['max_len']} --num_labels {self.cfg['num_labels']} --out {out}", check=False)

    def aggregate(self):
        sh(f"python -m dmthd.aggregate --runs {self.runs} --out {self.runs}/summary.csv", check=False)
        _, stag = self.student_list[0]
        # against the no-teacher control: does distillation help at all?
        for cand in ("dmthd", "uniform", "skd", "dmthd_spec", "uniform_spec", "dmthd_hetero", "dmthd_dis"):
            if os.path.isdir(f"{self.runs}/{stag}/{cand}") and os.path.isdir(f"{self.runs}/{stag}/ft"):
                sh(f"python -m dmthd.aggregate --compare {self.runs}/{stag}/ft {self.runs}/{stag}/{cand}", check=False)
        # and the comparisons that isolate one component at a time. `spec` against `homo` is the
        # specialist's contribution; `dmthd` against `uniform` is the weighting's; the two controls
        # ask whether the committee or the distillation is doing the work.
        for base, cand in (("uniform", "dmthd"), ("dmthd", "dmthd_spec"), ("uniform", "uniform_spec"),
                           ("dmthd_spec", "ablation_spec_only"), ("dmthd_spec", "ablation_implicit_pretrain"),
                           ("dmthd", "ablation_implicit_pretrain")):
            if os.path.isdir(f"{self.runs}/{stag}/{base}") and os.path.isdir(f"{self.runs}/{stag}/{cand}"):
                sh(f"python -m dmthd.aggregate --compare {self.runs}/{stag}/{base} {self.runs}/{stag}/{cand}", check=False)
        if self.dropped:
            print(f"NOTE: teachers dropped as collapsed: {self.dropped} (see {self.runs}/dropped_teachers.json)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tweets", choices=list(DATASETS))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONPATH", "src")
    raw = a.raw or {"tweets": "/kaggle/working/raw/cyberbullying_tweets.csv",
                "wikipedia": f"{ROOT}/raw_wikipedia", "implicit": IMPLICIT_RAW}[a.dataset]
    b = Bench(a.dataset, raw)
    b.resume()
    order = ["prepare", "specialist", "teachers", "cache", "students", "probes", "sweep", "robustness", "quant", "bench", "aggregate"]
    stages = order if a.stage == "all" else [a.stage]
    ran_out = None
    for st in stages:
        try:
            getattr(b, "cache_teachers" if st == "cache" else st)()
        except OutOfTime as e:
            ran_out = f"{st}: {e}"
            print(f"\nTIME BUDGET SPENT ({TIME_BUDGET_S / 3600:.1f} h) while starting {ran_out}", flush=True)
            break
    if ran_out:
        print("finishing the cheap reporting stages so this session leaves a usable output", flush=True)
        for st in ("probes", "aggregate"):
            try:
                getattr(b, st)()
            except OutOfTime:
                pass
        print("\nPARTIAL RUN: attach this version's output to the next one and run again; finished work is skipped.", flush=True)
    print(f"elapsed {(time.time() - START_T) / 3600:.2f} h", flush=True)
