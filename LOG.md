# Lab notebook

One entry per step: date, what was run, result, decision. Newest at the bottom. Numbers here are
copied from `results.json` / `report.json` files, never typed from memory. Times are Dhaka local.

## 2026-09-10

- **Review of Phase-2 report.** Findings: method nearly identical to MT-BERT; single seed, one
  dataset, no no-distillation control, no efficiency numbers; reference list with duplicates
  ([8]=[18], [10]=[11]), an anonymous entry [3], a mismatched "Luo et al." paragraph citing MiniLLM [13],
  Passalis et al. [33] cited for the opposite of its content, three venue/URL mismatches
  ([26], [27], [28]), fake-news papers used for KD claims ([31], [34], [36]). Decision: two-week plan,
  new student family, cross-task teachers, sarcasm probes. Plan documents: twelve-week artifact
  page and `D-MTHD_Phase3_Two-Week_Plan.docx`.
- **Phase-3 outputs inspected (team Drive folder).** Tweet corpus, six classes. DistilBERT
  no-distillation baseline 0.859 macro-F1 beat all three base-size teachers (0.819–0.828) and the
  D-MTHD student (0.830). Validation accuracy 0.94 vs test 0.86. Test set had 84 exact duplicates,
  57 with conflicting labels. Dynamic weights constant at about [0.08, 0.64, 0.28]. Latency numbers
  inconsistent (identical architectures differing 2.5x). Decision: rebuild data, larger teachers,
  smaller students, corrected benchmark.
- **Codebase written and smoke-tested** (`src/dmthd`, Kaggle driver, notebook). Repo:
  github.com/mahdihasanshadi/THESIS (public). No local GPU (AMD RX 6600, no CUDA): GPU work on Kaggle.
- **Tweet corpus prepared** (`prepare_tweets`, seed 42): 47,692 in; 758 dropped under two tokens;
  1,563 texts with conflicting labels (3,168 rows) dropped; 507 duplicates dropped; 43,259 out;
  train/val/test 34,607 / 4,326 / 4,326. `E:/dmthd-work/data/tweets/report.json`.
- **Classical floor, tweets** (`baseline_tfidf`): macro-F1 0.8798, acc 0.8925; per-class F1 age 0.98,
  ethnicity 0.98, religion 0.95, gender 0.91, other 0.75, not_cyberbullying 0.70.
- **Probe sets built** (`build_probes`): benign sarcasm 1,064 (iSarcasmEval), ironic abuse 1,560
  (797 Implicit Hate irony + 763 ISHate implicit, sources toxigen/ihc excluded), implicit abuse 763.
  Classical floor on probes: benign-sarcasm bullying rate 0.444 (0.331 on the screened set),
  ironic-abuse recall 0.599, implicit-abuse recall 0.611.
- **Probe screening** (`screen_probes`, threshold 0.8): 181 of 1,064 flagged for human review,
  883 kept unflagged.
- **Wikipedia corpus prepared** (`prepare_wikipedia`, Figshare files): 115,864 comments, median 10
  annotators; 522 dropped under three tokens; duplicates removed train 450 / dev 82 / test 114;
  dev/test overlap with train removed 199 / 202; test overlap with dev 42; train/val/test
  68,750 / 22,782 / 22,721; attack share 11.7 / 11.9 / 11.8%; 24.6% in the 0.2–0.8 agreement band.
- **Classical floor, Wikipedia**: macro-F1 0.8759, acc 0.9473, ROC-AUC 0.968, PR-AUC 0.868.
- **BERT-mini fine-tune-only, tweets, seed 1** (CPU): test macro-F1 0.8779, acc 0.8909; benign-sarcasm
  FPR 0.284 (screened set, n=883), ironic-abuse recall 0.673, ECE 0.039. Seeds 2–3 running.
- **Kaggle tweet run started by the user** (notebook `kaggle/dmthd_tweets.ipynb`, first driver
  version: homogeneous committee, three students, four modes, three seeds, ablations, probes, bench).
- **Novelty decisions.** "Homogeneous" is not a novelty claim; it becomes a measured finding via a
  2x2 (student family x hidden-state term) plus a heterogeneous teacher. Added to the plan (user
  agreed): disagreement-aware distillation, human-verified sarcastic-bullying test set with kappa,
  obfuscation and cross-dataset robustness, INT8 deployment table, posterior interpretation of
  the per-instance weights. Bengali extension goes to the thesis, not the Q1 paper.

## 2026-09-11

- **Implemented and smoke-tested** (`smoke2`, 18 stages passed): disagreement-aware loss
  (`--disagreement --kappa --reliability soft`), BiLSTM heterogeneous student (`--student bilstm`,
  10.4M params), `quantize_eval` (INT8), comparison-grid driver (`COMMITTEES=homo,hetero`,
  `HETERO_STUDENTS=deberta-xsmall,bilstm`, `dmthd_dis` on Wikipedia, `quant` stage), annotation kit
  (`annotation/guideline.md`, 300-row sheet = 181 flagged + 119 random, `compute_kappa.py`).
  Note: INT8 timing on the 4M-param smoke model was slower than fp32 while another job used the
  CPU; the deployment table must be measured on an idle CPU with BERT-mini/DistilBERT.
- **Epoch-level checkpoint resume** added to `train_teacher` and `train_student` (`ckpt_last.pt`,
  removed on success); verified on tiny runs. `RESUME_THESIS_JOBS.cmd` on the Desktop restarts
  local jobs after a power cut. Commits up to `6e9b824`.
- **BERT-mini fine-tune-only, tweets, seed 2** at epoch 4: val macro-F1 0.882 (running).
- **Bibliography verified against Crossref** (`paper/verify_refs.py`, `paper/refs.json`, 63 entries):
  38 matched by title search with DOI, 6 by known DOI checked through Crossref works/{doi},
  16 without DOI by design (arXiv-only, NeurIPS/ICLR/PMLR proceedings), 0 unresolved after fixes.
  Two Phase-2 errors surfaced: the IJCV survey DOI ends in `-z` (the report had `-5`), and the
  Zou et al. ESWA paper is volume 263, article 125599, DOI 10.1016/j.eswa.2024.125599; the
  volume 237 / 121393 DOI in the Phase-2 bibliography resolves to an unrelated pipe-jacking paper.
  Output: `paper/references.bib`, `paper/refs_report.csv`.
- **BERT-mini fine-tune-only, tweets, seed 2**: test macro-F1 0.8770, acc 0.8904. Seed 3 running.
- **Bibliography re-verified with retries**: 38 title-matched DOIs, 9 known DOIs confirmed through
  works/{doi}, 16 no-DOI-by-design, 0 unresolved (`paper/refs_report.csv`).
- **Analysis tools added** (`src/dmthd/analysis.py`): teacher complementarity (pairwise kappa, error
  overlap, oracle bound), per-agreement-band F1, dynamic-weight trajectories, Pareto join; all four
  exercised on the smoke-test outputs.
- **Robustness tooling** (`obfuscate.py`, `transfer_eval.py`, driver stages `sweep` and `robustness`).
  Tweets test obfuscated (leet/swap/space/mixed; 3,599–3,659 of 3,709 bullying items edited).
  BERT-mini fine-tune-only seed 1 on the mixed variant: macro-F1 0.7247 (clean 0.8779), a 15-point drop.
  Tweets model applied to Wikipedia (500-row check): binary macro-F1 0.273, ROC-AUC 0.511, i.e.
  chance; the six topic-defined classes do not transfer to personal attacks. To be reported as the
  domain-shift finding, not hidden.
- **Paper drafts written**: `paper/related_work_draft.md` (four themes, every citation from the
  verified list, gap statements), `paper/statements_draft.md` (ethics, data, reproducibility,
  limitations, CRediT template), `paper/method_draft.md` (notation table, per-instance reliability
  with its Bayesian-model-averaging reading, full objective, disagreement-aware variant, irony head,
  algorithm box, homogeneity definition).
- **Figures and reporting**: `figures.py` (Pareto, weight trajectories, agreement bands, complementarity
  heat-map, robustness drops), tested on smoke outputs; Wilcoxon added to `aggregate.py --compare`
  for five-seed comparisons; `paper/phase2_citation_map.md` maps all 36 Phase-2 references to
  keep / correct / delete with verified replacement keys.
- **BERT-mini fine-tune-only, tweets, three seeds complete** (`runs/tweets/bert-mini/ft`): test macro-F1
  0.8779 / 0.8770 / 0.8760, mean 0.8770 ± 0.0010; accuracy 0.8902. Against the classical floor
  (0.8798) the paired bootstrap on seed 1 gives −0.0018, 95% CI [−0.0108, +0.0070]: the compact
  student alone does not beat TF-IDF+LR. This is the headroom that distillation must fill; it is
  reported as such. Probes (screened benign set n=883; ironic abuse n=1,560): FPR 0.284 / 0.343 /
  0.251, recall 0.673 / 0.675 / 0.579; seed variance on the probes is large, so probe metrics are
  reported as mean ± std over seeds, never from one seed.
- **Wikipedia BERT-mini fine-tune-only** started automatically on the CPU (seed 1 of 3).
- **Wikipedia BERT-mini fine-tune-only, seed 1** (CPU, early stop at epoch 5): test macro-F1 0.8857,
  acc 0.9555, attack-class F1 0.796, ECE 0.014; above the classical floor (0.8759) and 0.008 below
  the Phase-2 BERT-large teacher (0.8936). Seeds 2–3 running.
- **Kaggle versions #1 and #2 (user's account) failed at data preparation**: the dataset file was not at
  `/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv` (not attached to the committed
  version, or a different path), and `!python` failures do not fail a Kaggle notebook, so both showed
  "Successful" after two minutes. Fix: the notebook now locates the CSV under `/kaggle/input` by name,
  downloads it from the Hugging Face mirror when absent, and raises so a failed benchmark shows as Failed.
- **Power cut (2026-09-11, morning).** State on return: tweets seeds 1–3 finished; Wikipedia seed 1
  finished; seed 2 had a checkpoint after epoch 2 (val macro-F1 0.8859); seed 3 not started. Local
  jobs relaunched with the detached wrappers: finished seeds skipped, seed 2 resumed from its
  checkpoint. The Kaggle run is unaffected by a local power cut.
- **Kaggle version #3 (tweets, new notebook): teachers trained on the T4.** Test macro-F1: BERT-large
  0.8973 (acc 0.9075; best val 0.9054 at epoch 3), HateBERT 0.8887 (val 0.8973), irony-RoBERTa adapted
  0.8926 (val 0.9065). All three beat the classical floor (0.8798) and the BERT-mini control
  (0.8770): **the Day-3 stop rule passes**, distillation has headroom. Per-class pattern identical
  across teachers: not_cyberbullying 0.73–0.76, other_cyberbullying 0.77–0.78, the rest 0.91–0.98.
- **DeBERTa-v3-base teacher collapsed** (loss NaN from epoch 1, test macro-F1 0.052, train-acc 0.18 in
  the cache). Autocast was already off for DeBERTa; most likely cause is the checkpoint being loaded
  in half precision by the newer transformers default. Fix: `load_classifier` now forces fp32
  weights; both training loops skip non-finite batches and report the count; the driver treats a
  teacher below MIN_TEACHER_F1 = 0.5 as collapsed, retrains it once (lr 1e-5, fp32) and drops it from
  every committee if still collapsed, rebuilding the cache.
- **Driver bug**: the run crashed entering the students stage (`'list' object is not callable`):
  the instance attribute `self.students` shadowed the `students()` method. Renamed to
  `student_list`. Root cause of it reaching Kaggle: the smoke test never exercised the driver.
  Added `scripts/smoke_driver.py` (full tiny grid + collapse-handling pass); it runs before push.
  The version's output still holds the three good teachers and their cache, reusable via RESUME_FROM.
- **Driver smoke test passed** (`scripts/smoke_driver.py`, CPU, 200 rows): ten stages of the full
  grid (two homogeneous tiny teachers + DeBERTa-v3-xsmall heterogeneous teacher, BERT-tiny and
  BiLSTM students, homo and hetero committees, five ablations, probes, ten sweeps, robustness,
  INT8, bench, aggregate), then a collapse-handling pass with MIN_TEACHER_F1=0.99 in which all
  three teachers were retrained once, dropped, recorded in `dropped_teachers.json`, and the
  students stage still ran the fine-tune-only control without error. DeBERTa-v3-xsmall trained
  with a finite loss (1.79) under the fp32 fix, confirming the NaN cause.
- **Cache/split integrity check added.** Teacher caches are indexed by training-row position, so
  pairing a cache with a different split would train on silently mismatched targets. `cache_teachers`
  now records a SHA-256 fingerprint of the training split (texts and labels, in order) in `meta.json`,
  and `train_student` refuses to start when it does not match. Verified both ways: the matching split
  trains, a different split exits with the mismatch message. This matters for the Kaggle resume, where
  the cache comes from an earlier session and the split is rebuilt from scratch.
- **Resume path**: Kaggle mounts a notebook output at `/kaggle/input/notebooks/<user>/<slug>`, not
  `/kaggle/input/<slug>`, so the first resume attempt exited immediately on the new guard (30 s lost
  instead of 1.5 h of retraining). Both notebooks now detect the path themselves by globbing
  `/kaggle/input/**/runs/<dataset>`; nothing has to be typed. Verified against a mock mount.
- **Wikipedia BERT-mini fine-tune-only, seed 2** (resumed after the power cut, early stop at epoch 5):
  test macro-F1 0.8861, acc 0.9558, attack-class F1 0.797. Seed 1 was 0.8857; the two agree to
  0.0004. Seed 3 running.
- **Paper tables generated from the runs tree** (`src/dmthd/tables.py`): teachers with the classical
  floor, main table (student x method, mean +- std, probe metrics), ablations with deltas,
  homogeneity (hidden-term gain by student family), committee (homogeneous vs heterogeneous),
  efficiency with INT8, robustness (obfuscation and transfer), sweeps, dataset counts. LaTeX with
  booktabs plus CSV; every number read from results.json / eval_*.json / bench_*.csv. Tables whose
  data is missing are skipped, so it runs mid-experiment. Fixture test `scripts/smoke_tables.py`
  builds a synthetic 111-run tree and asserts all nine tables render.
- **Annotation kit handed to the team** (`annotation/`): `make_sheets.py` produces one workbook per
  annotator (`sheets/annotator_A1..A4.xlsx` plus CSV twins) with the 300 items in fixed order, a
  drop-down restricting `label` to the four allowed values, a guideline worksheet, and no model
  scores so the classifier cannot steer the judgements. `merge_sheets.py` merges the returned files
  by id, reporting per-person completion and rejecting invalid labels. Round trip tested with
  simulated returns (three workbooks, one CSV, some rows skipped). Fixed the majority rule in
  `compute_kappa.py`: a majority is now counted over the votes cast on that item, so one person
  skipping a row no longer sends an otherwise unanimous item to adjudication; unit-checked on
  hand-made vote patterns (unanimous, 3-1, 2-2 tie, two skips, single vote, unsure-ignored).
