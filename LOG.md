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
- **Wikipedia BERT-mini fine-tune-only, three seeds complete**: test macro-F1 0.8857 / 0.8861 / 0.8935,
  mean 0.8884 +- 0.0044; attack-class F1 0.796 / 0.797 / 0.812; ROC-AUC 0.976; ECE 0.014-0.027.
  The compact student clears the classical floor (0.8759) by 0.013 and sits 0.005 below the Phase-2
  BERT-large teacher (0.8936). The contrast with tweets is now the paper's framing: on Wikipedia the
  11M student is already near teacher level, so distillation has little room; on tweets it is 2 points
  behind the teachers, which is where D-MTHD has to show its value. Seed 3 is 0.007 above the other
  two, so Wikipedia seed variance is larger than the two-seed estimate suggested (0.0044, not 0.0004).
- **Idle-CPU measurement block** (nothing else running, 8 threads, 50 warm-up batches, median of 5):
  - *Efficiency, BERT-mini*: 11.17M parameters, 42.6 MB on disk, 0.81 GFLOP per 128-token sequence.
    CPU latency 3.64 ms at batch 1 and 2.76 ms per sample at batch 32 (tweets, 128 tokens);
    6.30 and 5.82 ms on Wikipedia at 256 tokens.
  - *INT8 dynamic quantisation*: tweets 0.8779 -> 0.8722 (-0.0057), Wikipedia 0.8857 -> 0.8826
    (-0.0031); size 42.6 -> 33.5 MB in both cases; batch-32 latency 2.42 -> 1.39 ms (1.74x) and
    5.82 -> 3.71 ms (1.57x), but batch-1 latency gets slightly *worse* on tweets (4.35 -> 4.67 ms).
    The modest size drop is explained by the model shape: 7.8M of BERT-mini's 11.2M parameters are
    the 30,522 x 256 embedding table, which dynamic quantisation does not touch. Expect a larger
    gain on DistilBERT, where the encoder dominates; that contrast is the point of the table.
  - *Probes on Wikipedia models* (both probe sets are tweet-domain): benign-sarcasm FPR 0.036 and
    ironic-abuse recall 0.059. The model almost never fires on tweet-style text, so these numbers
    measure domain shift, not sarcasm awareness. **Probe metrics are therefore reported only for
    tweet-trained models**; the Wikipedia numbers go in the analysis as evidence of domain shift.
  - *Obfuscation* (bullying-class items only, benign untouched): tweets 0.8779 -> leet 0.7791,
    swap 0.7518, space 0.7488, mixed 0.7247 (a 10-15 point collapse). Wikipedia 0.8857 -> leet
    0.8649, swap 0.8613, mixed 0.8667, but **space 0.8977, above the clean score**: splitting words
    inside attack comments makes them easier to spot in a corpus whose benign text is tidy wiki
    prose. Reported as it stands, with the caveat that synthetic obfuscation is not real evasion.
    The tweets mixed figure reproduces the earlier standalone run exactly (0.7247).
  - *Cross-dataset transfer, both directions*: tweets model on Wikipedia gives binary macro-F1 0.273
    (ROC-AUC 0.562) and calls 80.9% of comments bullying against a true rate of 11.8%; Wikipedia
    model on tweets gives 0.425 (ROC-AUC 0.718) and calls 33.7% bullying against a true rate of
    85.7%. Neither direction transfers: one over-fires, the other under-fires. This is a finding
    about the two task definitions, not a defect of the students.
- **Kaggle tweets version 2 hit the 12-hour ceiling** (43,200 s, exit 137) with a 14.13 GB output.
  Work was saved, but the notebook's packaging cell never ran, so no small archive was produced.
  Three changes follow:
  1. `TIME_BUDGET_S` (default 39,600 s = 11 h): the driver refuses to *start* new work once the
     budget is spent, raises OutOfTime, then still runs the cheap reporting stages and exits 0, so
     the packaging cell always runs. Tested at zero budget (stops before the first teacher) and at
     a normal budget (trains, then finishes).
  2. `ABLATION_SEEDS` (set to 1 in the notebook): the five ablations run on one seed instead of
     three, saving about four GPU-hours. To be stated in Limitations.
  3. The packaging cell now writes `dmthd_<dataset>_results.tgz` containing only metrics, histories,
     predictions and the generated tables (tens of MB) instead of tarring the whole 14 GB tree;
     model weights stay in the Kaggle output for the next version to resume from.
- **Resume now merges every attached output.** RESUME_FROM accepts a comma-separated list and copies
  the sources richest-last (ordered by how many finished student runs each holds), so attaching the
  wrong or an extra previous notebook is harmless. The notebooks build the list automatically from
  `/kaggle/input/**/runs/<dataset>`. Verified: two fake sources merge to the richer count, and a
  single bad path in the list still exits with the list of what is missing.
- **Resume made disk-safe.** The version-3 log appeared to copy `runs/tweets` twice, but the timings
  show a single 182 s copy: Kaggle echoes each stdout line twice in its log viewer. The de-duplicating
  `sorted(set(...))` stays as a cheap guard. The real problem is disk: copying a
  finished grid of checkpoints back in costs about 14 GB of Kaggle's 20 GB working space. Resume now
  leaves model weights behind unless a run still needs them, i.e. it has `results.json` but no
  `eval_test.json`, so its probe evaluation is still pending (`RESUME_WEIGHTS=auto`, overridable with
  `all` or `none`). Caches are always copied in full, since student training reads them. Free disk is
  printed after the copy. Verified on a fixture: the probed run arrives without its checkpoint, the
  unprobed one keeps it.
- **Kaggle tweets version 3 resumed cleanly**: 4 finished teachers and **78 finished student runs**
  recovered from version 2 (which had used its whole 12-hour session). The grid is 120 runs with
  one-seed ablations, so about 42 remain, roughly six hours at the observed pace of ~9 minutes per
  run; the 11-hour budget should now cover the students, sweeps, robustness, quantisation, benchmark
  and aggregation. Version 3 is training `distilbert/skd_hetero/seed1`, i.e. it is already in the
  heterogeneous-committee block. First epoch of that run: validation macro-F1 0.8802, and the single
  teacher correctly carries weight 1.0.
- **Wasted work found in the version-3 log and fixed**: `skd_hetero` trains exactly the same model as
  `skd`, because single-teacher distillation uses only the committee's first teacher (BERT-large) and
  the heterogeneous committee differs only by appending DeBERTa. Both `ft` and `skd` are now run once,
  outside the committee loop; on the full grid that removes 15 redundant runs (about two GPU-hours).
  The already-finished `skd_hetero` runs stay on disk and are simply ignored by the tables.
- **DeBERTa-v3-base recovered.** Version 3 resumed four finished teachers and did not retrain any, so
  the collapsed DeBERTa teacher was successfully retrained in fp32 during version 2 and now scores
  above the 0.5 floor; it is in the heterogeneous committee, and the uniform run shows the expected
  0.25 weight on each of the four teachers.
- **DistilBERT with single-teacher distillation, three seeds**: test macro-F1 0.8963 / 0.8961 / 0.8965
  (mean 0.8963, spread 0.0004), against BERT-large's 0.8973. A 66M student is within 0.001 of its
  335M teacher. Per-class, the two hard classes remain the bottleneck: not_cyberbullying 0.75 and
  other_cyberbullying 0.78 against 0.91-0.99 elsewhere.

## 2026-09-12

- **The dynamic weighting is currently indistinguishable from uniform averaging.** In every
  heterogeneous-committee D-MTHD run the epoch-mean weights are 0.254 / 0.259 / 0.254 / 0.233 against
  a uniform 0.25, and they are *identical across epochs and across seeds*. That is not a bug: the
  teachers are frozen and cached, so w_k(i) = softmax(-CE(p_k(i), y_i)/tau) is a fixed function of
  the data. The weighting is per-instance, never per-epoch, and the paper must say so plainly.
  The problem is tau: at tau = 1 the weights sit within 0.03 of uniform, so D-MTHD and the uniform
  baseline optimise almost the same objective, which is exactly what the results show
  (DistilBERT: uniform 0.8976 vs D-MTHD 0.8966 on the heterogeneous committee).
  Added `src/dmthd/tune_tau.py`: it reads a cache and reports, per tau, the mean top weight, the
  entropy ratio, the share of decisive instances and which teacher wins. On a synthetic three-teacher
  cache with clearly different teacher quality it gives entropy ratio 0.94 at tau = 1 (nearly
  uniform) against 0.50 at tau = 0.1, confirming the diagnosis. The sweep grid now runs
  tau in {0.05, 0.1, 0.2, 0.5} and the driver runs the diagnostic before sweeping.
- **Heterogeneous-committee results, tweets** (three seeds each, test macro-F1):
  DistilBERT skd 0.8963 / uniform+DeBERTa 0.8976 / D-MTHD+DeBERTa 0.8966;
  DeBERTa-v3-xsmall skd 0.8865 / uniform 0.8847 / D-MTHD 0.8835 (two seeds so far).
  The heterogeneous *student* is both weaker and far slower to train (about 1,500 s per run against
  650 s for DistilBERT), which is the first evidence for the homogeneity claim, though confounded by
  DeBERTa-v3-xsmall simply being a weaker model on this task.
- **Why indirect abuse is missed: measured, and the usual explanation is wrong.** Ran
  `src/dmthd/implicit_analysis.py` on BERT-mini fine-tune-only (tweets, seed 1).
  - *Not lexical.* Recall on the ironic-abuse probe is 0.72 on the 43 items containing explicit
    profanity and 0.67 on the 1,517 without; on the implicit-abuse probe it is 0.45 with and 0.69
    without. The model is not a profanity detector: it sees indirect abuse about as well as explicit.
  - *The failure is discrimination, not detection.* Mean p(bullying) is 0.69 on ironic abuse against
    0.37 on benign sarcasm, giving a sarcasm-discrimination ROC-AUC of 0.776 (ironic abuse as
    positives, benign sarcasm as negatives). The distributions overlap heavily: at the 0.5 cut,
    recall 0.70 comes with a 32% false-positive rate on harmless sarcasm, and forcing false positives
    below 10% costs more than half the recall (threshold 0.9, recall 0.43). The model reacts to
    "this text is sarcastic and negative", not to "this text attacks someone".
  - *On the benchmark itself*, the same story: `other_cyberbullying` recall 0.79 (0.82 with
    profanity, 0.78 without), and its errors go mostly to `not_cyberbullying` (77), while
    `not_cyberbullying` recall is 0.68 with 133 of its errors going to `other_cyberbullying`. The two
    catch-all classes bleed into each other, which is a label-quality problem as much as a model one.
  - *Consequence for the paper*: **sarcasm-discrimination AUC is the metric that matches the claim**,
    and it is threshold-free, so it separates "cannot see it" from "cannot tell it apart". Added to
    `implicit_analysis.py`; the fine-tune-only baseline scores 0.776 and every distilled variant will
    be measured against it. Recall at a fixed 0.5 threshold should not be the headline number.
