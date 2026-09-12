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
- **Third benchmark built: indirect abuse becomes a label, not an inference-time probe.** Agreed with
  Mahdi that the paper's stated goal is untestable while the only two corpora bury implication inside
  a catch-all class (tweets) or a binary attack flag (Wikipedia). Added
  `src/dmthd/prepare_implicit.py` and the label scheme `implicit3 = not_hate | explicit_hate |
  implicit_hate`, from ISHate (`BenjaminOcampo/ISHate`) and Implicit Hate Corpus stage 1
  (`tasksource/implicit-hate-stg1`).
  Two schema traps found and fixed while building it, both of which would have produced a silently
  wrong corpus:
  - ISHate records the benign class **only** in `hateful_layer`; its `implicit_layer` holds
    {Explicit HS, Implicit HS} and is empty for every Non-HS row. Reading `implicit_layer` alone, as
    the first draft did, throws away all 17,869 benign examples. The loader now reads both columns.
  - The SALT-NLP `ImplicitHate` mirror ships **stage 2 only** (6,346 rows, no benign class). Stage 1,
    with all three classes and 21,480 rows, is at `tasksource/implicit-hate-stg1`.
  Exclusions, each reported in `report.json`: ISHate augmentations (`aug_method` other than `orig`,
  and `source == augmented` for the val/test files which carry no `aug_method` column) 63,758 -> 29,116;
  ToxiGen-sourced rows 29,116 -> 28,763 (provenance rule); then probe holdout.
  **Probe holdout.** The sarcasm probes were built from these same corpora, so training on them would
  contaminate every probe number in the paper, including the ones already reported. Every text
  occurring in any probe file is removed from this benchmark in all three splits: 1,581 rows
  (1,572 implicit_hate, 9 explicit_hate). The probes stay genuinely unseen, which is worth more than
  the rows.
  **Result** (SUPERSEDED the same day by the correction entry below, which rebuilds this from one
  corpus after finding the pooled version separable by source):
  50,243 combined -> 47,181 after cleaning; train 37,744 / val 4,718 / test 4,719.
  Train classes: not_hate 24,586, explicit_hate 8,385, implicit_hate 4,773. Corpus mix in train:
  ISHate 21,767, Implicit Hate 15,977. De-duplication removed 420 duplicates and, notably,
  **346 texts that the two corpora label differently** (710 rows) - a cross-corpus annotation
  disagreement worth reporting in its own right.
- **Classical floor on the implicit benchmark** (SUPERSEDED: these are the pooled-corpus numbers;
  the corrected floor is in the entry below)**, and the reason this benchmark is the right one.**
  TF-IDF + logistic regression: test macro-F1 **0.6809**, accuracy 0.7663, ECE 0.0449. Per class:
  not_hate 0.8421, explicit_hate 0.7474, **implicit_hate 0.4530**.
  A bag of n-grams is already decent at explicit abuse and collapses on implication - a 29-point F1
  gap inside one corpus, measured, with no model of ours involved. Compare the tweet corpus, where the
  same floor reaches 0.8798 and the fine-tune-only student cannot beat it (0.8770). This is the
  headroom the method needs, and it is exactly where the paper claims to contribute.
- **Implicit-abuse specialist teacher added to the committee.** New driver stage `specialist`:
  HateBERT is trained on the implicit benchmark first, then task-adapted onto the target benchmark
  like any other teacher (the classification head is re-initialised, `ignore_mismatched_sizes`).
  It joins the homogeneous committee of the tweets and Wikipedia benchmarks and is the only member
  that has ever seen abuse-by-implication labelled as such; on the implicit benchmark itself it is
  omitted, where it would duplicate the task teacher. `SPECIALIST=0` turns it off. Failure is not
  fatal: a missing corpus or a failed training run drops it from every committee and the grid
  continues, recorded in `dropped_teachers.json`.
- **Measurements wired into the driver rather than run by hand.** `implicit_analysis` now runs for
  every mode of the headline student, so sarcasm-discrimination AUC is produced for ft / skd /
  uniform / D-MTHD without anyone remembering to do it. Cross-corpus transfer now covers every other
  prepared benchmark instead of one, so **tweets -> implicit** is measured: a `--focus_class`
  option scores the implicit_hate rows on their own against the benign rows, because a collapsed
  transfer score hides precisely the case the paper is about.
- **Scheme plumbing.** `implicit3` added to `label_names`; `not_bullying_index` now resolves the
  benign class per scheme (`BENIGN`) instead of assuming `not_cyberbullying`, which also fixed
  `obfuscate.py`, where a non-tweet scheme would have obfuscated the benign rows as well.
  `implicit_analysis.py` takes its hard classes from the scheme.
- **The specialist forms its own committee (`spec`), rather than joining `homo`.** Adding a fourth
  teacher to the homogeneous committee would have changed what `homo` means, so every finished
  three-teacher run would have had to be repeated to stay comparable - about ten GPU-hours - and the
  specialist's contribution would then only be visible as a one-seed ablation. As a separate
  committee it is instead a controlled three-seed comparison of `homo` against `spec` on the same
  students, and nothing already finished is invalidated. `SPEC_STUDENTS` restricts it to the headline
  student by default, because the question is about the committee rather than the student. The
  `no_spec` ablation is therefore dropped: it is the `homo` run, measured over three seeds instead of
  one. Driver docstring, `aggregate --compare` list and `tables.py` mode labels updated with it.
- **Two controls added for the implicit claim.** `ablation_spec_only` distils from the specialist
  alone; `ablation_implicit_pretrain` takes the same student, fine-tuned on the implicit corpus and
  then on the task, with no teachers at all. The second is the one a reviewer asks for first: if it
  matches the `spec` committee then the implicit knowledge came from the data and the committee is
  decoration. Both are trained in the `specialist` stage and resumed across Kaggle sessions with it.
- **Tables extended to the metrics the claim is argued on.** `tables.py` now collects F1 on whichever
  class holds abuse-by-implication in each corpus, sarcasm-discrimination AUC, the operating point at
  0.5, the recall left when the false-positive rate is held under 10 per cent, and focus-class
  transfer. Two new tables: `implicit` and `routing`. Verified on the existing BERT-mini
  fine-tune-only run, which renders as macro-F1 0.8770, other_cyberbullying F1 0.741,
  sarcasm-discrimination AUC 0.776, recall 0.704 at FPR 0.317, recall 0.427 at FPR under 0.10. That
  row is the baseline every later variant is measured against.
- **`weight_routing.py` added.** For each tau it reports, per teacher, the mean weight on
  implicit-like rows minus explicit-like rows with a 2,000-sample percentile bootstrap interval. This
  is the mechanism evidence: a committee of specialists is only a committee if the weights go
  somewhere sensible. Tested on the smoke cache; at tau = 1 the contrasts are 0.01-0.04 and at
  tau = 0.1 they are 0.11-0.27, which is Finding 6 measured on routing rather than on entropy.
- **`paper/OPEN_DECISIONS.md` written**, at Mahdi's request that crucial decisions wait for him with
  the options and the logic set out. Six: what the paper is about; how the remaining GPU quota is
  spent (the only one needed now); whether "Homogeneous" and "Robust" stay in the title; what the
  paper says if the dynamic weighting turns out to do nothing; whether and how to release the
  implicit benchmark; and the target venue.
- **Corpus manifests published (`paper/manifests/`).** None of the three corpora may be re-hosted, so
  "reproducible" would otherwise rest on a build script nobody can check. `src/dmthd/manifest.py`
  writes one row per example - split, label, source corpus, SHA-1 of the normalised text - plus the
  split fingerprints, and `--verify` compares a rebuild row by row and names the rows that differ.
  No text, so nothing is redistributed. 2.7 MB compressed for all three corpora; round-trip verified
  (MATCH, 47,181 rows at the time) on the implicit corpus. Data and reproducibility statements updated.
- **ISHate's subtlety layer is not usable.** It looked like a free second axis (Subtle vs
  Non-Subtle), but in our test split only 10 of 4,719 rows carry it and all 10 are `explicit_hate`,
  while the implicit rows are almost entirely Non-Subtle. Whatever it annotates is not the
  implicit/explicit distinction, and there is not enough of it to measure anything. The column is
  carried for provenance and used by nothing; recorded as F14b so nobody rediscovers it and assumes
  it works.
- **CORRECTION, same day: the implicit benchmark is rebuilt from one corpus, not two.** Before
  training anything on the pooled corpus built earlier today, I checked whether its label is
  separable by source, because implicit and explicit examples come from opposite sides of it. It is.
  - A TF-IDF classifier predicts which corpus a text came from at **0.91 macro-F1**, and the class
    mix is lopsided: 95 per cent of implicit examples come from the Implicit Hate Corpus, 89 per cent
    of explicit examples from ISHate. A model trained on the pool can score well on
    implicit-versus-explicit by recognising the source and never read an implication.
  - It is not hypothetical. Scored on *identical* Implicit Hate test rows, a classical model trained
    on the pool gets implicit-hate F1 **0.473**; trained on that corpus alone it gets **0.548**
    (macro-F1 0.521 against 0.571). The pooled corpus scores higher in aggregate, 0.681 against
    0.562, precisely because the aggregate rewards the shortcut. Pooling bought a bigger headline
    number and cost 0.075 F1 on the only class the paper is about.
  - `prepare_implicit.py --corpora` now selects which corpora form the splits, default
    `ImplicitHate`. Whatever is left over is written as an out-of-domain test set instead of being
    discarded, with every text that also occurs in the splits removed first.
  **The benchmark as it now stands**: 20,637 rows from the Implicit Hate Corpus, split 16,509 /
  2,064 / 2,064; training classes not_hate 10,616, implicit_hate 5,029, explicit_hate 864. ISHate is
  held out entirely: `test_ood_ishate.csv`, 27,096 rows (17,691 benign, 9,404 explicit, 1 implicit)
  after dropping 613 that overlap the splits. Out-of-domain *implicit* recall is measured by the
  implicit-abuse probe, which is those ISHate implicit rows; the out-of-domain set measures whether a
  model trained to find implication starts firing on ordinary text when the domain changes.
  **Classical floor, corrected**: macro-F1 **0.5620**, accuracy 0.6972, ECE 0.0707; not_hate 0.7966,
  implicit_hate **0.5562**, explicit_hate 0.3333 (108 test rows, so that one is noise).
  **Headline metric added**: implicit-discrimination AUC, the threshold-free ranking of implied hate
  against ordinary text, 629 positives and 1,327 negatives. The floor scores **0.7610** (AP 0.6159).
  It sits almost exactly where the tweet student's sarcasm-discrimination AUC sits (0.776): the same
  difficulty, measured twice on different data. Macro-F1 is not the headline on this benchmark,
  because a three-class average over one large easy class and one 108-row class says little about
  implication.
- **The two corpora agree about hate and disagree about implication** (`dmthd.corpus_agreement`).
  They share 629 texts. On whether a text is hateful at all they agree on **99.7 per cent**. On
  whether the hate is stated or implied they agree on **48.2 per cent**: of the 624 both call
  hateful, the Implicit Hate Corpus labels essentially all implicit while ISHate labels 324 explicit
  and 300 implicit. The overlap is not a random sample of either corpus, so this is a comparison of
  two independent annotations of the same texts rather than a corpus-wide agreement estimate, and it
  is reported that way. It is the second reason not to pool them, it belongs beside our own
  annotation study, and it bounds how sharp any implicit-hate result on either corpus can be.
  This replaces the "346 conflicting texts" number in the earlier entry, which was a de-duplication
  count over the pooled corpus and not a measurement of annotator agreement.
- **Per-source reporting is now automatic** (`evaluate.py --group_col`, `baseline_tfidf.py
  --group_col`, on by default for the implicit benchmark). The source confound was found by looking;
  it should not have needed looking. Any benchmark assembled from more than one source can be gamed
  by style, and reporting the within-source numbers beside the aggregate every time is the cheapest
  guard against it.
- Documents corrected for the new numbers: `DECISIONS.md` (F13 and F14 rewritten with the superseded
  figures kept, F16 and Decisions 17 and 18 added), `paper/setup_draft.md`, `paper/q1_checklist.md`,
  `README.md`, and the implicit manifest regenerated (20,637 rows, new split fingerprints).
- **Bug found by re-reading `prepare_implicit.py`: the out-of-domain set's conflicting-label filter
  never fired.** Duplicates were dropped before conflicts were counted, so every key was unique by
  construction and `nunique()` was always 1. Fixed to count conflicts first. It was catching nothing
  and should have been catching 14 texts that ISHate labels two ways; those rows would have entered
  the out-of-domain test set with whichever label happened to come first. The set is 27,096 rows, not
  27,110. The train, validation and test splits are unaffected and their manifest still verifies
  (MATCH, 20,637 rows).
- **Driver smoke test, five passes, run against the committed code.** Every assertion passed: the
  full grid; every teacher collapsing, being retrained once and dropped; the implicit benchmark end
  to end under the `implicit3` scheme; the specialist trained on the implicit corpus and re-headed
  from three classes to six for the tweet committee; and the `spec` committee against `homo`. The
  checks that matter: `runs/tweets/tiny/dmthd_spec/seed1` lists teachers
  `[tiny-a, mini-b, implicit-spec]` while `runs/tweets/tiny/dmthd/seed1` lists `[tiny-a, mini-b]`,
  so the specialist reaches the committee it should and leaves untouched the one whose finished runs
  must stay valid. The cache carries the specialist; the adapted teacher has six classes; the
  implicit-pretrained control has three. The ablation log line `w_implicit-spec: 1.0` confirms
  `spec_only` distils from the specialist and nobody else.
  One bug, in the test script rather than the driver: the closing summary re-read
  `dropped_teachers.json` after pass 4 had cleared `runs/tweets`, so the script raised at its last
  line having already passed everything. The content is now captured when it is asserted. Re-running
  from scratch to confirm a clean exit.

## 2026-09-13

- **The tweet grid is finished: 120 student runs, 8.24 h, and the central claim does not survive it.**
  Kaggle version 4 completed the full comparison. Numbers below are test macro-F1, mean over three
  seeds, read from `summary_fixed.csv` after correcting the aggregation bug described further down.
  The classical floor is 0.8798 and the best teacher is BERT-large at 0.8973.

  | Student | params | ft | skd | uniform | D-MTHD | uniform+het | D-MTHD+het |
  |---|---|---|---|---|---|---|---|
  | BERT-mini | 11.2M | 0.8393 | 0.8405 | 0.8385 | 0.8378 | 0.8378 | 0.8373 |
  | BERT-small | 28.8M | 0.8469 | 0.8488 | 0.8505 | 0.8471 | 0.8509 | 0.8477 |
  | DistilBERT | 67.0M | 0.8907 | 0.8963 | 0.8960 | 0.8960 | 0.8975 | 0.8966 |
  | DeBERTa-v3-xsmall | 70.8M | 0.8825 | 0.8865 | 0.8862 | 0.8833 | 0.8847 | 0.8828 |
  | BiLSTM | 10.4M | 0.8693 | 0.8748 | 0.8761 | 0.8732 | 0.8760 | 0.8764 |

  **Distillation helps, and it is the only thing that does.** Every student gains from having a
  teacher: +0.0012 (BERT-mini), +0.0040 (BERT-small), +0.0068 (DistilBERT), +0.0040
  (DeBERTa-v3-xsmall), +0.0071 (BiLSTM).

  **D-MTHD does not beat uniform averaging on any student.** The differences are -0.0007, -0.0034,
  +0.0000, -0.0029 and -0.0029. Four negative, one tie, none positive. The best configuration is
  uniform or single-teacher for four of the five students; the only student where a D-MTHD variant
  wins is the BiLSTM, by 0.0004 over uniform with the heterogeneous committee, which is noise.

  **On the headline student nothing beats the no-teacher control at all**, and the paired bootstrap
  says so: ft versus D-MTHD -0.0015 [-0.0076, +0.0048]; versus uniform -0.0007 [-0.0074, +0.0058];
  versus single-teacher +0.0012 [-0.0057, +0.0079]; versus D-MTHD with the heterogeneous committee
  -0.0019 [-0.0082, +0.0041]. Every interval contains zero.

  This is Decision 4 of `paper/OPEN_DECISIONS.md` coming true, written down before the result arrived
  precisely so the response would not be improvised afterwards.

- **The tau sweep that ran was the old grid and cannot settle the weighting question.** The finished
  version tried tau = 0.5, 2.0 and 5.0, giving 0.8390, 0.8390 and 0.8389, and mean weights of
  0.330 / 0.340 / 0.330 at the sharpest of them. All three are effectively uniform, so all three
  score the same, which is arithmetic rather than evidence. The values that could make a difference,
  0.05, 0.1 and 0.2, are in the current grid and have not run. **Until they do, the correct statement
  is that D-MTHD has not been shown to differ from uniform averaging at any tau yet tested**, which
  is weaker than "it does not work" and stronger than nothing.

- **Everything else in the sweep is flat too.** T: 0.8386 / 0.8402 / 0.8391 at 1, 2 and 8.
  alpha 0.2 / 0.6: 0.8386 / 0.8394. delta 0.1 / 0.5: 0.8389 / 0.8399. Ablations on BERT-mini, one
  seed: no_dynamic 0.8386, per_batch 0.8389, no_hidden 0.8369, no_aux 0.8364, against full D-MTHD at
  0.8378. Removing the dynamic weighting *improves* the score by 0.0008. Only `from_scratch` moves
  anything, and it moves it down by 0.047, which says pre-training matters and nothing else does.

- **`skd_hetero` is bit-identical to `skd` across all five students**, as expected: single-teacher
  distillation uses the committee's first teacher, so the committee it nominally belongs to makes no
  difference. The redundancy was removed from the driver earlier; these runs predate that and are
  kept as a consistency check on the resume logic rather than deleted.

- **BUG: aggregate was pooling two different experiments into one row.** It grouped on the `mode`
  field inside results.json, which records the objective and not the committee, so `dmthd` and
  `dmthd_hetero` landed together and their means were silently averaged, as did `uniform` with
  `uniform_hetero` and `skd` with `skd_hetero`. Every "6 seeds" row in the first summary was two
  experiments of three seeds each. Fixed to group on the run directory, which is what distinguishes
  them; `tables.py` was already correct because it reads the directory name. The table above is from
  the corrected aggregation.

- **fp16 is exonerated as the cause of the BERT-mini discrepancy.** BERT-mini fine-tune-only scores
  0.8779 on this laptop and 0.8393 on Kaggle under a nominally identical configuration, and mixed
  precision was the obvious suspect. `scripts/fp16_check.py` trains the same student with and without
  it on the same seeds. Seed 1 tracks almost exactly: validation macro-F1 0.7993 / 0.8276 / 0.8423 /
  0.8499 with fp16 against 0.8008 / 0.8299 / 0.8438 / 0.8505 without, a difference of roughly 0.0015
  which is noise. The hypothesis was wrong.

- **The two runs are scoring on the same test set, so the data is not the explanation either.**
  Compared the gold-label sequence recorded in `test_labels.npy` by the local run against the one
  from Kaggle: 4,326 rows, identical row for row. The split is reproducible across environments,
  which is worth knowing in its own right.

- **What is left is the training itself, and the local history is missing a column the current code
  writes.** Local: validation 0.8431 / 0.8658 / 0.8750 / 0.8818 / 0.8833 / 0.8839, training CE 0.874
  falling to 0.208. Kaggle: 0.7993 / 0.8276 / 0.8423 / 0.8499 / 0.8487 / 0.8537, CE 1.019 falling to
  0.297. The Kaggle model is behind from the first epoch and stays behind, with a systematically
  higher training loss, which is a model learning more slowly rather than one generalising worse. The
  local `history.csv` has no `nan_batches` column, so those runs were produced by a version of
  `train_student.py` from before the NaN-guard commit. A re-run of the current code on this laptop,
  same seed and configuration, is under way and will say whether the code changed or the environment
  did. **Until it finishes, the local BERT-mini numbers and the Kaggle BERT-mini numbers must not
  appear in the same table.**

- **Driver smoke test passed on the committed code**, all five passes, exit 0: the full grid; every
  teacher collapsing and being retrained once then dropped; the implicit benchmark end to end under
  the `implicit3` scheme; the specialist re-headed from three classes to six for the tweet committee;
  and the `spec` committee running while `homo` stays clean.
- **Across the whole finished grid, no method discriminates sarcastic abuse better than any other.**
  114 evaluated models, teachers and students, every mode and seed. The false-positive rate on benign
  sarcasm and the recall on ironic abuse correlate at **+0.673**; recall minus false-positive rate has
  mean 0.352 and standard deviation 0.053 while its components range over 0.24-0.59 and 0.59-0.75.
  The parts move a lot, the difference barely moves. By student: DeBERTa-v3-xsmall 0.394, teachers
  0.382, DistilBERT 0.369, BERT-mini 0.368, BERT-small 0.333, BiLSTM 0.287; distillation mode does not
  appear in the ordering at all. Lowest margin in the grid is the randomly initialised student at
  0.082. Added `src/dmthd/tradeoff.py`. This is the strongest evidence yet that recall at a fixed
  threshold measures willingness to fire rather than understanding, and it is measured across a
  hundred models rather than one.
- **The BERT-mini discrepancy is the environment, and the rule it imposes matters more than the
  cause.** Three candidates tested, three eliminated. Not mixed precision: fp16 and fp32 track to
  within 0.0015 epoch by epoch on the same hardware. Not the data: `test_labels.npy` matches row for
  row across both environments, all 4,326 rows. Not the code: re-running the current trainer here
  with the same seed reproduces the old local run to four decimals, epoch 1 validation 0.8431 and
  loss 0.8736 against 0.8431 and 0.8736. What is left is the machine, CPU here against a T4 there.
  Both seed clusters are tight (0.8779 / 0.8770 / 0.8760 against 0.8394 / 0.8395 / 0.8389), so a
  four-point gap between them is systematic, and the Kaggle loss is higher at every epoch, so it is
  slower learning rather than worse generalisation.
  **Rule adopted: every number in a table comes from one environment.** The Kaggle grid is internally
  consistent so its comparisons stand; the local BERT-mini and BERT-small numbers are withdrawn
  unless re-run there. One Kaggle run with `GPU=0` would say whether it is the accelerator or the
  software stack, about thirty minutes, and is worth doing but blocks nothing.
