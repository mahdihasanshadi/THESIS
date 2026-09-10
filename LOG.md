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
