# Q1 readiness checklist

Status: `[x]` done, `[~]` in progress, `[ ]` not started, `[!]` blocked on a person or on GPU results.
Every `[x]` names where the evidence lives.

## A. Data and evaluation integrity
- [x] Primary corpus de-duplicated, conflicting labels removed, counts reported — `data/tweets/report.json`, LOG 2026-09-10
- [x] Second benchmark with an official split and leakage assertions — `data/wikipedia/report.json`
- [x] Teacher provenance rule written and enforced (no test set overlaps any teacher's training data) — `paper/setup_draft.md`, `build_probes.py`
- [x] Classical floor on every benchmark — TF-IDF+LR results.json under `runs/*/tfidf_lr`
- [~] Sarcasm probe sets: built and screened; human verification pending — `probes/`, `annotation/`
- [!] Human-verified sarcastic-bullying set with Fleiss' kappa — needs the four annotators (sheet sent 2026-09-11)
- [x] Obfuscation test variants (leet, swap, space, mixed) on both corpora, fine-tune-only baseline measured — tweets drops 10-15 points, Wikipedia 2 points (space variant rises); D-MTHD comparison pending Kaggle
- [x] Cross-dataset transfer, both directions, full test sets — tweets->Wikipedia 0.273 (over-fires), Wikipedia->tweets 0.425 (under-fires); reported as a finding about the task definitions
- [~] Hyper-parameter sweeps (tau, T, alpha/beta, delta) on validation — driver stage `sweep`, pending Kaggle

## B. Baselines, controls, ablations
- [~] Fine-tune-only control for every student, three seeds — BERT-mini done on both corpora (tweets 0.8770 ± 0.0010, Wikipedia 0.8884 ± 0.0044); BERT-small, DistilBERT, DeBERTa-xsmall, BiLSTM on Kaggle
- [~] Single-teacher KD, uniform-average multi-teacher, D-MTHD, three seeds, three students — teachers done on Kaggle (BERT-large 0.897, irony 0.893, HateBERT 0.889; stop rule passed); students pending the next Kaggle version
- [!] Ablations: no dynamic weights, no hidden term, no irony head, per-batch, from-scratch — Kaggle run 1/2
- [!] Homogeneity 2x2: BERT-mini vs DeBERTa-v3-xsmall student x hidden term; heterogeneous committee with DeBERTa-v3-base; BiLSTM student — Kaggle run 2 (driver ready)
- [!] Disagreement-aware variant on Wikipedia (`dmthd_dis`) — Kaggle Wikipedia run
- [ ] Optional: zero-shot LLM baseline on a 2,000-item test sample

## C. Statistics and reporting
- [x] Mean ± std over seeds and paired bootstrap 95% intervals implemented — `aggregate.py`
- [x] Wilcoxon across seeds when five seeds exist, otherwise bootstrap only — `aggregate.py --compare`
- [x] Per-class F1, ECE, ROC-AUC/PR-AUC for binary — `evaluate.py`
- [~] Per-agreement-band F1 on Wikipedia (0.2–0.8 band vs the rest) — `analysis.py agreement_bands` ready and tested; needs Wikipedia runs
- [~] Teacher complementarity: pairwise error overlap, Cohen's kappa, oracle-ensemble bound — `analysis.py complementarity` ready; needs Kaggle teachers
- [~] Weight trajectories per teacher per epoch — `analysis.py weights` ready; needs D-MTHD runs
- [~] Macro-F1 vs latency vs parameters table for the Pareto figure — `analysis.py pareto` ready; needs bench output

## D. Efficiency
- [x] Benchmark with warm-up, five repeats, median, batch 1 and 32, GPU and CPU — `bench.py`
- [x] INT8 dynamic quantisation measured on an idle CPU, both corpora — 1.74x / 1.57x at batch 32 for a 0.006 / 0.003 macro-F1 cost
- [~] Parameter/FLOP/latency/F1 Pareto figure — `figures.py pareto` tested on smoke output; needs Kaggle bench

## E. Method presentation
- [x] Loss fully specified with per-instance weights, T^2 KL, projections, soft term, irony head, disagreement variant — `losses.py` docstring, `paper/setup_draft.md`
- [x] Notation table and algorithm box — `paper/method_draft.md` (Sections 3.1, 3.5)
- [x] Posterior interpretation of softmax(-error/tau) — `paper/method_draft.md` Section 3.2
- [x] Homogeneity defined once (architecture family, not tokenizer) — `paper/method_draft.md` Section 3.6

## F. Related work and references
- [~] Thematic related work: cyberbullying/toxicity; KD for LMs; multi-teacher/adaptive KD; annotator disagreement — skeleton with placed citations in `paper/related_work_draft.md`; prose expansion by the writing lead
- [x] Verified bibliography: 63 entries, 47 with Crossref-verified DOIs, 16 arXiv/proceedings entries without DOI by design, no anonymous, duplicate or mismatched entries — `paper/references.bib`, `paper/refs_report.csv`
- [~] All Phase-2 citation faults mapped to keep/correct/delete with replacement keys — `paper/phase2_citation_map.md`; the writing lead applies it to the thesis text

## G. Reproducibility and ethics
- [x] Public code with fixed seeds and one-command drivers — github.com/mahdihasanshadi/THESIS
- [x] Lab notebook — `LOG.md`
- [ ] Checkpoints and cached teacher outputs released (Kaggle dataset or Zenodo)
- [~] Data statement; ethics statement; reproducibility statement; limitations — drafts in `paper/statements_draft.md`
- [ ] CRediT author contributions — template in `paper/statements_draft.md`, names to fill

## H. Claims discipline
- [ ] No "student exceeds teachers" claim unless seed statistics support it
- [ ] "Robust" in the title only if section A's robustness items are done
- [ ] "Homogeneous" in the abstract only as a measured finding
