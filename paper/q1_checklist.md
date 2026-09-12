# Q1 readiness checklist

Status: `[x]` done, `[~]` in progress, `[ ]` not started, `[!]` blocked on a person or on GPU results.
Every `[x]` names where the evidence lives.

## A. Data and evaluation integrity
- [x] Primary corpus de-duplicated, conflicting labels removed, counts reported — `data/tweets/report.json`, LOG 2026-09-10
- [x] Second benchmark with an official split and leakage assertions — `data/wikipedia/report.json`
- [x] Third benchmark in which abuse-by-implication is a label, not a hidden subset of a catch-all class — `data/implicit/report.json`, `prepare_implicit.py`, LOG 2026-09-12
- [x] Probe texts held out of the implicit benchmark so probe metrics stay uncontaminated (1,581 rows removed, reported) — `data/implicit/report.json`
- [x] Cross-corpus annotation disagreement measured: of 629 texts the two implicit corpora share, they agree on 99.7% about hate and 48.2% about whether it is implied — `python -m dmthd.corpus_agreement`
- [x] Source confound ruled out by construction: the splits come from one corpus, because the union is separable at 0.91 macro-F1 and 95% of implicit examples come from one side — `prepare_implicit.py --corpora`, LOG 2026-09-12
- [x] Out-of-domain test set (27,096 ISHate rows) held fully out of training — `data/implicit/test_ood_ishate.csv`
- [x] Every score on a multi-source benchmark also reported within each source — `evaluate.py --group_col`
- [x] Teacher provenance rule written and enforced (no test set overlaps any teacher's training data) — `paper/setup_draft.md`, `build_probes.py`
- [x] Classical floor on every benchmark — tweets 0.8798, Wikipedia 0.8759, implicit 0.5620 (implicit_hate F1 0.5562, implicit-discrimination AUC 0.7610)
- [~] Sarcasm probe sets: built and screened; human verification pending — `probes/`, `annotation/`
- [!] Human-verified sarcastic-bullying set with Fleiss' kappa — needs the four annotators (sheet sent 2026-09-11)
- [x] Obfuscation variants on both corpora, fine-tune-only and D-MTHD — tweets 0.8394 to 0.7116 mixed, and distillation does not help (D-MTHD 0.8385 to 0.7077); Wikipedia drops 2 points with the space variant rising
- [x] Cross-dataset transfer, both directions, full test sets — tweets->Wikipedia 0.273 (over-fires), Wikipedia->tweets 0.425 (under-fires); reported as a finding about the task definitions
- [~] Hyper-parameter sweeps — T, alpha, delta done and flat (spread 0.0016, 0.0008, 0.0010). tau done only at 0.5/2/5, where the weights are already uniform, so the sweep cannot yet answer the question it exists for; 0.05/0.1/0.2 running
- [x] Threshold-free sarcasm metric defined and implemented, replacing recall at an arbitrary cut — sarcasm-discrimination AUC, `implicit_analysis.py`; fine-tune-only baseline 0.776

## B. Baselines, controls, ablations
- [x] Fine-tune-only control for every student, three seeds — all five on tweets: BERT-mini 0.8393, BERT-small 0.8469, DistilBERT 0.8907, DeBERTa-v3-xsmall 0.8825, BiLSTM 0.8693. Wikipedia pending
- [x] Single-teacher KD, uniform-average multi-teacher, D-MTHD, three seeds, five students, three committees — 120 runs, `paper/tables/main.csv`. Distillation helps all five (+0.0012 to +0.0071); D-MTHD beats uniform on none
- [x] Ablations: no dynamic weights, no hidden term, no irony head, per-batch, from-scratch — `paper/tables/ablations.csv`. Only from-scratch moves anything (-0.0473); removing the weighting *improves* the score (+0.0008)
- [!] Implicit-specialist ablations: `spec_only` (the specialist alone) and `no_spec` (the committee without it). Without both, the claim that the *committee* helps is unfalsifiable — driver ready, pending Kaggle
- [!] Routing evidence: mean teacher weight on implicit rows minus explicit rows, bootstrap interval — `weight_routing.py` ready and tested, pending a real committee cache
- [!] Focus-class transfer: tweets model scored on implicit rows alone — `transfer_eval.py --focus_class`, pending Kaggle
- [~] Homogeneity 2x2 — the heterogeneous committee arm is done for all five students (effects -0.0005 to +0.0032, all inside noise) and the hidden-term arm has run only on BERT-mini (+0.0009). The DeBERTa and BiLSTM hidden-term cells are outstanding, and without them no causal claim about homogeneity is available
- [!] Disagreement-aware variant on Wikipedia (`dmthd_dis`) — Kaggle Wikipedia run
- [ ] Optional: zero-shot LLM baseline on a 2,000-item test sample

- [x] Threshold-free sarcasm metric validated across the whole grid — 114 models, false-positive rate and recall correlate at +0.673 with their difference standard deviation 0.053; no method discriminates better than any other — `python -m dmthd.tradeoff`
- [x] The paper's own method reported as a negative result where the evidence says so — `paper/results_draft.md` Sections 5.3 and 5.3a
- [!] Reproducibility caveat unresolved: the same code, data and seeds give 0.8779 on a laptop CPU and 0.8393 on a Kaggle T4. Precision, data and code all eliminated. Every table is restricted to one environment; one Kaggle CPU run would isolate it

## C. Statistics and reporting
- [x] Mean ± std over seeds and paired bootstrap 95% intervals implemented — `aggregate.py`
- [x] Wilcoxon across seeds when five seeds exist, otherwise bootstrap only — `aggregate.py --compare`
- [x] Per-class F1, ECE, ROC-AUC/PR-AUC for binary — `evaluate.py`
- [~] Per-agreement-band F1 on Wikipedia (0.2–0.8 band vs the rest) — `analysis.py agreement_bands` ready and tested; needs Wikipedia runs
- [x] Teacher complementarity — kappa 0.889-0.922, disagreement 6.4-9.2%, error overlap 0.673-0.730, oracle macro-F1 bound 0.9469 against best single 0.8973. This is the mechanism behind the negative result
- [x] Weight trajectories per teacher per epoch — 0.306 / 0.311 / 0.305 / 0.233, identical at every epoch, which is correct by construction with frozen teachers and is reported as such rather than plotted as a trend
- [x] Macro-F1 vs latency vs parameters — `paper/tables/efficiency.csv`; DistilBERT matches BERT-large at 5.1x fewer parameters and 5.8x lower batch-32 latency

## D. Efficiency
- [x] Benchmark with warm-up, five repeats, median, batch 1 and 32, GPU and CPU — `bench.py`
- [x] INT8 dynamic quantisation measured on an idle CPU, both corpora — 1.74x / 1.57x at batch 32 for a 0.006 / 0.003 macro-F1 cost
- [x] Parameter/FLOP/latency/F1 Pareto data — `runs/tweets/pareto.csv` from the finished grid; figure regenerates from it

## E. Method presentation
- [x] Loss fully specified with per-instance weights, T^2 KL, projections, soft term, irony head, disagreement variant — `losses.py` docstring, `paper/setup_draft.md`
- [x] Notation table and algorithm box — `paper/method_draft.md` (Sections 3.1, 3.5)
- [x] Posterior interpretation of softmax(-error/tau) — `paper/method_draft.md` Section 3.2
- [x] Homogeneity defined once (architecture family, not tokenizer) — `paper/method_draft.md` Section 3.6

## F. Related work and references
- [~] Thematic related work: cyberbullying/toxicity; KD for LMs; multi-teacher/adaptive KD; annotator disagreement — skeleton with placed citations in `paper/related_work_draft.md`; prose expansion by the writing lead
- [x] Verified bibliography: 63 entries, 47 with Crossref-verified DOIs, 16 arXiv/proceedings entries without DOI by design, no anonymous, duplicate or mismatched entries — `paper/references.bib`, `paper/refs_report.csv`
- [~] All Phase-2 citation faults mapped to keep/correct/delete with replacement keys — `paper/phase2_citation_map.md`; the writing lead applies it to the thesis text

- [x] Related-work theme on abuse carried by implication, with its own gap statement (2.1a) - `paper/related_work_draft.md`
- [x] Every cited key resolves against the verified bibliography; 63 entries, 63 cited, 0 dangling - `python paper/check_cites.py paper`
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
