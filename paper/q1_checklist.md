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
- [x] Implicit-abuse specialist trained and above the floor — HateBERT on `implicit3` scores 0.6029 macro-F1 and 0.8247 implicit-discrimination AUC against the 0.5620 / 0.7610 floor; task-adapted to tweets it scores 0.8931, inside a 0.008 band with the three existing teachers
- [x] Data-versus-distillation control trained — BERT-mini fine-tuned on the implicit corpus scores 0.5549 macro-F1 (*below* the 0.5620 floor) and 0.8196 AUC (well above 0.7610): it ranks implication and cannot threshold it, which is the benchmark-grounded form of the Section 5.7 argument
- [x] Classical floor on every benchmark — tweets 0.8798, Wikipedia 0.8759, implicit 0.5620 (implicit_hate F1 0.5562, implicit-discrimination AUC 0.7610)
- [~] Sarcasm probe sets: built and screened; human verification pending — `probes/`, `annotation/`
- [!] Human-verified sarcastic-bullying set with Fleiss' kappa — needs the four annotators (sheet sent 2026-09-11)
- [~] Obfuscation variants on both corpora, fine-tune-only and D-MTHD — tweets 0.8394 to 0.7116 mixed, and distillation does not help (D-MTHD 0.8385 to 0.7077). The Wikipedia figures (2 points, space variant rising) are from the laptop and wait for the Kaggle Wikipedia grid (DECISIONS F26)
- [~] Cross-dataset transfer, full test sets — tweets->implicit on Kaggle: implicit-hate AUC 0.600 against 0.820 in-domain, over-firing at 81% predicted abusive against 35.7% true (F25). Tweets<->Wikipedia (0.273 / 0.425) measured on the laptop only; waits for the Kaggle Wikipedia grid (F26)
- [x] Hyper-parameter sweeps — T, alpha, delta flat (spread 0.0016, 0.0008, 0.0010); tau from 0.05 to 5 flat too, 0.8385 to 0.8392, with the weights sharpening to 0.318 / 0.356 / 0.326 at 0.05 (F22)
- [x] Threshold-free sarcasm metric defined and implemented, replacing recall at an arbitrary cut — sarcasm-discrimination AUC, `implicit_analysis.py`; fine-tune-only baseline 0.773 on Kaggle (the 0.776 quoted before was the laptop model, F26)

## B. Baselines, controls, ablations
- [x] Fine-tune-only control for every student, three seeds — all five on tweets: BERT-mini 0.8393, BERT-small 0.8469, DistilBERT 0.8907, DeBERTa-v3-xsmall 0.8825, BiLSTM 0.8693. Wikipedia pending
- [x] Single-teacher KD, uniform-average multi-teacher, D-MTHD, three seeds, five students, three committees — 120 runs, `paper/tables/main.csv`. Distillation raises all five (+0.0012 to +0.0071) with no paired interval excluding zero (F27); D-MTHD beats uniform on none
- [x] Ablations: no dynamic weights, no hidden term, no irony head, per-batch, from-scratch — `paper/tables/ablations.csv`. Only from-scratch moves anything (-0.0473); removing the weighting *improves* the score (+0.0008)
- [x] Implicit-specialist committee and controls, tweets, Kaggle v4 — `spec` committee three seeds, `spec_only`, and the implicit-pretrained student; the committee without the specialist is `homo` itself. None helps: +0.0005 / -0.0005 against `homo`, AUC 0.768 against 0.773 (F23)
- [~] Routing evidence: mean teacher weight on implicit-like rows minus explicit-like rows, bootstrap interval — measured on Kaggle over the pooled five-teacher cache: the specialist gains a twentieth of the uniform weight at the trained tau, HateBERT slightly more (F24). Per-committee measurement queued for the next tweet session
- [x] Focus-class transfer: tweets model scored on implicit rows alone — implicit-hate AUC 0.600 (fine-tune-only) and 0.600 (D-MTHD), recall 0.85 at FPR 0.79 (F25)
- [~] Homogeneity 2x2 — the heterogeneous committee arm is done for all five students (effects -0.0005 to +0.0032, all inside noise) and the hidden-term arm has run only on BERT-mini (+0.0009). The DeBERTa and BiLSTM hidden-term cells are outstanding, and without them no causal claim about homogeneity is available
- [!] Disagreement-aware variant on Wikipedia (`dmthd_dis`) — Kaggle Wikipedia run
- [ ] Optional: zero-shot LLM baseline on a 2,000-item test sample

- [x] Threshold-free sarcasm metric validated across the whole grid — 133 models, false-positive rate and recall correlate at +0.712 with their difference standard deviation 0.051; over 25 BERT-mini variants the AUC itself stays within 0.756 to 0.777; no method discriminates better than any other — `python -m dmthd.tradeoff`, F19, F23
- [x] The paper's own method reported as a negative result where the evidence says so — `paper/results_draft.md` Sections 5.3 and 5.3a
- [!] Reproducibility caveat unresolved: the same code, data and seeds give 0.8779 on a laptop CPU and 0.8393 on a Kaggle T4. Precision, data and code all eliminated. Every table is restricted to one environment; one Kaggle CPU run would isolate it

## C. Statistics and reporting
- [x] Mean ± std over seeds and paired bootstrap 95% intervals implemented — `aggregate.py`
- [x] Wilcoxon across seeds when five seeds exist, otherwise bootstrap only — `aggregate.py --compare`
- [x] Per-class F1, ECE, ROC-AUC/PR-AUC for binary — `evaluate.py`
- [~] Per-agreement-band F1 on Wikipedia (0.2–0.8 band vs the rest) — `analysis.py agreement_bands` ready and tested; needs Wikipedia runs
- [x] Teacher complementarity — kappa 0.889-0.922, disagreement 6.4-9.2%, error overlap 0.673-0.730, oracle macro-F1 bound 0.9469 against best single 0.8973. This is the mechanism behind the negative result. With the implicit specialist: kappa 0.962 with HateBERT, disagreement 3.1%, oracle accuracy 0.9526 -> 0.9552 (F23)
- [x] Weight trajectories per teacher per epoch — 0.332 / 0.337 / 0.331 for the homogeneous committee and 0.254 / 0.259 / 0.254 / 0.233 with DeBERTa, identical at every epoch and seed, which is correct by construction with frozen teachers and is reported as such rather than plotted as a trend (corrected 17 September, F20)
- [x] Every comparison the paper states as a generated table — `python -m dmthd.significance`, `paper/tables/significance.csv`: 51 paired bootstraps, one interval excludes zero (F27)
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
