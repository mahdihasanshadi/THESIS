# Paper blueprint, 21 September 2026

The one document that says what the paper contains, where every measured thing goes, what is left
out and why. Derived from how the exemplar papers in Section 1 are built, checked against every
finding and decision in `DECISIONS.md` (Section 5). Deadlines: thesis draft 26 September, slides 29
September, defense 3 October. V7 (D21) lands on 22 September and changes exactly one subsection (5.8).

## 0. Title and thesis sentence

**Out-of-Sample Knowledge Distillation for Compact Cyberbullying Detection: More Unlabelled Text
Helps, More Teachers Do Not**

*Thesis sentence.* Distilling a compact student from large teachers on the labelled split the teachers
were fine-tuned on adds nothing over fine-tuning, because there the teachers' soft labels are the gold
labels; distilling on unlabelled tweets the teachers have not seen improves the student monotonically
with the amount of text; teacher committees, reliability weighting and specialist teachers add nothing
at any point; and none of it transfers the ability to tell implied abuse from harmless sarcasm.

Four pillars, each with its own evidence and its own section:

| Pillar | Claim | Evidence | Section |
|---|---|---|---|
| P1 constructive | Out-of-sample distillation helps, grows with the text, survives matched compute, and holds on three compact students | F31, F33, F34 | 5.7, 5.8 |
| P2 audit | In sample nothing helps; committees, weighting, specialist add nothing; the reason is measured | F3, F6, F22, F23, F27, F28, F29 | 5.2 to 5.6 |
| P3 implication | Discrimination of implied abuse is fixed by pre-training; methods differ only in firing rate | F10, F19, F25, F26, F32 | 5.9 |
| P4 protocol | Threshold-free probes, single-source implicit benchmark, manifests, pre-registered predictions, paired bootstrap | F13 to F16, D19 to D21, 4.6 | 4, statements |

Why only the constructive pillar is in the title: a title carries one claim and its sharpest contrast.
P2 is the subtitle's second clause. P3 and P4 are what the abstract's last two sentences and two
section titles carry. Putting three findings in a title makes three things to defend in the first
minute; the abstract has room for all four.

## 1. What the exemplars teach, and where we apply it

| Exemplar | Convention we adopt | Where |
|---|---|---|
| Stanton et al. 2021, *Does knowledge distillation really work?* (NeurIPS) | Ask one question; vary one factor at a time; separate what the student matches from what it gains; attribute the cause with a targeted experiment rather than an argument | 5.2 to 5.6 vary weighting, committee, teacher, data in turn; F28 is the targeted cause; the steps control is the targeted confound |
| Beyer et al. 2022, *A good teacher is patient and consistent* (CVPR) | Distillation is function matching: the student must see the teacher's function on inputs the teacher has not fitted; long schedules and matched compute are part of the recipe, so report them | 3.5 defines in-sample vs out-of-sample as where the function is sampled; 5.8 reports the matched-steps controls; 7 lists the 168k-length control |
| Turc et al. 2019, *Well-read students learn better* | Compact pre-trained students; the unlabelled transfer set's size and properties are experimental variables, not fixed choices | BERT-mini family as headline; the size curve and the composition control; F27 (pre-training worth 40 times any component) |
| Hinton et al. 2015; Tang et al. 2019; Jiao et al. 2020 | The transfer set is the classical vehicle; augmentation is its usual source | 2.2 places our transfer set in that line and says what is new here: the controls, the curve, the domain |
| Wu et al. 2021, MT-BERT (Findings of ACL) | Instance-level teacher weighting and hidden-state distillation from several teachers; the claim that several beat one | 3.1 to 3.3 audit exactly these components; 5.3 and 5.4 give the controlled answer |
| Röttger et al. 2021, HateCheck (ACL) | Targeted functional tests expose what held-out accuracy hides; validate the tests | 4.3 probes (benign sarcasm vs ironic abuse) and 4.6 threshold-free AUC; 7 states the human validation still pending |
| ElSherief et al. 2021, Latent Hatred (EMNLP) | Implicit hate needs its own label and its own benchmark | 4.2 the implicit benchmark, built from that corpus alone (D17); F13 to F16 |
| Dodge et al. 2019; Card et al. 2020; Dror et al. 2018; Koehn 2004 | Report seeds, search budget and compute; test with paired bootstrap; state the minimum detectable effect | 4.7: three seeds, paired bootstrap, 1,000 resamples; the 4,326-item test set resolves about 0.007 macro-F1 and we say so; GPU-hours per run |
| Vidgen and Derczynski 2020 | Document abusive-language data: provenance, cleaning, what was removed | 4.1, 4.4 and the data statement; manifests (D16 to D18) |
| Prasomphan 2025; the Phase-2 report | The claim type we audit: multi-teacher distillation reported without a no-distillation control, a teacher score, seeds or intervals | 2.3 names the gap neutrally; 5.2 and 5.4 supply the controls |

## 2. Claims ledger

Every sentence the paper asserts, with what licenses it. Nothing else is claimed.

| # | Claim | Evidence | Table / figure |
|---|---|---|---|
| C1 | In sample, distillation raises every student by 0.001 to 0.007 and no paired interval excludes zero | F3, F27 | T4, T5 |
| C2 | Per-instance reliability weighting equals uniform averaging at every temperature from 0.05 to 5, on five students and three committees | F6, F22 | T4, T6, Fig 3 |
| C3 | The committee predicts better than its best member (+0.5 to +0.8) and the student receives none of it; teachers agree at kappa 0.89 to 0.96 | F20, F29 | Fig 2 |
| C4 | The implicit specialist, alone, in a committee, or replaced by pre-training on its corpus, adds nothing; after adaptation it is a near-copy of HateBERT (kappa 0.962) | F23, F24 | T7 |
| C5 | Why: on the training split the teachers' soft labels are the gold labels (train loss 0.03 to 0.08), so the reliability signal is saturated and distillation reduces to fine-tuning | F28 | Fig 1 (schematic), T3 |
| C6 | Out of sample, one teacher's soft labels on 42,013 unlabelled tweets raise the 11M student by +0.007, every seed above every fine-tuning seed; committee, corrected weighting and hard pseudo-labels add nothing | F31 | T8 |
| C7 | The gain grows with the text: monotone over six sizes, +0.0123 [+0.0035, +0.0211] at 168,000 rows; generic tweets work nearly as well; part of it is optimisation length (+0.0027 at matched steps), the rest is the text | F33 | T9, Fig 4 |
| C7b | Fine-tuning for the 168k arm's number of updates gains nothing (-0.0013); the 168k arm keeps +0.0136 [+0.0023, +0.0240] over it; the endpoint holds on BiLSTM (+0.0163, interval clear of zero) and BERT-small (+0.0057, interval including zero) | F34 | T9b |
| C8 | Implication does not transfer: sarcasm-discrimination AUC stays within 0.75 to 0.78 across every variant; recall and false positives move together across the whole grid (r = 0.71); tweet-trained students rank implied hate at 0.60 AUC against 0.82 trained in domain | F10, F19, F25, F26, F32 | T10, Fig 5, Fig 6 |
| C9 | Pooling implicit-hate corpora rewards source recognition; the two corpora agree on hate (99.7%) and not on implication (48.2%) | F14 to F16 | T1 |
| C10 | Deployment: a 67M student within 0.001 of a 335M teacher at a fifth of the latency; INT8 gains depend on shape; obfuscation costs 10 to 15 points and distillation does not protect against it | F4, F7, F8, F11 | T11 |

## 3. Structure and what each section must contain

**Abstract** (Version C in `abstract_draft.md`): problem, the usual proposal, the audit's answer, the
out-of-sample result with its curve, the implication result, what is released. One number per claim.

**1 Introduction.** The deployment problem; the usual answer (committees of task-adapted teachers,
weighted); the question (what does the student actually receive, and from where); the answer in one
paragraph; four contributions as bullets (audit, mechanism, curve with controls, implication protocol);
a sentence on what is not claimed (no new method; one corpus; one headline student until V7).

**2 Related work.** 2.1 Cyberbullying, toxicity, abuse by implication (corpora, the implicit gap, annotator
disagreement F14 folded in). 2.2 Distilling compact language models, including transfer sets. 2.3
Multi-teacher and adaptive distillation, ending with the unaudited-claim pattern. 2.4 Evaluation
practice: functional tests, seeds and intervals, power.

**3 The audited method and the out-of-sample recipe.** 3.1 Committee and notation. 3.2 Per-instance
reliability (the formula, tau, per-batch variant). 3.3 Objective (KL, CE, hidden term, irony head).
3.4 The specialist and its two controls. 3.5 In sample and out of sample: the definition, the label
mask, the transfer set, the kNN reliability. 3.6 What is measured about the weighting (routing
contrast, tau diagnostic). Every symbol defined once; the algorithm box once.

**4 Data, probes and protocol.** 4.1 Tweet corpus and cleaning (F1). 4.2 Implicit benchmark (D16 to
D18, F13 to F16). 4.3 Probes and screening. 4.4 Transfer set: sources, screens, counts (transfer
reports). 4.5 Teachers, students, classical floor. 4.6 Metrics, including the threshold-free AUC and
why recall at 0.5 is not reported as a headline (D13). 4.7 Protocol and statistics: seeds, bootstrap,
minimum detectable effect, compute, the two-environment caveat (F17, F30).

**5 Results**, in the order the argument runs, not the order the runs happened:
5.1 Teachers and floor (F2). 5.2 In-sample distillation on five students (C1). 5.3 Weighting against
averaging at every temperature; ablations and sweeps (C2). 5.4 Committee against single teacher,
heterogeneous committee, oracle (C3). 5.5 The specialist and its controls (C4). 5.6 Why nothing moved
(C5): the pivot of the paper. 5.7 Out of sample: the five arms (C6). 5.8 The size curve and its controls
(C7, C7b): the headline table and figure. 5.9 What never transferred: the implication audit (C8, C9).
5.10 Robustness and efficiency (C10).

**6 Discussion.** What the evidence licenses and what it does not; when a committee could matter
(diversity that task adaptation does not spend); why implication needs labelled implication in the
transfer text, not more of it; the recipe's cost in GPU-hours and its inference cost of zero.

**7 Threats to validity.** Environment discrepancy; optimisation length at 168k (until V7); one corpus;
heuristic probes pending human validation; single teacher runs; the heterogeneous confound; the
minimum detectable effect.

**8 Conclusion.** The thesis sentence, restated with the numbers.

**Statements.** Ethics, data availability, reproducibility (manifests, seeds, commit), compute.

**Appendices.** A generated tables (full significance table, routing, tau diagnostics, sweeps); B
transfer-set report; C hyper-parameters (Table 3.1 of Phase 2 carried over, with what changed); D
the per-model sarcasm analysis; E how the study evolved (DECISIONS Section 7, condensed), which
examiners value and journals cut.

## 4. Thesis chapter map (department format)

| Thesis chapter | Paper sections | Notes |
|---|---|---|
| 1 Introduction | 1 | Add objectives, scope, and a one-page outline |
| 2 Literature review | 2 | Add the Phase-2 citation corrections (`phase2_citation_map.md`) |
| 3 Methodology | 3, 4 | The thesis can carry the full algorithm box and the transfer-set construction in prose |
| 4 Results and analysis | 5 | Same order as the paper; the thesis may keep every generated table in the chapter |
| 5 Discussion and limitations | 6, 7 | Add Appendix E here as "how the study evolved" |
| 6 Conclusion and future work | 8 | Future work: implication-labelled transfer text; second corpus; human-validated probes |

## 5. Coverage matrix: every finding and decision has a home

| Findings | Section |
|---|---|
| F1 | 4.1 |
| F2 | 5.1 |
| F3, F27 | 5.2 |
| F4, F7, F8, F11 | 5.10 |
| F5, F9 | 5.10 (cross-corpus), 7 |
| F6, F22 | 5.3 |
| F10, F12, F19, F25, F26, F32 | 5.9 |
| F13, F14, F15, F16 | 4.2, 5.9 |
| F17, F30 | 4.7, 7 |
| F18 | 4.7 (a sentence on aggregation by run directory) |
| F20, F29 | 5.4 |
| F21, F23, F24 | 5.5 |
| F28 | 5.6 |
| F31 | 5.7 |
| F33 (+ D21) | 5.8 |

| Decisions | Where stated |
|---|---|
| D1, D16, D17, D18 | 4.1, 4.2 as design rationale |
| D2, D3, D4, D5, D11, D15 | 3.1 to 3.4 |
| D6 | 4.7 |
| D7, D9 | 1 (what is not claimed), 5.10 |
| D8 | not in the paper (language of the thesis) |
| D10, D12 | Statements, Appendix E |
| D13 | 4.6 |
| D14 | 4.2 |
| D19, D20, D21 | 3.5, 5.7, 5.8 as pre-registered predictions, scored |

Nothing measured is left out; the only material that stays out of the journal version is Appendix E's
narrative, which goes to the thesis.

## 6. Tables and figures, and the redundancy rule

| Table | Content | Source |
|---|---|---|
| T1 | Corpora, splits, cleaning counts, implicit benchmark, agreement of the two corpora | `dataset.csv`, F14 |
| T2 | Transfer set: sources, rows in, rows dropped per screen, rows out | `transfer_report.json`, `transfer_big_report.json` |
| T3 | Teachers, floor, train losses (the F28 column) | `teachers.csv` + F28 |
| T4 | Main grid: five students by objective and committee | `main.csv` condensed |
| T5 | Selected paired comparisons (the full 94 rows to Appendix A) | `significance.csv` |
| T6 | Ablations and sweeps on the headline student, merged | `ablations.csv`, `sweeps.csv` |
| T7 | Specialist committee and its two controls; routing contrast | `implicit.csv` subset, `routing.csv` |
| T8 | Out-of-sample arms (five) | 5.3c table |
| T9 | Size curve with the two controls; T9b the other students | `scaling.csv`, `scaling_students.csv` |
| T10 | Implication audit: AUC, operating points, implicit-benchmark transfer | `implicit.csv`, F25 |
| T11 | Robustness and efficiency, merged | `robustness.csv`, `efficiency.csv` |

| Figure | Content |
|---|---|
| Fig 1 | In-sample against out-of-sample distillation, one schematic: where the teacher's function is sampled |
| Fig 2 | Teacher agreement and the oracle (complementarity) |
| Fig 3 | Tau sweep: score flat, weights flat |
| Fig 4 | The size curve with intervals, the two controls as horizontal lines: the paper's figure |
| Fig 5 | False-positive rate against recall across every model (F19) |
| Fig 6 | Sarcasm-discrimination AUC band across variants, with the randomly initialised student outside it |
| Fig 7 (optional) | Validation curves: 13- and 35-epoch fine-tuning against the 168k arm |

Redundancy rule: a number appears once in prose and once in a table; a table appears once (main text
or appendix, never both); a result is restated only in the Discussion, by pointer; the four
contributions are listed once, in the introduction, and answered once, in the conclusion.

## 7. Defense questions, with the answer's evidence

1. Why did the multi-teacher method not work? F28 and F20: memorised labels, no diversity to weight.
2. Is +0.012 just longer training? No: 13 epochs alone gain +0.0027, 35 epochs alone gain -0.0013 (F33, F34).
3. Why "out of sample" and not "out of domain"? Generic tweets gain within 0.002 of abuse-domain ones.
4. Is out-of-sample distillation new? No: Hinton's transfer set, Turc, Tang. New: the controls, the curve, the domain, the implication result.
5. What does the student learn, then? The boundary between not-cyberbullying and other-cyberbullying (F33 per-class).
6. Why does implication not move? F19 and F32: firing rate changes, discrimination does not; nothing in the transfer text labels implication.
7. Why one seed for ablations? D6; the intervals are the test, single-seed rows are marked indicative.
8. Why do the Kaggle and laptop numbers differ? F17, F30; every table from one environment.
9. Why single-source implicit benchmark? F16: pooled corpora are separable by source at 0.91.
10. Is a 1.2-point gain worth a paper? The audit is the paper; the gain is where the evidence says the gain is, with the curve and the controls; the rest of the literature reports larger gains without the controls.
11. What about the teachers' own variance? Trained once; stated in 7.
12. What would you do next? Implication-labelled transfer text; a second corpus; human-validated probes; the LLM baseline.

## 8. Schedule

| Date | Deliverable |
|---|---|
| 22 Sep | V7 analysed (done); tables regenerated (done); 5.8 and T9b final (done); Fig 4 to draw |
| 22 to 23 Sep | Sections 1 to 8 assembled from the drafts under the filed title as `paper/PAPER.md` (first full draft done 22 Sep); figures to draw |
| 24 Sep | Coverage matrix ticked against the draft; figures 1 to 6 drawn; conversion into the department's template |
| 25 Sep | Sections 6 to 8, abstract, statements, figures; the redundancy pass |
| 26 Sep | Thesis draft submitted in the department format (chapter map, Section 4) |
| 27 to 29 Sep | Slides: 15 slides following Sections 0, 5.6, 5.8, 5.9; Q&A drill from Section 7 |
| 30 Sep to 2 Oct | Rehearsal; one-page defense sheet with the ten numbers |
| 3 Oct | Defense |
