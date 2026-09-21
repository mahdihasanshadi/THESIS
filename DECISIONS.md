# D-MTHD: decisions, observations and definitions

This is the reasoning record for the paper. `LOG.md` is the lab notebook: what ran, when, with which
numbers. **This file is organised by decision and by finding**, so that when the paper is written
nobody has to reconstruct why a choice was made or what an observation actually showed.

Rules for this file. Every number is copied from a `results.json`, `report.json` or an analysis
output, never from memory. Every decision records what was rejected and why. Every claim carries the
evidence that licenses it, and claims we cannot yet make are listed as such. When a decision is
reversed, the old entry stays and the reversal is added underneath, because the reversal is often
the more interesting fact.

---

## 1. Glossary

Terms are used in exactly these senses throughout the code, this file and the paper. Where a term is
commonly used loosely in the literature, the narrow sense we adopt is stated.

| Term | Definition as used here |
|---|---|
| **Teacher** | A large pre-trained encoder, fine-tuned on the target task, then frozen. Its outputs are computed once and cached; it never trains alongside the student. |
| **Student** | The compact model being trained. It is the deployable artefact; everything else is scaffolding. |
| **Committee** | The set of teachers whose outputs are combined for one student run. `homo` = BERT-large + HateBERT + irony-RoBERTa. `hetero` = those three plus DeBERTa-v3-base. |
| **Task adaptation** | Initialising a teacher from a specialist checkpoint and fine-tuning it on the target task so every committee member shares one label space. Without it, a specialist's logits cannot enter the KL term at all. |
| **Homogeneous** | Same *architecture family* (BERT-lineage encoders), not the same tokenizer and not the same size. RoBERTa counts as BERT-lineage; DeBERTa-v3 does not. This is a description of the setup, **not a claim of novelty** (see Decision 7). |
| **Dynamic weighting** | Per-instance weights `w_k(i) = softmax_k(-CE(p_k(i), y_i)/tau)` over committee teachers. Because teachers are frozen, these weights are a fixed function of the data: they vary across instances, never across epochs or seeds (Finding 6). "Dynamic" therefore means instance-adaptive, and the paper must say so. |
| **tau** | Temperature of the weighting softmax. Small tau sharpens toward selecting one teacher; large tau flattens toward uniform averaging. Distinct from **T**, the distillation temperature that softens teacher and student distributions in the KL term. |
| **Auxiliary irony head** | A second, two-way head on the student, distilled from the *un-adapted* irony teacher. It exists because that teacher's labels are irony/not-irony, a different label space from the task, so it cannot join the committee. Discarded at inference; its purpose is to shape the shared representation. |
| **Classical floor** | TF-IDF (word 1-2 grams + char 3-5 grams) with class-balanced logistic regression. Every neural result is reported relative to it. A method that cannot beat it does not deserve a GPU. |
| **Stop rule** | The pre-registered condition: if no retrained teacher beats the fine-tune-only student on the clean test split, distillation has nothing to transfer and the student family is changed. Passed on the tweets corpus (Finding 2). |
| **Provenance rule** | No test set may overlap the training data of any teacher, including data a specialist checkpoint saw before we touched it. This is why ToxiGen and Founta data are excluded everywhere. The Implicit Hate Corpus is excluded from the *probes* but used as training data in the implicit benchmark, which is consistent: no teacher of ours was pre-trained on it. |
| **Probe set** | An inference-only evaluation set, never used for training by any model on any benchmark. Three exist: benign sarcasm (negatives), ironic abuse (positives), implicit abuse (positives). |
| **Benign sarcasm** | Sarcastic or ironic text that attacks nobody. 1,064 items from iSarcasmEval; 883 survive automatic screening, and a 300-item human annotation is in progress. |
| **Ironic abuse** | Abuse delivered through irony or implication. 1,560 items: 797 from the Implicit Hate Corpus stage-2 irony category, 763 original ISHate rows labelled Implicit HS. |
| **Sarcasm-discrimination AUC** | ROC-AUC with ironic abuse as positives and benign sarcasm as negatives, scored by p(bullying). Threshold-free, so it separates "cannot see indirect abuse" from "sees it but cannot tell it from harmless sarcasm". **This is the headline metric for the paper's sarcasm claim** (Decision 13). |
| **Disagreement-aware variant** | On corpora with annotator fractions, the hard-label term is scaled by annotator agreement `a_i` and the teacher term by `1 + kappa(1 - a_i)`, so the student trusts the committee more where annotators disagreed. Applies to Wikipedia only. |
| **Implicit benchmark** | The third corpus, built here: `not_hate`, `explicit_hate`, `implicit_hate`, from the Implicit Hate Corpus stage-1 release. The only one of our three where abuse-by-implication is a label rather than a hidden subset. ISHate is held out of it as an out-of-domain test set (Decision 17). |
| **Implicit-discrimination AUC** | ROC-AUC with implicit-hate rows as positives and benign rows as negatives, scored by p(implicit hate). The headline metric of the implicit benchmark, for the same reason sarcasm-discrimination AUC is the headline of the sarcasm claim: it is threshold-free and it is not diluted by classes the paper is not about. |
| **Out-of-domain test set** | A corpus annotated for the same task by other people, kept entirely out of training. Here: 27,096 ISHate rows. It answers whether a model trained to find implication still behaves when the domain changes. |
| **Explicit hate** | Abuse that states its target and its hostility outright, typically with slurs or direct insult. |
| **Implicit hate** | Abuse carried by implication, stereotype, irony or coded reference, with no lexical marker of hostility. This is what the paper is about. |
| **Implicit specialist** | HateBERT trained on the implicit benchmark, then task-adapted onto the target benchmark like any other teacher. The only committee member that has seen implication labelled as such. |
| **Probe holdout** | Deleting every probe text from the implicit benchmark, all three splits, so probe metrics stay measured on text no model has trained on. Costs 1,581 rows and keeps every previously reported probe number valid. |
| **Focus-class transfer** | Cross-corpus transfer scored on one target class against the benign class, instead of collapsing everything to abusive-vs-not. A collapsed score hides a model that catches explicit abuse and misses every implication. |
| **Cache fingerprint** | A hash of the training split stored with the teacher cache. Teacher logits are indexed by row position, so pairing a cache with a different split would train on silently mismatched targets; the student refuses to start on a mismatch. |
| **Collapsed teacher** | A teacher whose test macro-F1 falls below `MIN_TEACHER_F1` (0.5), as happens on a NaN loss. It is retrained once conservatively and dropped from every committee if it stays collapsed. |

---

## 2. Where the project started

*Corrected 21 September: the report's own numbers and the team's later tweet-corpus outputs were
run together here; they are two different experiments.*

**The Phase-2 report** (February 2026; Wikipedia personal-attacks corpus, test set 22,303, one seed,
per-batch weighting at tau = 1) reported, in its Table 4.2, the multi-teacher DistilBERT student at
**0.8947** macro-F1 against its teachers BERT-large 0.8936 and RoBERTa-base 0.8904 and the
single-teacher student 0.8896, and called "student exceeds teachers" its key empirical finding. The
report itself limits that claim in two places. Its Section 4.6, "Statistical caveat", says the figures
are single runs and that the margins, multi over single and student over teachers, are within typical
seed-to-seed variation. And the comparison "with models without distillation" promised in Section 1.4
does not appear in Chapter 4: the same student trained with no teacher was never measured, so
whether distillation contributed anything was never tested. Section 4.6 also anticipates F20: the two
teachers "are individually close and largely agree, so the ensemble adds limited diversity".

**The team's follow-up on the tweet corpus** (Drive folder, inspected 10 September) had that control,
and it showed the method did not work:

| Team's tweet-corpus outputs, before Phase 3 | Macro-F1 |
|---|---|
| DistilBERT, **no distillation** | 0.859 |
| Teacher: BERT-base general | 0.819 |
| Teacher: BERT-base "implicit" | 0.828 |
| Teacher: RoBERTa-base | 0.827 |
| D-MTHD student | 0.830 |

The student trained *without* teachers beat every teacher and beat the distilled student by three
points. Distillation was actively harmful. Three further problems: validation accuracy 0.94 against
test 0.86 (duplicate leakage), 84 exact duplicates in the test set of which 57 carried conflicting
labels, and dynamic weights frozen at roughly [0.08, 0.64, 0.28] so the committee was effectively one
teacher. Latency numbers were unusable: two identical BERT-base teachers differed by 2.5x.

**The single most important consequence:** every claim in this project is now checked against a
control that uses no teachers at all. That control is the reason we know the method works this time.

---

## 3. Decisions

### D1. Rebuild the data before anything else
*Taken 2026-09-10.* De-duplicate on normalised text, drop every text carrying more than one label
(all copies), assert splits disjoint, report all counts.
**Why:** the Phase-2 validation-test gap was a data artefact, not a modelling result.
**Evidence:** tweets 47,692 → 43,259 (758 under two tokens, 3,168 rows across 1,563 conflicting
texts, 507 duplicates). The 0.94/0.86 gap disappeared on the clean split.
**Rejected:** keeping the published split "for comparability". Comparability to a leaking split is
worth nothing, and the counts let any reader reproduce ours.
**Paper:** setup section, with the full table. This is also a finding in its own right: anyone using
this corpus without de-duplication is over-reporting.

### D2. Larger teachers, smaller students
**Why:** in Phase 2 the teachers (base-size) and the student (DistilBERT) sat at the same capacity,
so there was nothing to distil. **Evidence:** Finding 2. **Consequence:** BERT-large, HateBERT and an
adapted irony RoBERTa as teachers; BERT-mini, BERT-small, DistilBERT as students.

### D3. A cross-task specialist committee, not three general teachers
General (BERT-large), explicit-abuse specialist (HateBERT), sarcasm specialist (irony RoBERTa), each
task-adapted. **Why:** a committee of near-identical teachers has no complementary knowledge to
combine, which is exactly what went wrong in Phase 2. **Paper:** contribution 1.

### D4. The irony teacher gets an auxiliary head, not a committee seat
**Why:** its label space is irony/not-irony. The KL term and the reliability weights are undefined
for a teacher that does not predict the task's classes. **Rejected:** task-adapting it and only using
it in the committee; we do that too, but the un-adapted model carries the sarcasm knowledge that
adaptation would overwrite. **Paper:** contribution 2, and the mechanism behind every sarcasm claim.

### D5. Per-instance weights, standard KL direction, temperature-squared scaling
**Why:** per-batch weights cannot express that different instances need different experts; the
missing T² factor silently rescales the distillation term against the others.
**Paper:** method section; the per-batch variant is an ablation.

### D6. Three seeds, one-seed ablations
**Why:** time. Five seeds everywhere does not fit the schedule. **Consequence:** stated in
Limitations; bootstrap intervals carry the significance argument, with Wilcoxon only where five
seeds exist.

### D7. "Homogeneous" is a measured finding, never a novelty claim
**Why:** same-family distillation is the default setting in the literature (DistilBERT, TinyBERT,
Patient-KD, MT-BERT). Claiming it as novelty invites a one-line rejection. **Instead:** measure it
with a 2x2 (BERT-mini versus DeBERTa-v3-xsmall student, hidden-state term on and off) plus a
non-transformer BiLSTM student and a heterogeneous committee. If the hidden-state term helps
same-family students more, that is a result; if not, homogeneity is a design choice and the word
leaves the abstract.

### D8. Bengali goes to the thesis, not to the Q1 paper
**Why:** no Bengali sarcasm teacher exists, so the paper's central claim cannot be tested there; it
would add two to three GPU-days and split the paper's focus. **Consequence:** a thesis chapter, and
a candidate second paper.

### D9. The word "Robust" leaves the title unless robustness is measured
**Status:** robustness is now measured (Findings 8 and 9), so the word can return if the results
support it. The obfuscation results are synthetic edits, not real evasion, and the paper must say so.

### D10. Report failures, do not hide them
Applies to: the cross-corpus transfer collapse (Finding 9), the obfuscation fragility (Finding 8),
the one obfuscation variant that *improved* Wikipedia results, the near-uniform weights (Finding 6),
and the fine-tune-only student failing to beat the classical floor on tweets (Finding 3).
**Why:** each is a real property of the problem, and a reviewer who finds a hidden one stops trusting
everything else.

### D11. Single-teacher distillation is not repeated per committee
**Why:** `skd` uses only the committee's first teacher, so `skd_hetero` trains an identical model.
Fifteen redundant runs on the full grid. **Found by** reading the version-3 log.

### D12. Guards over vigilance
A teacher below 0.5 macro-F1 is retrained once and then dropped; a cache whose fingerprint does not
match the split refuses to train; a wrong resume path exits in seconds instead of silently retraining
everything; the session stops launching work at eleven hours so the packaging step always runs.
**Why:** every one of these was written after the corresponding failure cost real time, and each has
since paid for itself.

### D13. The headline sarcasm metric is discrimination, not recall at 0.5
*Taken 2026-09-12, after Finding 10.* **Why:** recall at an arbitrary threshold conflates "cannot
see indirect abuse" with "sees it but cannot separate it from harmless sarcasm", and the measurement
says it is the second. **Consequence:** sarcasm-discrimination AUC becomes the number every variant
is compared on; the threshold curve is reported alongside it.

### D14. Train on implicit abuse, not only test on it
*Taken 2026-09-12, on Mahdi's instruction that indirect abuse is the paper's whole point.*
Until now the implicit corpora were inference-only probes, so no student had ever seen an example
labelled as abuse-by-implication. Two changes:
- **A third benchmark**, `implicit`, with the label scheme `not_hate | explicit_hate | implicit_hate`
  built from ISHate and Implicit Hate Corpus stage 1 (F13).
- **An implicit-abuse specialist teacher**: HateBERT trained on that benchmark, then task-adapted
  onto the target benchmark like every other committee member. It is the only teacher that has seen
  implication labelled as such, and it is the mechanism the implicit claim rests on.

**Why it is clean:** none of our teachers was trained on those corpora, so the provenance rule holds.
**Why the probes survive:** every probe text is deleted from the new benchmark, in all three splits.
**What it buys:** the paper's stated goal becomes something the method is trained for and measured
on, rather than something it is only tested against. **Cost:** six to ten GPU-hours.

### D15. The implicit specialist forms its own committee rather than joining the homogeneous one
*Taken 2026-09-12.* The specialist could have been added as a fourth member of `homo`. Instead
`spec` = `homo` + specialist is a separate committee.
**Why:** adding a teacher to `homo` would change what `homo` means, so every finished run that used
the three-teacher committee would have to be re-run to stay comparable, which is about ten GPU-hours.
More importantly it turns "does the specialist help?" into a controlled comparison of two full
committees over three seeds, which is stronger evidence than a single-seed ablation.
**Cost:** one extra committee in the grid, restricted by default to the headline student, because the
question is about the committee and not about the student.
**Rejected:** adding it to `homo` (invalidates finished work, weaker evidence); leaving it out of the
tweet corpus entirely (then the paper's mechanism is never tested where the corpus lacks implicit
labels, which is the interesting case).

### D16. The probes are held out of the new benchmark, not merely split away
The probes were drawn from these corpora. Three options existed: route probe texts into the test
split, re-derive the probes from the test split, or delete them from the benchmark entirely. The
first two make every probe number in the paper conditional on a split, and one of them invalidates
results already reported. The third costs 1,581 rows out of 48,662 and keeps every probe number
comparable across all three benchmarks and across everything already run. Taken: delete them.
### D17. The implicit benchmark is built from one corpus, and the other is held out
*Taken 2026-09-12, on the evidence of F16, replacing the pooled corpus built earlier the same day.*
Train, validation and test come from the Implicit Hate Corpus alone. ISHate is kept entirely out of
them and becomes a 27,096-row out-of-domain test set.
**Why:** pooling makes the label partly predictable from the source (F16), and costs 0.075 F1 on
implied hate measured on identical test rows. A benchmark whose headline number can be earned by
recognising which corpus a text came from cannot support this paper's claim.
**Rejected:** pooling and reporting per-source metrics as a correction. Per-source reporting is kept
anyway, because it costs nothing and catches this class of problem, but it does not fix a *training*
signal that rewards the shortcut.
**Cost:** a smaller corpus (20,637 rows against 47,181) and a thin explicit class (864 training
rows), which is why macro-F1 is not the headline. The implicit-discrimination AUC is.
**Reversible in one flag:** `prepare_implicit.py --corpora ImplicitHate,ISHate` rebuilds the pooled
version, and the rebuild is deterministic, so the comparison can be redone at any time.

### D18. Every score on a multi-source benchmark is also reported within each source
*Taken 2026-09-12.* `evaluate.py --group_col` and `baseline_tfidf.py --group_col`.
**Why:** F16 was found by looking; it should not have needed looking. Any benchmark assembled from
more than one source can be gamed by style, and the cheapest guard is to report the within-source
numbers alongside the aggregate, every time, automatically.


---

### D19. Distil out of sample: an unlabelled transfer set, the committee as labeller, and a reliability the teachers cannot have memorised
*Decided 17 September, after F28 and F29; predictions written here before the run.*

F28 says the teachers' outputs on the training split are the gold labels and their reliability there
is saturated; F29 says the committee is worth 0.5 to 0.8 points over its best member on text the
teachers have not fitted, and that no re-weighting of the same knowledge recovers more than 0.3. The
one lever those two findings leave inside knowledge distillation is the data the student imitates
the committee on. That is also where the recipes that report large gains for small students put it
(Tang et al. 2019; Jiao et al. 2020; Turc et al. 2019), and it is what this grid never had.

**What was built** (`prepare_transfer.py`, `knn_reliability.py`, `train_student.py --transfer`, driver
stage `transfer`, all smoke-tested on CPU):
- A transfer set of unlabelled in-domain tweets: OLID and HatEval through TweetEval, Davidson et al.
  2017, used as text only, plus the 1,563 tweets our own cleaning dropped for carrying more than one
  label. The TweetEval irony configuration is excluded because it is the irony teacher's fine-tuning
  data, and Founta et al. 2018 under the provenance rule. Every text is matched on the splits' own key
  against train, validation, test, every probe set and the implicit benchmark, and dropped on any
  hit. About 50,000 texts survive, 1.5 times the training split.
- On transfer rows the student trains on the committee alone: the KL term, the hidden term and, where
  used, the irony head, with the hard-label term masked out. The teachers' outputs there are cached
  like the training split's.
- A gold-free, out-of-sample reliability: for each teacher, its accuracy on the k = 20 validation texts
  nearest the instance in the teacher's own representation space, Laplace-smoothed, turned into
  weights by a softmax of its log at tau = 1. It is defined on unlabelled text, it cannot be memorised
  because the teachers never trained on validation, and it is what F28 says the original signal
  should have been.

**The arms**, BERT-mini, three seeds each, against the in-sample runs already in the grid:
`skd_transfer` (BERT-large with the transfer set), `uniform_transfer` (the homogeneous committee
averaged), `uniform_hetero_transfer` (the DeBERTa committee, the best labeller in F29),
`dmthd_knn_transfer` (the homogeneous committee under the corrected weights), and `pseudo_transfer`
(the committee's hard argmax as a pseudo-label with cross-entropy only: the control that separates
soft labels from simply having more labelled-looking data). Every pairwise question is a row of
`significance.csv`.

**Predictions, in the order they would be read.**
1. `uniform_transfer` beats fine-tune-only by at least 0.010 macro-F1 on BERT-mini, with an interval
   that excludes zero. This is the prediction the route stands or falls on. If it fails, the transfer
   route is closed and the paper stays as the branch `paper-b2-audit` has it.
2. `uniform_transfer` beats `skd_transfer`: the committee is the better labeller, by roughly the 0.5
   points F29 measured on the test set. An interval that includes zero here is expected and would not
   retire the claim, which rests on F29's direct measurement.
3. `dmthd_knn_transfer` does not beat `uniform_transfer` by more than 0.003: F29's ceiling. The
   corrected reliability is reported as the completion of the audit, not as a method.
4. `pseudo_transfer` gains less than `uniform_transfer`; if they tie, the gain is "more data" rather
   than "dark knowledge", and the paper says so.
5. Sarcasm-discrimination AUC on the transfer arms stays inside 0.756 to 0.777 (F23), because nothing
   in the transfer set teaches the distinction the probes measure.

**Cost.** Caching five teachers on 55,000 texts, about ten minutes on a T4; fifteen runs of about
seven minutes; probes and the implicit analysis on them; about 2.5 GPU-hours in all, inside one
resumed session of the tweet notebook.

**What it would mean.** If prediction 1 holds, the paper has its positive result and it is a
distillation result: a compact student improves on data the teachers labelled and the annotators did
not, and a committee is the right thing to label it with. The audit then explains why the in-sample
grid found nothing, which is a stronger paper than either half alone. If prediction 1 fails, the
audit gains one more measured negative and the branch manuscript stands.

**Outcome, 21 September (Kaggle v5, 2.11 h, 146 finished runs).** Scored by
`scripts/transfer_verdict.py` against the predictions above; BERT-mini fine-tune-only is 0.8393 ± 0.0003.

| Arm | Macro-F1 | Against fine-tune-only |
|---|---|---|
| Single-teacher KD + transfer set | 0.8466 ± 0.0016 | +0.0073 [-0.0003, +0.0150] |
| Uniform committee + transfer set | 0.8462 ± 0.0016 | +0.0070 [-0.0001, +0.0144] |
| Uniform committee + DeBERTa + transfer set | 0.8477 ± 0.0026 | +0.0084 [+0.0007, +0.0164] |
| Out-of-sample (kNN) weighting + transfer set | 0.8484 ± 0.0011 | +0.0091 [+0.0020, +0.0165] |
| Hard pseudo-labels + transfer set | 0.8412 ± 0.0013 | +0.0020 [-0.0062, +0.0104] |

1. **Not met as written.** The pre-registered arm gains +0.0070, below the 0.010 asked for, and its
   interval touches zero. Two arms of the same family clear zero (the DeBERTa committee and the
   corrected weighting), and every one of the twelve soft-label seeds (0.8444 to 0.8501) sits above
   every fine-tune-only seed (0.8389 to 0.8395). The effect is real and smaller than predicted.
2. **Failed.** The committee is not the better labeller for the student: -0.0003 [-0.0061, +0.0056]
   against one teacher.
3. **Held.** The corrected weighting adds +0.0021 [-0.0017, +0.0082] over averaging, inside the ceiling
   F29 set. On the transfer set its weights stay near uniform (entropy ratio 0.993) because the
   teachers' local accuracies there differ by 0.03 (0.767 / 0.798 / 0.794).
4. **Held.** Hard pseudo-labels gain +0.0020 where soft labels gain +0.0070; the difference, -0.0050
   [-0.0118, +0.0016], is the soft labels' share, with an interval that includes zero.
5. **Failed downward.** Sarcasm-discrimination AUC falls for three of the five arms (0.753, 0.753,
   0.727; the single-teacher and DeBERTa arms stay at 0.765 and 0.762). See F32.

By the rule written above, the route as a *multi-teacher* result is closed: the committee, the
weighting and the specialist add nothing out of sample either. What the run found instead is F31.

### D20. Measure the size curve before deciding what the transfer result is
*Decided 21 September, after F31; predictions written before the run (Kaggle v6).*

F31 measures one transfer set, 42,013 abuse-domain tweets, and finds +0.007 to +0.009 with one
teacher labelling. Whether that is a footnote to the audit or the paper's result depends on one thing
the grid has not measured: the slope. Three confounds go with it: size against composition (larger
sets can only be built by adding generic tweets), soft labels against extra text (the hard-label arm
bounds this but does not isolate it), and extra text against extra optimisation steps (the transfer
arms take 2.2 times the updates of fine-tuning).

**What v6 runs**, on BERT-mini, three seeds, one teacher (BERT-large) labelling, six epochs as
everywhere else (`kaggle/run_benchmark.py`, stage `transfer_scale`):
- nested prefixes of the shuffled transfer set: 5,000, 10,000, 21,000 and 42,013 rows (abuse-domain
  text, the composition held fixed while the size changes; the 42,013 arm reproduces F31's);
- the composition control: 42,013 generic tweets (TweetEval sentiment, emoji and emotion
  configurations, text only, screened like the base set), the size held fixed while the composition
  changes;
- larger sets: 84,000 and 168,000 rows, the base set plus generic tweets;
- the steps control: fine-tuning alone for 13 epochs, as many updates as the 42,013 arm, early stopping
  off, best validation epoch kept.
Every arm against fine-tuning, each size against the next smaller, generic against abuse-domain at
42,013, and the base arm against the matched-steps control, in `significance.csv`; the curve in
`paper/tables/scaling.csv`.

**Predictions.**
1. The gain rises with size at fixed composition: the 5k, 10k, 21k and 42k means are ordered, with at
   most one inversion smaller than 0.002, and 42k reproduces F31 (+0.006 to +0.009 over fine-tuning).
2. Composition matters more than size: 42k generic tweets gain at least 0.003 less than 42k
   abuse-domain tweets, and each generic row added beyond 42k adds less than the abuse-domain rows did.
3. Steps are not the explanation: fine-tuning for 13 epochs gains at most +0.002 over fine-tuning for
   six, and the 42k arm beats it by at least 0.004.
4. The decision rule, fixed now. If the largest arm gains at least +0.012 over fine-tuning with an
   interval that excludes zero and the curve is still rising between 84k and 168k, the constructive
   result leads the paper and the thesis title changes to say so. If the curve is flat beyond 42k
   (168k within 0.003 of 42k), the recipe is the audit's coda, and B2 stands under a title that does not
   claim it.

**Cost.** About 3.5 GPU-hours in one resumed session: caching one teacher on about 200,000 texts, 24
training runs from 3 to 17 minutes each, probes and the implicit analysis on them.

## 4. Findings

Numbered so the paper can cite them. Each states what was measured, on what, and what it licenses.

### F1. The corpus is dirtier than its users assume
1,563 tweet texts carry more than one label; 507 are exact duplicates. On the clean split the
validation-test gap vanishes. Wikipedia needed 522 short comments, 646 within-split duplicates and
443 cross-split overlaps removed.

### F2. Teachers clear the floor; the stop rule passes
Tweets, test macro-F1: BERT-large 0.8973, irony-RoBERTa (adapted) 0.8926, HateBERT 0.8887, against a
classical floor of 0.8798 and a fine-tune-only BERT-mini at 0.8770. Distillation has headroom.

### F3. Distillation helps every student, and it is the only component that does
On the tweet corpus, best distilled configuration minus fine-tune-only: BERT-mini **+0.0012**,
BERT-small **+0.0040**, DistilBERT **+0.0068**, DeBERTa-v3-xsmall **+0.0040**, BiLSTM **+0.0071**.
Five out of five positive. Against that, the per-instance weighting contributes nothing (F6), the
hidden-state term contributes 0.0009 on the headline student, and the auxiliary head contributes
0.0014.

The one component that matters enormously is pre-training: a randomly initialised student scores
0.7905 against 0.8378, a gap of 0.047, forty times anything else in the grid.

**On the headline student the gain does not clear noise.** Paired bootstrap against fine-tune-only:
D-MTHD -0.0015 [-0.0076, +0.0048]; uniform -0.0007 [-0.0074, +0.0058]; single-teacher +0.0012
[-0.0057, +0.0079]; D-MTHD with the heterogeneous committee -0.0019 [-0.0082, +0.0041]. Every
interval contains zero, so on BERT-mini the correct statement is that no method has been shown to
beat training without teachers at all.

### F4. A 67M student lands within 0.001 of a 335M teacher
DistilBERT with single-teacher distillation reaches 0.8963 ± 0.0002 against BERT-large's 0.8973, and
0.8975 ± 0.0031 with the uniform heterogeneous committee, which is *above* the teacher. Five times
fewer parameters, 5.2 ms per example against 26.6 ms at batch 1 on the same GPU. This is the paper's
efficiency claim and it is the one result that survived the grid intact.

### F5. The two corpora tell opposite stories, and that is the framing
On Wikipedia the same 11M student reaches 0.8884 ± 0.0044 on its own, above the floor (0.8759) and
0.005 below the Phase-2 BERT-large teacher (0.8936). Where a compact model is already near teacher
level, distillation has little room; where it is two points behind, it has a lot. The paper should
say when distillation is worth doing, not only that it works.

### F6. The dynamic weighting does not beat uniform averaging on any student
*Settled 2026-09-13 by the finished grid; the earlier diagnostic version of this entry predicted it
from the weights alone and is summarised at the end.*

Test macro-F1, three seeds, tweet corpus. Classical floor 0.8798, best teacher 0.8973.

| Student | params | ft | skd | uniform | D-MTHD | uniform+het | D-MTHD+het |
|---|---|---|---|---|---|---|---|
| BERT-mini | 11.2M | 0.8393 | 0.8405 | 0.8385 | 0.8378 | 0.8378 | 0.8373 |
| BERT-small | 28.8M | 0.8469 | 0.8488 | 0.8505 | 0.8471 | 0.8509 | 0.8477 |
| DistilBERT | 67.0M | 0.8907 | 0.8963 | 0.8960 | 0.8960 | 0.8975 | 0.8966 |
| DeBERTa-v3-xsmall | 70.8M | 0.8825 | 0.8865 | 0.8862 | 0.8833 | 0.8847 | 0.8828 |
| BiLSTM | 10.4M | 0.8693 | 0.8748 | 0.8761 | 0.8732 | 0.8760 | 0.8764 |

D-MTHD minus uniform, per student: **-0.0007, -0.0034, +0.0000, -0.0029, -0.0029**. Four negative,
one tie, none positive. The best configuration is uniform or single-teacher for four of the five
students. Removing the weighting entirely (`ablation_no_dynamic`, 0.8386) scores *above* full D-MTHD
(0.8378) on the headline student.

**Why this is not yet the final word.** Every tau the finished run tried was 0.5 or larger, where the
mean weights are 0.330 / 0.340 / 0.330, so all those configurations optimise nearly the same
objective and scoring the same is arithmetic rather than evidence. The values that could separate
them, 0.05, 0.1 and 0.2, are in the current grid and have not run. The honest present claim is
**"not shown to differ from uniform averaging at any tau yet tested"**, which is weaker than "it does
not work" and is what the paper should say until the sharp values return.

*Superseded diagnostic:* epoch-mean weights in every heterogeneous run were 0.254 / 0.259 / 0.254 /
0.233 against a uniform 0.25, identical across epochs and seeds, because with frozen teachers the
weights are a fixed function of the data. On a synthetic cache the weight entropy ratio was 0.94 at
tau = 1 and 0.50 at tau = 0.1. That diagnosis predicted this result before the runs existed.

### F7. The heterogeneous student is weaker and slower
DeBERTa-v3-xsmall: skd 0.8865, uniform 0.8847, D-MTHD 0.8835 (two seeds), against DistilBERT's
0.8963 / 0.8976 / 0.8966, and about 1,500 s per run against 650 s. Evidence toward D7, though
confounded by DeBERTa-v3-xsmall being a weaker model on this task; the hidden-term 2x2 is the
controlled version of the question.

### F8. Obfuscation hurts short text far more than long text
Tweets, BERT-mini fine-tune-only: clean 0.8779 → leetspeak 0.7791, character swaps 0.7518, inserted
spaces 0.7488, mixed 0.7247. Wikipedia: clean 0.8857 → 0.8649 / 0.8613 / **0.8977** / 0.8667. The
space variant *improves* Wikipedia, because splitting words inside attack comments makes them stand
out against tidy wiki prose. Synthetic obfuscation is not real evasion and the paper says so.

### F9. Neither corpus transfers to the other
Tweets model on Wikipedia: binary macro-F1 0.273, ROC-AUC 0.562, calling 80.9% of comments bullying
against a true rate of 11.8%. Wikipedia model on tweets: 0.425, ROC-AUC 0.718, calling 33.7% bullying
against a true rate of 85.7%. One over-fires, the other under-fires. This is a fact about two
different task definitions, not a defect of the students.

*17 September:* these come from the laptop models, which F17's rule keeps out of the paper. They stay
here as a record and leave the results draft until the Wikipedia grid measures them on Kaggle (F26).

### F10. Indirect abuse is missed through poor discrimination, not blindness
Measured on BERT-mini fine-tune-only, tweets, seed 1.
- **Not lexical.** Ironic-abuse recall is 0.72 on the 43 probe items containing explicit profanity and
  0.67 on the 1,517 without; on implicit abuse it is 0.45 with and 0.69 without. The model is not a
  profanity detector.
- **Discrimination is the failure.** Mean p(bullying) is 0.693 on ironic abuse and 0.375 on benign
  sarcasm; sarcasm-discrimination AUC 0.776. At the 0.5 cut, recall 0.704 comes with a 31.7%
  false-positive rate on harmless sarcasm; holding false positives under 10% (threshold 0.9) drops
  recall to 0.427. The model responds to "sarcastic and negative", not to "attacks someone".
- **The benchmark shows the same thing.** `other_cyberbullying` recall 0.786 (0.82 with profanity,
  0.78 without), 77 of its errors going to `not_cyberbullying`; `not_cyberbullying` recall 0.681 with
  133 errors going the other way. The two catch-all classes bleed into each other.

*17 September:* measured on the laptop model. On the Kaggle models the diagnosis stands, but the lexical
split and the 10 per cent operating point change materially; see F26 for the numbers the paper uses.

### F11. INT8 quantisation pays off by model shape, not by model size
BERT-mini, idle CPU: 42.6 → 33.5 MB, batch-32 latency 2.42 → 1.39 ms (1.74x) for a 0.0057 macro-F1
cost on tweets; 5.82 → 3.71 ms (1.57x) for 0.0031 on Wikipedia; batch-1 latency slightly *worse* on
tweets. The modest size drop is structural: 7.8M of BERT-mini's 11.2M parameters are the embedding
table, which dynamic quantisation does not touch.

### F12. Probe metrics only mean something within the probe's own domain
Wikipedia-trained models score a 3.6% false-positive rate on benign sarcasm and a 5.9% recall on
ironic abuse: they almost never fire on tweet-style text. Those numbers measure domain shift, not
sarcasm awareness. **Probe metrics are therefore reported for models trained on Twitter-domain
corpora only**, which is the tweet corpus and the implicit benchmark (the Implicit Hate Corpus is
also Twitter data, and every probe text is held out of it), and never for Wikipedia.

### F13. On a corpus that labels implication, there is room for a method to earn its place
*Revised 2026-09-12 after F16; the superseded numbers are kept at the end of this entry because the
difference between them is itself the evidence for Decision 17.*

The implicit benchmark, TF-IDF plus logistic regression, test macro-F1 **0.5620**, accuracy 0.6972,
ECE 0.0707. Per class: not_hate 0.7966, **implicit_hate 0.5562**, explicit_hate 0.3333. The headline
number is **implicit-discrimination AUC 0.7610** (average precision 0.6159) over 629 implicit and
1,327 benign test rows: ranked by p(implicit hate), does implied abuse come above ordinary text?
Well above chance, a long way from solved, and strikingly close to the 0.776 sarcasm-discrimination
AUC of the tweet student (F10), which is the same difficulty measured on a different corpus.

Macro-F1 is not the headline here. The explicit class has 108 test rows, so its F1 is noise, and a
three-class macro average over one large easy class and one tiny one says little about implication.

Corpus construction, from `report.json`: the splits come from the Implicit Hate Corpus alone,
20,684 rows after probe holdout, 20,637 after cleaning, split 16,509 / 2,064 / 2,064. Training
classes: not_hate 10,616, implicit_hate 5,029, explicit_hate 864. ISHate is held out entirely as a
27,096-row out-of-domain test set (17,691 benign, 9,404 explicit, 1 implicit).

*Superseded:* pooling both corpora gave 47,181 rows and a higher aggregate score, macro-F1 0.6809
with implicit_hate 0.4530. The higher aggregate is an artefact and the lower implicit F1 is the real
cost; see F16.

### F14. The two corpora agree about hate and disagree about implication
They share 629 texts. On whether a text is hateful at all they agree on **99.7 per cent** of them. On
whether the hate is stated or implied they agree on **48.2 per cent**: of the 624 both call hateful,
the Implicit Hate Corpus labels essentially all implicit, while ISHate labels 324 explicit and 300
implicit. Two expert annotation efforts placing the boundary in materially different places is a
direct measurement of how hard this label is. It bounds how sharp any implicit-hate result on either
corpus can be, it belongs in the paper beside our own annotation study, and it is one of the two
reasons the corpora are not pooled (`python -m dmthd.corpus_agreement`).

### F14b. ISHate's subtlety layer is not usable, and the paper should not pretend otherwise
ISHate ships a `subtlety_layer` (Subtle / Non-Subtle) that looks like a natural second axis for this
paper. It is not: in our test split only 10 of 4,719 rows carry it, and all 10 are `explicit_hate`,
while the implicit rows are almost entirely Non-Subtle. Whatever that column annotates, it is not the
implicit/explicit distinction and there is nowhere near enough of it to measure anything. The column
is carried through `prepare_implicit.py` for provenance and is used by nothing. Recorded so that a
later reader does not rediscover it and assume it works.

### F15. Two schema traps in the source corpora, either of which yields a silently wrong benchmark
Recorded because anyone reproducing this will hit them.
- ISHate records the benign class **only** in `hateful_layer`. Its `implicit_layer` holds
  {Explicit HS, Implicit HS} and is empty for every Non-HS row, so reading that column alone deletes
  all 17,869 benign examples and leaves a corpus with no negative class.
- The SALT-NLP `ImplicitHate` mirror ships **stage 2 only**: 6,346 rows, no benign class. Stage 1,
  the three-class set with 21,480 rows, is at `tasksource/implicit-hate-stg1`.

### F16. Pooling the two corpora buys an aggregate score and costs the thing we care about
The union is separable: a TF-IDF classifier predicts which corpus a text came from at **0.91
macro-F1**. The classes are drawn very unevenly from the two sides: 95 per cent of implicit examples
come from the Implicit Hate Corpus and 89 per cent of explicit examples from ISHate. A model trained
on the union can therefore score well on implicit-versus-explicit by recognising the source, and the
evidence says it does. Scored on *identical* Implicit Hate test rows, a classical model trained on
the union reaches 0.473 F1 on implicit hate; the same model trained on that corpus alone reaches
0.548, and macro-F1 goes from 0.521 to 0.571. The pooled corpus scores higher in aggregate (0.681
against 0.562) precisely because the aggregate rewards the shortcut.

This is the failure mode that would have sunk the paper quietly: a strong headline number, a
plausible story, and a reviewer asking why implicit and explicit examples come from different places.

### F17. The same code and data give different results on two machines, and every number must come from one of them
BERT-mini fine-tune-only scores **0.8779** here and **0.8393** on a Kaggle T4 under a configuration
that is identical in every respect we can name: same six epochs, batch 32, maximum length 128,
learning rate 3e-5, same seeds. Three candidate explanations were tested and three were eliminated.

**Not mixed precision, and now measured to completion on two students.** Twelve runs on the same
hardware, three seeds each:

| Student | fp32 seeds | fp32 mean | fp16 seeds | fp16 mean | cost of fp16 |
|---|---|---|---|---|---|
| BERT-mini | 0.8385 / 0.8433 / 0.8403 | 0.8407 | 0.8394 / 0.8395 / 0.8389 | 0.8393 | 0.0014 |
| BERT-small | 0.8516 / 0.8463 / 0.8461 | 0.8480 | 0.8511 / 0.8465 / 0.8431 | 0.8469 | 0.0011 |

Two students agree on about a thousandth, which is inside seed noise. Mixed precision is exonerated,
and every fp16 number in the grid stands as reported.

**This also splits the gap cleanly.** The headline comparison was local fp32 against Kaggle fp16,
which confounded two things. With Kaggle fp32 now measured at 0.8407, the environment alone accounts
for **0.0372** (0.8779 against 0.8407 at matched precision) and mixed precision for the remaining
0.0014. The environment is 27 times the size of the precision effect.

**Not the data.** The gold-label sequence recorded in `test_labels.npy` matches row for row across
the two environments, all 4,326 rows. The preparation pipeline is reproducible across machines,
which is worth knowing on its own.

**Not the code, and the check was stronger than it needed to be.** Re-running the current trainer on
this laptop with the same seed reproduces the original run *exactly*: test macro-F1
0.8779444975134871 against 0.8779444975134871, and validation macro-F1 identical to sixteen
significant figures at all six epochs (0.8431027360303543, 0.8658475690577240, 0.8750055333337056,
0.8818449709201831, 0.8832502397917246, 0.8839131421739026). Only the training-loss accumulation
differs, in the ninth significant figure. The pipeline is bit-reproducible on fixed hardware, and
every change made to the trainer since those runs is behaviour-preserving.

What is left is the machine, and the elimination is now complete. The local runs are CPU on this
laptop; the grid is a T4 on Kaggle. The gap is present from the first epoch, the Kaggle training loss
is higher at every epoch (1.019 against 0.874, ending 0.297 against 0.208), and both sets of seeds
are internally tight: 0.8779 / 0.8770 / 0.8760 locally against 0.8394 / 0.8395 / 0.8389 on Kaggle.
One environment is deterministic to sixteen digits across re-runs; the other lands **0.0372** away
from it at matched precision on the same script, data and seed. That is a systematic difference, not seed variance, and
the higher loss at every epoch says it is slower optimisation rather than worse generalisation.

Note that it does not rescue the student: at its local best, 0.8779, BERT-mini fine-tune-only is
still below the 0.8798 classical floor. The discrepancy changes the absolute numbers and not the
conclusion that a compact model needs help to beat a bag of n-grams on this corpus.

**The rule this imposes, which matters more than the cause:** every number in a table must come from
one environment. The Kaggle grid is internally consistent, so all of its *comparisons* stand; the
local BERT-mini and BERT-small numbers may not be quoted beside them and are withdrawn from the
paper unless re-run there. This also means the students may be capable of more than the grid shows,
which affects how the efficiency claim is phrased but not the ranking of the methods.

**What would isolate it:** one run on Kaggle with `GPU=0`, same configuration, about thirty minutes
of CPU. If it lands near 0.878, the difference is the accelerator; if near 0.839, it is the software
stack. Neither answer changes the rule above, which is why this is a curiosity to resolve rather
than a blocker.

### F21. The implicit specialist exists, and the first implicit numbers say the metric was the right choice
Three models have now been trained on the implicit benchmark (16,509 / 2,064 / 2,064, single-source
ImplicitHate, `implicit3` labels).

| Model | macro-F1 | not_hate F1 | explicit_hate F1 | implicit_hate F1 | implicit-discrimination AUC |
|---|---|---|---|---|---|
| Classical floor (TF-IDF, linear) | 0.5620 | - | - | 0.5562 | 0.7610 |
| BERT-mini, fine-tune only | 0.5549 | 0.8347 | 0.2857 | 0.5443 | 0.8196 |
| HateBERT specialist | **0.6029** | 0.8352 | 0.3662 | 0.6072 | **0.8247** |

**The compact student is below a bag of n-grams on macro-F1 and far above it on ranking.** BERT-mini
scores 0.5549 against the floor's 0.5620, and 0.5443 against 0.5562 on the implicit class itself,
yet its implicit-discrimination AUC is 0.8196 against 0.7610. It separates implied abuse from benign
text substantially better than the classical model and still loses on F1, because it cannot place a
threshold. This is precisely the failure a threshold-free metric was introduced to expose (F19), and
it is the first time the two measures have disagreed in direction rather than degree.

**The explicit class is the hard one here, which is the opposite of the intuition.** Both neural
models score in the 0.29 to 0.37 range on `explicit_hate` against 0.83 on `not_hate`. The class has
864 training rows against 10,616, so this is mostly scarcity; it also means macro-F1 on this
benchmark is dominated by a class the paper does not argue about, which is a further reason to lead
with the discrimination AUC.

The specialist clears the floor (0.6029 against 0.5620) and is the model the committee will route to.

**Adapted to the tweet task it scores 0.8931**, against BERT-large 0.8973, irony 0.8930 and HateBERT
0.8890. A teacher trained on a different corpus under a different label scheme lands inside a
0.008 band with the other three. That is not yet evidence about diversity, which needs the pairwise
kappa, but it is one more observation consistent with F20: adaptation to a shared label space pulls
cross-task specialists towards each other.

*17 September:* the kappa is measured, and it is the strongest form of that observation: 0.962 with
HateBERT (F23).

### F18. Aggregation was silently averaging two different experiments
`aggregate.py` grouped on the `mode` field inside `results.json`, which records the objective and not
the committee, so `dmthd` and `dmthd_hetero` were pooled into one row, as were `uniform` with
`uniform_hetero` and `skd` with `skd_hetero`. Every "6 seeds" row in the first summary was two
experiments of three seeds each, averaged. Found by noticing that a three-seed grid was reporting six
seeds. `tables.py` was unaffected because it reads the run directory. Recorded because it is the
class of bug that produces a plausible number nobody questions.

### F19. No method in the grid discriminates better; they differ only in how readily they fire
Across all **114 evaluated models** in the finished tweet grid, teachers and students, every mode and
every seed, the false-positive rate on benign sarcasm and the recall on ironic abuse are correlated
at **+0.673**. The quantity that would separate genuine discrimination from a shifted threshold,
recall minus false-positive rate, has a mean of **0.352** and a standard deviation of **0.053**,
while its two components range over 0.24 to 0.59 and 0.59 to 0.75 respectively. The parts move a
great deal; the difference between them barely moves.

By student, the mean margin is DeBERTa-v3-xsmall 0.394, teachers 0.382, DistilBERT 0.369, BERT-mini
0.368, BERT-small 0.333, BiLSTM 0.287. Distillation mode does not appear in that ordering at all:
what varies is the architecture, not the method. DistilBERT has both the lowest false-positive rate
(0.238) and the lowest recall (0.607); BERT-mini has the highest of each (0.385 and 0.753). They sit
at different points on one curve.

The single worst margin in the grid is 0.082, for the randomly initialised student, which is the same
answer F3 gives from the other direction: whatever ability these models have to tell implied abuse
from harmless sarcasm comes from pre-training, and nothing we added to the objective changes it.

**This is the strongest single piece of evidence for the reframing.** It says, across a hundred
models rather than one, that reporting recall at a fixed threshold measures willingness to fire and
not understanding, and that a threshold-free metric is not a stylistic preference but the only honest
way to ask the question. `python -m dmthd.tradeoff runs/tweets`.

### F20. The committee is not complementary enough for per-instance weighting to have anything to do
This is the mechanism behind F6, and it is measurable without training a single student.

The four task-adapted teachers agree with each other on their *predictions* at Cohen's kappa
**0.889 to 0.922**, pairwise. Their disagreement rate is **6.4 to 9.2 per cent**. Where they are
wrong they are mostly wrong together: pairwise error overlap **0.673 to 0.730**, and all four are
wrong on the same 4.7 per cent of the test set while all four are right on 83.2 per cent.

A per-instance weighting scheme can only express a preference where its members disagree. On this
committee that is at most one instance in eleven, and on two-thirds of the errors there is no correct
teacher to prefer. There is very little for the weights to do, and the grid shows them doing it: the
mean weights over the whole training set are 0.332 / 0.337 / 0.331 for the homogeneous committee and
0.254 / 0.259 / 0.254 / 0.233 with DeBERTa-v3, identical at every epoch and seed because the teachers
are frozen, and never far from uniform. (Corrected 17 September. The figures first given here,
0.306 / 0.311 / 0.305 / 0.233, averaged runs from committees of different sizes and did not sum to one.)

**The headroom exists but the weights cannot reach it.** An oracle that picked the right teacher for
every instance would reach **0.9526** accuracy and a macro-F1 upper bound of **0.9469**, against the
best single teacher's 0.9075 and 0.8973. So a five-point gain is theoretically available from
selection. Reliability measured by cross-entropy against the training label does not find any of it,
which is a more interesting negative result than "the method did not help": it says the *signal* used
to select experts is the wrong signal, not that selection is worthless.

This also tells us what a committee would have to look like for the idea to work: members that
disagree far more than these do. Ours were chosen for complementary *expertise*, a general encoder, an
abuse specialist and an irony specialist, and after task adaptation on the same corpus they converged
to near-identical behaviour. Task adaptation is what made their logits comparable enough to combine,
and it is also what destroyed the diversity the combination was supposed to exploit. That tension is
worth stating in the paper, because every cross-task committee will meet it.

### F22. Sharper weighting changes nothing, at any temperature from 0.05 to 5
The standing objection to F6 was that tau = 1 leaves the weights about 0.004 from uniform, so D-MTHD and
uniform averaging optimise nearly the same objective and scoring alike proves nothing. Kaggle v4 (17
September) extended the sweep down to tau = 0.05 on BERT-mini, one seed each.

The weights do sharpen: BERT-large / HateBERT / irony move from 0.332 / 0.337 / 0.331 at tau = 1 to
0.327 / 0.344 / 0.329 at 0.2, 0.323 / 0.349 / 0.328 at 0.1 and 0.318 / 0.356 / 0.326 at 0.05. The score
does not: test macro-F1 0.8385 / 0.8388 / 0.8392 at 0.2 / 0.1 / 0.05, against 0.8385 at the default,
0.8372 for uniform averaging and 0.8394 for fine-tune-only on the same seed. At tau = 0.05 the paired
intervals are +0.0007 [-0.0008, +0.0023] against the default and +0.0020 [-0.0042, +0.0077] against
uniform averaging. Across a hundredfold range of temperature the score stays inside 0.0007.

This closes open question 1 and Decision 4 in `paper/OPEN_DECISIONS.md`. The claim loses the
qualification it carried on 13 September: the weighting does not differ from uniform averaging at any
temperature tested.

### F23. The implicit specialist does not help on the tweet corpus, and neither control does
BERT-mini on the tweet corpus, three seeds unless stated (Kaggle v4):

| Configuration | Macro-F1 | F1 other_cyberbullying | Sarcasm AUC, seed 1 |
|---|---|---|---|
| Fine-tune only | 0.8393 +- 0.0003 | 0.707 | 0.773 |
| Uniform, homogeneous | 0.8385 +- 0.0018 | 0.700 | 0.774 |
| Uniform + specialist | 0.8390 +- 0.0035 | 0.699 | 0.768 |
| D-MTHD, homogeneous | 0.8378 +- 0.0011 | 0.705 | 0.773 |
| D-MTHD + specialist | 0.8373 +- 0.0025 | 0.697 | 0.768 |
| Specialist alone (1 seed) | 0.8392 | 0.709 | 0.767 |
| Pre-trained on the implicit corpus, no teacher (1 seed) | 0.8377 | 0.705 | 0.756 |

Adding the specialist: +0.0005 [-0.0059, +0.0064] under averaging, -0.0005 [-0.0069, +0.0055] under
D-MTHD. Class F1 and discrimination AUC both fall slightly. The pre-trained control has the lowest AUC
of any pre-trained model analysed.

**The mechanism is F20's, now measured on the specialist itself.** Adapted to the tweet task, the
specialist agrees with HateBERT, the model it was built from, at kappa **0.962** and disagrees with it on
**3.1 per cent** of the test set; every other pair in the committee sits at 0.889 to 0.924. 84 per cent
of HateBERT's errors are the specialist's too. It lifts the oracle over the committee only from 0.9526
to 0.9552 accuracy. A teacher trained on different data under a different label scheme came back from
task adaptation as a near-copy of its base model, which is "task adaptation buys comparability and
spends diversity" in its sharpest form.

**And nothing else moves discrimination either.** Over the 25 pre-trained BERT-mini models analysed,
which cover every objective and committee, four temperatures, the T, alpha and delta sweeps, every
ablation and both controls, sarcasm-discrimination AUC has mean 0.771, standard deviation 0.005, range
0.756 to 0.777. The randomly initialised student scores 0.613. The auxiliary irony head is part of this:
without it the AUC is 0.775, with it 0.773.

This answers open question 5 on the tweet corpus, the tweet half of open question 4, and open question
2. What remains is the implicit benchmark, where the specialist's knowledge is in-domain rather than
transferred.

### F24. The weights shift towards the specialist, slightly, and as much towards HateBERT
`weight_routing` on the training split: mean weight on `other_cyberbullying` minus mean weight on the
four targeted classes, 2,000-sample bootstrap, uniform weight 0.200 over the five cached teachers.

| Teacher | tau = 0.05 | tau = 1 |
|---|---|---|
| HateBERT | +0.0486 [+0.0460, +0.0512] | +0.0105 [+0.0094, +0.0116] |
| Implicit specialist | +0.0441 [+0.0416, +0.0465] | +0.0090 [+0.0080, +0.0100] |
| RoBERTa-irony | +0.0184 [+0.0159, +0.0210] | +0.0053 [+0.0043, +0.0064] |
| BERT-large | -0.0063 [-0.0089, -0.0035] | +0.0011 [-0.0002, +0.0023] |
| DeBERTa-v3-base | -0.1048 [-0.1067, -0.1028] | -0.0259 [-0.0274, -0.0245] |

In the four-teacher committee the students trained with, the specialist's mean weight at tau = 1 is
0.2519 against 0.25. So the weighting does route by class, in the direction D15 intended, but:
- with tens of thousands of training instances almost any difference excludes zero, so the size is the evidence,
  and at the trained temperature the specialist's shift is a twentieth of the uniform weight;
- it is not specific to the specialist, which F23's kappa predicts;
- the weights are computed from cross-entropy against the gold label, so a shift says a teacher fits
  `other_cyberbullying` better, not that it reads implication;
- **the measurement pooled all five cached teachers**, a committee no student trained with. The driver
  called `weight_routing` without `--teachers`, so the softmax spanned the whole cache. Which teacher
  leads on an instance is the same in any sub-committee; the sizes are not. Fixed on 17 September: the
  driver measures each trained committee separately, into `weight_routing/<committee>/`, the tau
  diagnostic likewise runs on the homogeneous committee, and the tables prefer the per-committee files.

*21 September, measured per trained committee (v5).* In the `spec` committee at tau = 1 the
specialist's contrast is +0.0027 [+0.0018, +0.0036] on a uniform weight of 0.25, HateBERT's +0.0044,
BERT-large's -0.0060, irony's -0.0011; at tau = 0.05 they are +0.0187, +0.0235, -0.0347 and -0.0075.
In the homogeneous committee HateBERT is the only teacher that gains on indirect abuse (+0.0081 at
tau = 1, +0.0449 at 0.05). The pooled estimate above overstated the specialist's share: in the
committee the students trained with, it is a hundredth of the uniform weight.

### F25. What the tweet students know about abuse does not include implication
BERT-mini trained on tweets, applied with no further training to the implicit benchmark's 2,064 test
posts (seed 1). It calls 81.0 per cent of them abusive against a true rate of 35.7 per cent (binary
macro-F1 0.423, ROC-AUC 0.608). On implicit hate against not-hate it catches 85.1 per cent at a
78.8 per cent false-positive rate and ranks the two at an AUC of **0.600**, against **0.820** for the
same architecture trained on that corpus (F21). D-MTHD: binary macro-F1 0.411, implicit-hate AUC 0.600.
It over-fires, as the tweet model did on Wikipedia (F9), and it cannot rank the one distinction the
paper is about.

### F26. Two sections of the results draft quoted the laptop, and one conclusion changes on Kaggle
F17's rule is that every number in the paper comes from the Kaggle grid. Two did not. The diagnostic in
F10 (results 5.6) came from the laptop's fine-tune-only model (macro-F1 0.8770), and the tweet-Wikipedia
transfer in F9 (results 5.8) from the laptop's tweet and Wikipedia models. Found while replacing the
draft's numbers with the v4 analysis, which runs the same measurement on the Kaggle models.

BERT-mini fine-tune-only on Kaggle, seed 1, against the laptop figures in brackets:
- sarcasm-discrimination AUC 0.773 (0.776); mean p(abusive) 0.748 on ironic abuse, 0.449 on benign
  sarcasm (0.693, 0.375);
- at the 0.5 cut, recall 0.792 at a false-positive rate of 0.395 (0.704 at 0.317): nearly the same
  margin at a different operating point, which is F19 again;
- **no threshold up to 0.90 brings the false-positive rate under 10 per cent**, on this model or on any
  of the 26 analysed (the laptop model reached it at 0.90, with recall 0.427);
- **the lexical split changes.** Ironic-abuse recall is 0.91 on the 43 items with profanity and 0.76 on
  the 1,517 without (0.72 and 0.67). Across the 25 pre-trained Kaggle models the with-profanity advantage
  is 6 to 16 points and never negative. "Not a profanity detector" survives, since items without
  profanity are still caught about three times in four on every model, but "sees indirect abuse about as
  well as explicit" does not. The implicit-abuse split rests on 11 items and supports nothing;
- `other_cyberbullying` recall 0.710 with 109 errors to `not_cyberbullying`; `not_cyberbullying` recall
  0.648 with 126 the other way (0.786 / 77 and 0.681 / 133).

Results 5.6 now uses the Kaggle numbers, and 5.8 drops the tweet-Wikipedia transfer until the Wikipedia
grid runs. The laptop AUC stays in 5.10, where the laptop is the subject: a 0.0386 gap in macro-F1 is a
0.003 gap in discrimination.

### F27. Of 51 paired comparisons in the grid, one interval excludes zero: removing pre-training
`python -m dmthd.significance` recomputes every comparison the paper states from the saved test
predictions, with the test `aggregate --compare` uses (deterministic, so it reproduces the Kaggle
console exactly). The draft had copied intervals from that console, whose log capture prints each
comparison's output under the next command's line; the copied ones turn out to be correct, but the
table is now generated rather than transcribed.

Of the 51 comparisons, only the randomly initialised student against fine-tune-only excludes zero:
-0.0488 [-0.0603, -0.0376]. **None of the 27 comparisons of a distilled student against fine-tuning
does**, on any of the five students, including the best configuration per student: BERT-mini +0.0012
[-0.0057, +0.0079], BERT-small +0.0040 [-0.0058, +0.0126], DistilBERT +0.0068 [-0.0013, +0.0139],
DeBERTa-v3-xsmall +0.0040 [-0.0024, +0.0109], BiLSTM +0.0071 [-0.0039, +0.0195]. F3's "distillation
helps every student" was a statement about point estimates. Five of five positive is still worth
reporting, as consistency of direction, but a 4,326-post test set cannot resolve differences this size.

### F28. The reliability signal is measured where every teacher has memorised the label
The weights `w_k(i) = softmax_k(-CE_k(i)/tau)` are computed from cached teacher logits on the *training
split*, the same split each teacher was fine-tuned on. By their last epoch the teachers' training losses
are 0.033 (BERT-large), 0.059 (HateBERT), 0.077 (irony), 0.061 (implicit specialist) and 0.227
(DeBERTa-v3-base): on the data the weights are read from, every task-adapted teacher assigns the gold
label a probability near one on nearly every instance. Cross-entropy differences between teachers are
therefore a few hundredths, which is why the weights stay within 0.03 of uniform even at tau = 0.05
(F22), and why DeBERTa, the one teacher that did not memorise the split, is the one whose weight moves.

The same fact bears on the distillation itself. On the training split the teachers' soft labels are
close to the one-hot gold labels, so the KL term carries little that the cross-entropy term does not,
and single-teacher, uniform and weighted distillation all reduce to fine-tuning with a slightly
smoothed target (F27). This is a known hazard of distilling a memorising teacher on its own training
data (Stanton et al. 2021; Beyer et al. 2022) and it applies with full force to every error-weighted
multi-teacher scheme that measures reliability against the gold label on the training set, MT-BERT
and CA-MKD included. Two consequences: reliability must be estimated out of sample (cross-fitted
teachers, or a validation-fitted gate), and distillation needs data the teachers have not fitted
(held-out folds or an unlabelled transfer set) before its soft labels carry information.

### F29. The committee is worth 0.5 to 0.8 points over its best member, and the student receives none of it
Scored directly from the teachers' saved test probabilities (`scratchpad/teacher_ensemble.py`, 17
September), uniform averaging of the committee beats the best single teacher: homogeneous 0.9026,
with the specialist 0.9000, with DeBERTa 0.9051, all five 0.9036, against BERT-large's 0.8973. No
gold-free combination improves on the uniform mean by more than 0.003: confidence weighting 0.9018 to
0.9055, entropy weighting 0.9011 to 0.9052, taking the most confident teacher 0.9000 to 0.9015, and a
stacked logistic-regression gate fitted by five-fold cross-validation over the test set 0.9041 to
0.9055. The oracle sits at 0.945 to 0.955 accuracy, but where the teachers disagree (10 to 14 per cent
of items) their own probabilities do not say which of them is right: there the uniform mean is right
on 55 to 62 per cent and some teacher on about 90 per cent, and the gate cannot tell them apart.

So even a corrected reliability signal (F28) has a ceiling of about +0.3 macro-F1 over uniform
averaging on this committee, and the five-point oracle gap is not reachable from teacher outputs. The
students, meanwhile, do not receive even the 0.5 to 0.8 the committee has: uniform multi-teacher
distillation against single-teacher distillation is -0.0019, +0.0017, -0.0003, -0.0003 and +0.0014
across the five students, every interval containing zero (`significance.csv`, rows "does a committee
beat one teacher"). A better label on the same 34,607 training texts does not make a better student.
What a small student needs is more texts to imitate the committee on, which is where the transfer-set
literature (Tang et al. 2019; Jiao et al. 2020; Turc et al. 2019) locates the gains, and where a
committee has a job a single teacher does not: it is the better labeller.

### F30. The Kaggle students learn more slowly from the first epoch, not just at the end
On BERT-mini fine-tune-only, seed 1, the training loss after epoch 1 is 1.019 on Kaggle against 0.874
on the laptop, and validation macro-F1 0.799 against 0.843; the gap persists at every epoch and ends at
0.854 against 0.884 on validation. Same steps, learning rate, schedule, clipping and data; mixed
precision accounts for 0.0014 (F17). The local environment is torch 2.14 with transformers 5.17; the
Kaggle image is older. Two experiments isolate the cause without a GPU: pin transformers 4.4x locally
and re-run seed 1 (30 minutes of CPU), and run the Kaggle notebook with `GPU=0` (F17). Until one of
them runs, every absolute BERT-mini and BERT-small number on Kaggle should be read as possibly
under-trained by four points, which is also why both sit below the classical floor; the comparisons
within the grid remain valid because every run shares the condition.

### F31. Distillation's gain is in the data the student imitates on, not in the committee or its weighting
Kaggle v5, 21 September: the D19 arms on BERT-mini, three seeds each, beside the in-sample runs of
the same objectives (F3, F6, F22).

| Objective | Labelled split only (in sample) | + 42,013 unlabelled tweets (out of sample) |
|---|---|---|
| Single teacher (BERT-large) | 0.8405 ± 0.0017 | 0.8466 ± 0.0016 |
| Uniform committee | 0.8385 ± 0.0018 | 0.8462 ± 0.0016 |
| Uniform committee + DeBERTa | 0.8378 ± 0.0024 | 0.8477 ± 0.0026 |
| Reliability-weighted committee | 0.8378 ± 0.0011 (in-sample weights) | 0.8484 ± 0.0011 (out-of-sample weights) |
| Hard labels only | 0.8393 ± 0.0003 (gold; fine-tune only) | 0.8412 ± 0.0013 (gold + committee pseudo-labels) |

Every soft-label objective gains 0.006 to 0.011 from the transfer set (one teacher +0.0061 [-0.0009,
+0.0133]; the committee +0.0077 [+0.0000, +0.0155]), and the gain is the same whatever the labeller:
the four soft-label arms lie within 0.0022 of each other. Hard pseudo-labels on the same text, with
the same number of optimisation steps, give +0.0020, so most of the gain is carried by the soft labels
rather than by the extra text or the extra steps. Over the whole tweet grid the intervals that exclude
zero are now three: removing pre-training, and two of these arms.

The gain lands where the in-sample runs never moved: F1 on `other_cyberbullying` 0.707 -> 0.710 to
0.720 and on `not_cyberbullying` 0.644 -> 0.655 to 0.663, the two classes that confuse each other.
Calibration goes the other way: ECE 0.042 for fine-tuning and for the hard-label arm, 0.076 to 0.080
for every soft-label arm, in sample or out.

This is the constructive half of F28 and F29. On the training split the teachers' soft labels are the
gold labels and distillation reduces to fine-tuning; on text they have not fitted, their soft labels
carry information the gold labels do not, and the student receives it. It is a distillation result,
not a multi-teacher result, and it is modest: +0.7 to +0.9 macro-F1 on an 11M student from 1.2 times
the training split in unlabelled text, in one session. Whether it grows with more text is the open
question that decides whether it can lead a paper (open question 7).

Two limits. The transfer arms take 2.2 times the optimisation steps of fine-tuning (76,620 rows per
epoch against 34,607); the hard-label arm shares those steps and gains 0.002, which bounds the
steps-alone explanation but does not measure it, and a fine-tune-only run at matched steps should.
And the absolute BERT-mini numbers remain subject to F30.

### F32. The transfer set moves the student's operating point on sarcasm, and not towards implication
Seed 1 of each transfer arm against fine-tune-only (sarcasm-discrimination AUC 0.773, recall 0.792
at a false-positive rate of 0.395 at the 0.5 threshold): one teacher 0.765 (0.692 / 0.308), the
DeBERTa committee 0.762 (0.712 / 0.350), the uniform committee 0.753 (0.657 / 0.302), the corrected
weighting 0.753 (0.658 / 0.304), hard pseudo-labels 0.727 (0.700 / 0.390). Every transfer-trained
student fires less on sarcasm of both kinds, and three of the five rank ironic abuse against benign
sarcasm slightly worse than any of the 28 in-sample variants, whose AUC spans 0.756 to 0.777 (mean
0.771, sd 0.004; the three sharp-tau runs, analysed in v5, sit at 0.772). The transfer text is
offensive-language data (Davidson, OLID) in which abuse is mostly explicit, so what the teachers'
soft labels teach there is explicit abuse; nothing in the set carries the distinction the probes
measure. The gain of F31 is a task gain, not an implication gain, which is what F19 and F23 predict,
and it says that if a transfer set is to help with implication, its composition is the first thing
to change, not its size.

---

## 5. What the paper may and may not claim

**May claim, with the evidence above:**
- A 67M student distilled from a single large teacher reaches 0.8963 against that teacher's 0.8973,
  and 0.8975 with a uniform committee, at a fifth of the parameters and a fifth of the latency (F4,
  F11).
- Distillation raises the score of all five students, from +0.0012 to +0.0071, but no paired interval
  excludes zero (F3, F27). Pre-training matters forty times more than any component of the method,
  and removing it is the only difference in the grid whose interval excludes zero (F27).
- **Per-instance dynamic weighting does not beat uniform averaging on any of five students at any tau
  from 0.05 to 5** (F6, F22), and we can say why: the committee members agree at kappa 0.889 to 0.922
  and disagree on at most one instance in eleven, so there is almost nothing for a weighting to
  express, while an oracle over the same teachers would gain five points (F20). The signal used to
  select experts is the wrong signal, which is a more useful negative result than "it did not help".
- A specialist teacher that learned implication from labelled data adds nothing on the tweet corpus,
  whether in a committee, alone, or replaced by pre-training on its data, and the reason is measurable:
  task adaptation turns it into a near-copy of its base model, kappa 0.962 (F23). The weights shift
  towards it by class, but by a twentieth of the uniform weight and no more than towards that base
  model (F24).
- Indirect-abuse detection fails through discrimination, and the right metric is threshold-free (F10,
  F26). The detector leans on profanity, 6 to 16 points of recall, without depending on it (F26).
  Across 133 models, false-positive rate and recall on the sarcasm probes correlate at +0.712 while
  their difference has a standard deviation of 0.051: no method discriminates better, they differ only
  in how readily they fire (F19). Over 25 variants of the headline student the threshold-free AUC stays
  between 0.756 and 0.777 (F23), and no threshold holds false positives on harmless sarcasm under 10
  per cent on any model (F26).
- On a corpus that labels implication, a lexical model ranks implied hate above ordinary text at an
  AUC of 0.761, close to the 0.773 the tweet student reaches on ironic abuse against benign sarcasm:
  the same difficulty, measured twice on different data (F13, F26). A tweet-trained student ranks that
  corpus's implicit hate against not-hate at 0.600, against 0.820 for the same architecture trained on
  it (F25).
- Two expert corpora agree on 99.7 per cent of shared texts about whether they are hateful and on
  48.2 per cent about whether the hate is implied (F14), both carry schema traps that silently
  produce a wrong benchmark (F15), and pooling them creates a source shortcut that inflates the
  aggregate score while costing 0.075 F1 on implied hate (F16).
- One of the two widely used corpora is materially dirty (F1). That the two do not transfer to each
  other (F9) is measured on the laptop only and waits for the Wikipedia grid (F26).
- A deployment profile a practitioner can act on (F11).
- Where distillation's gain is: on 42,013 unlabelled in-domain tweets a single teacher's soft labels
  raise the 11M student by 0.006 to 0.009 macro-F1, every soft-label seed above every no-teacher
  seed, while the same objectives on the labelled split gave nothing; the committee, the corrected
  weighting and hard pseudo-labels add nothing to that (F31). The gain is a task gain and leaves
  sarcasm discrimination where it was or slightly lower (F32).

**May not claim:**
- That D-MTHD beats uniform averaging. It does not, at any tau from 0.05 to 5 (F22).
- That the method beats the no-teacher control on the headline student. Six paired bootstrap
  intervals, all containing zero (F3, F23).
- That in-sample distillation improves any student. It raises every one, within test-set noise
  (F27). Out of sample it does, on the headline student, by less than a point (F31); that the gain
  grows with more text, or holds on the other students, has not been measured.
- That a committee labels a transfer set better than one teacher, for the student. It does not
  (F31), although the committee's own predictions are better (F29).
- Anything about BERT-mini or BERT-small in absolute terms, until the environment discrepancy in F17
  is resolved. The *comparisons within* the Kaggle grid are valid because everything in it was
  trained the same way; the absolute numbers are not yet trustworthy.
- That the implicit specialist helps, or that the weighting routes implication to it (F23, F24).
  Nothing has yet been measured on the implicit benchmark's own student grid, which is the one place
  that claim could still be earned.
- That the auxiliary irony head improves sarcasm discrimination. Without it the AUC is 0.775, with it
  0.773, one seed (F23).
- That homogeneity causes anything. The hidden-term contribution on the headline student is 0.0009.
- "Student exceeds teachers", except in the narrow sense of F4, where DistilBERT with a uniform
  heterogeneous committee reaches 0.8975 against BERT-large's 0.8973, a difference far inside noise.

---

## 6. Open questions

1. Does D-MTHD beat uniform averaging once tau is chosen properly? If not, the paper's story becomes
   "a cross-task specialist committee and a sarcasm head work; the dynamic weighting adds nothing",
   which is still publishable but a smaller claim. The decision waits on the sweep, not on a guess.
   **Answered 17 September: no, at any tau from 0.05 to 5 (F22).**
2. Does the auxiliary irony head raise sarcasm-discrimination AUC above the 0.776 baseline?
   **Answered 17 September: no.** Without it 0.775, with it 0.773, against a Kaggle baseline of 0.773
   (F23, F26).
3. Does the hidden-state term help BERT-lineage students more than DeBERTa or BiLSTM?
4. **The question the paper now turns on.** Does the implicit specialist in the committee raise
   implicit-hate F1 above the fine-tune-only student, on the implicit benchmark and through transfer
   to the tweet corpus? Four numbers decide it: implicit_hate F1 against the 0.453 floor;
   sarcasm-discrimination AUC against the 0.776 baseline; focus-class transfer from tweets to the
   implicit test set; and whether the committee routes implicit-looking instances to the specialist
   rather than spreading weight evenly (which is F6's question asked where it should finally bite).
   **The tweet half is answered, and the answer is no** (F23 to F25): no gain in F1 or AUC, near-chance
   transfer, and routing too small to matter. The implicit benchmark half is now the whole question.
5. Does distilling from the specialist beat simply fine-tuning the student on the implicit corpus?
   If it does not, the specialist is a data argument rather than a distillation one, and the paper
   must say so. This control has to be in the grid from the start, not added after a reviewer asks.
   **On the tweet corpus neither helps** (F23); on the implicit benchmark it is still open.
6. What is the human agreement on sarcastic abuse? The 300-item annotation answers this, and it also
   tells us how much of the benign-sarcasm probe is mislabelled. F14's 346 disagreements are the
   same question measured on a different corpus.
7. Does the out-of-sample gain grow with the transfer set? F31 measures one size, 1.2 times the
   training split, and gets +0.007 to +0.009. A curve over about 10,000, 42,000 and 100,000 or more
   unlabelled tweets (TweetEval's other configurations and whatever the provenance rule allows) on
   BERT-mini, with a fine-tune-only run at matched optimisation steps as the control, decides
   whether the constructive result can lead the paper or closes it. About three GPU-hours.

---

## 7. How the story changed

Worth keeping because the paper's framing came out of these turns, and because a reviewer asking
"why this design?" is really asking for this section.

1. **"Multi-teacher distillation for cyberbullying."** The Phase-2 framing. Killed by its own numbers:
   the no-teacher control beat everything.
2. **"Fix the pipeline first."** Clean data, bigger teachers, smaller students, a real control. This
   produced F2 and F4 and made the project viable again.
3. **"Homogeneity is our novelty."** Dropped on inspection: it is the field's default setting. Turned
   into a measurable question instead (D7).
4. **"Sarcasm is the differentiator."** The auxiliary irony head and the probe sets. Still the
   strongest novelty candidate.
5. **"The model cannot see indirect abuse."** The natural reading of the low recall, and wrong: F10
   shows it sees indirect abuse but cannot separate it from harmless sarcasm. This changed the metric
   (D13) and reframed the contribution from detection to discrimination.
6. **"The dynamic part may be doing nothing."** F6, found by reading weights that nobody had looked
   at. The paper's central mechanism is on hold pending a hyper-parameter it always had.
7. **"We are testing for implication but never training for it."** The current turn. Both benchmarks
   label topic, not indirectness, so no student had ever seen an example marked as abuse-by-
   implication and no teacher could hold that knowledge. A committee cannot route to an expertise
   none of its members has. Hence the third benchmark and the implicit specialist (D14), and hence
   the classical floor that splits 0.747 / 0.453 and shows the gap is real (F13). The paper's centre
   of gravity moves from "a small model can match a large one" to "a small model can be taught to
   read implication, and here is the mechanism that teaches it".
8. **"The specialist does not teach it either."** 17 September. The committee with the specialist,
   the specialist alone and pre-training on its data all land where the no-teacher student is, and the
   reason is measurable: task adaptation made the specialist a near-copy of HateBERT (F23). Sharper
   weighting changes nothing (F22), and no distilled student beats fine-tuning by more than the test
   set can resolve (F27). With F19, the grid's clearest message is about what distillation does not
   transfer: every objective, committee and temperature leaves the ability to tell implied abuse from
   harmless sarcasm where pre-training put it. The centre of gravity moves again, from "here is the
   mechanism that teaches implication" to "here is what distillation does and does not transfer about
   implication, measured", with the implicit benchmark as the one place a constructive result could
   still come from.
9. **"The gain is in the data, not the committee."** 21 September. The transfer set was the one
   lever the audit left, and it moved the student: +0.007 to +0.009 with every seed above every
   no-teacher seed, from soft labels on text the teachers had not fitted. One teacher does it as well
   as three; the weighting and the specialist add nothing here either; and the gain is in the task,
   not in implication. The paper's constructive sentence is now "distil where the teacher is
   uncertain", and whether that sentence leads the paper or closes it depends on one more run.

---

## 8. Where everything lives

| What | Where |
|---|---|
| Chronological record of every run and number | `LOG.md` |
| Reasoning, definitions, claims | this file |
| Q1 requirement tracker with evidence pointers | `paper/q1_checklist.md` |
| Method with notation, equations, algorithm | `paper/method_draft.md` |
| Setup section with all corpus counts | `paper/setup_draft.md` |
| Related work skeleton, four themes | `paper/related_work_draft.md` |
| Ethics, data, reproducibility, limitations | `paper/statements_draft.md` |
| Verified bibliography, 63 entries | `paper/references.bib`, `paper/refs_report.csv` |
| What happens to each Phase-2 citation | `paper/phase2_citation_map.md` |
| Tables generated from results | `python -m dmthd.tables` |
| Human annotation kit | `annotation/` |
| Decisions that belong to the team, with options and logic | `paper/OPEN_DECISIONS.md` |
| Corpus manifests, so a rebuild can be verified without redistributing text | `paper/manifests/` |
| Implicit benchmark construction and its counts | `src/dmthd/prepare_implicit.py`, `data/implicit/report.json` |
| Sarcasm and implicit-abuse measurements | `src/dmthd/implicit_analysis.py` |
| Weighting temperature diagnostic | `src/dmthd/tune_tau.py` |
| Driver smoke test, all four passes | `scripts/smoke_driver.py` |
| Paired bootstrap for every comparison the paper states | `python -m dmthd.significance`, `paper/tables/significance.csv` |
| Teacher agreement and oracle with the implicit specialist | `paper/complementarity_with_specialist.json` |
| Per-model sarcasm analysis, first seed, Kaggle v4 | `paper/results_tweets_implicit_seed1.csv` |
