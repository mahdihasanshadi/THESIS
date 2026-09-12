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

The Phase-2 report claimed a working method. Inspecting its own outputs showed it did not work, and
that is the fact everything since has been built to fix.

| Phase-2 model | Macro-F1 |
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

**Not mixed precision.** Trained with and without it on the same seeds and hardware: validation
macro-F1 by epoch 0.7993 / 0.8276 / 0.8423 / 0.8499 with fp16 against 0.8008 / 0.8299 / 0.8438 /
0.8505 without. About 0.0015 apart, which is noise.

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
One environment is deterministic to sixteen digits across re-runs; the other lands **0.0386** away
from it on the same script, data and seed. That is a systematic difference, not seed variance, and
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
mean weights over the whole training set are 0.306 / 0.311 / 0.305 / 0.233, identical at every epoch
because the teachers are frozen, and never far from uniform.

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

---

## 5. What the paper may and may not claim

**May claim, with the evidence above:**
- A 67M student distilled from a single large teacher reaches 0.8963 against that teacher's 0.8973,
  and 0.8975 with a uniform committee, at a fifth of the parameters and a fifth of the latency (F4,
  F11).
- Distillation helps all five students, from +0.0012 to +0.0071, and pre-training matters forty times
  more than any component of the method (F3).
- **Per-instance dynamic weighting does not beat uniform averaging on any of five students at any tau
  tested so far** (F6), and we can say why: the committee members agree at kappa 0.889 to 0.922 and
  disagree on at most one instance in eleven, so there is almost nothing for a weighting to express,
  while an oracle over the same teachers would gain five points (F20). The signal used to select
  experts is the wrong signal, which is a more useful negative result than "it did not help".
- Indirect-abuse detection fails through discrimination, not lexical blindness, and the right metric
  is threshold-free (F10). Across 114 models in the finished grid, false-positive rate and recall on
  the sarcasm probes correlate at +0.673 while their difference has a standard deviation of 0.053:
  no method discriminates better, they differ only in how readily they fire (F19).
- On a corpus that labels implication, a lexical model ranks implied hate above ordinary text at an
  AUC of 0.761, close to the 0.776 the tweet student reaches on ironic abuse against benign sarcasm:
  the same difficulty, measured twice on different data (F10, F13).
- Two expert corpora agree on 99.7 per cent of shared texts about whether they are hateful and on
  48.2 per cent about whether the hate is implied (F14), both carry schema traps that silently
  produce a wrong benchmark (F15), and pooling them creates a source shortcut that inflates the
  aggregate score while costing 0.075 F1 on implied hate (F16).
- Two widely used corpora do not transfer to each other, and one of them is materially dirty (F1, F9).
- A deployment profile a practitioner can act on (F11).

**May not claim:**
- That D-MTHD beats uniform averaging. The grid says it does not, at every tau tried. The sharp tau
  values may change this and have not run; nothing may be claimed in either direction until they do.
- That the method beats the no-teacher control on the headline student. Four paired bootstrap
  intervals, all containing zero (F3).
- Anything about BERT-mini or BERT-small in absolute terms, until the environment discrepancy in F17
  is resolved. The *comparisons within* the Kaggle grid are valid because everything in it was
  trained the same way; the absolute numbers are not yet trustworthy.
- Anything at all about the implicit specialist. The corpus, the scheme, the teacher and the
  measurements exist and are smoke-tested; not one model has been trained on a GPU yet.
- That the auxiliary irony head improves sarcasm discrimination. Measured only on the fine-tune-only
  baseline so far.
- That homogeneity causes anything. The hidden-term contribution on the headline student is 0.0009.
- "Student exceeds teachers", except in the narrow sense of F4, where DistilBERT with a uniform
  heterogeneous committee reaches 0.8975 against BERT-large's 0.8973, a difference far inside noise.

---

## 6. Open questions

1. Does D-MTHD beat uniform averaging once tau is chosen properly? If not, the paper's story becomes
   "a cross-task specialist committee and a sarcasm head work; the dynamic weighting adds nothing",
   which is still publishable but a smaller claim. The decision waits on the sweep, not on a guess.
2. Does the auxiliary irony head raise sarcasm-discrimination AUC above the 0.776 baseline?
3. Does the hidden-state term help BERT-lineage students more than DeBERTa or BiLSTM?
4. **The question the paper now turns on.** Does the implicit specialist in the committee raise
   implicit-hate F1 above the fine-tune-only student, on the implicit benchmark and through transfer
   to the tweet corpus? Four numbers decide it: implicit_hate F1 against the 0.453 floor;
   sarcasm-discrimination AUC against the 0.776 baseline; focus-class transfer from tweets to the
   implicit test set; and whether the committee routes implicit-looking instances to the specialist
   rather than spreading weight evenly (which is F6's question asked where it should finally bite).
5. Does distilling from the specialist beat simply fine-tuning the student on the implicit corpus?
   If it does not, the specialist is a data argument rather than a distillation one, and the paper
   must say so. This control has to be in the grid from the start, not added after a reviewer asks.
6. What is the human agreement on sarcastic abuse? The 300-item annotation answers this, and it also
   tells us how much of the benign-sarcasm probe is mislabelled. F14's 346 disagreements are the
   same question measured on a different corpus.

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
