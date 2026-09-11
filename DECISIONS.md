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
| **Implicit benchmark** | The third corpus, built here: `not_hate`, `explicit_hate`, `implicit_hate`, from ISHate original rows and Implicit Hate Corpus stage 1. The only one of our three where abuse-by-implication is a label rather than a hidden subset. |
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

### F3. A compact student alone does not beat the classical floor on tweets
BERT-mini fine-tune-only: 0.8779 / 0.8770 / 0.8760, mean 0.8770 ± 0.0010, against the floor's 0.8798;
paired bootstrap on seed 1 gives −0.0018, 95% CI [−0.0108, +0.0070]. This is the gap distillation has
to fill, and it is reported as such rather than buried.

### F4. Distillation closes almost the whole teacher-student gap
DistilBERT with single-teacher distillation: 0.8963 / 0.8961 / 0.8965 (mean 0.8963, spread 0.0004)
against BERT-large's 0.8973. A 66M student lands within 0.001 of a 335M teacher. This is the paper's
central efficiency claim and it is now evidence, not hope.

### F5. The two corpora tell opposite stories, and that is the framing
On Wikipedia the same 11M student reaches 0.8884 ± 0.0044 on its own, above the floor (0.8759) and
0.005 below the Phase-2 BERT-large teacher (0.8936). Where a compact model is already near teacher
level, distillation has little room; where it is two points behind, it has a lot. The paper should
say when distillation is worth doing, not only that it works.

### F6. The dynamic weighting is currently indistinguishable from uniform averaging
Epoch-mean weights in every heterogeneous D-MTHD run: 0.254 / 0.259 / 0.254 / 0.233 against a uniform
0.25, identical across epochs and seeds. Not a bug: with frozen teachers the weights are a fixed
function of the data. The cause is tau: on a synthetic three-teacher cache the weight entropy ratio
is 0.94 at tau = 1 (near-uniform) and 0.50 at tau = 0.1. The sweep now covers tau in
{0.05, 0.1, 0.2, 0.5}. **Until this is resolved, no claim about dynamic weighting may be made.**
Current evidence: DistilBERT uniform 0.8976 versus D-MTHD 0.8966 on the heterogeneous committee.

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
sarcasm awareness. **Probe metrics are therefore reported for tweet-trained models only.**

### F13. On a corpus that labels implication, the classical floor splits in two
The implicit benchmark, TF-IDF plus logistic regression, test macro-F1 0.6809 and accuracy 0.7663.
Per class: not_hate 0.8421, explicit_hate 0.7474, **implicit_hate 0.4530**. A bag of n-grams handles
explicit abuse and collapses on implication, a 29-point F1 gap inside a single corpus with no model
of ours involved. On the tweet corpus the same floor reaches 0.8798 and our fine-tune-only student
cannot beat it (F3); here there is room for a method to earn its place, and the room is exactly where
the paper says it contributes.

Corpus construction, from `report.json`: ISHate 63,758 raw, 29,116 after dropping augmentations,
28,763 after dropping ToxiGen provenance; Implicit Hate Corpus stage 1 contributes 21,480; 1,581 rows
removed as probe holdout; 48,662 combined, 47,181 after cleaning, split 37,744 / 4,718 / 4,719.

### F14b. ISHate's subtlety layer is not usable, and the paper should not pretend otherwise
ISHate ships a `subtlety_layer` (Subtle / Non-Subtle) that looks like a natural second axis for this
paper. It is not: in our test split only 10 of 4,719 rows carry it, and all 10 are `explicit_hate`,
while the implicit rows are almost entirely Non-Subtle. Whatever that column annotates, it is not the
implicit/explicit distinction and there is nowhere near enough of it to measure anything. The column
is carried through `prepare_implicit.py` for provenance and is used by nothing. Recorded so that a
later reader does not rediscover it and assume it works.

### F14. The two implicit corpora disagree with each other on 346 texts
De-duplicating the combined corpus found 346 texts present in both sources with different labels,
710 rows in total, all removed. That disagreement rate is a measurement of how hard the implicit
label is for humans, obtained for free, and it belongs in the paper next to the annotation study.

### F15. Two schema traps in the source corpora, either of which yields a silently wrong benchmark
Recorded because anyone reproducing this will hit them.
- ISHate records the benign class **only** in `hateful_layer`. Its `implicit_layer` holds
  {Explicit HS, Implicit HS} and is empty for every Non-HS row, so reading that column alone deletes
  all 17,869 benign examples and leaves a corpus with no negative class.
- The SALT-NLP `ImplicitHate` mirror ships **stage 2 only**: 6,346 rows, no benign class. Stage 1,
  the three-class set with 21,480 rows, is at `tasksource/implicit-hate-stg1`.

---

## 5. What the paper may and may not claim

**May claim, with the evidence above:**
- A 66M student distilled from a single large teacher lands within 0.001 macro-F1 of that teacher, at
  a fraction of its cost (F4, F11).
- Distillation is worth doing exactly where a compact model is behind, and barely worth it where it
  is not (F3, F5).
- Indirect-abuse detection fails through discrimination, not lexical blindness, and the right metric
  is threshold-free (F10).
- Two widely used corpora do not transfer to each other, and one of them is materially dirty (F1, F9).
- A deployment profile that a practitioner can act on (F11).
- On a corpus that labels implication, a lexical model reaches 0.747 F1 on explicit abuse and 0.453 on
  implicit abuse, so the difficulty is in the implication and not in the topic (F13).
- Two public implicit-hate corpora disagree on 346 shared texts (F14), and both carry schema traps
  that silently produce a wrong benchmark (F15).

**May not claim yet:**
- That dynamic weighting beats uniform averaging. Blocked by F6 until the tau sweep resolves it.
- That the auxiliary irony head improves sarcasm discrimination. The measurement exists (D13) but has
  only been run on the fine-tune-only baseline; the distilled variants are still on Kaggle.
- That homogeneity causes anything. Blocked until the hidden-term 2x2 completes.
- "Student exceeds teachers." No seed-tested evidence.
- Anything at all about the implicit specialist. The corpus, the scheme, the teacher and the
  measurements exist and are smoke-tested; not one model has been trained on a GPU yet. Every number
  for the implicit benchmark in this file is the classical floor and nothing more.

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
