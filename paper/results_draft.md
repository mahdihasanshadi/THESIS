# Results (draft, 17 September 2026)

Every number here is read from a `results.json` or an `eval_*.json` produced by the committed code.
Nothing is typed from memory and nothing is rounded in our favour. Where a result contradicts what we
set out to show, it is reported in the same voice as the results that support it.

The tweet-corpus grid is complete: 131 student runs over five students, three seeds per
configuration, a homogeneous and a heterogeneous committee on every student and a third committee,
which adds an implicit-abuse specialist, on the headline student, plus ablations, controls and
hyper-parameter sweeps. The first 120 runs took 8.2 GPU-hours; a further 1.4 added the specialist, its
committee, two controls and the three sharpest weighting temperatures. The Wikipedia benchmark and the
student grid on the implicit benchmark are in progress and their sections are marked as such.

**One rule constrains what may appear in these tables.** The same code, data and seeds produce
systematically different results on a CPU and on a T4 (Section 5.10), so every number in a table comes
from the Kaggle grid alone. Comparisons within it are valid because everything in it was trained
identically. Numbers produced elsewhere are excluded rather than reconciled.

**Every difference is tested, not read off a table.** Intervals are paired bootstraps over the test
set: seeds paired by index, 1,000 resamples per seed, 95 per cent interval of the pooled differences
(`python -m dmthd.significance`, Table `significance`). The grid makes 51 such comparisons. One
interval excludes zero: the one that removes pre-training.

---

## 5.1 The teachers clear the bar, and the stop rule passes

| Teacher | Macro-F1 | Accuracy | ECE | Benign-sarcasm FPR | Ironic-abuse recall |
|---|---|---|---|---|---|
| BERT-large | 0.8973 | 0.9075 | 0.056 | 0.224 | 0.585 |
| Implicit specialist, task-adapted | 0.8931 | 0.9043 | 0.072 | 0.263 | 0.678 |
| RoBERTa-irony, task-adapted | 0.8926 | 0.9050 | 0.067 | 0.294 | 0.718 |
| HateBERT, task-adapted | 0.8887 | 0.9011 | 0.076 | 0.317 | 0.696 |
| DeBERTa-v3-base | 0.8720 | 0.8867 | 0.053 | 0.409 | 0.773 |
| TF-IDF + logistic regression | 0.8798 | 0.8930 | -- | -- | -- |

The pre-registered stop rule required at least one task-adapted teacher to beat the fine-tune-only
student on the clean test split. Three of the original four do, and three of four beat the classical
floor. The fourth, DeBERTa-v3-base, sits below the floor and is retained only as the heterogeneous
committee member, which is what it is there to test.

The fifth teacher exists for the question in Section 5.3b. It is HateBERT fine-tuned on the implicit
corpus (Section 5.6a) and then adapted to this task like the others, and it lands at 0.8931, inside a
0.009 band with BERT-large, RoBERTa-irony and HateBERT, although it learned from different data under a
different label scheme.

The probe columns already show the pattern that Section 5.7 makes precise: ordered by false-positive
rate on harmless sarcasm, the teachers are almost exactly ordered by recall on ironic abuse (Spearman
0.9 over five teachers; only RoBERTa-irony and HateBERT swap places). The teacher that fires most
readily on one probe fires most readily on the other.

## 5.2 Distillation raises every student's score, by less than the test set can resolve

Test macro-F1, mean over three seeds. The classical floor is 0.8798.

| Student | Params | Fine-tune only | Single-teacher | Uniform | D-MTHD | Uniform + DeBERTa | D-MTHD + DeBERTa |
|---|---|---|---|---|---|---|---|
| BERT-mini | 11.2M | 0.8393 | 0.8405 | 0.8385 | 0.8378 | 0.8378 | 0.8373 |
| BERT-small | 28.8M | 0.8469 | 0.8488 | 0.8505 | 0.8471 | 0.8509 | 0.8477 |
| DistilBERT | 67.0M | 0.8907 | 0.8963 | 0.8960 | 0.8960 | 0.8975 | 0.8966 |
| DeBERTa-v3-xsmall | 70.8M | 0.8825 | 0.8865 | 0.8862 | 0.8833 | 0.8847 | 0.8828 |
| BiLSTM | 10.4M | 0.8693 | 0.8748 | 0.8761 | 0.8732 | 0.8760 | 0.8764 |

On BERT-mini the committee that adds the implicit specialist scores 0.8390 with uniform averaging and
0.8373 with D-MTHD (Section 5.3b).

Best distilled configuration against fine-tune-only:

| Student | Best configuration | Difference | 95% interval |
|---|---|---|---|
| BERT-mini | Single-teacher | +0.0012 | [-0.0057, +0.0079] |
| BERT-small | Uniform + DeBERTa | +0.0040 | [-0.0058, +0.0126] |
| DistilBERT | Uniform + DeBERTa | +0.0068 | [-0.0013, +0.0139] |
| DeBERTa-v3-xsmall | Single-teacher | +0.0040 | [-0.0024, +0.0109] |
| BiLSTM | D-MTHD + DeBERTa | +0.0071 | [-0.0039, +0.0195] |

Five out of five positive, which is the result the control was added to test and which the Phase-2
version of this work failed. **None of the five intervals excludes zero**, and neither does any other
distillation-against-fine-tuning comparison in the grid, twenty-seven in all. Taking the best of six
configurations per student also flatters the difference. The claim this supports is the modest one:
distillation moves every student in the same direction, by 0.001 to 0.007 macro-F1, which is less
than a test set of 4,326 posts can resolve.

The efficiency claim sits in this table. DistilBERT with a uniform heterogeneous committee reaches
**0.8975** against BERT-large's **0.8973**, with a fifth of the parameters and a fifth of the latency
(Section 5.9). We state that as parity, not as the student beating the teacher: the difference is far
inside seed variance.

## 5.3 The per-instance weighting does not beat uniform averaging

This is the paper's own method and it does not do what it was designed to do.

| Student | D-MTHD minus uniform | 95% interval | Same, with DeBERTa | 95% interval |
|---|---|---|---|---|
| BERT-mini | -0.0007 | [-0.0074, +0.0061] | -0.0005 | [-0.0067, +0.0054] |
| BERT-small | -0.0034 | [-0.0106, +0.0036] | -0.0032 | [-0.0100, +0.0033] |
| DistilBERT | +0.0000 | [-0.0079, +0.0082] | -0.0010 | [-0.0087, +0.0069] |
| DeBERTa-v3-xsmall | -0.0029 | [-0.0086, +0.0029] | -0.0019 | [-0.0108, +0.0108] |
| BiLSTM | -0.0029 | [-0.0116, +0.0056] | +0.0004 | [-0.0072, +0.0081] |

Four negative, one tie, none meaningfully positive, and every interval contains zero. The best
configuration for four of the five students is uniform averaging or single-teacher distillation. On
the headline student the ablation that removes the weighting altogether scores **0.8386** against
**0.8385** for full D-MTHD on the same seed.

**On BERT-mini nothing beats the no-teacher control at all.** Candidate minus fine-tune-only:

| Comparison | Difference | 95% interval |
|---|---|---|
| D-MTHD | -0.0015 | [-0.0076, +0.0048] |
| Uniform | -0.0007 | [-0.0074, +0.0058] |
| Single-teacher | +0.0012 | [-0.0057, +0.0079] |
| D-MTHD + DeBERTa | -0.0019 | [-0.0082, +0.0041] |
| Uniform + implicit specialist | -0.0003 | [-0.0080, +0.0075] |
| D-MTHD + implicit specialist | -0.0019 | [-0.0093, +0.0049] |

Every interval contains zero.

**Sharper weighting was the obvious objection, and it changes nothing.** Because the teachers are
frozen and cached, the weights `w_k(i) = softmax_k(-CE_k(i)/tau)` are a fixed function of the data:
they vary across instances and never across epochs or seeds. Their sharpness is set entirely by tau.
At the default tau = 1 the mean weights are 0.332 / 0.337 / 0.331 against a uniform 0.333, close enough
that D-MTHD and uniform averaging would score alike by arithmetic rather than by evidence. So the sweep
was extended down to tau = 0.05.

| tau | Mean weights, BERT-large / HateBERT / irony | Test macro-F1, seed 1 |
|---|---|---|
| 0.05 | 0.318 / 0.356 / 0.326 | 0.8392 |
| 0.1 | 0.323 / 0.349 / 0.328 | 0.8388 |
| 0.2 | 0.327 / 0.344 / 0.329 | 0.8385 |
| 0.5 | 0.330 / 0.340 / 0.330 | 0.8390 |
| 1 (default) | 0.332 / 0.337 / 0.331 | 0.8385 |
| 2 | 0.332 / 0.336 / 0.332 | 0.8390 |
| 5 | 0.333 / 0.334 / 0.333 | 0.8389 |
| Uniform averaging | 0.333 each | 0.8372 |
| Fine-tune only | -- | 0.8394 |

Across a hundredfold range of temperature the score stays inside a 0.0007 band, below fine-tune-only
at every value. The sharpest setting differs from the default by +0.0007 [-0.0008, +0.0023] and from
uniform averaging by +0.0020 [-0.0042, +0.0077], one seed each. Even at tau = 0.05 the heaviest teacher
averages 0.356, 0.023 above uniform, and the student's score does not move. **The claim is now that
per-instance reliability weighting does not differ from uniform averaging at any temperature from
0.05 to 5.**

## 5.3a Why it does not work: the committee is not diverse enough

The negative result in 5.3 has a mechanism, and it can be measured without training a single student.

| Teacher pair | Cohen's kappa | Disagreement | Error overlap |
|---|---|---|---|
| BERT-large, HateBERT | 0.920 | 6.7% | 0.710 |
| BERT-large, irony | 0.922 | 6.4% | 0.712 |
| BERT-large, DeBERTa-v3 | 0.893 | 8.9% | 0.712 |
| HateBERT, irony | 0.916 | 7.0% | 0.673 |
| HateBERT, DeBERTa-v3 | 0.889 | 9.2% | 0.673 |
| irony, DeBERTa-v3 | 0.900 | 8.3% | 0.730 |
| BERT-large, implicit specialist | 0.924 | 6.4% | 0.710 |
| **HateBERT, implicit specialist** | **0.962** | **3.1%** | **0.843** |
| irony, implicit specialist | 0.918 | 6.8% | 0.689 |
| implicit specialist, DeBERTa-v3 | 0.889 | 9.2% | 0.691 |

The original four teachers are all right on **83.2%** of the test set and all wrong on the same
**4.7%**. A per-instance weighting can only express a preference where its members disagree, which
here is at most one instance in eleven, and on roughly two-thirds of the errors there is no correct
teacher to prefer. The learned weights behave accordingly: averaged over the training set they are
0.332 / 0.337 / 0.331 for the homogeneous committee and 0.254 / 0.259 / 0.254 / 0.233 once DeBERTa-v3
joins, identical at every epoch and every seed, because with frozen teachers they are a fixed function
of the data.

**The headroom is real; the signal is wrong.** An oracle that picked the best teacher for each
instance would reach **0.9526** accuracy and a macro-F1 upper bound of **0.9469**, against the best
single teacher's 0.9075 and 0.8973. Selection over exactly these teachers is worth five macro-F1
points. Reliability estimated by cross-entropy against the training label recovers none of it.

We think this is the more useful form of the negative result. It is not that combining specialists is
a bad idea; it is that the quantity this literature uses to decide *whom to trust on this instance*
carries almost no information once the teachers have been adapted to a shared label space. And that
adaptation is not optional: it is what makes a sarcasm model's logits comparable with an abuse
model's in the first place. **Task adaptation buys comparability and spends diversity**, and any
cross-task committee will face the same trade.

The implicit specialist is the sharpest case of it. It learned from a different corpus under a
different label scheme, and it comes back from adaptation to this task as the closest thing in the
committee to HateBERT, the model it started from: kappa 0.962, disagreement on 3.1 per cent of the
test set, and 84 per cent of HateBERT's errors made by the specialist as well. Adding it raises the five-teacher oracle only from
0.9526 to 0.9552 accuracy.

## 5.3b A teacher that has seen implication labelled does not help either

The tweet corpus labels topics, not indirectness, so no teacher adapted to it alone could hold
knowledge of implication. The implicit specialist was built to supply it. If committee composition
matters where the weighting does not, this is where it should show. BERT-mini:

| Configuration | Seeds | Macro-F1 | F1 on `other_cyberbullying` | Sarcasm-discrimination AUC (seed 1) |
|---|---|---|---|---|
| Fine-tune only | 3 | 0.8393 +- 0.0003 | 0.707 | 0.773 |
| Uniform, homogeneous committee | 3 | 0.8385 +- 0.0018 | 0.700 | 0.774 |
| Uniform, + implicit specialist | 3 | 0.8390 +- 0.0035 | 0.699 | 0.768 |
| D-MTHD, homogeneous committee | 3 | 0.8378 +- 0.0011 | 0.705 | 0.773 |
| D-MTHD, + implicit specialist | 3 | 0.8373 +- 0.0025 | 0.697 | 0.768 |
| Specialist alone, no committee | 1 | 0.8392 | 0.709 | 0.767 |
| Pre-trained on the implicit corpus, no teacher | 1 | 0.8377 | 0.705 | 0.756 |

Adding the specialist changes macro-F1 by +0.0005 [-0.0059, +0.0064] under uniform averaging and by
-0.0005 [-0.0069, +0.0055] under D-MTHD. F1 on the class that holds indirect abuse falls slightly under
both, and sarcasm-discrimination AUC falls by 0.005 to 0.006. Neither control does better. The
specialist on its own sits within 0.0002 of the fine-tune-only student, and putting the implicit
corpus into the student directly, by pre-training on it before the tweet task, gives the lowest
sarcasm-discrimination AUC of the 25 pre-trained models analysed.

**The weights do move towards the specialist, a little, and no more than towards HateBERT.** In the
committee the students trained with, at tau = 1, the specialist's mean weight is 0.2519 against a
uniform 0.25. Split by class, measured on the training split over all five cached teachers, as mean
weight on `other_cyberbullying` minus mean weight on the four targeted classes (uniform weight 0.200):

| Teacher | tau = 0.05 | tau = 1 |
|---|---|---|
| HateBERT | +0.0486 [+0.0460, +0.0512] | +0.0105 [+0.0094, +0.0116] |
| Implicit specialist | +0.0441 [+0.0416, +0.0465] | +0.0090 [+0.0080, +0.0100] |
| RoBERTa-irony | +0.0184 [+0.0159, +0.0210] | +0.0053 [+0.0043, +0.0064] |
| BERT-large | -0.0063 [-0.0089, -0.0035] | +0.0011 [-0.0002, +0.0023] |
| DeBERTa-v3-base | -0.1048 [-0.1067, -0.1028] | -0.0259 [-0.0274, -0.0245] |

Three things limit what this shows. At the temperature the students trained with, the specialist's
shift is a twentieth of the uniform weight; it reaches a fifth only at tau = 0.05, a sharpness that
left the homogeneous committee's score unchanged (Section 5.3) and has not been run with the specialist. The shift is not specific to the specialist: HateBERT, which never saw the
implicit corpus, gains slightly more, as the near-identical predictions in Section 5.3a would predict.
And the table pools all five cached teachers, a committee no student trained with. Which teacher leads
on an instance is the same in any sub-committee, but the sizes are not, and the measurement on the
committees actually trained is queued for the next session. Because the weights are computed from
cross-entropy against the gold label, a positive shift also means only that a teacher fits
`other_cyberbullying` better than it fits the targeted classes, not that it reads implication.

**So on this corpus the answer to the question the specialist was built for is no.** A teacher that
has learned implication from labelled data passes nothing a compact student's decisions can use,
whether it is weighted, averaged, used alone, or replaced by pre-training on its data. The implicit
benchmark, where that knowledge is in-domain rather than transferred, is the test that remains.

## 5.4 Ablations and controls: only pre-training matters

BERT-mini, one seed, against full D-MTHD at 0.8378.

| Configuration | Macro-F1 | Difference |
|---|---|---|
| Randomly initialised student | 0.7905 | **-0.0473** |
| No auxiliary irony head | 0.8364 | -0.0014 |
| No hidden-state term | 0.8369 | -0.0009 |
| Uniform weights instead of per-instance | 0.8386 | **+0.0008** |
| Per-batch instead of per-instance weights | 0.8389 | **+0.0011** |
| Control: implicit specialist alone, no committee | 0.8392 | +0.0014 |
| Control: pre-trained on the implicit corpus, no teacher | 0.8377 | -0.0001 |

Removing components of the method costs at most 0.0014 and in two cases *improves* the score. Removing
pre-training costs 0.047, forty times more than anything else in the grid, and against fine-tune-only
on the same seed it is the one difference in the grid whose interval excludes zero: -0.0488
[-0.0603, -0.0376]. Hyper-parameters behave the same way: over the whole sweep, T in {1, 2, 8} gives
0.8386 / 0.8402 / 0.8391, alpha in {0.2, 0.6} gives 0.8386 / 0.8394, delta in {0.1, 0.5} gives
0.8389 / 0.8399, and tau in {0.05, 0.1, 0.2, 0.5, 2, 5} gives 0.8392 / 0.8388 / 0.8385 / 0.8390 /
0.8390 / 0.8389. The objective is flat in every direction we can move it.

## 5.5 Homogeneity and committee composition

Adding a DeBERTa-v3 teacher to the homogeneous committee changes the student by -0.0005, +0.0006,
+0.0032, -0.0005 and +0.0006. Only the BiLSTM, the one student that is not a transformer, gains
anything, and 0.0032 is inside its seed spread.

The controlled form of the homogeneity question is the hidden-state term, which is the only part of
the objective whose effect should depend on architectural similarity. On BERT-mini it is worth
**+0.0009**. The same 2x2 on the DeBERTa and BiLSTM students has not run.

We therefore do not claim that homogeneity causes anything, and the word is used in this paper only
to describe the setup (Section 3.6).

## 5.6 Where indirect abuse is missed, and why

Diagnostic on BERT-mini fine-tune-only, tweet corpus, seed 1. Recall on the ironic-abuse probe is
**0.91** on the 43 items containing explicit profanity and **0.76** on the 1,517 without; on the
implicit-abuse probe it is 0.91 on 11 items with and 0.74 on 752 without. Profanity helps: recall is
6 to 16 points higher with it on every one of the 25 pre-trained models analysed. But the items
without it are still caught about three times in four on every one of them, so the detector leans on
profanity without being a profanity detector. The with-profanity groups are small, and the 11-item
one supports no estimate at all.

The failure is discrimination. Mean p(abusive) is **0.748** on ironic abuse and **0.449** on benign
sarcasm, and the two distributions overlap heavily: sarcasm-discrimination AUC is **0.773**. At a 0.5
threshold, recall of 0.792 costs a **39.5%** false-positive rate on sarcasm that attacks nobody, and
**no threshold up to 0.90 brings that rate under 10%**, on this model or on any of the 26 analysed.

The benchmark's own labels show the same confusion: `other_cyberbullying` recall 0.710 with 109 of its
errors going to `not_cyberbullying`, and `not_cyberbullying` recall 0.648 with 126 going the other
way. The two catch-all classes bleed into each other, which is a label-quality problem as much as a
model one.

## 5.6a The same failure on a corpus that labels implication

Section 5.6 infers a discrimination failure from probe sets we wrote. The implicit benchmark tests it
against other people's labels. It is the Implicit Hate Corpus reduced to one source and three classes
(16,509 / 2,064 / 2,064 rows; train `not_hate` 10,616, `implicit_hate` 5,029, `explicit_hate` 864),
and the question is whether a model that ranks implied abuse correctly also thresholds it correctly.

| Model | macro-F1 | `not_hate` F1 | `explicit_hate` F1 | `implicit_hate` F1 | implicit-discrimination AUC |
|---|---|---|---|---|---|
| Classical floor (TF-IDF, linear) | 0.5620 | - | - | 0.5562 | 0.7610 |
| BERT-mini, fine-tune only | 0.5549 | 0.8347 | 0.2857 | 0.5443 | 0.8196 |
| HateBERT specialist | **0.6029** | 0.8352 | 0.3662 | 0.6072 | **0.8247** |

**The compact student loses to a bag of n-grams on F1 and beats it decisively on ranking.** BERT-mini
scores 0.5549 macro-F1 against the floor's 0.5620, and 0.5443 against 0.5562 on the implicit class
itself, while separating implied abuse from benign text at 0.8196 AUC against the floor's 0.7610. The
two measures disagree in direction, not merely in degree. A reader looking only at F1 would conclude
the neural model had learned nothing the n-grams had not; the ranking says it has learned a great
deal and cannot convert it into a decision. This is the clearest evidence we have for the argument of
Section 5.7, and unlike the probe evidence it does not depend on any set we wrote ourselves.

**Two honest qualifications.** First, `explicit_hate` has 864 training rows and both neural models
score between 0.29 and 0.37 on it, against 0.83 on `not_hate`. Macro-F1 here is therefore dominated
by a scarce class the paper does not argue about, which is a further reason to lead with the
threshold-free measure rather than a reason to distrust it. Second, these are single-seed results and
the full student grid on this benchmark has not yet run; the direction is what we rely on, not the
third decimal.

The specialist clears the floor on both measures. Adapted to the tweet task it is the fifth teacher of
Section 5.1 and the committee member tested in Section 5.3b, where it does not help. Whether distilling
from it helps a student on this benchmark, where its knowledge is in-domain, is what the student grid
here will show.

## 5.7 No method in the grid discriminates better than any other

This is the strongest result in the paper and it is not the one we expected.

Across all **133 evaluated models** in the grid, teachers and students, every mode and every seed, the
false-positive rate on benign sarcasm and the recall on ironic abuse correlate at **+0.712**. The
quantity that separates real discrimination from a shifted decision threshold, recall minus
false-positive rate, has mean **0.357** and standard deviation **0.051**, while its two components
range over 0.14 to 0.59 and 0.54 to 0.81 respectively. The parts move a great deal. The difference
between them barely moves.

| Model family | n | Benign FPR | Ironic recall | Margin |
|---|---|---|---|---|
| DeBERTa-v3-xsmall | 21 | 0.334 | 0.728 | 0.394 |
| Teachers | 5 | 0.301 | 0.690 | 0.388 |
| BERT-mini | 44 | 0.390 | 0.765 | 0.375 |
| DistilBERT | 21 | 0.238 | 0.607 | 0.369 |
| BERT-small | 21 | 0.380 | 0.713 | 0.333 |
| BiLSTM | 21 | 0.336 | 0.623 | 0.287 |

Distillation mode does not appear in this ordering. Architecture does. DistilBERT has both the lowest
false-positive rate and the lowest recall; BERT-mini has the highest of each. They occupy different
points on a single trade-off curve rather than different curves. The worst margin in the entire grid,
0.082, belongs to the randomly initialised student, which agrees with Section 5.4 from the other
direction: whatever ability these models have to tell implied abuse from harmless sarcasm comes from
pre-training, and nothing we added to the objective moves it.

**The threshold-free measure says the same thing about the headline student.** Over the 25
pre-trained BERT-mini models analysed, which cover every objective and committee, four temperatures,
the T, alpha and delta sweeps, every ablation and both controls, sarcasm-discrimination AUC has mean
0.771 and standard deviation 0.005, from 0.756 to 0.777. The randomly initialised student scores
0.613. Over the same 25 models the operating point at a 0.5 threshold moves far more: recall on
ironic abuse from 0.758 to 0.821 and the false-positive rate from 0.343 to 0.448, correlated at +0.87.

**This is why the paper reports a threshold-free area and not a recall.** Recall at a fixed cut
measures how readily a model fires. Across a hundred models, that is all it measures.

## 5.8 Robustness and transfer

Synthetic obfuscation of the abusive rows only, BERT-mini, seed 1:

| Method | Clean | Leetspeak | Character swap | Inserted spaces | Mixed |
|---|---|---|---|---|---|
| Fine-tune only | 0.8394 | 0.7475 | 0.7417 | 0.7275 | 0.7116 |
| D-MTHD | 0.8385 | 0.7460 | 0.7361 | 0.7250 | 0.7077 |

Twelve to thirteen points lost on short text, and distillation does not help. These are synthetic
character edits, not evasion by people trying to evade, and the paper says so rather than calling this
robustness.

**Transfer to the corpus that labels implication.** Applied without further training to the implicit
benchmark's 2,064 test posts, the tweet-trained BERT-mini calls 81.0% of them abusive against a true
rate of 35.7% (binary macro-F1 0.423, ROC-AUC 0.608). On the case this paper is about, implicit hate
against posts that are not hateful, it catches 85.1% of the implicit hate at a 78.8% false-positive
rate and ranks the two at an AUC of **0.600**, barely above chance, against **0.820** for the same
architecture trained on that corpus (Section 5.6a). D-MTHD transfers no better: binary macro-F1 0.411,
implicit-hate AUC 0.600. Whatever the tweet students have learned about abuse, it does not include what
makes an implication hateful, which is consistent with Section 5.3b: there was little of that
knowledge in the committee for them to receive.

Transfer between the tweet and Wikipedia corpora is reported with the Wikipedia grid.

## 5.9 Efficiency

Measured on an idle machine, median of five timed passes after warm-up.

| Student | Params | Latency b1 | Latency b32 | GFLOPs/seq | fp32 size | INT8 size | INT8 macro-F1 |
|---|---|---|---|---|---|---|---|
| BiLSTM | 10.4M | 2.57 ms | 0.18 ms | ~0 | 40 MB | 32 MB | 0.8736 |
| BERT-mini | 11.2M | 3.47 ms | 0.37 ms | 0.87 | 43 MB | 33 MB | 0.8319 |
| BERT-small | 28.8M | 3.74 ms | 1.25 ms | 3.36 | 110 MB | 73 MB | 0.8448 |
| DistilBERT | 67.0M | 5.22 ms | 3.72 ms | 11.17 | 255 MB | 132 MB | 0.8931 |
| DeBERTa-v3-xsmall | 70.8M | 23.49 ms | 3.25 ms | 10.57 | 270 MB | 209 MB | 0.8765 |
| BERT-large (teacher) | 335.1M | 26.56 ms | 21.65 ms | 78.92 | 1,279 MB | -- | -- |

DistilBERT matches BERT-large's accuracy at **5.1x** fewer parameters and **5.8x** lower batch-32
latency. INT8 quantisation halves DistilBERT's size for 0.003 macro-F1. BERT-mini's size falls only
from 43 to 33 MB because 7.8M of its 11.2M parameters are the embedding table, which dynamic
quantisation does not touch: the saving is structural, not proportional.

## 5.10 Threats to validity

**Two machines disagree about the same configuration, by 0.0386 macro-F1.** BERT-mini fine-tune-only
scores 0.8779 on a laptop CPU and 0.8393 on a Kaggle T4 with identical code, data, seeds, epochs,
batch size, maximum length and learning rate. We eliminated three explanations.

Mixed precision accounts for only 0.0014 of the gap: trained in full precision on the same Kaggle
hardware and seeds, BERT-mini scores 0.8407 (BERT-small loses 0.0011 to mixed precision likewise), so
0.0372 belongs to the environment. The data is not the cause: the gold-label sequences recorded by the
two runs match row for row across all 4,326 test examples, so both score the same split. The code is
not the cause: re-running the current trainer on the original machine reproduces the original result
exactly, test macro-F1 agreeing to sixteen significant figures and validation macro-F1 agreeing at
every epoch, with only the loss accumulation differing in the ninth digit.

So one environment is bit-reproducible across re-runs and the other lands four points away from it on
the same script. Both seed clusters are internally tight (0.8779 / 0.8770 / 0.8760 against 0.8394 /
0.8395 / 0.8389), and the training loss is higher at every epoch on the slower machine, which makes
this slower optimisation rather than worse generalisation. The gap is also a gap in fit rather than in
discrimination: the laptop model separates ironic abuse from benign sarcasm at an AUC of 0.776, the
Kaggle model at 0.773.

Every number in this paper therefore comes from one environment, and comparisons are made only within
it. The discrepancy does not change any conclusion: at its local best the student is still below the
classical floor, and the ranking of methods is unaffected because everything in the grid was trained
identically. We report it because a gap this large between two runs of the same script is a fact
about reproducibility that a field publishing three-decimal differences should not leave unstated.

**Seeds.** Three per configuration, one for ablations, controls and sweeps. Differences are read from
bootstrap intervals, not from seed variance, and single-seed rows are treated as indicative only. The
threshold-free probe analysis (Sections 5.3b, 5.6 and the end of 5.7) covers the first seed of each
configuration; the three sharpest temperatures were trained after it ran and are not yet in it.

**Teacher variance is not estimated.** Each teacher is trained once.

**The heterogeneous comparison is confounded.** DeBERTa-v3-xsmall is simply a weaker model on this
task than the BERT-lineage students it is compared against, so the student swap mixes architecture
family with model quality. The hidden-state term is the controlled version of that question.

**The routing measurement pools every cached teacher.** Section 5.3b says what that does and does not
change; the per-committee measurement replaces it in the next session.

**Probes are rule-built, not hand-verified.** About 1,000 items each, with a 300-item human-annotated
subset in progress. Probe metrics are reported only for models trained on Twitter-domain data; on a
Wikipedia-trained model they measure domain shift and are omitted.

**The label itself is contested.** The two public corpora that annotate implication agree on 99.7% of
their 629 shared texts about whether a text is hateful and on 48.2% about whether the hate is stated
or implied. Every result about implicit abuse, ours included, is bounded by that.
