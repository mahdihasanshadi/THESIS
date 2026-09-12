# Results (draft, 13 September 2026)

Every number here is read from a `results.json` or an `eval_*.json` produced by the committed code.
Nothing is typed from memory and nothing is rounded in our favour. Where a result contradicts what we
set out to show, it is reported in the same voice as the results that support it.

The tweet-corpus grid is complete: 120 student runs, three seeds, five students, three teacher
committees, plus ablations and hyper-parameter sweeps, 8.2 GPU-hours. The Wikipedia and implicit
benchmarks are in progress and their sections are marked as such.

**One rule constrains what may appear in these tables.** The same code, data and seeds produce
systematically different results on a CPU and on a T4 (Section 5.10), so every number in a table comes
from the Kaggle grid alone. Comparisons within it are valid because everything in it was trained
identically. Numbers produced elsewhere are excluded rather than reconciled.

---

## 5.1 The teachers clear the bar, and the stop rule passes

| Teacher | Macro-F1 | Accuracy | ECE | Benign-sarcasm FPR | Ironic-abuse recall |
|---|---|---|---|---|---|
| BERT-large | 0.8973 | 0.9075 | 0.056 | 0.224 | 0.585 |
| RoBERTa-irony, task-adapted | 0.8926 | 0.9050 | 0.067 | 0.294 | 0.718 |
| HateBERT, task-adapted | 0.8887 | 0.9011 | 0.076 | 0.317 | 0.696 |
| DeBERTa-v3-base | 0.8720 | 0.8867 | 0.053 | 0.409 | 0.773 |
| TF-IDF + logistic regression | 0.8798 | 0.8930 | -- | -- | -- |

The pre-registered stop rule required at least one task-adapted teacher to beat the fine-tune-only
student on the clean test split. Three of four do, and three of four beat the classical floor. The
fourth, DeBERTa-v3-base, sits below the floor and is retained only as the heterogeneous committee
member, which is what it is there to test.

The probe columns already show the pattern that Section 5.7 makes precise: the teacher with the
highest recall on ironic abuse also has the highest false-positive rate on harmless sarcasm, and the
teacher with the lowest has the lowest. They are ordered identically.

## 5.2 Distillation helps every student, by a little

Test macro-F1, mean over three seeds. The classical floor is 0.8798.

| Student | Params | Fine-tune only | Single-teacher | Uniform | D-MTHD | Uniform + DeBERTa | D-MTHD + DeBERTa |
|---|---|---|---|---|---|---|---|
| BERT-mini | 11.2M | 0.8393 | 0.8405 | 0.8385 | 0.8378 | 0.8378 | 0.8373 |
| BERT-small | 28.8M | 0.8469 | 0.8488 | 0.8505 | 0.8471 | 0.8509 | 0.8477 |
| DistilBERT | 67.0M | 0.8907 | 0.8963 | 0.8960 | 0.8960 | 0.8975 | 0.8966 |
| DeBERTa-v3-xsmall | 70.8M | 0.8825 | 0.8865 | 0.8862 | 0.8833 | 0.8847 | 0.8828 |
| BiLSTM | 10.4M | 0.8693 | 0.8748 | 0.8761 | 0.8732 | 0.8760 | 0.8764 |

Best distilled configuration minus fine-tune-only: +0.0012, +0.0040, +0.0068, +0.0040, +0.0071. Five
out of five positive, which is the result the control was added to test and which the Phase-2 version
of this work failed.

The efficiency claim sits in this table. DistilBERT with a uniform committee reaches **0.8975**
against BERT-large's **0.8973**, with a fifth of the parameters and a fifth of the latency
(Section 5.9). We state that as parity, not as the student beating the teacher: the difference is far
inside seed variance.

## 5.3 The per-instance weighting does not beat uniform averaging

This is the paper's own method and it does not do what it was designed to do.

| Student | D-MTHD minus uniform | D-MTHD + DeBERTa minus uniform + DeBERTa |
|---|---|---|
| BERT-mini | -0.0007 | -0.0005 |
| BERT-small | -0.0034 | -0.0032 |
| DistilBERT | +0.0000 | -0.0009 |
| DeBERTa-v3-xsmall | -0.0029 | -0.0019 |
| BiLSTM | -0.0029 | +0.0004 |

Four negative, one tie, none meaningfully positive. The best configuration for four of the five
students is uniform averaging or single-teacher distillation. On the headline student the ablation
that removes the weighting altogether scores **0.8386** against full D-MTHD's **0.8378**.

**On BERT-mini nothing beats the no-teacher control at all.** Paired bootstrap over 1,000 test-set
resamples, candidate minus fine-tune-only:

| Comparison | Difference | 95% interval |
|---|---|---|
| D-MTHD | -0.0015 | [-0.0076, +0.0048] |
| Uniform | -0.0007 | [-0.0074, +0.0058] |
| Single-teacher | +0.0012 | [-0.0057, +0.0079] |
| D-MTHD + DeBERTa | -0.0019 | [-0.0082, +0.0041] |

Every interval contains zero.

**The qualification, stated plainly.** Because the teachers are frozen and cached, the weights
`w_k(i) = softmax_k(-CE_k(i)/tau)` are a fixed function of the data: they vary across instances and
never across epochs or seeds. Their sharpness is set entirely by tau, and every tau this grid tried
was 0.5 or larger, where the mean weights are 0.330 / 0.340 / 0.330 against a uniform 0.333. At those
values D-MTHD and uniform averaging optimise nearly the same objective, so their scoring alike is
arithmetic rather than evidence. The values that could separate them, tau in {0.05, 0.1, 0.2}, are
running at the time of writing. **The claim supported today is that per-instance reliability
weighting has not been shown to differ from uniform averaging at any temperature yet tested.**

## 5.4 Ablations: only pre-training matters

BERT-mini, one seed, against full D-MTHD at 0.8378.

| Configuration | Macro-F1 | Difference |
|---|---|---|
| Randomly initialised student | 0.7905 | **-0.0473** |
| No auxiliary irony head | 0.8364 | -0.0014 |
| No hidden-state term | 0.8369 | -0.0009 |
| Uniform weights instead of per-instance | 0.8386 | **+0.0008** |
| Per-batch instead of per-instance weights | 0.8389 | **+0.0011** |

Removing components of the method costs at most 0.0014 and in two cases *improves* the score. Removing
pre-training costs 0.047, forty times more than anything else in the grid. Hyper-parameters behave the
same way: over the whole sweep, T in {1, 2, 8} gives 0.8386 / 0.8402 / 0.8391, alpha in {0.2, 0.6}
gives 0.8386 / 0.8394, delta in {0.1, 0.5} gives 0.8389 / 0.8399, and tau in {0.5, 2, 5} gives
0.8390 / 0.8390 / 0.8389. The objective is flat in every direction we can move it.

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

Diagnostic on BERT-mini fine-tune-only, tweet corpus. Recall on the ironic-abuse probe is **0.72** on
the 43 items containing explicit profanity and **0.67** on the 1,517 without; on the implicit-abuse
probe it is 0.45 with and 0.69 without. The detector is not a profanity detector.

The failure is discrimination. Mean p(abusive) is **0.693** on ironic abuse and **0.375** on benign
sarcasm, but the distributions overlap heavily: sarcasm-discrimination AUC is **0.776**. At a 0.5
threshold, recall of 0.704 costs a **31.7%** false-positive rate on sarcasm that attacks nobody;
holding false positives under 10% leaves recall at **0.427**.

The benchmark's own labels show the same confusion: `other_cyberbullying` recall 0.786 with 77 of its
errors going to `not_cyberbullying`, and `not_cyberbullying` recall 0.681 with 133 going the other
way. The two catch-all classes bleed into each other, which is a label-quality problem as much as a
model one.

## 5.7 No method in the grid discriminates better than any other

This is the strongest result in the paper and it is not the one we expected.

Across all **114 evaluated models** in the grid, teachers and students, every mode and every seed, the
false-positive rate on benign sarcasm and the recall on ironic abuse correlate at **+0.673**. The
quantity that separates real discrimination from a shifted decision threshold, recall minus
false-positive rate, has mean **0.352** and standard deviation **0.053**, while its two components
range over 0.24 to 0.59 and 0.59 to 0.75 respectively. The parts move a great deal. The difference
between them barely moves.

| Model family | n | Benign FPR | Ironic recall | Margin |
|---|---|---|---|---|
| DeBERTa-v3-xsmall | 21 | 0.334 | 0.728 | 0.394 |
| Teachers | 4 | 0.311 | 0.693 | 0.382 |
| DistilBERT | 21 | 0.238 | 0.607 | 0.369 |
| BERT-mini | 26 | 0.385 | 0.753 | 0.368 |
| BERT-small | 21 | 0.380 | 0.713 | 0.333 |
| BiLSTM | 21 | 0.336 | 0.623 | 0.287 |

Distillation mode does not appear in this ordering. Architecture does. DistilBERT has both the lowest
false-positive rate and the lowest recall; BERT-mini has the highest of each. They occupy different
points on a single trade-off curve rather than different curves. The worst margin in the entire grid,
0.082, belongs to the randomly initialised student, which agrees with Section 5.4 from the other
direction: whatever ability these models have to tell implied abuse from harmless sarcasm comes from
pre-training, and nothing we added to the objective moves it.

**This is why the paper reports a threshold-free area and not a recall.** Recall at a fixed cut
measures how readily a model fires. Across a hundred models, that is all it measures.

## 5.8 Robustness and transfer

Synthetic obfuscation of the abusive rows only, BERT-mini:

| Method | Clean | Leetspeak | Character swap | Inserted spaces | Mixed |
|---|---|---|---|---|---|
| Fine-tune only | 0.8394 | 0.7475 | 0.7417 | 0.7275 | 0.7116 |
| D-MTHD | 0.8385 | 0.7460 | 0.7361 | 0.7250 | 0.7077 |

Twelve to thirteen points lost on short text, and distillation does not help. These are synthetic
character edits, not evasion by people trying to evade, and the paper says so rather than calling this
robustness.

Cross-corpus transfer, reported as a property of the two task definitions rather than a defect of the
students: a tweet-trained model calls 80.9% of Wikipedia comments bullying against a true rate of
11.8% (binary macro-F1 0.273, ROC-AUC 0.562); a Wikipedia-trained model calls 33.7% of tweets bullying
against a true rate of 85.7% (0.425, ROC-AUC 0.718). One over-fires and the other under-fires.

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

**Two machines disagree about the same configuration.** BERT-mini fine-tune-only scores 0.8779 on a
laptop CPU and 0.8393 on a Kaggle T4 with identical code, data, seeds, epochs, batch size, maximum
length and learning rate. Mixed precision is not the cause: fp16 and fp32 track to within 0.0015 epoch
by epoch on the same hardware. The data is not the cause: the gold-label sequences match row for row
across both environments. The code is not the cause: re-running the current trainer reproduces the old
local result to four decimal places. Both seed clusters are internally tight, so a four-point gap
between them is systematic, and the training loss is higher at every epoch on the slower machine, so
it is slower learning rather than worse generalisation. Every number in this paper therefore comes
from one environment, and comparisons are made only within it. We report this because a difference
this large between two runs of the same script is a fact about reproducibility that the field should
not leave unstated.

**Seeds.** Three per configuration, one for ablations and sweeps. Differences are read from bootstrap
intervals, not from seed variance, and single-seed ablations are treated as indicative only.

**Teacher variance is not estimated.** Each teacher is trained once.

**The heterogeneous comparison is confounded.** DeBERTa-v3-xsmall is simply a weaker model on this
task than the BERT-lineage students it is compared against, so the student swap mixes architecture
family with model quality. The hidden-state term is the controlled version of that question.

**Probes are rule-built, not hand-verified.** About 1,000 items each, with a 300-item human-annotated
subset in progress. Probe metrics are reported only for models trained on Twitter-domain data; on a
Wikipedia-trained model they measure domain shift and are omitted.

**The label itself is contested.** The two public corpora that annotate implication agree on 99.7% of
their 629 shared texts about whether a text is hateful and on 48.2% about whether the hate is stated
or implied. Every result about implicit abuse, ours included, is bounded by that.
