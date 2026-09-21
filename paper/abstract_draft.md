# Abstract, three versions

Decision 1 in `OPEN_DECISIONS.md` is about framing, and framing is hard to judge in the abstract, so
here are the versions written out. Every number is a placeholder marked `[ ]` except the ones
already measured, which are given as they stand.

**Updated 17 September, after the specialist runs.** Version B's bet did not pay on the tweet corpus:
the implicit specialist helps neither in a committee nor alone, and the reason is measured (DECISIONS
F23). Version A has lost its last qualification, since no weighting temperature from 0.05 to 5
separates D-MTHD from averaging (F22). Version B2 is new and is the recommended one: the same subject
as B, claimed as a measurement rather than a method. Every sentence in it is already measured. The one
experiment still to run, the student grid on the implicit benchmark, could upgrade it back to B and
cannot break it.

**Updated 13 September, after the tweet grid finished.** The measured numbers below are now the real
ones. Version A has changed the most, because the result it was built on did not arrive: the dynamic
weighting beats uniform averaging on none of five students. Version A is now an abstract about
efficiency plus a negative result, which is honest and smaller than it was. Version B still depends on
the implicit specialist, which is untested.

---

## Version A: the current framing

*A Dynamic Multi-Teacher Homogeneous Knowledge Distillation Framework for Robust Cyberbullying
Detection*

Cyberbullying detection is usually deployed under tight latency and memory budgets, which rules out
the large encoders that perform best. Knowledge distillation is the standard answer, and the
multi-teacher variants of it are usually presented with a weighting scheme as their contribution. We
built such a scheme, measured it against the one baseline these papers rarely include, the same
compact student trained with no teachers at all, and report that it does not work.

We distil a committee of task-adapted teachers with complementary expertise, a general encoder, an
abusive-language specialist and an irony specialist, into five students spanning 10M to 71M parameters
and three architecture families, weighting the committee per instance by each teacher's reliability.
Across 131 runs on a de-duplicated, leakage-checked corpus, distillation raises every student's score,
by 0.001 to 0.007 macro-F1, less than the test set can resolve. **Per-instance reliability weighting
beats uniform averaging on none of them**, the five differences being -0.0007, -0.0034, +0.0000,
-0.0029 and -0.0029, at no temperature from 0.05 to 5, and removing the weighting entirely does not
lower the headline student's score. Among all components of the objective, only pre-training matters:
ablating it costs 0.047 where nothing else costs more than 0.0014.

What survives is an efficiency result and a measurement problem. A 67M student reaches 0.8975 against
a 335M teacher's 0.8973 at a fifth of the latency, and INT8 quantisation halves its size for 0.003
macro-F1. And across all 133 evaluated models, the false-positive rate on benign sarcasm and the recall
on ironic abuse correlate at +0.712 with their difference nearly constant: no method discriminates
sarcastic abuse better than any other, they differ only in how readily they fire. Reporting recall at
a fixed threshold, as this literature does, measures willingness to fire rather than understanding.

**What this version is betting on:** that a carefully measured negative result about a mechanism the
field keeps proposing, plus an efficiency result and a metric correction, is enough. It is honest and
it is smaller than the paper we set out to write. Its risk is that a reviewer reads it as a paper
without a method, under a title that still names that method.

---

## Version B: the implication framing, as a method

*Teaching a small model to read implication: cross-task specialist distillation for implicit abuse
detection*

Abuse that says what it means is largely solved; abuse carried by implication is not. On a corpus in
which implication is annotated as such, a bag of n-grams reaches 0.75 F1 on explicitly hateful text
and 0.56 on implied hate, and ranks implied hate above ordinary text at an AUC of only 0.76. We show
that the difficulty is not mainly lexical: a compact detector still catches ironic abuse three times in
four when no profanity is present, and fails instead at discrimination, assigning mean p(abusive) 0.75
to ironic abuse and 0.45 to harmless sarcasm, so that recall of 0.79 costs a 40% false-positive rate on
sarcasm that attacks nobody. Across 133 models spanning three architecture families and four
distillation objectives, that trade-off is the same one: false-positive rate and recall correlate at
+0.712 and their difference is nearly constant, so no existing method discriminates better than any
other.

We treat this as a transfer problem. Neither established cyberbullying benchmark labels indirectness,
so no teacher fine-tuned on them holds the knowledge and no weighting scheme has an expert to route
to. We therefore add one: an implicit-abuse specialist, trained on a corpus that makes the
distinction and then task-adapted into a committee alongside a general encoder, an abusive-language
specialist and an irony specialist, with the committee weighted per instance by each teacher's
reliability. The student is compact enough to deploy: [ ]M parameters, [ ] ms per example on CPU.

We report the comparison that decides the claim, not a score in isolation. The same committee without
the specialist, the specialist distilled alone, and the same student simply pre-trained on the
implicit corpus with no distillation are all measured across three seeds; [result]. We measure
whether the weighting routes implication to the specialist rather than averaging over the committee
[result], and we evaluate on a corpus annotated by a different team on a different platform that no
model in this paper trains on [result]. We also report that the two public corpora annotating
implication agree on 99.7% of their shared texts about whether a text is hateful and on 48.2% about
whether the hate is implied, which bounds what any result on either of them can mean.

**What this version is betting on:** that the specialist actually helps. If it does not, the honest
abstract is the first two paragraphs plus a negative result, which is a genuine and publishable
contribution but a smaller one, and Version A is then the better home for the efficiency material.

**What happened, 17 September.** On the tweet corpus the bet lost. With the specialist the committee
scores +0.0005 [-0.0059, +0.0064] under averaging and -0.0005 [-0.0069, +0.0055] under D-MTHD; neither
control does better; routing towards the specialist is a twentieth of the uniform weight and no larger
than towards HateBERT, which the specialist has become a near-copy of (kappa 0.962). The first
`[result]` above is therefore "no difference", and this version survives only if the implicit
benchmark's student grid shows something the tweet corpus did not. Version B2 is the form of it that
the evidence supports today.

---

## Version B2: the implication framing, as a measurement (recommended)

*Distillation Does Not Teach Implication: A Threshold-Free Audit of Multi-Teacher Knowledge
Distillation for Abusive Language Detection*

Abuse carried by implication, rather than by what a text says outright, is where compact detectors
fail, and distillation from committees of specialist teachers is the usual proposal for closing that
gap cheaply. We test the proposal under the controls it is rarely given. Five students from 10M to 71M
parameters are distilled from committees of task-adapted teachers, a general encoder, an
abusive-language specialist, an irony specialist and a teacher trained on a corpus that labels
implication, under single-teacher, uniform and per-instance reliability-weighted objectives: 131 runs
on a de-duplicated tweet corpus, three seeds per main configuration, every difference tested by paired
bootstrap.

Distillation raises every student's macro-F1 by 0.001 to 0.007, and no interval excludes zero.
Reliability weighting matches uniform averaging at every temperature from 0.05 to 5. The implicit-abuse
teacher adds nothing, whether in a committee, alone, or replaced by pre-training on its corpus, and the
reason is measurable: adapted to the task, it agrees with the model it was built from at Cohen's kappa
0.962, so the committee never held the knowledge it was added to supply. Measured without a threshold,
the ability to tell ironic abuse from harmless sarcasm is fixed by pre-training: across 25 variants of
one student it spans only 0.756 to 0.777 AUC, while a randomly initialised student scores 0.613. Across
133 models, recall on ironic abuse and false positives on harmless sarcasm rise together (r = 0.71), so
recall at a fixed threshold, the usual report, measures readiness to fire rather than understanding.
And tweet-trained students rank implied hate on a corpus that annotates it at 0.60 AUC, against 0.82
for the same architecture trained there. The one intervention that moves the student is the data: on
42,013 unlabelled in-domain tweets a single teacher's soft labels raise it by 0.007 to 0.009 macro-F1,
with every seed above every no-teacher seed, while the committee, its weighting and hard
pseudo-labels add nothing to that; the gain is in the task, not in implication.

We release the evaluation protocol, a single-source implicit-abuse benchmark with per-row manifests
that verify a rebuild without redistributing text, and evidence that pooling sources into such a
benchmark rewards recognising the source instead of the implication.

**What this version is betting on:** that a controlled negative result with a measured mechanism and a
reusable protocol is a contribution a Q1 venue will take. Its risk is venue fit: journals that want a
method will read it as a paper without one, so the target list should lean towards *Information
Processing and Management* or a Findings venue (Decision 6). Its strength is that nothing in it waits
on an unmeasured number. The implicit benchmark's student grid, still to run, is the one experiment
that bears on it, and the abstract makes a prediction there that could fail: implicit-discrimination
AUC will not move beyond test-set noise. If it does, the paper becomes Version B with this evidence as
its motivation.

---

## What is already true in every version

These sentences are measured and can be written now, whichever version wins.

- The fine-grained cyberbullying tweet corpus contains 1,563 texts carrying more than one label and
  507 exact duplicates; removing them eliminates a validation-to-test accuracy gap of 0.94 against
  0.86 that earlier work reports on the raw corpus.
- A 67M-parameter student distilled from a single large teacher reaches 0.8963 macro-F1 against that
  teacher's 0.8973, and 0.8975 with a uniform heterogeneous committee.
- Distillation raises all five students, from +0.0012 to +0.0071, with no paired interval excluding
  zero; per-instance weighting helps none of them at any temperature from 0.05 to 5; and ablating
  pre-training costs 0.047, the one difference in the grid whose interval excludes zero.
- Across 133 evaluated models, false-positive rate on benign sarcasm and recall on ironic abuse
  correlate at +0.712 while their difference has a standard deviation of 0.051.
- A task-adapted implicit-abuse specialist agrees with its base model at kappa 0.962 and adds nothing
  to a committee, alone, or as pre-training.
- INT8 dynamic quantisation gives 1.74x at batch 32 for 0.006 macro-F1 on tweets and 1.57x for 0.003
  on Wikipedia; the size reduction is modest because 7.8M of BERT-mini's 11.2M parameters are the
  embedding table, which dynamic quantisation does not touch.
- Models trained on the tweet corpus rank implicit hate against not-hate on the implicit benchmark at
  an AUC of 0.600, against 0.820 for the same architecture trained on it. (Transfer between the tweet
  and Wikipedia corpora was measured on the laptop only and waits for the Kaggle Wikipedia grid.)
- Splits are released as per-row manifests that let a reader verify a rebuild without either party
  redistributing the underlying text.
