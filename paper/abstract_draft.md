# Abstract, two versions

Decision 1 in `OPEN_DECISIONS.md` is about framing, and framing is hard to judge in the abstract, so
here are both versions written out. Every number is a placeholder marked `[ ]` except the ones
already measured, which are given as they stand.

**Updated 13 September, after the tweet grid finished.** The measured numbers below are now the real
ones. Version A has changed the most, because the result it was built on did not arrive: the dynamic
weighting beats uniform averaging on none of five students. Version A is now an abstract about
efficiency plus a negative result, which is honest and smaller than it was. Version B still depends on
the implicit specialist, which is untested.

Pick one after the implicit results arrive. Do not pick one before: version B promises something we
cannot yet show, and version A now promises less than it did.

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
Across 120 runs on a de-duplicated, leakage-checked corpus, distillation helps every student, by
0.001 to 0.007 macro-F1. **Per-instance reliability weighting beats uniform averaging on none of
them**, the five differences being -0.0007, -0.0034, +0.0000, -0.0029 and -0.0029, and removing the
weighting entirely improves the headline student. Among all components of the objective, only
pre-training matters: ablating it costs 0.047 where nothing else costs more than 0.0014.

What survives is an efficiency result and a measurement problem. A 67M student reaches 0.8975 against
a 335M teacher's 0.8973 at a fifth of the latency, and INT8 quantisation halves its size for 0.003
macro-F1. And across all 114 evaluated models, the false-positive rate on benign sarcasm and the recall
on ironic abuse correlate at +0.673 with their difference nearly constant: no method discriminates
sarcastic abuse better than any other, they differ only in how readily they fire. Reporting recall at
a fixed threshold, as this literature does, measures willingness to fire rather than understanding.

**What this version is betting on:** that a carefully measured negative result about a mechanism the
field keeps proposing, plus an efficiency result and a metric correction, is enough. It is honest and
it is smaller than the paper we set out to write. Its risk is that a reviewer reads it as a paper
without a method.

---

## Version B: the implication framing

*Teaching a small model to read implication: cross-task specialist distillation for implicit abuse
detection*

Abuse that says what it means is largely solved; abuse carried by implication is not. On a corpus in
which implication is annotated as such, a bag of n-grams reaches 0.75 F1 on explicitly hateful text
and 0.56 on implied hate, and ranks implied hate above ordinary text at an AUC of only 0.76. We show
that the difficulty is not lexical: a compact detector responds to ironic abuse about as often with
profanity present as without, and fails instead at discrimination, assigning mean p(abusive) 0.69 to
ironic abuse and 0.38 to harmless sarcasm, so that recall of 0.70 costs a 32% false-positive rate on
sarcasm that attacks nobody. Across 114 models spanning three architecture families and four
distillation objectives, that trade-off is the same one: false-positive rate and recall correlate at
+0.673 and their difference is nearly constant, so no existing method discriminates better than any
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

---

## What is already true in both

These sentences are measured and can be written now, whichever version wins.

- The fine-grained cyberbullying tweet corpus contains 1,563 texts carrying more than one label and
  507 exact duplicates; removing them eliminates a validation-to-test accuracy gap of 0.94 against
  0.86 that earlier work reports on the raw corpus.
- A 67M-parameter student distilled from a single large teacher reaches 0.8963 macro-F1 against that
  teacher's 0.8973, and 0.8975 with a uniform committee.
- Distillation helps all five students, from +0.0012 to +0.0071; per-instance weighting helps none of
  them; and ablating pre-training costs 0.047 where no component of the objective costs more than
  0.0014.
- Across 114 evaluated models, false-positive rate on benign sarcasm and recall on ironic abuse
  correlate at +0.673 while their difference has a standard deviation of 0.053.
- INT8 dynamic quantisation gives 1.74x at batch 32 for 0.006 macro-F1 on tweets and 1.57x for 0.003
  on Wikipedia; the size reduction is modest because 7.8M of BERT-mini's 11.2M parameters are the
  embedding table, which dynamic quantisation does not touch.
- Models trained on the tweet corpus call 80.9% of Wikipedia comments bullying against a true rate of
  11.8%; models trained on Wikipedia call 33.7% of tweets bullying against a true rate of 85.7%.
- Splits are released as per-row manifests that let a reader verify a rebuild without either party
  redistributing the underlying text.
