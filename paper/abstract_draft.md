# Abstract, two versions

Decision 1 in `OPEN_DECISIONS.md` is about framing, and framing is hard to judge in the abstract, so
here are both versions written out. Every number is a placeholder marked `[ ]` except the ones
already measured, which are given as they stand today so the shape of each claim is visible.

Pick one after the implicit results arrive. Do not pick one before: version B promises something we
cannot yet show.

---

## Version A: the current framing

*A Dynamic Multi-Teacher Homogeneous Knowledge Distillation Framework for Robust Cyberbullying
Detection*

Cyberbullying detection is usually deployed under tight latency and memory budgets, which rules out
the large encoders that perform best. Knowledge distillation is the standard answer, but published
multi-teacher schemes are rarely checked against the one baseline that matters: the same compact
student trained with no teachers at all. We build that check into every experiment and report what it
shows.

We distil a committee of task-adapted teachers with complementary expertise, a general encoder, an
abusive-language specialist and an irony specialist, into compact BERT-lineage students, weighting the
committee per instance by each teacher's reliability on that instance. An auxiliary head distilled
from the un-adapted irony model shapes the student's representation without introducing sarcasm
labels into the training corpora.

On three corpora, with de-duplicated and leakage-checked splits released as verifiable manifests, a
66M-parameter student reaches within 0.001 macro-F1 of a 335M-parameter teacher, and INT8
quantisation gives a 1.74x speed-up at batch 32 for 0.006 macro-F1. We also report where the method
does not help: where a compact model already sits near teacher level, distillation adds little, and
[the per-instance weighting either does or does not beat uniform averaging once its temperature is
chosen properly]. Two widely used corpora do not transfer to each other in either direction, and one
of them contains [ ] duplicate and conflicting-label rows that inflate previously published results.

**What this version is betting on:** that efficiency plus an honest negative result is enough. It is
a safe abstract and a slightly dull one, and it invites the question of what separates the method
from existing multi-teacher distillation.

---

## Version B: the implication framing

*Teaching a small model to read implication: cross-task specialist distillation for implicit abuse
detection*

Abuse that says what it means is largely solved; abuse carried by implication is not. On a corpus in
which implication is annotated as such, a bag of n-grams reaches 0.80 F1 on explicitly hateful text
and 0.56 on implied hate, and ranks implied hate above ordinary text at an AUC of only 0.76. We show
that the difficulty is not lexical: a compact detector responds to ironic abuse about as often with
profanity present as without, and fails instead at discrimination, assigning mean p(abusive) 0.69 to
ironic abuse and 0.38 to harmless sarcasm, so that recall of 0.70 costs a 32% false-positive rate on
sarcasm that attacks nobody.

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
- A 66M-parameter student distilled from a single large teacher reaches 0.8963 macro-F1 against that
  teacher's 0.8973.
- INT8 dynamic quantisation gives 1.74x at batch 32 for 0.006 macro-F1 on tweets and 1.57x for 0.003
  on Wikipedia; the size reduction is modest because 7.8M of BERT-mini's 11.2M parameters are the
  embedding table, which dynamic quantisation does not touch.
- Models trained on the tweet corpus call 80.9% of Wikipedia comments bullying against a true rate of
  11.8%; models trained on Wikipedia call 33.7% of tweets bullying against a true rate of 85.7%.
- Splits are released as per-row manifests that let a reader verify a rebuild without either party
  redistributing the underlying text.
