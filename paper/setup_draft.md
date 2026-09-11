# Experimental setup (draft for the paper, updated 12 Sep 2026)

Numbers below are produced by the scripts in `src/dmthd/` and are reproducible from the
committed code; every count comes from a `report.json` written by the pipeline.

## Datasets

| Corpus | Task | Train / Val / Test | Labels | Notes |
|---|---|---|---|---|
| Fine-grained cyberbullying tweets (Wang, Fu and Lu, 2020) | six-class | 34,607 / 4,326 / 4,326 | age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying | primary benchmark |
| Wikipedia Talk personal attacks (Wulczyn, Thain and Dixon, 2017) | binary | 68,750 / 22,782 / 22,721 | attack / not attack, plus the annotator fraction as a soft label | official split; about ten annotators per comment |
| Implicit abuse (Implicit Hate Corpus stage 1, ElSherief et al. 2021) | three-class | 16,509 / 2,064 / 2,064 | not_hate, explicit_hate, implicit_hate | built here; the only corpus of the three in which abuse-by-implication is a label rather than a hidden subset. ISHate (Ocampo et al. 2023) is held out as a 27,110-row out-of-domain test set |

### Tweet corpus: de-duplication

The published corpus contains 47,692 tweets. Repeated tweets are common and some repeats carry
different labels. We apply, in order: (1) drop tweets shorter than two tokens after whitespace
normalisation (758 removed); (2) drop every tweet whose normalised text appears with more than
one label, all copies included (1,563 distinct texts, 3,168 rows removed); (3) drop exact
duplicates on normalised text, keeping the first occurrence (507 removed). The 43,259 remaining
tweets are split 80/10/10, stratified by class, with seed 42, and the three splits are asserted
disjoint on normalised text. Normalisation for matching lower-cases the text and strips URLs,
`@user` placeholders and punctuation; training text keeps its original form.

Class counts after cleaning are no longer balanced: religion 7,956; age 7,934; ethnicity 7,838;
gender 7,526; not_cyberbullying 6,175; other_cyberbullying 5,830 (train+val+test). The two
classes that lose the most rows are exactly the two the literature reports as confusable,
which is consistent with label noise concentrating there.

A validation-to-test accuracy gap of 0.94 versus 0.86 observed in earlier runs on the raw
corpus disappears on the cleaned split, indicating that the gap was caused by duplicate tweets
shared between training and validation data.

### Wikipedia corpus

Soft label = mean attack vote over the comment's annotators; hard label = soft label > 0.5, so
ties are not-attack. Cleaning removes NEWLINE/TAB tokens, HTML tags and entities, URLs, IP
addresses, timestamps, signatures and wiki markup, and collapses whitespace; case is kept.
After applying the official train/dev/test split: comments under three tokens removed (522);
within-split exact duplicates removed (train 450, dev 82, test 114); dev and test comments whose
text also occurs in train removed (199 and 202), and test comments also in dev removed (42).
The splits are asserted disjoint on text and on revision id. Attack share is 11.7% / 11.9% /
11.8%; 24.6% of comments have an annotator fraction between 0.2 and 0.8.

### Implicit corpus

Neither of the two corpora above labels indirectness. The tweet corpus places abuse-by-implication
inside `other_cyberbullying`, a catch-all that also holds unrelated material, and the Wikipedia
corpus records only whether a comment is an attack. A model trained on either therefore never sees
an example annotated as implicit, which makes the central claim of this paper untestable on them.

We build a third benchmark in which the distinction is the label. Two corpora annotate it: the
Implicit Hate Corpus stage-1 release, 21,480 posts across exactly these three classes (the widely
mirrored stage-2 file has no benign class and cannot be used here), and ISHate, whose original rows
number 29,116 after excluding its augmentations and 28,763 after excluding ToxiGen-sourced rows under
the provenance rule. ISHate records its benign class only in `hateful_layer`, its `implicit_layer`
being empty for every Non-HS row.

**The splits come from one corpus, not both, and the reason is measured.** Ninety-five per cent of
implicit examples in the union come from the Implicit Hate Corpus and 89 per cent of explicit
examples come from ISHate, while a TF-IDF classifier separates the two corpora at 0.91 macro-F1. A
model trained on the union can therefore score well on implicit-versus-explicit by recognising which
corpus a text came from, without reading any implication at all. It also does measurably worse where
it matters: scored on identical Implicit Hate test rows, a classical model trained on the union
reaches 0.473 F1 on implicit hate against 0.548 for the same model trained on that corpus alone.
We therefore build train, validation and test from the Implicit Hate Corpus and keep ISHate entirely
out of them, as a 27,110-row out-of-domain test set.

Every text occurring in any of the three probe sets below is removed first, 1,581 rows across the
union, so that probe metrics remain measured on text no model has trained on. The Implicit Hate rows
that remain then go through the same pipeline as the tweet corpus: 10 rows under two tokens, 12 texts
carrying more than one label (25 rows), 12 exact duplicates. The 20,637 survivors are split 80/10/10
stratified by class with seed 42, giving 16,509 / 2,064 / 2,064 with training classes not_hate
10,616, implicit_hate 5,029 and explicit_hate 864. The held-out ISHate rows lose a further 613 that
also occur in the splits, leaving 27,110.

Note what the out-of-domain set can and cannot test. It holds 17,697 benign and 9,412 explicitly
hateful texts but only one implicit one, because ISHate's implicit rows are almost all in the probe
sets already. Out-of-domain implicit *recall* is therefore measured by the implicit-abuse probe,
which is exactly those 763 ISHate rows; the out-of-domain set measures whether a model trained to
find implication starts calling ordinary text hateful when the domain changes.

**The two corpora disagree about implication far more than they disagree about hate.** They share
629 texts. On whether a text is hateful at all they agree on 99.7 per cent of them. On whether the
hate is stated or implied they agree on 48.2 per cent: of the 624 shared texts both call hateful, the
Implicit Hate Corpus labels essentially all of them implicit while ISHate labels 324 explicit and 300
implicit. Two expert annotation efforts placing the boundary in materially different places is a
direct measurement of how hard this label is, it is the reason we do not pool the corpora, and it
bounds how sharp any implicit-hate result on either of them can be.

## Targeted test sets for the sarcasm claim (inference only)

| Set | Size | Source | Metric |
|---|---|---|---|
| Benign sarcasm | 1,064 | sarcastic tweets from iSarcasmEval (SemEval-2022 Task 6), train and test, author-labelled | false-positive rate: share predicted as any bullying class |
| Ironic abuse | 1,560 | 797 posts labelled `irony` in the Implicit Hate Corpus stage-2 data (ElSherief et al., 2021) plus 763 original ISHate rows labelled Implicit HS (Ocampo et al., 2023) | recall: share predicted as a bullying class |
| Implicit abuse | 763 | the ISHate subset alone | recall |

Augmented ISHate rows are never used anywhere. For the probe sets specifically, ISHate rows whose
source is ToxiGen or the Implicit Hate Corpus are also excluded, so that the probes stay independent
of the corpus they are used to test transfer into. This is not in tension with the implicit
benchmark's use of the Implicit Hate Corpus as training data: the provenance rule concerns what a
*teacher* was trained on before we touched it, and no teacher of ours was pre-trained on either
corpus. The direction of the rule is one-way, and it is enforced in both places by the same source
filter. The benign set receives one manual screening pass restricted to rows a classical classifier
flags as abusive with probability at least 0.8 (see `screen_probes.py`).

Because the probes are drawn from the same corpora as the implicit benchmark, every probe text is
deleted from that benchmark, in all three splits. The probes are therefore unseen by every model in
the paper, on every corpus, and probe numbers are comparable across all three benchmarks and across
runs made before the third benchmark existed.

### The metric that matches the claim

Recall on ironic abuse at a fixed decision threshold cannot distinguish a model that fails to see
indirect abuse from one that sees it but cannot separate it from harmless sarcasm, and the
measurement in Section [findings] says it is the second. We therefore report
**sarcasm-discrimination AUC**: the ROC-AUC of p(abusive) with the ironic-abuse probe as positives
and the benign-sarcasm probe as negatives. It is threshold-free, it is defined identically on all
three corpora, and it is the number against which the auxiliary irony head and the implicit
specialist are judged. The recall-versus-false-positive curve is reported alongside it, because a
deployment has to pick a threshold even though an evaluation should not.

## Teacher provenance rule

No test set may overlap the training data of any teacher, including the data a specialist
checkpoint was trained on before task adaptation: ToxiGen and Founta et al. (2018) for the
ToxiGen RoBERTa, SemEval-2018 Task 3 for the Cardiff irony model, RAL-E Reddit posts for HateBERT.

The implicit specialist is the one teacher we train ourselves from scratch in this sense: HateBERT
fine-tuned on the implicit benchmark, then task-adapted onto the target benchmark like every other
committee member, its classification head re-initialised for the target label space. It is the only
member of the committee that has seen abuse-by-implication annotated as such, and it is the
mechanism behind the implicit claim rather than an incidental extra teacher. The rule it must obey
is the same one: it is never used on the corpus it was trained on, and the implicit benchmark's test
split never reaches the tweet or Wikipedia evaluations.

## Classical floor

TF-IDF over word 1-2 grams and character 3-5 grams with class-balanced logistic regression (C = 4).

| Corpus | Macro-F1 | Accuracy | Other |
|---|---|---|---|
| Tweets | 0.880 | 0.893 | per-class F1: age 0.98, ethnicity 0.98, religion 0.95, gender 0.91, other 0.75, not_cyberbullying 0.70 |
| Wikipedia | 0.876 | 0.947 | ROC-AUC 0.968, PR-AUC 0.868, attack-class F1 0.78 |
| Implicit | 0.562 | 0.697 | per-class F1: not_hate 0.797, implicit_hate 0.556, explicit_hate 0.333; implicit-discrimination AUC 0.761; ECE 0.071 |

Every neural result is reported relative to this floor. The third row is the reason the third
corpus exists. On the tweet corpus the floor is high enough that our own fine-tune-only student does
not beat it, which leaves distillation little room to demonstrate anything; on the implicit corpus a
bag of n-grams reaches 0.56 F1 on implied hate and ranks implied hate above ordinary text at an AUC
of 0.76, which is well above chance and a long way from solved. The explicit class is small in this
corpus (108 test rows) and its F1 is correspondingly noisy, which is why macro-F1 is not the headline
here: the number the claim is argued on is implicit-discrimination AUC, the threshold-free ranking of
implied hate against ordinary text.

For the record, pooling both corpora gives a higher aggregate score, 0.681 macro-F1 with 0.453 on
implicit hate, and that is precisely the artefact described above: the extra accuracy comes from the
model being able to tell the corpora apart.

## Protocol (fixed before any GPU run)

Three seeds per configuration; mean and standard deviation; paired bootstrap of 1,000 test-set
resamples for the 95% interval of every D-MTHD-versus-baseline difference. Hyper-parameters:
teachers 5 epochs (3 on Wikipedia), learning rate 2e-5, batch 32 (16 on Wikipedia), 10% warm-up,
linear decay, best validation macro-F1 epoch kept; students 6 epochs, learning rate 3e-5,
early stopping with patience 2 on validation macro-F1. Maximum length 128 tokens for tweets,
256 for Wikipedia. Latency is the median of five timed passes after twenty warm-up batches, at
batch sizes 1 and 32, on GPU and CPU.

Two measurements are specific to the implicit claim and are fixed here so that they cannot be chosen
after seeing results. First, cross-corpus transfer is scored not only on the collapsed
abusive-versus-benign task but on the implicit rows alone against the benign rows, because a
collapsed score rewards a model that catches explicit abuse and misses every implication. Second,
the per-instance teacher weights are examined for routing: the mean weight each teacher receives on
implicit rows minus the mean weight it receives on explicit rows, with a percentile bootstrap
interval over 2,000 resamples. A contrast whose interval excludes zero is evidence that the
weighting selects an expert; one that spans zero means the committee treats both kinds of abuse
alike, and we report that instead.
