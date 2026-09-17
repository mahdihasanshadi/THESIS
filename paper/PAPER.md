# Distillation Does Not Teach Implication: A Threshold-Free Audit of Multi-Teacher Knowledge Distillation for Abusive Language Detection

*Manuscript assembled 17 September 2026 from the measured results in this repository. Every number is
produced by the committed code from the Kaggle grid (tables in `paper/tables/`); numbers in square
brackets are 95 per cent paired-bootstrap intervals. Sections that wait on runs not yet made are
marked. Authors and affiliations to be filled: Mahdi Hasan Shadi and co-authors, Department of Computer
Science and Engineering, BRAC University; supervised by Dr. Muhammad Iqbal Hossain and Sheikh Araf
Noshin.*

## Abstract

Abuse carried by implication, rather than by what a text says outright, is where compact detectors
fail, and distillation from committees of specialist teachers is the usual proposal for closing that
gap cheaply. We test the proposal under the controls it is rarely given. Five students from 10M to 71M
parameters are distilled from committees of task-adapted teachers, a general encoder, an
abusive-language specialist, an irony specialist and a teacher trained on a corpus that labels
implication, under single-teacher, uniform and per-instance reliability-weighted objectives: 131 runs
on a de-duplicated tweet corpus, three seeds per main configuration, every difference tested by paired
bootstrap. Distillation raises every student's macro-F1 by 0.001 to 0.007, and no interval excludes
zero. Reliability weighting matches uniform averaging at every temperature from 0.05 to 5, and we show
why: the reliability is read on the split the teachers were fine-tuned on, where each of them assigns
the gold label a probability near one, so there is nothing to weigh; and even a gate fitted to the
teachers' outputs cannot recover more than 0.3 points of the five that an oracle over the committee
would gain. The implicit-abuse teacher adds nothing, in a committee, alone, or as pre-training, because
task adaptation turns it into a near-copy of the model it was built from (Cohen's kappa 0.962).
Measured without a threshold, the ability to tell ironic abuse from harmless sarcasm is fixed by
pre-training: across 25 variants of one student it spans 0.756 to 0.777 AUC, while a randomly
initialised student scores 0.613. Across 133 models, recall on ironic abuse and false positives on
harmless sarcasm rise together (r = 0.71), so recall at a fixed threshold, the usual report, measures
readiness to fire rather than understanding. Tweet-trained students rank implied hate on a corpus that
annotates it at 0.60 AUC, against 0.82 for the same architecture trained there. We release the
evaluation protocol, a single-source implicit-abuse benchmark with per-row manifests that verify a
rebuild without redistributing text, and evidence that pooling sources into such a benchmark rewards
recognising the source instead of the implication.

**Keywords:** knowledge distillation, multi-teacher distillation, implicit hate speech, cyberbullying
detection, sarcasm, evaluation methodology, compact models.

## 1. Introduction

Detectors of abusive language are deployed under latency and memory budgets that exclude the large
encoders that score best on the benchmarks \cite{schwartz2020green, patterson2021carbon, zhou2019edge}.
Knowledge distillation is the standard response \cite{hinton2015distilling, bucilua2006model}: a
compact student is trained to imitate a large teacher's outputs and, in the multi-teacher variants that
this literature increasingly favours, the outputs of several teachers combined by some weighting
\cite{you2017learning, wu2021one, zhang2022confidence, yuan2021reinforced}. The promise is specific.
A committee of teachers with complementary expertise, a general encoder, a specialist in abusive
vocabulary, a specialist in irony, should hold more of what the task requires than any one of them,
and a weighting that trusts each teacher where it is reliable should pass that knowledge to a student
small enough to run on a phone.

The knowledge that matters most is the hardest to hold. Abuse that names its target and its hostility
is largely solved by lexical models; abuse carried by implication, stereotype, irony or coded reference
is where detectors and annotators both fail \cite{elsherief2021latent, ocampo2023indepth,
hartvigsen2022toxigen}. Every survey of cyberbullying detection names sarcastic and indirect abuse as
the open case \cite{rosa2019automatic, salawu2020approaches, emmery2021current}, and none of the
widely used benchmarks labels it: the fine-grained cyberbullying corpus files it under a catch-all
class \cite{wang2020sosnet} and the Wikipedia corpus records only whether a comment is an attack
\cite{wulczyn2017exmachina}. If distillation can move the ability to read implication into a compact
model, that is the result worth having.

This paper set out to obtain that result and reports, with the controls that would have been needed
to believe it, that it does not obtain. We built the method the literature suggests, D-MTHD, a
committee of task-adapted specialist teachers weighted per instance by reliability, with an auxiliary
irony head and a fourth teacher trained on a corpus in which implication is a label; we measured it on
a cleaned benchmark against the one baseline these papers rarely include, the same student trained
with no teachers at all; and we measured the thing the method exists for, discrimination of implied
abuse from harmless sarcasm, with a threshold-free metric rather than recall at an arbitrary cut. The
method does not beat its controls, and we can say why in enough detail that the reasons generalise
beyond it.

The contributions are the audit and what it found.

1. **A controlled grid for multi-teacher distillation on abusive language.** Five students spanning
   three architecture families, three committees, four objectives, three seeds, and 66 paired
   comparisons with bootstrap intervals. Distillation raises every student by 0.001 to 0.007 macro-F1,
   and not one interval excludes zero; per-instance reliability weighting never differs from uniform
   averaging, at any temperature from 0.05 to 5 (Sections 5.2 and 5.3).
2. **The mechanism.** The reliability signal is read where every teacher has memorised the label, so
   it is saturated; the committee is worth half a point over its best member but no gate fitted to the
   teachers' outputs recovers more than 0.3 of the five points an oracle would; and task adaptation,
   which is what makes specialists comparable enough to combine, removes the diversity the committee
   was assembled for, to the point that a teacher trained on a corpus of implication comes back as a
   copy of its base model (Sections 5.4 to 5.6).
3. **A threshold-free protocol for implication.** Sarcasm-discrimination AUC on held-out probes and
   implicit-discrimination AUC on a corpus that labels implication. Across 133 models, recall at a fixed
   threshold and the false-positive rate on harmless sarcasm rise together, so the usual report measures
   readiness to fire; across 25 variants of one student the threshold-free measure barely moves, and
   only removing pre-training moves it (Section 5.8).
4. **A single-source implicit-abuse benchmark** with the evidence that motivates its construction: the
   two public corpora that annotate implication agree on 99.7 per cent of their shared texts about hate
   and on 48.2 per cent about whether it is implied, and pooling them yields a benchmark a classifier
   can solve by recognising the source (Section 4.2).
5. **What survives for practice.** A 67M student matches a 335M teacher at a fifth of the parameters
   and latency, and INT8 quantisation halves its size for 0.003 macro-F1 (Section 5.11).

We think the negative result is more useful stated than avoided. The weighting we audited is the one
this literature keeps proposing, the controls we ran are the ones it keeps omitting, and the mechanism
we found applies to any committee whose members are fine-tuned on the same data and whose reliability
is scored against the labels they were fine-tuned on.

## 2. Related work

### 2.1 Cyberbullying and toxicity detection

Surveys frame cyberbullying detection as text classification with three persistent problems: scarce
and inconsistent data, weak evaluation practice, and poor transfer across platforms
\cite{rosa2019automatic, salawu2020approaches, emmery2021current}. The wider abusive-language
literature documents the same issues, including training-data quality and its effect on
generalisation \cite{fortuna2018survey, vidgen2020directions, yin2021towards}. The benchmarks used here
are the Wikipedia personal-attacks corpus with crowd annotations \cite{wulczyn2017exmachina}, the
fine-grained cyberbullying tweet corpus \cite{wang2020sosnet}, and, for implicit and ironic abuse, the
Implicit Hate Corpus \cite{elsherief2021latent} and ISHate \cite{ocampo2023indepth}; ToxiGen generates
implicit examples adversarially \cite{hartvigsen2022toxigen} and HateXplain adds rationales
\cite{mathew2021hatexplain}. Sarcasm resources are iSarcasm and iSarcasmEval \cite{oprea2020isarcasm,
abufarha2022semeval} and the SemEval-2018 irony task \cite{vanhee2018semeval}. Domain-specialised
encoders include HateBERT \cite{caselli2021hatebert}, the TweetEval models \cite{barbieri2020tweeteval}
and the ToxDect RoBERTa \cite{zhou2021challenges}. Earlier datasets and the evaluation pitfalls of
duplicate-heavy corpora are documented in \cite{vanhee2018automatic, founta2018large,
davidson2017automated, borkan2019nuanced}.

Sarcastic and ironic abuse is named as the hard case in every survey, but compact detectors are
evaluated on aggregate F1 alone; no work reports, for a deployable model, the false-positive rate on
sarcasm that attacks nobody beside the recall on abuse that is ironic. Section 5.8 shows why the two
must be reported together and, once they are, why neither is the right headline.

### 2.2 Abuse carried by implication

A separate line of work argues that the hard case is not offensive vocabulary but its absence.
ElSherief et al. \cite{elsherief2021latent} build a taxonomy of implicit hate and a corpus in which
implication is annotated as such, and report that models competent on explicit hate lose most of their
accuracy on it. Ocampo et al. \cite{ocampo2023indepth} separate hate speech into explicit and implicit
layers and show that the implicit layer is where detectors and annotators both struggle. ToxiGen
\cite{hartvigsen2022toxigen} makes the point that implicit abuse is scarce in naturally collected data.
HateXplain \cite{mathew2021hatexplain} supplies rationales, which help where the evidence is lexical
and least where it is not. The annotation literature adds that the texts annotators disagree on are
disproportionately the ones whose hostility is implied rather than stated \cite{uma2021learning,
davani2022dealing, plank2022problem}.

Two things follow for a paper about compact models. A corpus that does not label implication cannot be
used to train for it, and the domain-specialised encoders a distillation committee would naturally
recruit were themselves trained on corpora of that kind. Implicit abuse has been studied as a benchmark
problem for large models and as an annotation problem, not as a transfer problem: whether the ability
to read implication can be moved into a model small enough to deploy, and by what mechanism, has not
been asked. This paper asks it, and Section 4.2 measures, on the two corpora that annotate the
distinction, how far the label itself can be trusted.

### 2.3 Knowledge distillation for pre-trained language models

Distillation transfers a large model's softened outputs to a smaller one \cite{hinton2015distilling,
bucilua2006model}; intermediate-feature transfer follows FitNets \cite{romero2015fitnets}; surveys
organise response-, feature- and relation-based variants \cite{gou2021knowledge}. For BERT-style
encoders the line runs through DistilBERT \cite{sanh2019distilbert}, Patient KD \cite{sun2019patient},
TinyBERT \cite{jiao2020tinybert}, MiniLM \cite{wang2020minilm} and MobileBERT \cite{sun2020mobilebert}.
Two findings shaped our design and, in the end, our explanation. Pre-training the compact student
matters more than the distillation recipe \cite{turc2019wellread}, and non-transformer students such as
BiLSTMs can be distilled from BERT \cite{tang2019distilling}. Distillation does not always work as
expected \cite{stanton2021does, cho2019efficacy}: students fail to match teachers they have the
capacity to match, and the fidelity of the imitation depends on the data it is practised on. Beyer et
al. \cite{beyer2022knowledge} make the same point from the other side, showing that a teacher must be
consistent and patient, distilled over many passes of augmented data, before the student learns its
function rather than its labels. The task-specific recipes that report the largest gains for small
students \cite{tang2019distilling, jiao2020tinybert, turc2019wellread} all distil on a transfer set
larger than the labelled training data. Our audit was run on the labelled data alone, which is the
common practice in this application area, and Section 6 argues that this is where the gains went.

### 2.4 Multi-teacher and adaptive distillation

Ensembles of teachers \cite{you2017learning, fukuda2017efficient} are combined with fixed or learned
weights: adaptive multi-level weighting \cite{liu2020adaptive}, gradient-space adaptation
\cite{du2020agree}, reinforced teacher selection \cite{yuan2021reinforced} and confidence-aware
weighting \cite{zhang2022confidence}. For language models, MT-BERT co-fine-tunes several teachers and
weights their soft labels by prediction error while aligning hidden states through learned projections
\cite{wu2021one}; dynamic weighting by teacher confidence has also been used for semantic parsing
\cite{zou2025dynamic}. Heterogeneous distillation, where the architectures differ, needs an explicit
information-flow model \cite{passalis2020heterogeneous}.

The method we audit belongs to the error-weighted family: its per-instance weights follow
\cite{wu2021one, zhang2022confidence}, and the combination of error-weighted soft labels with projected
hidden-state alignment is MT-BERT's. We do not claim that combination as novel. What the family shares,
and what none of its papers tests, is a pair of assumptions: that the teachers collectively know more
than any one of them in a way their outputs reveal, and that reliability scored against the gold label
on the training data measures anything once the teachers have been fine-tuned on that data. Section 5.5
tests both and finds both false on a committee assembled in the usual way. The finding is not specific
to our implementation, because the weighting rule is the family's.

### 2.5 Learning from annotator disagreement

Crowd labels carry information beyond the majority vote \cite{peterson2019human}, and surveys argue for
modelling disagreement rather than discarding it \cite{uma2021learning, davani2022dealing,
plank2022problem}. The Wikipedia corpus provides about ten votes per comment
\cite{wulczyn2017exmachina}; the disagreement-aware variant of the objective (Section 3.3) uses them,
and its evaluation belongs to the Wikipedia grid, which is not yet run. The measurement we do report
from this literature is the one in Section 4.2: two expert annotation efforts agree about whether a text
is hateful and disagree about whether the hate is implied, which bounds every result on the implicit
class, ours included.

## 3. The method under audit

### 3.1 Notation and committee

Let $x_i$ be a text with majority label $y_i \in \{1,\dots,C\}$. A committee of $K$ teachers produces
logits $z_k(x_i) \in \mathbb{R}^C$ and masked-mean-pooled last-layer states $h_k(x_i) \in
\mathbb{R}^{d_k}$; the student produces $z_s(x_i)$ and $h_s(x_i) \in \mathbb{R}^{d_s}$, plus an
auxiliary irony head $z_s^{\text{irony}}(x_i) \in \mathbb{R}^2$. $W_k \in \mathbb{R}^{d_k \times d_s}$
is a learned projection from the student width to teacher $k$'s width; $T$ is the distillation
temperature and $\tau$ the weight temperature.

Teachers are *task-adapted*: each is initialised from a specialist checkpoint and fine-tuned on the
target training split, so that every committee member shares the task's label space. Without this
step a sarcasm model's logits are not comparable with an abuse model's and cannot enter a KL term. The
homogeneous committee $\mathcal{K}_{\text{homo}}$ is BERT-large \cite{devlin2019bert}, HateBERT
\cite{caselli2021hatebert} and a Twitter irony RoBERTa \cite{barbieri2020tweeteval, liu2019roberta}: a
general encoder, a specialist in abuse that says what it means, and a specialist in saying one thing
and meaning another. The heterogeneous committee adds DeBERTa-v3-base \cite{he2023debertav3}. The
third committee adds the implicit specialist of Section 3.4. Teachers are frozen after adaptation and
their outputs on the training split cached once; the student never sees a teacher at run time.

### 3.2 Per-instance reliability

For each training instance the committee is weighted by how reliable each teacher is on it:

$$
\ell_k(i) = \mathrm{CE}\big(\mathrm{softmax}(z_k(x_i)),\, y_i\big), \qquad
w_k(i) = \frac{\exp(-\ell_k(i)/\tau)}{\sum_{j} \exp(-\ell_j(i)/\tau)} .
$$

$w_k(i)$ is the posterior probability that teacher $k$ is the right expert for $x_i$ under a uniform
prior and a likelihood $p(y_i \mid k, x_i)^{1/\tau}$: at $\tau = 1$ it is Bayesian model averaging over
the committee on that instance; $\tau \to \infty$ recovers uniform averaging and $\tau \to 0$ hard
selection of the single best teacher. Per-batch weighting, as in earlier work, replaces $\ell_k(i)$ by
its batch mean. Because the teachers are frozen and cached, $w_k(i)$ is a fixed function of the data:
it varies across instances and never across epochs or seeds, so "dynamic" means instance-adaptive and
nothing else. Where a corpus provides annotator fractions $s_i$, reliability can be measured against
them instead ($\ell_k(i) = \mathrm{BCE}(p_k^{\text{attack}}(x_i), s_i)$).

### 3.3 Objective

$$
\bar p(i) = \sum_k w_k(i)\, \mathrm{softmax}\!\big(z_k(x_i)/T\big), \qquad
\mathcal{L}_{\mathrm{KL}}(i) = T^2\, \mathrm{KL}\!\big(\bar p(i)\,\|\,\mathrm{softmax}(z_s(x_i)/T)\big),
$$
$$
\mathcal{L}_{\mathrm{hid}}(i) = \sum_k w_k(i)\, \big\| W_k h_s(x_i) - h_k(x_i) \big\|_2^2, \qquad
\mathcal{L}_{\mathrm{irony}}(i) = T^2\, \mathrm{KL}\!\big(\mathrm{softmax}(z_{\text{irony}}(x_i)/T)\,\|\,\mathrm{softmax}(z_s^{\text{irony}}(x_i)/T)\big),
$$
$$
\mathcal{L} = \frac{1}{B}\sum_i \Big[\alpha\, \mathcal{L}_{\mathrm{KL}}(i) + \beta\, \mathrm{CE}(z_s(x_i), y_i)
 + \gamma\, \mathcal{L}_{\mathrm{hid}}(i) + \delta\, \mathcal{L}_{\mathrm{irony}}(i)\Big],
$$

with $\alpha = \beta = 0.4$, $\gamma = 0.2$, $\delta = 0.3$, $T = 4$ and $\tau = 1$ unless stated. The
irony teacher cannot join the committee, because its label space is irony against non-irony, so its
softened output is distilled into a second head on the student's pooled state; the head is discarded at
inference and exists to shape the shared representation. On corpora with annotator fractions a soft
term $\mathrm{BCE}(\sigma(z_s^{\text{attack}}), s_i)$ shares $\gamma$ with the hidden term, and the
disagreement-aware variant scales the hard-label term by annotator agreement $a_i$ and the teacher term
by $1 + \kappa(1 - a_i)$; both belong to the Wikipedia grid and are not evaluated here. Uniform
multi-teacher distillation sets $w_k(i) = 1/K$; single-teacher distillation uses BERT-large alone;
fine-tune-only sets $\alpha = \gamma = \delta = 0$.

Training: 6 epochs, AdamW \cite{loshchilov2019decoupled} at $3 \times 10^{-5}$ ($10^{-3}$ for the
BiLSTM), 10 per cent warm-up and linear decay, batch 32, gradient clipping at 1.0, mixed precision on
GPU, best validation macro-F1 epoch kept, implemented in PyTorch and Transformers
\cite{paszke2019pytorch, wolf2020transformers}.

### 3.4 The implicit specialist and its controls

No public checkpoint and neither established corpus holds knowledge of abuse by implication, so a
committee assembled from them cannot route to it. We therefore train the fourth member: HateBERT
fine-tuned on the implicit benchmark of Section 4.2, then task-adapted onto the tweet corpus like every
other teacher, its classification head re-initialised for the six-class label space. It forms its own
committee, $\mathcal{K}_{\text{spec}} = \mathcal{K}_{\text{homo}} \cup \{\text{specialist}\}$, so
that its effect is a three-seed comparison of two full committees rather than a single ablation, and
every result obtained with the three-teacher committee remains valid.

Two controls accompany it, fixed before it was trained. The first distils from the specialist alone,
asking whether the committee contributes anything the specialist does not. The second fine-tunes the
same student on the implicit corpus and then on the tweet task with no teacher, asking whether the
knowledge, if any arrives, came from distillation or from the data. If either control matches the full
committee, the corresponding claim is unavailable and the paper says so.

### 3.5 Does the weighting select?

A committee of specialists is only a committee if the weights go somewhere. For each teacher we report
the *routing contrast*: its mean weight on instances of the class that holds indirect abuse minus its
mean weight on the classes that name their target, with a percentile bootstrap interval over 2,000
resamples. With tens of thousands of training instances almost any contrast excludes zero, so its size
is read against the uniform weight $1/K$: a weighting that selects an expert moves a large share of it.
And because the weights are a fixed function of the data, the sweep over $\tau$ reaches 0.05, where
they are as sharp as the reliability signal allows.

## 4. Benchmarks, probes and protocol

### 4.1 The tweet corpus

The fine-grained cyberbullying corpus \cite{wang2020sosnet} contains 47,692 tweets in six classes: age,
ethnicity, gender, religion, other_cyberbullying and not_cyberbullying. Repeated tweets are common and
some repeats carry different labels. We drop tweets shorter than two tokens (758), every tweet whose
normalised text appears with more than one label, all copies included (1,563 texts, 3,168 rows), and
exact duplicates (507), then split the 43,259 survivors 80/10/10 stratified by class with seed 42,
asserting the splits disjoint on normalised text: 34,607 / 4,326 / 4,326. The two classes that lose
the most rows, other_cyberbullying and not_cyberbullying, are the two the literature reports as
confusable. A validation-to-test accuracy gap of 0.94 against 0.86 observed on the raw corpus
disappears on the cleaned split.

### 4.2 The implicit benchmark

Neither the tweet corpus nor the Wikipedia corpus labels indirectness, so a model trained on either
never sees an example annotated as implicit. Two corpora annotate the distinction: the Implicit Hate
Corpus stage-1 release, 21,480 posts in three classes (not_hate, explicit_hate, implicit_hate)
\cite{elsherief2021latent}, and ISHate, 28,763 original rows after excluding its augmentations and its
ToxiGen-sourced rows \cite{ocampo2023indepth}.

**The splits come from one corpus, and the reason is measured.** In the union, 95 per cent of implicit
examples come from the Implicit Hate Corpus and 89 per cent of explicit examples from ISHate, and a
TF-IDF classifier tells the two corpora apart at 0.91 macro-F1. A model trained on the union can
therefore score well on implicit-against-explicit by recognising the source. It also does worse where
it matters: on identical Implicit Hate test rows, a classical model trained on the union reaches 0.473
F1 on implicit hate against 0.548 for the same model trained on that corpus alone, while its aggregate
macro-F1 is higher, 0.681 against 0.562, which is exactly the artefact. We therefore build train,
validation and test from the Implicit Hate Corpus and hold ISHate out whole as a 27,096-row
out-of-domain test set. Every text occurring in any probe set (Section 4.3) is removed first, 1,581
rows, so that probe metrics stay measured on text no model has trained on; the survivors then go
through the tweet pipeline (10 short rows, 12 multi-label texts, 12 duplicates removed) and are split
80/10/10 with seed 42: 16,509 / 2,064 / 2,064, training classes not_hate 10,616, implicit_hate 5,029,
explicit_hate 864.

**The label is contested.** The two corpora share 629 texts. On whether a text is hateful at all they
agree on 99.7 per cent; on whether the hate is stated or implied they agree on 48.2 per cent, the
Implicit Hate Corpus labelling essentially all of the 624 shared hateful texts implicit and ISHate
labelling 324 explicit and 300 implicit. Two expert annotation efforts placing the boundary in
materially different places bounds how sharp any result on the implicit class can be, and is the
reason the benchmark's headline metric is a threshold-free ranking (Section 4.4) rather than a
decision.

The benchmark is released as a build script plus a per-row manifest (split, label, source, SHA-1 of
the normalised text), so a reader can verify a rebuild row by row without either party redistributing
text.

### 4.3 Probes

Three inference-only sets, never used for training by any model on any corpus. *Benign sarcasm*:
1,064 sarcastic tweets from iSarcasmEval \cite{abufarha2022semeval, oprea2020isarcasm},
author-labelled, of which 883 survive a screening pass over the rows a classical classifier flags as
abusive; the metric is the false-positive rate, the share predicted as any abusive class. *Ironic
abuse*: 1,560 items, 797 posts labelled irony in the Implicit Hate Corpus stage-2 data and 763 original
ISHate rows labelled implicit; the metric is recall. *Implicit abuse*: the 763 ISHate rows alone. ISHate
rows sourced from ToxiGen or the Implicit Hate Corpus are excluded from the probes so that they stay
independent of the corpus they test transfer into. The probes are rule-built from published corpora,
not hand-verified; a 300-item human annotation with inter-annotator agreement \cite{fleiss1971measuring}
is in progress. Probe metrics are only meaningful for models trained on Twitter-domain data.

### 4.4 Metrics

Recall on ironic abuse at a fixed threshold cannot distinguish a model that fails to see indirect abuse
from one that sees it and cannot separate it from harmless sarcasm. We report **sarcasm-discrimination
AUC**: the ROC-AUC of $p(\text{abusive})$ with the ironic-abuse probe as positives and the
benign-sarcasm probe as negatives, threshold-free and defined identically on every corpus, with the
recall-against-false-positive curve beside it because a deployment must choose a threshold even though
an evaluation should not. On the implicit benchmark the analogue is **implicit-discrimination AUC**,
implicit-hate rows against benign rows ranked by $p(\text{implicit hate})$; macro-F1 there is dominated
by an explicit class of 108 test rows and is reported but not argued on. Elsewhere: macro-F1, accuracy,
expected calibration error \cite{guo2017calibration}, and per-class F1.

### 4.5 Teachers, students, floor

Teachers: BERT-large-uncased (335M), HateBERT (110M), Twitter-RoBERTa-base-irony (125M), DeBERTa-v3-base
(184M), and the implicit specialist (HateBERT, 110M), each fine-tuned on the tweet training split for
5 epochs at $2 \times 10^{-5}$, batch 32, best validation epoch kept, trained once. A provenance rule
excludes from every test set any data a teacher's checkpoint was trained on before adaptation.
Students: BERT-mini (11.2M, the headline student), BERT-small (28.8M) \cite{turc2019wellread},
DistilBERT (67.0M) \cite{sanh2019distilbert}, DeBERTa-v3-xsmall (70.8M) \cite{he2023debertav3}, and a
two-layer BiLSTM over a WordPiece vocabulary (10.4M) \cite{tang2019distilling}; the last two are the
heterogeneous students. The classical floor is TF-IDF over word 1-2 grams and character 3-5 grams with
class-balanced logistic regression: 0.8798 macro-F1 on tweets, 0.5620 on the implicit benchmark
(implicit_hate F1 0.5562, implicit-discrimination AUC 0.7610). Every neural result is read against it.

### 4.6 Protocol and statistics

Three seeds per main configuration, one for ablations, controls and sweeps. Every difference the paper
states is a paired bootstrap over the test set \cite{koehn2004statistical, dror2018hitchhiker}: seeds
paired by index, 1,000 resamples per seed, the 95 per cent interval of the pooled differences. The grid
makes 66 such comparisons and they are generated as a table, not transcribed. A pre-registered stop
rule required at least one task-adapted teacher to beat the fine-tune-only student on the clean test
split before any distillation ran.

**One rule constrains every table.** The same code, data and seeds produce systematically different
results on a laptop CPU and on a Kaggle T4 (Section 7), so every number comes from the Kaggle grid
alone. Comparisons within it are valid because everything in it was trained identically; numbers
produced elsewhere are excluded rather than reconciled. The tweet grid is complete: 131 student runs,
9.6 GPU-hours. The Wikipedia grid and the student grid on the implicit benchmark are not yet run and
are marked where they would appear.

## 5. Results

### 5.1 The teachers clear the bar

| Teacher | Macro-F1 | Accuracy | ECE | Benign-sarcasm FPR | Ironic-abuse recall |
|---|---|---|---|---|---|
| BERT-large | 0.8973 | 0.9075 | 0.056 | 0.224 | 0.585 |
| Implicit specialist, task-adapted | 0.8931 | 0.9043 | 0.072 | 0.263 | 0.678 |
| RoBERTa-irony, task-adapted | 0.8926 | 0.9050 | 0.067 | 0.294 | 0.718 |
| HateBERT, task-adapted | 0.8887 | 0.9011 | 0.076 | 0.317 | 0.696 |
| DeBERTa-v3-base | 0.8720 | 0.8867 | 0.053 | 0.409 | 0.773 |
| TF-IDF + logistic regression | 0.8798 | 0.8930 | -- | -- | -- |

Three of the original four teachers beat the fine-tune-only student and the classical floor, so the
stop rule passes. DeBERTa-v3-base sits below the floor and is retained only as the heterogeneous
committee member, which is what it is there to test. The implicit specialist, trained on different
data under a different label scheme, lands within 0.009 of the other three after adaptation. The probe
columns already show the pattern Section 5.8 makes precise: ordered by false-positive rate on harmless
sarcasm the teachers are almost exactly ordered by recall on ironic abuse.

### 5.2 Distillation raises every student by less than the test set can resolve

Test macro-F1, mean over three seeds; classical floor 0.8798.

| Student | Params | Fine-tune only | Single-teacher | Uniform | D-MTHD | Uniform + DeBERTa | D-MTHD + DeBERTa |
|---|---|---|---|---|---|---|---|
| BERT-mini | 11.2M | 0.8393 | 0.8405 | 0.8385 | 0.8378 | 0.8378 | 0.8373 |
| BERT-small | 28.8M | 0.8469 | 0.8488 | 0.8505 | 0.8471 | 0.8509 | 0.8477 |
| DistilBERT | 67.0M | 0.8907 | 0.8963 | 0.8960 | 0.8960 | 0.8975 | 0.8966 |
| DeBERTa-v3-xsmall | 70.8M | 0.8825 | 0.8865 | 0.8862 | 0.8833 | 0.8847 | 0.8828 |
| BiLSTM | 10.4M | 0.8693 | 0.8748 | 0.8761 | 0.8732 | 0.8760 | 0.8764 |

Best distilled configuration against fine-tune-only: BERT-mini +0.0012 [-0.0057, +0.0079], BERT-small
+0.0040 [-0.0058, +0.0126], DistilBERT +0.0068 [-0.0013, +0.0139], DeBERTa-v3-xsmall +0.0040 [-0.0024,
+0.0109], BiLSTM +0.0071 [-0.0039, +0.0195]. Five of five positive, which is the result the control was
added to test and which an earlier version of this work failed; and none of the five intervals
excludes zero, nor does any of the 27 distillation-against-fine-tuning comparisons in the grid. Taking
the best of six configurations per student also flatters the difference. The supportable claim is the
modest one: distillation moves every student in the same direction, by 0.001 to 0.007 macro-F1, which
a test set of 4,326 posts cannot resolve.

Two students deserve a remark. The most heterogeneous student, the BiLSTM, gains the most, and the
student that shares the teachers' architecture most closely, BERT-mini, gains the least; the hidden-state
term, the only part of the objective that architectural similarity could help, is worth +0.0009 on
BERT-mini. Nothing in this grid supports a preference for homogeneous student and teachers. And on
BERT-mini nothing beats the no-teacher control at all: candidate minus fine-tune-only is -0.0015
[-0.0076, +0.0048] for D-MTHD, -0.0007 [-0.0074, +0.0058] for uniform, +0.0012 [-0.0057, +0.0079] for
single-teacher, -0.0019 [-0.0082, +0.0041] with the DeBERTa committee, -0.0003 [-0.0080, +0.0075] and
-0.0019 [-0.0093, +0.0049] with the specialist committee under averaging and weighting.

The efficiency claim sits in this table. DistilBERT with a uniform heterogeneous committee reaches
0.8975 against BERT-large's 0.8973 at a fifth of the parameters and latency (Section 5.11). We state
that as parity, not as the student beating the teacher.

### 5.3 Per-instance weighting does not differ from uniform averaging at any temperature

| Student | D-MTHD minus uniform | Same, with DeBERTa |
|---|---|---|
| BERT-mini | -0.0007 [-0.0074, +0.0061] | -0.0005 [-0.0067, +0.0054] |
| BERT-small | -0.0034 [-0.0106, +0.0036] | -0.0032 [-0.0100, +0.0033] |
| DistilBERT | +0.0000 [-0.0079, +0.0082] | -0.0010 [-0.0087, +0.0069] |
| DeBERTa-v3-xsmall | -0.0029 [-0.0086, +0.0029] | -0.0019 [-0.0108, +0.0108] |
| BiLSTM | -0.0029 [-0.0116, +0.0056] | +0.0004 [-0.0072, +0.0081] |

Four negative, one tie, every interval containing zero. On the headline student the ablation that
removes the weighting altogether scores 0.8386 against 0.8385 for full D-MTHD on the same seed.

The obvious objection is that at $\tau = 1$ the weights are nearly uniform, 0.332 / 0.337 / 0.331
against 0.333, so the two objectives coincide by arithmetic. The sweep therefore reaches $\tau = 0.05$.

| $\tau$ | Mean weights, BERT-large / HateBERT / irony | Test macro-F1, seed 1 |
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

Across a hundredfold range of temperature the score stays inside 0.0007 and below fine-tune-only at
every value; the sharpest setting differs from the default by +0.0007 [-0.0008, +0.0023] and from
uniform averaging by +0.0020 [-0.0042, +0.0077]. The weights barely sharpen: even at $\tau = 0.05$ the
heaviest teacher averages 0.356. Section 5.5 says why.

### 5.4 The committee is worth more than its best member, and the student receives none of it

Scored directly from the teachers' saved test probabilities, before any student is involved:

| Committee | Best single | Uniform mean | Confidence-weighted | Entropy-weighted | Most confident teacher | Stacked gate (5-fold CV) | Oracle accuracy |
|---|---|---|---|---|---|---|---|
| Homogeneous (3) | 0.8973 | 0.9026 | 0.9018 | 0.9023 | 0.9015 | 0.9046 | 0.9454 |
| + implicit specialist (4) | 0.8973 | 0.9000 | 0.9005 | 0.9011 | 0.9011 | 0.9043 | 0.9494 |
| + DeBERTa (4) | 0.8973 | 0.9051 | 0.9055 | 0.9052 | 0.9005 | 0.9055 | 0.9526 |
| All five | 0.8973 | 0.9036 | 0.9046 | 0.9030 | 0.9000 | 0.9041 | 0.9552 |

Uniform averaging of the committee beats the best teacher by 0.3 to 0.8 macro-F1. No combination that
needs no gold label at inference improves on the uniform mean by more than 0.003, and neither does a
stacked logistic-regression gate fitted to the concatenated teacher probabilities by five-fold
cross-validation over the test set, which is as favourable a test of learned routing as can be made
without touching the training data. The oracle that picks a correct teacher whenever one exists sits
at 0.945 to 0.955 accuracy, against 0.9075 for BERT-large. That headroom is real and it is
unreachable from the teachers' outputs: where the teachers disagree, 10 to 14 per cent of the test
set, some teacher is right on about 90 per cent of items and the uniform mean on 55 to 62, and their
probabilities do not say which. A corrected reliability signal therefore has a ceiling of roughly +0.3
points over uniform averaging on this committee.

The students do not receive even the 0.3 to 0.8 that the committee has. Uniform multi-teacher against
single-teacher distillation is -0.0019 [-0.0077, +0.0038], +0.0017 [-0.0070, +0.0115], -0.0003
[-0.0070, +0.0060], -0.0003 [-0.0072, +0.0070] and +0.0014 [-0.0073, +0.0099] across the five students,
and adding DeBERTa to the averaged committee changes them by -0.0007, +0.0004, +0.0016, -0.0015 and
-0.0002, all inside noise. A better label on the same 34,607 training texts does not make a better
student.

### 5.5 Why: the reliability is read where every teacher has memorised the label

The weights are computed from cached teacher logits on the training split, the split each teacher was
fine-tuned on. By their last epoch the teachers' training losses are 0.033 (BERT-large), 0.059
(HateBERT), 0.077 (irony), 0.061 (implicit specialist) and 0.227 (DeBERTa-v3-base). On the data the
weights are read from, every task-adapted teacher assigns the gold label a probability near one on
nearly every instance, cross-entropy differences between them are a few hundredths, and the softmax
of Section 3.2 is flat at any temperature that does not amplify noise. The one teacher that did not
memorise the split, DeBERTa, is the one whose weight moves (0.233 in a committee of four). The
reliability signal is measuring memorisation, not reliability, and this is a property of the
error-weighted family \cite{wu2021one, zhang2022confidence}, not of our implementation: any scheme that
scores reliability against the gold label on the training data will find every fine-tuned teacher
equally reliable there.

The same fact bears on the distillation itself. On the training split the teachers' soft labels are
close to the one-hot gold labels, so the KL term carries little that the cross-entropy term does not
and single-teacher, uniform and weighted distillation all reduce to fine-tuning with a slightly
smoothed target. This is the hazard of distilling a memorising teacher on its own training data
\cite{stanton2021does, beyer2022knowledge}, and it explains Section 5.4 from the student's side: the
committee's half point exists on the test set, where the teachers are uncertain, and not on the
training set, where they are not.

The teachers are also not diverse enough for a weighting to have work to do, and adaptation is what
made them so.

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

The original four teachers are all right on 83.2 per cent of the test set and all wrong on the same
4.7 per cent. A per-instance weighting can only express a preference where its members disagree, at
most one instance in eleven here, and on two-thirds of the errors there is no correct teacher to
prefer. The committee was chosen for complementary expertise, a general encoder, an abuse specialist,
an irony specialist, and after task adaptation on the same corpus its members converged to
near-identical behaviour. Task adaptation is what makes a sarcasm model's logits comparable with an
abuse model's, and it is also what removes the diversity the combination was meant to exploit. **Task
adaptation buys comparability and spends diversity**, and any cross-task committee faces the trade.

### 5.6 A teacher that has seen implication labelled does not help either

The implicit specialist is the sharpest case. It learned from a different corpus under a different
label scheme, and it comes back from adaptation as the closest thing in the committee to HateBERT, the
model it started from: kappa 0.962, disagreement on 3.1 per cent of the test set, 84 per cent of
HateBERT's errors made by the specialist as well. It raises the oracle over the committee only from
0.9526 to 0.9552. BERT-mini:

| Configuration | Seeds | Macro-F1 | F1 on other_cyberbullying | Sarcasm-discrimination AUC (seed 1) |
|---|---|---|---|---|
| Fine-tune only | 3 | 0.8393 +- 0.0003 | 0.707 | 0.773 |
| Uniform, homogeneous committee | 3 | 0.8385 +- 0.0018 | 0.700 | 0.774 |
| Uniform, + implicit specialist | 3 | 0.8390 +- 0.0035 | 0.699 | 0.768 |
| D-MTHD, homogeneous committee | 3 | 0.8378 +- 0.0011 | 0.705 | 0.773 |
| D-MTHD, + implicit specialist | 3 | 0.8373 +- 0.0025 | 0.697 | 0.768 |
| Specialist alone, no committee | 1 | 0.8392 | 0.709 | 0.767 |
| Pre-trained on the implicit corpus, no teacher | 1 | 0.8377 | 0.705 | 0.756 |

Adding the specialist changes macro-F1 by +0.0005 [-0.0059, +0.0064] under averaging and -0.0005
[-0.0069, +0.0055] under weighting; F1 on the class that holds indirect abuse and the discrimination
AUC both fall slightly. Neither control does better: the specialist alone sits within 0.0002 of the
no-teacher student, and putting the implicit corpus into the student directly, by pre-training on it,
gives the lowest discrimination AUC of the 25 pre-trained models analysed.

The weights do lean towards the specialist, a little, and no more than towards HateBERT. Measured on
the training split over all five cached teachers (uniform weight 0.200), the mean weight on
other_cyberbullying minus the mean weight on the four targeted classes is +0.0486 [+0.0460, +0.0512]
for HateBERT and +0.0441 [+0.0416, +0.0465] for the specialist at $\tau = 0.05$, +0.0105 and +0.0090 at
$\tau = 1$; RoBERTa-irony +0.0184 and +0.0053; BERT-large -0.0063 and +0.0011; DeBERTa -0.1048 and
-0.0259. At the temperature the students trained with, the specialist's shift is a twentieth of the
uniform weight; it is not specific to the specialist, as the kappa predicts; and because the weights
come from cross-entropy against the gold label, a shift says only that a teacher fits
other_cyberbullying better than the targeted classes, not that it reads implication. (This measurement
pools all five teachers, a committee no student trained with; the per-committee measurement is queued.)

### 5.7 Ablations and sweeps: only pre-training matters

BERT-mini, one seed, against full D-MTHD at 0.8378: randomly initialised student 0.7905 (-0.0473); no
auxiliary irony head 0.8364 (-0.0014); no hidden-state term 0.8369 (-0.0009); uniform weights 0.8386
(+0.0008); per-batch weights 0.8389 (+0.0011); the specialist alone 0.8392 (+0.0014); pre-trained on
the implicit corpus 0.8377 (-0.0001). Removing components of the method costs at most 0.0014 and in
two cases improves the score; removing pre-training costs 0.047, and against fine-tune-only on the
same seed it is the one difference in the grid whose interval excludes zero, -0.0488 [-0.0603,
-0.0376]. Hyper-parameters behave the same way: $T \in \{1, 2, 8\}$ gives 0.8386 / 0.8402 / 0.8391,
$\alpha \in \{0.2, 0.6\}$ 0.8386 / 0.8394, $\delta \in \{0.1, 0.5\}$ 0.8389 / 0.8399, and $\tau$ as in
Section 5.3. The objective is flat in every direction we can move it. Adding a DeBERTa-v3 teacher to
the homogeneous committee changes the five students by -0.0005, +0.0006, +0.0032, -0.0005 and +0.0006.

### 5.8 Discrimination is fixed by pre-training, and recall at a threshold measures readiness to fire

Diagnostic on BERT-mini fine-tune-only, seed 1. Recall on the ironic-abuse probe is 0.91 on the 43
items containing explicit profanity and 0.76 on the 1,517 without; on the implicit-abuse probe 0.91 on
11 items with and 0.74 on 752 without. Profanity helps, by 6 to 16 points of recall on every one of the
25 pre-trained models analysed, but the items without it are still caught about three times in four
on every model: the detector leans on profanity without being a profanity detector. The failure is
discrimination. Mean $p(\text{abusive})$ is 0.748 on ironic abuse and 0.449 on benign sarcasm, the
distributions overlap heavily, and sarcasm-discrimination AUC is 0.773. At a 0.5 threshold, recall of
0.792 costs a 39.5 per cent false-positive rate on sarcasm that attacks nobody, and no threshold up to
0.90 brings that rate under 10 per cent, on this model or on any of the 26 analysed. The benchmark's own
labels show the same confusion: other_cyberbullying recall 0.710 with 109 of its errors going to
not_cyberbullying, and not_cyberbullying recall 0.648 with 126 going the other way.

**Across the grid.** Over the 25 pre-trained BERT-mini models analysed, which cover every objective
and committee, seven temperatures, the $T$, $\alpha$ and $\delta$ sweeps, every ablation and both
controls, sarcasm-discrimination AUC has mean 0.771 and standard deviation 0.005, from 0.756 to 0.777.
The randomly initialised student scores 0.613. The auxiliary irony head, the component built for this
measurement, is inside that band: without it 0.775, with it 0.773. Over the same models the operating
point at 0.5 moves far more, recall from 0.758 to 0.821 and false-positive rate from 0.343 to 0.448,
correlated at +0.87.

Across all 133 evaluated models in the grid, teachers and students, every mode and seed, the
false-positive rate on benign sarcasm and the recall on ironic abuse correlate at +0.712. Recall minus
false-positive rate has mean 0.357 and standard deviation 0.051, while its components range over 0.14
to 0.59 and 0.54 to 0.81.

| Model family | n | Benign FPR | Ironic recall | Margin |
|---|---|---|---|---|
| DeBERTa-v3-xsmall | 21 | 0.334 | 0.728 | 0.394 |
| Teachers | 5 | 0.301 | 0.690 | 0.388 |
| BERT-mini | 44 | 0.390 | 0.765 | 0.375 |
| DistilBERT | 21 | 0.238 | 0.607 | 0.369 |
| BERT-small | 21 | 0.380 | 0.713 | 0.333 |
| BiLSTM | 21 | 0.336 | 0.623 | 0.287 |

Distillation mode does not appear in this ordering; architecture does. The models occupy different
points on one trade-off curve rather than different curves, and the worst margin in the grid, 0.082,
belongs to the randomly initialised student. Whatever ability these models have to tell implied abuse
from harmless sarcasm comes from pre-training, and nothing added to the objective moves it. This is why
the paper reports a threshold-free area: recall at a fixed cut measures how readily a model fires, and
across a hundred models that is all it measures.

### 5.9 The same failure on a corpus that labels implication

| Model | macro-F1 | not_hate F1 | explicit_hate F1 | implicit_hate F1 | Implicit-discrimination AUC |
|---|---|---|---|---|---|
| Classical floor | 0.5620 | - | - | 0.5562 | 0.7610 |
| BERT-mini, fine-tune only | 0.5549 | 0.8347 | 0.2857 | 0.5443 | 0.8196 |
| HateBERT specialist | 0.6029 | 0.8352 | 0.3662 | 0.6072 | 0.8247 |

The compact student loses to a bag of n-grams on F1 and beats it decisively on ranking: it separates
implied abuse from benign text at 0.8196 AUC against 0.7610 while scoring below the floor on the
implicit class itself. The two measures disagree in direction, not degree. A reader looking only at F1
would conclude the neural model had learned nothing the n-grams had not; the ranking says it has
learned a great deal and cannot convert it into a decision, which is the argument of Section 5.8 made
on other people's labels. Both neural models score 0.29 to 0.37 on the 864-row explicit class, which is
why macro-F1 is not the headline here. These are single-seed results; the student grid on this
benchmark is not yet run.

**Transfer.** Applied without further training to the benchmark's 2,064 test posts, the tweet-trained
BERT-mini calls 81.0 per cent of them abusive against a true rate of 35.7 per cent (binary macro-F1
0.423, ROC-AUC 0.608). On implicit hate against posts that are not hateful it catches 85.1 per cent at a
78.8 per cent false-positive rate and ranks the two at an AUC of 0.600, barely above chance, against
0.820 for the same architecture trained on the corpus. D-MTHD transfers no better (0.411, 0.600).
Whatever the tweet students have learned about abuse does not include what makes an implication
hateful, which is consistent with Section 5.6: there was little of that knowledge in the committee to
receive.

### 5.10 Robustness

Synthetic obfuscation of the abusive rows, BERT-mini, seed 1, macro-F1: fine-tune only 0.8394 clean,
0.7475 leetspeak, 0.7417 character swap, 0.7275 inserted spaces, 0.7116 mixed; D-MTHD 0.8385, 0.7460,
0.7361, 0.7250, 0.7077. Twelve to thirteen points lost on short text, and distillation does not help.
These are character edits, not evasion by people trying to evade, and we do not call the result
robustness.

### 5.11 Efficiency

Median of five timed passes after warm-up on an idle machine.

| Student | Params | Latency b1 | Latency b32 | GFLOPs/seq | fp32 size | INT8 size | INT8 macro-F1 |
|---|---|---|---|---|---|---|---|
| BiLSTM | 10.4M | 2.57 ms | 0.18 ms | ~0 | 40 MB | 32 MB | 0.8736 |
| BERT-mini | 11.2M | 3.47 ms | 0.37 ms | 0.87 | 43 MB | 33 MB | 0.8319 |
| BERT-small | 28.8M | 3.74 ms | 1.25 ms | 3.36 | 110 MB | 73 MB | 0.8448 |
| DistilBERT | 67.0M | 5.22 ms | 3.72 ms | 11.17 | 255 MB | 132 MB | 0.8931 |
| DeBERTa-v3-xsmall | 70.8M | 23.49 ms | 3.25 ms | 10.57 | 270 MB | 209 MB | 0.8765 |
| BERT-large (teacher) | 335.1M | 26.56 ms | 21.65 ms | 78.92 | 1,279 MB | -- | -- |

DistilBERT matches BERT-large at 5.1 times fewer parameters and 5.8 times lower batch-32 latency; INT8
dynamic quantisation halves its size for 0.003 macro-F1. BERT-mini's size falls only from 43 to 33 MB
because 7.8M of its 11.2M parameters are the embedding table, which dynamic quantisation does not
touch.

## 6. Discussion

**What the audit says about error-weighted multi-teacher distillation.** The family's weighting rule
scores each teacher's reliability against the gold label on the training set. For teachers fine-tuned
on that set the score is saturated, and the weights it produces are uniform whatever the temperature.
Fixing the estimate, by cross-fitting the teachers or by fitting a gate on held-out data, is necessary
for the rule to mean anything, and on a committee assembled in the usual way it is not sufficient: the
teachers' outputs do not carry the information needed to pick the right one where they disagree, so
the reachable gain over uniform averaging is a few tenths of a point. The five-point oracle gap is
real, and it is made of instances on which the teachers, and very likely the annotators, are
uncertain.

**What it says about the committee.** Assembling specialists for complementary expertise and then
fine-tuning them on one corpus produces teachers that agree at kappa 0.9 and share their errors. The
adaptation cannot be skipped, because it is what makes the specialists' outputs comparable; it can only
be paid for. A committee that keeps its diversity needs at least one member whose knowledge was not
obtained by fitting the target training set, and such a member's reliability, measured on that set,
would for once be an out-of-sample quantity.

**What it says about where distillation's gains are.** The half point that the committee holds on the
test set is absent on the training set, where the teachers are near one-hot, and the student is trained
only there. The recipes that report large gains for small students distil on transfer sets larger than
the labelled data \cite{tang2019distilling, jiao2020tinybert, turc2019wellread}; on such data the
teachers are uncertain, their soft labels informative, and a committee has a job a single teacher does
not, since it is the better labeller. Our grid did not include a transfer set, and we read the absence
of gains as evidence about that omission rather than about distillation as such. It is the first thing
we would change.

**What it says about evaluation.** Recall at a fixed threshold and the false-positive rate on
harmless sarcasm move together across 133 models; a method that reports only the first can claim
progress on implication by lowering its threshold. The threshold-free measure shows the ability to be
fixed by pre-training and unmoved by every component we tested. We think this is the paper's most
transferable finding, and the one most likely to change how the next method in this area is reported.

**What would change the answer.** The student grid on the implicit benchmark, where every teacher is
itself trained on labelled implication, is the fairest test distillation can be given, and the audit
makes a prediction there that can fail: implicit-discrimination AUC will not move beyond test-set
noise, while F1, which depends on where the threshold falls, may. A gain in AUC there would be the
constructive result this paper looked for on the tweet corpus and did not find.

## 7. Threats to validity

**Two machines disagree about the same configuration.** BERT-mini fine-tune-only scores 0.8779 on a
laptop CPU and 0.8393 on a Kaggle T4 with identical code, data, seeds and hyper-parameters. Mixed
precision accounts for 0.0014 (full-precision Kaggle: 0.8407); the gold-label sequences match row for
row; re-running the current code on the laptop reproduces its result to sixteen significant figures.
The Kaggle run learns more slowly from the first epoch (training loss 1.02 against 0.87, validation
macro-F1 0.80 against 0.84), which points at the software stack, since the library versions differ,
rather than at the accelerator; the isolating experiments are pinned library versions on the laptop
and a CPU run on Kaggle. Until one runs, every absolute BERT-mini and BERT-small number here may be
under-trained by up to four points, which is also why both sit below the classical floor. Every table
therefore comes from one environment and comparisons are made only within it; the ranking of methods
is unaffected because every run shares the condition, and the discrimination AUC of the two
environments' models differs by 0.003 (0.776 against 0.773).

**Test-set size and seeds.** 4,326 test posts give paired intervals about 0.007 wide, which is larger
than every distillation effect in the grid; three seeds per main configuration and one for ablations,
controls and sweeps. Differences are read from intervals, not from seed variance, and single-seed rows
are indicative only. The threshold-free probe analysis covers the first seed of each configuration.

**Teacher variance is not estimated**; each teacher is trained once. **The heterogeneous student
comparison is confounded**: DeBERTa-v3-xsmall is a weaker model on this task than the BERT-lineage
students, so the swap mixes family with quality; the hidden-state term is the controlled version of
that question. **The routing measurement pools every cached teacher**, a committee no student trained
with. **The probes are rule-built**, about 1,000 items each, with a 300-item human annotation in
progress; probe metrics are reported only for models trained on Twitter-domain data. **The label is
contested**: the two corpora that annotate implication agree on 48.2 per cent of shared hateful texts
about whether the hate is implied, and every result on that class is bounded by it. **Robustness** was
measured with synthetic character edits. **The Wikipedia grid and the implicit student grid are not yet
run**, so the disagreement-aware variant, the second corpus, and the out-of-domain ISHate evaluation
are absent from this version. English only.

## 8. Conclusion

We built the multi-teacher distillation method that the abusive-language literature keeps proposing,
gave it the teacher it was missing, and measured it under the controls it is rarely given. It does not
beat a compact student trained with no teachers, its per-instance weighting does not differ from
averaging at any temperature, and a teacher trained on labelled implication adds nothing, for reasons
we could measure: reliability read where every teacher has memorised the label, diversity spent by the
adaptation that makes teachers comparable, and distillation practised on data where the teachers'
soft labels are already the gold labels. Measured without a threshold, the ability to tell implied
abuse from harmless sarcasm is set by pre-training and unmoved by every component we tested, and the
usual report, recall at a fixed cut, measures readiness to fire. What stands is a protocol for asking
the question honestly, a benchmark built so that the question can be asked at all, an account of
where the gains of distillation are not, and, for practice, a 67M student that matches a 335M teacher
at a fifth of the cost.

## Statements

**Ethics.** The corpora contain offensive, hateful and harassing text and are existing public research
datasets used under their licences (Wikipedia personal attacks CC0; the tweet corpus CC0 as
distributed; ISHate BSL-1.0; the Implicit Hate Corpus and iSarcasmEval for research use). No new user
data was collected and no attempt was made to identify authors. Automated moderation can harm the
people it is meant to protect, misclassifying dialects, reclaimed language, sarcasm and
self-deprecation; the benign-sarcasm probe measures one such failure directly. The models are released
for research; deployment requires human review, calibration to the platform and monitoring for
disparate impact, which we do not evaluate.

**Data.** English social-media text throughout; author demographics are not available and were not
inferred. The implicit benchmark is not redistributed: neither source is ours to re-host. It is
released as a build script plus a per-row manifest (split, label, source, SHA-1 of the normalised text)
that lets a reader verify a rebuild against ours without either party redistributing text.

**Reproducibility.** Code, configuration, seeds and one-command drivers are public
(github.com/mahdihasanshadi/THESIS). Every reported number is produced by a script that writes a
results file; the lab notebook records each run; every table and every interval in this paper is
generated. Data preparation is deterministic (seed 42) and reports every removal count. Checkpoints and
cached teacher outputs will be released on acceptance. Compute: 9.6 GPU-hours on a Kaggle T4 for the
tweet grid.

**Author contributions.** To be completed (CRediT). Supervision: Dr. Muhammad Iqbal Hossain, Sheikh
Araf Noshin.

## Appendix A. Generated tables

The complete tables are in `paper/tables/`: `main`, `teachers`, `implicit`, `routing`, `ablations`,
`sweeps`, `committee`, `homogeneity`, `robustness`, `efficiency`, `significance` (all 66 paired
comparisons), `dataset`, and the raw per-run collection `all_runs.csv`. The per-model sarcasm analysis
for the first seed is `results_tweets_implicit_seed1.csv`; teacher agreement with and without the
specialist is `complementarity.json` and `complementarity_with_specialist.json`.
