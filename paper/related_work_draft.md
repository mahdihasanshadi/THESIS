# Related work (draft skeleton with citation keys from `references.bib`)

Target length 1,800 words. Five themes, each ending with the gap this paper fills. Every claim
below is tied to a key in `paper/references.bib`; do not add a citation that is not in the
verified list without running `verify_refs.py` on it first.

## 2.1 Cyberbullying and toxicity detection

Surveys frame cyberbullying detection as text classification with three persistent problems:
scarce and inconsistent data, weak evaluation practice, and poor transfer across platforms
\cite{rosa2019automatic, salawu2020approaches, emmery2021current}. The wider abusive-language
literature documents the same issues, including training-data quality and its effect on
generalisation \cite{fortuna2018survey, vidgen2020directions, yin2021towards}. Benchmarks used
here: the Wikipedia personal-attacks corpus with crowd annotations \cite{wulczyn2017exmachina},
the fine-grained cyberbullying tweet corpus \cite{wang2020sosnet}, and, for implicit and ironic
abuse, the Implicit Hate Corpus \cite{elsherief2021latent}, ISHate \cite{ocampo2023indepth}, and
ToxiGen \cite{hartvigsen2022toxigen}; HateXplain adds rationales \cite{mathew2021hatexplain}.
Sarcasm resources: iSarcasm and iSarcasmEval \cite{oprea2020isarcasm, abufarha2022semeval} and
SemEval-2018 irony \cite{vanhee2018semeval}. Domain-specialised encoders: HateBERT
\cite{caselli2021hatebert}, the TweetEval models \cite{barbieri2020tweeteval}, and the ToxDect
RoBERTa used by ToxiGen \cite{zhou2021challenges}. Earlier cyberbullying datasets and the
evaluation pitfalls of duplicate-heavy corpora \cite{vanhee2018automatic, founta2018large,
davidson2017automated, borkan2019nuanced}.

**Gap.** Sarcastic and ironic abuse is named as the hard case in every survey, but compact
detectors are evaluated on aggregate F1 only; no work reports the false-positive rate on benign
sarcasm and the recall on ironic abuse of a deployable model. [Our contribution 3.]

## 2.1a Abuse carried by implication

A separate line of work argues that the hard case is not offensive vocabulary but its absence.
ElSherief et al. \cite{elsherief2021latent} build a taxonomy of implicit hate and a corpus in which
implication is annotated as such, and report that models competent on explicit hate lose most of
their accuracy on it. Ocampo et al. \cite{ocampo2023indepth} take the same position from the other
direction, separating hate speech into explicit and implicit layers and showing that the implicit
layer is where detectors and annotators both struggle. ToxiGen \cite{hartvigsen2022toxigen}
approaches the problem by generating implicit examples adversarially, which makes the point that
implicit abuse is scarce in naturally collected data. HateXplain \cite{mathew2021hatexplain}
supplies rationales, which help where the evidence is lexical and help least where it is not.
Related evidence comes from the annotation literature: the texts annotators disagree on are
disproportionately the ones whose hostility is implied rather than stated
\cite{uma2021learning, davani2022dealing, plank2022problem}.

Two things follow for a paper about compact models. First, a corpus that does not label implication
cannot be used to train for it: the widely used cyberbullying and toxicity benchmarks
\cite{wang2020sosnet, wulczyn2017exmachina, davidson2017automated, founta2018large} record topic or
a binary attack flag, so implicit abuse is present in them but invisible to the objective. Second,
the domain-specialised encoders a distillation committee would naturally recruit
\cite{caselli2021hatebert, barbieri2020tweeteval, zhou2021challenges} were themselves trained on
corpora of the first kind, so none of them holds the knowledge either.

**Gap.** Implicit abuse is studied as a benchmark problem for large models and as an annotation
problem, but not as a transfer problem: nobody asks whether the ability to read implication can be
moved into a model small enough to deploy, or by what mechanism. That is what this paper does, and
it is why one member of our committee has to be trained rather than downloaded. [Our contributions
1 and 2.]

## 2.2 Knowledge distillation for pre-trained language models

Distillation transfers a large model's softened outputs to a smaller one \cite{hinton2015distilling,
bucilua2006model}; intermediate-feature transfer follows FitNets \cite{romero2015fitnets}; surveys
organise response-, feature- and relation-based variants \cite{gou2021knowledge}. For BERT-style
encoders: DistilBERT \cite{sanh2019distilbert}, Patient KD \cite{sun2019patient}, TinyBERT
\cite{jiao2020tinybert}, MiniLM \cite{wang2020minilm}, MobileBERT \cite{sun2020mobilebert}.
Two findings shape our design: pre-training the compact student matters more than the
distillation recipe \cite{turc2019wellread}, and non-transformer students such as BiLSTMs can be
distilled from BERT \cite{tang2019distilling}. Distillation does not always work as expected
\cite{stanton2021does, cho2019efficacy}, and patient, consistent teaching matters
\cite{beyer2022knowledge}.

**Gap.** These methods assume one teacher and a task with a single label source; none uses
task-specialist teachers or annotator disagreement.

## 2.3 Multi-teacher and adaptive distillation

Ensembles of teachers \cite{you2017learning, fukuda2017efficient} are combined with fixed or
learned weights: adaptive multi-level weighting \cite{liu2020adaptive}, gradient-space
adaptation \cite{du2020agree}, reinforced teacher selection \cite{yuan2021reinforced},
confidence-aware weighting \cite{zhang2022confidence}. For language models, MT-BERT co-fine-tunes
several teachers and weights their soft labels by prediction error while aligning hidden states
through learned projections \cite{wu2021one}; dynamic weighting by teacher confidence has also
been used for semantic parsing \cite{zou2025dynamic}. Heterogeneous distillation, where the
architectures differ, needs an explicit information-flow model \cite{passalis2020heterogeneous}.

**Gap and positioning.** This is the section where the paper is most exposed, so the overlap is
stated before the difference. Our per-instance weights follow the error-based family
\cite{wu2021one, zhang2022confidence}, and the combination of error-weighted soft labels with
projected hidden-state alignment is MT-BERT's \cite{wu2021one}. We do not claim that combination as
novel and we do not present it as our contribution.

What the existing work assumes, and what we do not, is that the teachers collectively know the task.
Every weighting scheme above decides *how much* to trust each teacher on an instance; none of them
can supply an expertise that no teacher has. On the problem this paper addresses that assumption
fails: the corpora from which teachers are fine-tuned record topic or a binary attack flag, so no
member of a committee assembled in the usual way has ever seen abuse-by-implication marked as such,
and there is nothing for a weighting scheme to route to. Our contribution is therefore about what is
in the committee rather than about how it is combined: a cross-task committee whose members are
chosen for complementary expertise, including one member trained specifically on the distinction the
task does not label, plus an auxiliary head distilled from a teacher whose label space differs from
the task and which therefore cannot be weighted at all.

Two further differences are methodological rather than architectural. Reliability is measured against
annotator fractions rather than majority labels where a corpus provides them, which connects this
family to the disagreement literature in Section 2.4. And the weighting is examined rather than
assumed: because the teachers are frozen and cached, the weights are a fixed function of the data,
and we report where they actually go, which of the cited papers does not.

[Contributions 1, 2, 4.]

## 2.4 Learning from annotator disagreement

Crowd labels carry information beyond the majority vote \cite{peterson2019human}; surveys and
position papers argue for modelling disagreement rather than discarding it \cite{uma2021learning,
davani2022dealing, plank2022problem}. The Wikipedia corpus provides about ten votes per comment
\cite{wulczyn2017exmachina}.

**Gap.** No multi-teacher distillation method uses annotator agreement to decide how much to
trust the human label versus the teacher committee. [Contribution 3.]

## Supporting citations for the setup and statistics

Models and tooling \cite{devlin2019bert, liu2019roberta, he2023debertav3, loshchilov2019decoupled,
wolf2020transformers, paszke2019pytorch}; efficiency motivation \cite{schwartz2020green,
brown2020language, patterson2021carbon, zhou2019edge}; agreement and significance testing
\cite{fleiss1971measuring, koehn2004statistical, dror2018hitchhiker}; calibration
\cite{guo2017calibration}.
