# Related work (draft skeleton with citation keys from `references.bib`)

Target length 1,500 words. Four themes, each ending with the gap this paper fills. Every claim
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

**Gap and positioning.** Our per-instance weights follow the error-based family
\cite{wu2021one, zhang2022confidence}; what is new is (i) a committee of *cross-task* teachers
(general, explicit abuse, sarcasm) with an auxiliary head distilled from a teacher whose label
space differs from the task, (ii) reliability measured against annotator fractions rather than
majority labels, and (iii) a controlled comparison of homogeneous and heterogeneous teacher
committees and student families, including a non-transformer student. [Contributions 1, 2, 4.]

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
