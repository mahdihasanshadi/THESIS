# Statements for the paper (drafts)

## Ethics statement

This work builds classifiers for abusive language. The training and evaluation corpora contain
offensive, hateful and harassing text; all are existing public research datasets used under their
licences (Wikipedia personal attacks: CC0 via Figshare; the tweet corpus: CC0 as distributed on
Kaggle; ISHate: BSL-1.0; Implicit Hate Corpus: research use per its authors; iSarcasmEval: research
use per SemEval-2022). No new user data was collected, and no attempt was made to identify authors.
Word clouds and example lists in the thesis appendix reproduce slurs only where needed to explain
model behaviour; the paper quotes examples sparingly and masks slurs.

Automated moderation models can harm the people they are meant to protect: they misclassify
dialects and reclaimed language, they flag sarcasm and self-deprecation as abuse, and they can be
used to silence rather than protect. Our targeted evaluation on benign sarcasm exists to measure
one of these failure modes directly. The models are released for research; deploying them for
moderation requires human review, calibration to the platform, and monitoring for disparate
impact across groups, which we do not evaluate here beyond the identity-independent metrics reported.

Annotation for the human-verified sarcastic-bullying set was done by the authors; the guideline
warns annotators about the content, allows breaks, and imposes no time limit.

## Data statement

| Corpus | Source | Licence | Size used | Purpose |
|---|---|---|---|---|
| Wikipedia Talk personal attacks | Figshare (Wulczyn et al. 2017) | CC0 | 114,253 comments after cleaning | training and evaluation, binary, soft labels |
| Fine-grained cyberbullying tweets | Kaggle (Wang et al. 2020) | CC0 | 43,259 tweets after de-duplication | training and evaluation, six-class |
| iSarcasmEval | GitHub (Abu Farha et al. 2022) | research | 1,064 sarcastic tweets | inference-only probe |
| Implicit Hate Corpus, stage 2 | Hugging Face mirror of ElSherief et al. 2021 | research | 797 irony posts | inference-only probe |
| ISHate | Hugging Face (Ocampo et al. 2023) | BSL-1.0 | 763 original Implicit HS rows, ToxiGen and IHC sources excluded | inference-only probe |
| ISHate | Hugging Face (Ocampo et al. 2023) | BSL-1.0 | 28,763 original rows, augmentations and ToxiGen-sourced rows excluded | training and evaluation, implicit benchmark |
| Implicit Hate Corpus, stage 1 | Hugging Face (`tasksource/implicit-hate-stg1`, ElSherief et al. 2021) | research | 21,480 posts | training and evaluation, implicit benchmark |

Language: English. Text of every corpus is user-generated social-media content; demographic
information about authors is not available and was not inferred.

The implicit benchmark is derived from the last two rows and is not redistributed: neither source
is ours to re-host. It is released as a build script plus a manifest (see the reproducibility
statement). Every text used by a probe set is deleted from it, in all three splits, so no model in
this paper is evaluated on a probe it has trained on.

## Reproducibility statement

Code, configuration, seeds and one-command drivers are public at
github.com/mahdihasanshadi/THESIS. Every reported number is produced by a script that writes a
`results.json`; the lab notebook (`LOG.md`) records each run. Data preparation is deterministic
(seed 42) and reports every removal count.

None of the three corpora may be re-hosted. In place of the data we publish, for each corpus, a
manifest: one row per example giving its split, its label, its source corpus and a SHA-1 of the
normalised text, with the split fingerprints beside it. It contains no text and therefore
redistributes nothing, and `python -m dmthd.manifest --verify` compares a reader's rebuild against
ours row by row and names the rows that differ. Because teacher logit caches are indexed by row
position, publishing the split fingerprints also lets a reader confirm that a released cache belongs
to the split they reconstructed. Student and teacher checkpoints and cached teacher
outputs are released as a Kaggle dataset [link on acceptance]. Hardware: Kaggle T4 GPUs for
teachers and students, a 16-core CPU for the BiLSTM and BERT-mini controls and for CPU latency.
Approximate compute: [fill from `train_time_s` in results.json] GPU-hours.

## CRediT author contributions (to fill)

- Conceptualisation: [ ]
- Methodology: [ ]
- Software: [ ]
- Data curation and annotation: [ ]
- Formal analysis: [ ]
- Writing, original draft: [ ]
- Writing, review and editing: [ ]
- Supervision: Dr. Muhammad Iqbal Hossain, Sheikh Araf Noshin

## Limitations (draft bullets)

Ordered by how much they should change a reader's confidence, not by how comfortable they are.

**About the label itself.** The distinction this paper is built on is not one that expert annotators
agree about. The two public corpora that annotate implication share 629 texts; they agree on 99.7
per cent of them about whether the text is hateful and on 48.2 per cent about whether the hate is
stated or implied. Any result here, including ours, is bounded by that. It is the reason we report a
threshold-free ranking metric rather than a single decision, and the reason a human annotation study
is part of the paper rather than an appendix to it.

**About the implicit benchmark.** It comes from a single corpus, which removes a source confound but
means the in-domain result rests on one collection effort and one platform. The explicit class in it
has 864 training and 108 test examples, so per-class results for that class are noisy and are not
used to support any claim. Out-of-domain evidence comes from a second corpus that contains almost no
implicit examples, so out-of-domain *implicit recall* is measured on the probe set rather than on a
full held-out corpus.

**About the probes.** They are built from published corpora by rule rather than by hand, about 1,000
items each, and the human-verified subset is 300 items. The benign-sarcasm set is author-labelled for
sarcasm, not checked by us for abusiveness beyond one screening pass over the rows a classical model
flags. Probe metrics are only meaningful for models trained on Twitter-domain data; on a
Wikipedia-trained model they measure domain shift and we do not report them.

**About the experimental grid.** Three seeds per configuration and one seed for ablations, so
ablation differences are read from bootstrap intervals and not from seed variance. Each teacher is
trained once, so teacher-side variance is not estimated. The heterogeneous comparison confounds
architecture family with model quality, since DeBERTa-v3-xsmall is simply weaker on these tasks than
the BERT-lineage students it is compared against; the controlled version of that question is the
hidden-state term rather than the student swap.

**About robustness.** Obfuscation uses synthetic character edits, not evasion attempts by people who
want to evade. One variant improves the Wikipedia score rather than hurting it, which we report as it
stands and which should be read as a caution about the method rather than a result about robustness.

**About scope.** English only. Bengali, which motivated this line of work, has no sarcasm teacher
available and is therefore left to the thesis rather than this paper.
