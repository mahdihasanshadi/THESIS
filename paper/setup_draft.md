# Experimental setup (draft for the paper, 10 Sep 2026)

Numbers below are produced by the scripts in `src/dmthd/` and are reproducible from the
committed code; every count comes from a `report.json` written by the pipeline.

## Datasets

| Corpus | Task | Train / Val / Test | Labels | Notes |
|---|---|---|---|---|
| Fine-grained cyberbullying tweets (Wang, Fu and Lu, 2020) | six-class | 34,607 / 4,326 / 4,326 | age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying | primary benchmark |
| Wikipedia Talk personal attacks (Wulczyn, Thain and Dixon, 2017) | binary | 68,750 / 22,782 / 22,721 | attack / not attack, plus the annotator fraction as a soft label | official split; about ten annotators per comment |

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

## Targeted test sets for the sarcasm claim (inference only)

| Set | Size | Source | Metric |
|---|---|---|---|
| Benign sarcasm | 1,064 | sarcastic tweets from iSarcasmEval (SemEval-2022 Task 6), train and test, author-labelled | false-positive rate: share predicted as any bullying class |
| Ironic abuse | 1,560 | 797 posts labelled `irony` in the Implicit Hate Corpus stage-2 data (ElSherief et al., 2021) plus 763 original ISHate rows labelled Implicit HS (Ocampo et al., 2023) | recall: share predicted as a bullying class |
| Implicit abuse | 763 | the ISHate subset alone | recall |

ISHate rows whose source is ToxiGen or the Implicit Hate Corpus are excluded so that no
teacher's pre-training data reaches a test set; augmented ISHate rows are never used. The benign
set receives one manual screening pass restricted to rows a classical classifier flags as
abusive with probability at least 0.8 (see `screen_probes.py`).

## Teacher provenance rule

No test set may overlap the training data of any teacher, including the data a specialist
checkpoint was trained on before task adaptation: ToxiGen and Founta et al. (2018) for the
ToxiGen RoBERTa, SemEval-2018 Task 3 for the Cardiff irony model, RAL-E Reddit posts for HateBERT.

## Classical floor

TF-IDF over word 1-2 grams and character 3-5 grams with class-balanced logistic regression (C = 4).

| Corpus | Macro-F1 | Accuracy | Other |
|---|---|---|---|
| Tweets | 0.880 | 0.893 | per-class F1: age 0.98, ethnicity 0.98, religion 0.95, gender 0.91, other 0.75, not_cyberbullying 0.70 |
| Wikipedia | 0.876 | 0.947 | ROC-AUC 0.968, PR-AUC 0.868, attack-class F1 0.78 |

Every neural result is reported relative to this floor.

## Protocol (fixed before any GPU run)

Three seeds per configuration; mean and standard deviation; paired bootstrap of 1,000 test-set
resamples for the 95% interval of every D-MTHD-versus-baseline difference. Hyper-parameters:
teachers 5 epochs (3 on Wikipedia), learning rate 2e-5, batch 32 (16 on Wikipedia), 10% warm-up,
linear decay, best validation macro-F1 epoch kept; students 6 epochs, learning rate 3e-5,
early stopping with patience 2 on validation macro-F1. Maximum length 128 tokens for tweets,
256 for Wikipedia. Latency is the median of five timed passes after twenty warm-up batches, at
batch sizes 1 and 32, on GPU and CPU.
