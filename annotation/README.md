# Annotation: sarcastic tweets, bullying or not

## Why this exists

The paper claims that a sarcasm-aware student separates sarcasm from abuse. Right now that claim is
measured on a probe set built by a heuristic. A small set where four people, not a heuristic, decided
which sarcastic tweets are abusive is far more convincing, and the agreement statistic (Fleiss'
kappa) tells a reviewer how well-defined the task is. 300 items is enough for that.

## For each annotator

1. Take your own workbook from `sheets/`: `annotator_A1.xlsx` … `annotator_A4.xlsx`. Agree who is
   which; it does not matter, but two people must not use the same file.
2. Read the `guideline` worksheet inside the workbook once (it is the full instruction, two minutes).
3. On the `annotation` worksheet fill only the **label** column. Click a cell and a drop-down offers
   the four allowed values; typing something else is rejected, which keeps the statistics valid.
4. `notes` is optional: use it when an item needs discussion later.
5. Do not reorder, sort, insert or delete rows. Rows are matched back by `id`.
6. **Work alone.** Do not discuss items until everyone has finished. Disagreement is the thing being
   measured, not a mistake.
7. About 300 tweets at 20 seconds each is under two hours. There is no time limit; take breaks.
   Some tweets contain slurs and harassment.

Prefer Google Sheets? Use `annotator_<you>.csv` instead, upload it, fill the `label` column with the
same four values spelled exactly, and download it back as CSV with the same name.

## The four labels

| Label | Meaning |
|---|---|
| `benign_sarcasm` | Sarcastic or ironic, attacks nobody: self-mockery, jokes about situations, complaints about weather, exams, companies, politics in general. |
| `sarcastic_abuse` | The sarcasm carries an attack on a person or a group: insult, contempt, demeaning stereotype, threat, harassment. The attack may be indirect; ask whether the target would reasonably feel attacked. |
| `not_sarcastic` | Not sarcastic at all; the source label was wrong. |
| `unsure` | You genuinely cannot decide after reading twice. Use sparingly. |

Calibration examples (invented, not from the data):

- "Oh great, another Monday. Truly the highlight of my existence." → `benign_sarcasm`
- "Wow, you managed to tie your shoes today, someone give this genius a medal." (to a person) → `sarcastic_abuse`
- "Love how the airline lost my bag again, world-class service." → `benign_sarcasm`
- "Sure, women are great at driving, that's why my insurance is cheap." → `sarcastic_abuse` (group stereotype)
- "My phone died. Again." → `not_sarcastic`

Rules that decide the hard cases:

1. Judge the tweet on its own text. Do not search for context or guess who wrote it.
2. Profanity alone is not abuse.
3. Naming a public figure is not automatically abuse. Mocking a decision is benign; calling the
   person subhuman, or attacking their group, is abuse.
4. Abuse toward a group counts the same as abuse toward a person.
5. Quoting someone else's abuse to mock it is benign, unless the mockery itself attacks.
6. `@user` placeholders and hashtags are part of the text; a hashtag can carry the attack.

## When everyone is done

Put the four returned files in one folder, then:

```bash
python annotation/merge_sheets.py --sheets annotation/sheets --sheet annotation/sarcbully_sheet.csv
python annotation/compute_kappa.py --sheet annotation/sarcbully_sheet_filled.csv --out probes
```

The first command reports how many items each person labelled and rejects invalid labels; the second
prints Fleiss' kappa and pairwise Cohen's kappa, and writes:

- `probes/benign_sarcasm_human.csv` — items whose majority label is benign sarcasm (the false-positive test set)
- `probes/sarcastic_abuse_human.csv` — items whose majority label is sarcastic abuse (the recall test set)
- `annotation/adjudicate.csv` — items with no majority, to settle together in one short meeting

Majority means more than half of the votes actually cast on that item; an even split goes to
adjudication. `unsure` counts as no vote.

## What the paper reports

The kappa value, the number of items, how many needed adjudication, and the two resulting test-set
sizes. Both numbers go in the setup section; the sets are used for inference only and never for
training.

## Files

| File | What it is |
|---|---|
| `sarcbully_sheet.csv` | the 300 items: 181 that a classical classifier flagged as abusive, plus 119 random ones, shuffled |
| `sarcbully_key.csv` | source and model score per item, kept out of the annotators' sheets so the model cannot steer them |
| `sheets/annotator_*.xlsx` | one workbook per person, with the drop-down and the guideline |
| `make_sheets.py` | rebuilds the per-person workbooks from the sheet |
| `merge_sheets.py` | merges the returned files, checking ids and labels |
| `compute_kappa.py` | agreement statistics and the gold test sets |
| `guideline.md` | the same guideline as prose |
