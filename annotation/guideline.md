# Annotation guideline: sarcastic tweets, bullying or not

Purpose. The paper claims that a sarcasm-aware student separates sarcasm from abuse. To test
that claim we need a small test set where people, not heuristics, decided which sarcastic tweets
are abusive. Four annotators label the same 300 tweets independently; agreement is reported as
Fleiss' kappa and the majority label becomes the gold label.

Time. About 300 tweets at 20 seconds each is under two hours per annotator. Do it in one or two
sittings, alone, without discussing items with the others until everyone has finished.

## Labels

Give every tweet exactly one label.

| Label | Meaning |
|---|---|
| `benign_sarcasm` | The tweet is sarcastic or ironic, and it does not attack, demean, threaten or harass a person or a group. Self-mockery, jokes about situations, complaints about weather, exams, companies, politics in general. |
| `sarcastic_abuse` | The tweet is sarcastic or ironic, and the sarcasm carries an attack on a person or a group: insult, contempt, demeaning stereotype, threat, or harassment. The attack may be indirect; the test is whether the target would reasonably feel attacked. |
| `not_sarcastic` | The tweet is not sarcastic at all (the source dataset's label was wrong). Decide abuse separately in the notes column if you like, but the label stays `not_sarcastic`. |
| `unsure` | You genuinely cannot decide after reading twice. Use sparingly; a set full of `unsure` tells us nothing. |

## Decision rules

1. Judge the tweet on its own text. Do not search for context, do not guess who the author is.
2. Profanity alone is not abuse. "This exam was f***ing wonderful" is `benign_sarcasm`.
3. Naming a public figure is not automatically abuse. Mocking a politician's decision is benign;
   calling the politician subhuman, or attacking their group, is abuse.
4. Abuse toward a group counts the same as abuse toward a person.
5. If the tweet quotes someone else's abuse to mock it, and the mockery is not itself abusive, it is `benign_sarcasm`.
6. `@user` placeholders and hashtags are part of the text; a hashtag can carry the attack.
7. When the sarcasm is clear but the target is ambiguous, ask: would a reasonable member of the
   plausible target group feel demeaned? Yes: `sarcastic_abuse`. No: `benign_sarcasm`.

## Examples (invented, for calibration)

- "Oh great, another Monday. Truly the highlight of my existence." -> `benign_sarcasm`
- "Wow, you managed to tie your shoes today, someone give this genius a medal." (addressed to a specific person) -> `sarcastic_abuse`
- "Love how the airline lost my bag again, world-class service." -> `benign_sarcasm`
- "Sure, women are great at driving, that's why my insurance is cheap." -> `sarcastic_abuse` (group stereotype)
- "My phone died. Again." -> `not_sarcastic`

## Procedure

1. Open `annotation/sarcbully_sheet.csv` in Google Sheets or Excel. Do not reorder rows.
2. Fill only your own column (`A1`, `A2`, `A3` or `A4`) with one of the four labels, spelled exactly.
3. Use `notes` for anything you want adjudicated later. Keep it short.
4. Save as CSV with the same file name and column order.
5. When all four are done, run:

    python annotation/compute_kappa.py --sheet annotation/sarcbully_sheet.csv --out probes

   It prints Fleiss' kappa and pairwise Cohen's kappa, and writes `probes/benign_sarcasm_human.csv`
   and `probes/sarcastic_abuse_human.csv` from the majority labels. Items with no majority go to
   `annotation/adjudicate.csv`; the four of you settle those together in one short meeting.

The sheet contains no model scores on purpose; the model's opinion must not steer yours.
