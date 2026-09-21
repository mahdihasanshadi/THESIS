# Decisions that are not mine to take

Every technical choice made so far is recorded in `DECISIONS.md` with the evidence behind it. The
ones below are different: they change what the paper *is*, where it is sent, or how the team's
remaining GPU quota is spent. Each one states the options, the argument for and against each, the
evidence that will settle it, and when it has to be settled by.

**Updated 21 September, night, after the size curve.** D20's decision rule, written before the run,
fires: the out-of-sample gain grows with the transfer set, monotone over six sizes, +0.0123
[+0.0035, +0.0211] at 168,000 rows, from generic tweets nearly as well as from abuse-domain ones
(DECISIONS F33). The recommendation for Decision 1 is now Version C in `abstract_draft.md`, the
constructive framing with the audit as its evidence, and Decision 3 gains a title question that is
now urgent. Two controls the constructive framing still lacks are listed in Decision 2 as v7.

**Updated 17 September, after the specialist runs.** Three of the four numbers Decision 1 was waiting
for are in, and they change the recommendation: the specialist does not help on the tweet corpus, the
reason is measured, and sharper weighting does not rescue the method either, which also closes
Decision 4. The recommendation is now a measurement paper about implication (B2 below), with the
implicit benchmark's student grid as the last experiment that could make it constructive again.
Decision 2's table is updated to what has run.

**Updated 13 September, after the tweet grid finished.** Decision 4 has triggered: the per-instance
dynamic weighting does not beat uniform averaging on any of five students. That removes the main
argument for Option A in Decision 1 and makes Decision 4A the live question rather than a
contingency. Decision 2, the quota, is still the only one needing an answer today.

---

## Decision 1. What the paper is about

The current title is *A Dynamic Multi-Teacher Homogeneous Knowledge Distillation Framework for
Robust Cyberbullying Detection*. The evidence collected so far does not support that title evenly:
the strongest measured results are about implication, and the weakest are about the dynamic
weighting.

**Option A — keep the current framing.** D-MTHD is the contribution; three benchmarks; implicit
abuse is one section among several.
- For: no rewriting; matches what the team has already written.
- Against, and this is now measured rather than feared: the method is close to MT-BERT (Wu et al.
  2021), and the part that would have distinguished it does not work. The dynamic weighting loses to
  uniform averaging on four of five students and ties on the fifth (Finding 6), and on the headline
  student no distillation variant beats training with no teacher at all (Finding 3). Under this
  framing the paper's two headline mechanisms are its two weakest results, and a reviewer will see
  that immediately.

**Option B — reframe around implication.** Something like *Teaching a small model to read
implication: cross-task specialist distillation for implicit abuse detection*. The implicit
benchmark becomes the primary one; the tweet and Wikipedia corpora become generalisation evidence;
the discrimination AUCs are the headline metrics.
- For: there is a real, measured gap to close, and the same gap shows up twice on different data. On
  the corpus that labels implication, a bag of n-grams ranks implied hate above ordinary text at an
  AUC of 0.761; on the tweet corpus, our student separates ironic abuse from benign sarcasm at 0.776
  (the laptop model; 0.773 on Kaggle, see below).
  Well above chance, a long way from solved. The failure has a diagnosis rather than just a number:
  the model sees implication but cannot separate it from harmless sarcasm (Finding 10). There is a
  mechanism aimed at exactly that (the specialist teacher, the irony head), a way to show the
  mechanism works (weight routing), and controls that could falsify it.
- Against: it depends on results we do not have yet. If the specialist does not help, this framing
  has nothing to stand on. It also means rewriting the introduction and abstract.

**Option C — two papers.** Implication in one, efficiency and deployment in the other.
- Against: not in two weeks, and each half is thin alone.

**My recommendation: B, and the grid has strengthened it considerably.** Option A's case rested on
the method working, and on the tweet corpus it does not. Option B's case rests on the specialist
working, which is untested, so the decision still waits on the implicit results rather than being
made today. What has changed is that A is no longer the safe choice: those results are in the paper
either way, and under A they sit where the contribution should be.

**What settles it:** four numbers from the implicit benchmark and the `spec` committee run.
1. Implicit-discrimination AUC on the implicit benchmark against the classical floor's 0.761, and
   implicit-hate F1 against its 0.556.
2. Sarcasm-discrimination AUC on the tweet corpus against the fine-tune-only baseline (0.776 on the
   laptop, 0.773 on Kaggle).
3. `spec` committee against `homo` committee, three seeds, on the tweet corpus.
4. Whether `spec` beats the two controls: the specialist alone, and the same student pre-trained on
   the implicit corpus without any distillation.

**To make the choice concrete**, the abstracts are written out in `paper/abstract_draft.md`,
with every unmeasured number marked as a placeholder and a note under each saying what that
version is betting on. Since 17 September there are three, including B2 below. Read them before
deciding; the difference between them is easier to judge as prose than as an argument.

**By when:** as soon as the implicit Kaggle run finishes. Until then the writing that is safe to do
is the related work, the setup section and the limitations, none of which depend on the framing.

### What the numbers said, 17 September

Kaggle version 4 of the tweet notebook ran the specialist, its committee on three seeds, both controls
and the sharp temperatures. Against the four numbers above:
1. **Implicit benchmark:** the student grid has not run. The specialist itself clears the floor
   (implicit-discrimination AUC 0.8247 against 0.761; implicit-hate F1 0.607 against 0.556).
2. **Sarcasm-discrimination AUC on tweets:** the fine-tune-only baseline on Kaggle is 0.773; the 0.776
   quoted above was the laptop model (DECISIONS F26). Both specialist committees score 0.768, and all 25
   pre-trained variants of the headline student fall between 0.756 and 0.777 (F23).
3. **`spec` against `homo`, three seeds:** +0.0005 [-0.0059, +0.0064] under uniform averaging, -0.0005
   [-0.0069, +0.0055] under D-MTHD (F23).
4. **Against the controls:** the specialist alone 0.8392, the implicit-pretrained student 0.8377, both
   inside the same noise as everything else (F23).

Option B as written bet on number 3 or 4 moving. Neither did, and the reason is measured rather than
guessed: adapted to the tweet task, the specialist agrees with HateBERT at kappa 0.962, so the committee
never held the knowledge the framing needed it to pass on (F23). The weights lean towards it by a
twentieth of the uniform weight, and as much towards HateBERT (F24). Tweet-trained students rank the
implicit corpus's implicit hate against not-hate at 0.600, against 0.820 in-domain (F25). And across
the grid, of 51 paired comparisons, the only interval that excludes zero is removing pre-training (F27).

**Option A is now closed.** Its contribution was the dynamic weighting, and the last way out has been
tried: no temperature from 0.05 to 5 separates it from averaging (F22).

**Option B2 — what distillation does and does not transfer about implication.** Same subject as B,
different claim. Something like *Distillation Does Not Teach Implication: A Threshold-Free Audit of
Multi-Teacher Knowledge Distillation for Abusive Language Detection*. The contribution is evidence and
protocol, not a method:
- a threshold-free way to measure implication, and the demonstration over 133 models that recall at a
  fixed threshold measures readiness to fire rather than understanding (F19);
- a controlled grid in which no objective, committee, temperature, specialist teacher or control moves
  discrimination, and only pre-training does (F22, F23, F27);
- the mechanism, measured: task adaptation removes the diversity a cross-task committee is built for,
  down to a specialist that becomes a copy of its base model, so reliability weighting has nothing to
  select (F20, F23, F24);
- the data results that stand whatever else happens: the single-source implicit benchmark and the
  source shortcut that pooling creates (F16), the corpora's 48 per cent agreement about implication
  (F14), the schema traps (F15);
- efficiency parity as a secondary, practical result (F4, F11).
- For: every claim in it is already measured, and it survives either outcome of the implicit grid. It
  also removes the overlap risk with MT-BERT, because the paper no longer proposes that method. A
  negative result with a measured mechanism and a reusable protocol is publishable.
- Against: journals prefer methods. A negative result is an easier sell at *Information Processing and
  Management* or a Findings venue than at *Expert Systems with Applications* or *Knowledge-Based
  Systems* (Decision 6). The implicit grid still has to run, because under this framing the implicit
  benchmark is the primary one.

**My recommendation now: B2, with the implicit grid as the one test that could upgrade it.** On that
benchmark every teacher is itself trained on labelled implication, so it is the fairest chance
distillation has. B2 makes a prediction there that could fail: implicit-discrimination AUC will not
move beyond test-set noise, while F1, which depends on where the threshold falls, may. If AUC does
move, the paper becomes the constructive form of B, with the tweet corpus as the case where transfer
fails and the reason why. If it does not, B2 stands as written and loses nothing. There is little room
at the top either way: the best teacher ranks at 0.8247 and the fine-tune-only student at 0.8196.

**By when:** the framing can be adopted now, because both outcomes of the implicit grid fit it. The
introduction and abstract should be rewritten around implication; the results section already has
been. The title waits for the implicit grid.

**17 September, evening: Mahdi's decision.** He does not want the negative-result paper as the paper,
and asked for the strongest positive route inside knowledge distillation. The B2 manuscript is
preserved on the branch `paper-b2-audit`. The route chosen is out-of-sample distillation with an
unlabelled transfer set (DECISIONS D19), which attacks the mechanism the audit found (F28, F29) rather
than re-weighting the same knowledge. Its predictions are written down; one resumed Kaggle session of
the tweet notebook decides them. If prediction 1 holds, Decision 1 becomes a constructive paper whose
audit explains why the in-sample grid found nothing; if it fails, B2 stands.

**Outcome, 21 September (v5).** Prediction 1 was not met as written: the committee arm gains +0.0070
[-0.0001, +0.0144], below the 0.010 asked for, though two arms of the family clear zero and every
soft-label seed sits above every no-teacher seed (DECISIONS F31). The committee is not a better
labeller than one teacher, the corrected weighting adds +0.002, hard pseudo-labels +0.002, and the
sarcasm AUC does not rise (F32). So the multi-teacher method is closed out of sample too. What the run
found is a distillation recipe: the student gains only when it imitates a teacher on text the teacher
has not fitted. The recommendation keeps its shape and changes its tone: B2, with F31 as its
constructive last section ("where the gain is"), and one more run, the transfer-set scaling curve of
open question 7 (about three GPU-hours, with a matched-steps fine-tuning control), before deciding
whether that section can lead. If the gain grows with the set, the title can turn constructive, for
example *Distil Where the Teacher Is Uncertain: An Audit of Multi-Teacher Distillation for
Abusive-Language Detection*; if it does not, B2 stands as written with the recipe as its coda.

---

## Decision 2. How the remaining GPU quota is spent — this one is needed now

Kaggle gives 30 GPU-hours per account per week. Four accounts is 120. My estimate of what is left:

| Work | Estimate | Status |
|---|---|---|
| Finish the tweet grid | done, 8.24 h | complete: 120 runs, 13 September |
| Implicit specialist + the implicit-pretrained control | 1 h, once, inside the tweet run | done, 17 September (v4) |
| `spec` committee on the tweet corpus, headline student, three seeds | 2 h | done, 17 September (v4) |
| Sharp tau values, 0.05 / 0.1 / 0.2, one seed each | 20 min | done, 17 September (v4) |
| fp16 control for the narrow students | 40 min | done (v2) |
| v4 in total: specialist, `spec` committee, both controls, sharp tau, analyses | 1.4 h | complete, 131 runs |
| v5: five out-of-sample arms, per-committee routing, probes for the sharp tau runs | 2.11 h | done, 21 September |
| v6: transfer-set size curve on BERT-mini (5k to 168k), a composition control and a matched-steps control | 3.5 h | ran 21 September: monotone, +0.0123 [+0.0035, +0.0211] at 168k; D20's rule fires (F33) |
| v7 (proposed): fine-tune-only for 35 epochs (the 168k arm's number of updates) and the 168k arm on BERT-small, three seeds each | ~2.7 h | the two controls the constructive framing still lacks; recommended before the paper is written |
| Implicit benchmark, full grid | 8-10 h, one session | not started |
| Wikipedia, full grid | 20 h or more, 256-token inputs | not started, needs a second account |

The implicit estimate halved on 12 September because the benchmark is now built from one corpus
rather than two (see "Decisions I have already taken" below): 20,637 rows instead of 47,181.

**Option A — everything.** Roughly 45 GPU-hours. Fits in one week across three accounts if they run
in parallel starting now.

**Option B — tweets and implicit in full; Wikipedia cut to the headline student only.** Roughly
30 hours. Wikipedia's unique contribution is the annotator-agreement data and the disagreement-aware
variant, and both need only one student. The rest of the Wikipedia grid repeats what the tweet grid
already shows.

**Option C — drop Wikipedia entirely**, and say so in Limitations.
- Against: a second corpus is close to mandatory at Q1, and Wikipedia is the only one of the three
  with per-annotator agreement, which is what the disagreement-aware variant is built on.

**My recommendation: B.** It keeps every claim that needs a second corpus and spends nothing on
repeating a result.

**What I need from the team:** which Kaggle accounts are free, so the three runs start in parallel
rather than in series. The notebooks are ready: `kaggle/dmthd_implicit.ipynb`,
`kaggle/dmthd_tweets.ipynb`, `kaggle/dmthd_wikipedia.ipynb`.

---

## Decision 3. Whether "Homogeneous" and "Robust" stay in the title

Recorded as Decisions 7 and 9 in `DECISIONS.md`; repeated here because it is a title change and the
supervisors should see it coming.

- *Homogeneous* is the default setting in the distillation literature, not a novelty. It stays in the
  title only if the 2x2 (BERT-lineage student against DeBERTa-v3-xsmall, hidden-state term on and
  off) shows the hidden-state term helping same-family students more. If it does not, homogeneity is
  a design choice we describe once and do not claim.
- *Robust* stays only if the robustness numbers support it. They partly do: obfuscation costs the
  tweet student 10 to 15 points (Finding 8), and one Wikipedia variant actually *improves* the score,
  which we report as-is. Synthetic character edits are not real evasion, and the paper will say so.

**What settles it:** the homogeneity 2x2, which is already in the grid.

**Updated 21 September, night: the whole title, not two words.** The 2x2 ran (hidden-state term worth
0.0009 on the headline student, DECISIONS F23), so *Homogeneous* goes; *Robust* is unsupported (10 to
15 points lost to obfuscation); *Dynamic* and *Multi-Teacher* name the two things the grid measured as
adding nothing (F22, F29, F31). With D20's rule fired, the title should say what the paper shows.
Three candidates, in order of preference:

1. *Distil Where the Teacher Is Uncertain: Out-of-Sample Knowledge Distillation for Compact
   Cyberbullying Detection*. Names the mechanism and the result; the audit is the evidence inside.
2. *Out-of-Sample Knowledge Distillation for Cyberbullying Detection: Unlabelled Text, Not Teacher
   Committees, Improves Compact Students*. Plainer, states the contrast the audit measured.
3. *From Multi-Teacher to Out-of-Sample Knowledge Distillation for Cyberbullying Detection: An Audit
   and the One Lever That Moves the Student*. Keeps continuity with the registered title for a
   department that wants to see the lineage.

Any of the three is defensible today; none of them claims what v7 has not yet measured, because
"compact" and "cyberbullying detection" are true of the one student and corpus measured. If the
department requires the registered wording to survive in part, option 3.

---

## Decision 4. What the paper says if the dynamic weighting turns out to do nothing

This is the largest single risk and it is worth deciding the response before seeing the number, so
the decision is not made under pressure.

Today the per-instance weights sit within 0.03 of uniform, because tau is too large for the spread of
teacher errors (Finding 6). The sweep over tau in {0.05, 0.1, 0.2, 0.5} will resolve it.

**What happened.** The grid finished on 13 September and the answer is no, at every tau it tried.
D-MTHD minus uniform averaging, per student: -0.0007, -0.0034, +0.0000, -0.0029, -0.0029. Four
negative, one tie, none positive. Removing the weighting entirely scores *above* keeping it on the
headline student. On BERT-mini nothing beats training with no teacher at all, with all four paired
bootstrap intervals containing zero.

One qualification, and it is real rather than a consolation. Every tau the finished run tried was 0.5
or larger, where the mean weights sit at a third each; those configurations optimise nearly the same
objective, so scoring alike is arithmetic and not evidence. The sharp values, 0.05, 0.1 and 0.2, are
in the current grid and have not run. **The claim available today is "not shown to differ from
uniform averaging at any tau yet tested."**

**What happened when they ran, 17 September.** Nothing changed. Mean weights sharpen from 0.332 /
0.337 / 0.331 at tau = 1 to 0.318 / 0.356 / 0.326 at tau = 0.05, and test macro-F1 stays at 0.8385 to
0.8392 (DECISIONS F22). Paired intervals at tau = 0.05: +0.0007 [-0.0008, +0.0023] against the
default, +0.0020 [-0.0042, +0.0077] against uniform averaging. **This decision is resolved: 4A**, and
Decision 1 now carries the consequence.

**If the sharp tau values change it:** the paper keeps its central claim and gains an honest
negative-to-positive story about a hyper-parameter that matters more than the literature suggests.
That is publishable in itself, and the diagnostic that predicted the whole thing from the weights
alone becomes a contribution rather than a footnote.

**If they do not, which is the way to bet:** two options.
- **4A. Report it and move the contribution.** The paper becomes about committee composition (a
  cross-task specialist committee including an implicit-abuse specialist) and the auxiliary irony
  head, with per-instance weighting reported as a component that did not pay for itself. Smaller
  claim, still a paper, and the negative result is genuinely useful to the field.
- **4B. Drop the weighting from the title and keep it as an ablation.** Same content, less exposure.

**My recommendation: 4A, and I now think this is close to forced.** A reviewer who sees uniform and
dynamic scoring within noise of each other and no acknowledgement of it will distrust everything else
in the paper. Saying it first is worth more than hiding it, and there is a real paper underneath: the
committee helps, the specialist is untested but motivated, the efficiency result is intact, and a
carefully measured negative result about a mechanism the field keeps proposing is worth publishing on
its own. What is gone is the version of this paper whose contribution was the weighting.

---

## Decision 5. Whether to release the implicit benchmark

The corpus is derived from ISHate and the Implicit Hate Corpus, both of which have their own
licences and neither of which we can simply re-host.

**Option A — release the build script only** (`prepare_implicit.py`), which is already public in the
repository. Anyone with access to the two source corpora reproduces our exact splits from it.
**Option B — release the script plus a manifest**: for each row, its source corpus, its identifier
and a hash of the normalised text, so a reader can verify they rebuilt the same corpus without us
redistributing anyone's text.

**My recommendation: B**, and I will prepare the manifest unless told otherwise; it costs nothing and
it is the difference between "reproducible in principle" and "verifiably reproducible". Re-hosting
the texts is not on the table without checking both licences, which is a question for the
supervisors.

---

## Decision 6. Target venue

Depends on Decision 1.

- If the paper stays on Option A (efficiency and distillation): *Expert Systems with Applications* or
  *Information Processing and Management*, with *Knowledge-Based Systems* and *Neurocomputing* as
  fallbacks.
- If it moves to Option B (implication): the same journals remain open, and ACL/EMNLP Findings
  becomes a realistic alternative, because implicit abuse and the sarcasm-discrimination framing are
  closer to that community's interests. A Findings paper is faster to a decision, shorter, and
  better read by the people who work on this problem.

**Not needed until the framing is settled.** Worth asking the supervisors now which they would prefer
in principle, because a journal submission and a conference submission want different amounts of
writing and the team's two weeks should be spent accordingly.

---

## Decisions I have already taken, which you can overrule

These were technical rather than strategic, so I took them rather than waiting, and each is
reversible by one flag or one rebuild. They are listed here because two of them change numbers you
may already have seen.

**The implicit benchmark is built from the Implicit Hate Corpus alone**, with ISHate held out as a
27,096-row out-of-domain test set. The first version pooled both, and I checked before training
anything on it: a TF-IDF classifier tells the two corpora apart at 0.91 macro-F1, 95 per cent of
implicit examples come from one side and 89 per cent of explicit examples from the other, so a model
could have scored well by recognising the source instead of reading an implication. On identical
test rows, pooled training gives 0.473 F1 on implicit hate against 0.548 for single-corpus training.
The pooled version scored higher in aggregate, 0.681 against 0.562, which is exactly the trap.
*To overrule:* `python -m dmthd.prepare_implicit --corpora ImplicitHate,ISHate`. Everything else
follows automatically. *What changes if you do:* a bigger corpus and a bigger headline number that I
would not be willing to defend.

**The headline metric on that benchmark is implicit-discrimination AUC, not macro-F1.** The explicit
class has 108 test rows, so its F1 is noise and a three-class average is dominated by it and by the
large benign class. The AUC asks the question directly: ranked by p(implicit hate), does implied
abuse come above ordinary text? The classical floor scores 0.761. *To overrule:* report macro-F1
instead; both are in every results file.

**The implicit specialist is a separate committee (`spec`) rather than a fourth member of `homo`.**
This keeps every run already finished with the three-teacher committee valid, worth about ten
GPU-hours, and makes "does the specialist help?" a three-seed comparison of two full committees
instead of a one-seed ablation. *To overrule:* `COMMITTEES=homo,hetero` and add the specialist to
`TEACHERS`, at the cost of re-running the finished committee work.

**The specialist is off by default on the Wikipedia benchmark.** It would cost about three GPU-hours
there to answer a question the tweet benchmark answers more cheaply. *To overrule:* `SPECIALIST=1`
in that notebook.
