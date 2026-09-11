# Decisions that are not mine to take

Every technical choice made so far is recorded in `DECISIONS.md` with the evidence behind it. The
ones below are different: they change what the paper *is*, where it is sent, or how the team's
remaining GPU quota is spent. Each one states the options, the argument for and against each, the
evidence that will settle it, and when it has to be settled by. Nothing here is urgent today except
Decision 2.

---

## Decision 1. What the paper is about

The current title is *A Dynamic Multi-Teacher Homogeneous Knowledge Distillation Framework for
Robust Cyberbullying Detection*. The evidence collected so far does not support that title evenly:
the strongest measured results are about implication, and the weakest are about the dynamic
weighting.

**Option A — keep the current framing.** D-MTHD is the contribution; three benchmarks; implicit
abuse is one section among several.
- For: no rewriting; matches what the team has already written.
- Against: the method is close to MT-BERT (Wu et al. 2021), and a reviewer who knows that paper will
  ask what is new. On the tweet corpus the dynamic weighting is currently indistinguishable from
  uniform averaging (Finding 6), and the fine-tune-only student does not beat a TF-IDF baseline
  (Finding 3). Under this framing the paper's two headline mechanisms are the two weakest results.

**Option B — reframe around implication.** Something like *Teaching a small model to read
implication: cross-task specialist distillation for implicit abuse detection*. The implicit
benchmark becomes the primary one; the tweet and Wikipedia corpora become generalisation evidence;
sarcasm-discrimination AUC is the headline metric.
- For: there is a real, measured gap to close. A bag of n-grams scores 0.747 F1 on explicit hate and
  0.453 on implicit hate in the same corpus. The failure has a diagnosis, not just a number: the
  model sees implication but cannot separate it from harmless sarcasm (Finding 10). There is a
  mechanism aimed at exactly that (the specialist teacher, the irony head), a way to show the
  mechanism works (weight routing), and controls that could falsify it.
- Against: it depends on results we do not have yet. If the specialist does not help, this framing
  has nothing to stand on. It also means rewriting the introduction and abstract.

**Option C — two papers.** Implication in one, efficiency and deployment in the other.
- Against: not in two weeks, and each half is thin alone.

**My recommendation: B, decided after the implicit results arrive, not before.** The honest order is
evidence first, framing second. Note that Option A's weakness is not fixed by choosing A: those
results will be in the paper either way, and a reviewer will find them.

**What settles it:** four numbers from the implicit benchmark and the `spec` committee run.
1. implicit_hate F1 against the 0.453 classical floor.
2. Sarcasm-discrimination AUC against the 0.776 fine-tune-only baseline.
3. `spec` committee against `homo` committee, three seeds, on the tweet corpus.
4. Whether `spec` beats the two controls: the specialist alone, and the same student pre-trained on
   the implicit corpus without any distillation.

**By when:** as soon as the implicit Kaggle run finishes. Until then the writing that is safe to do
is the related work, the setup section and the limitations, none of which depend on the framing.

---

## Decision 2. How the remaining GPU quota is spent — this one is needed now

Kaggle gives 30 GPU-hours per account per week. Four accounts is 120. My estimate of what is left:

| Work | Estimate | Status |
|---|---|---|
| Finish the tweet grid | 11 h, one more session | version 3 running now |
| Implicit specialist + the implicit-pretrained control | 1.5 h, once | not started |
| `spec` committee on the tweet corpus, headline student, three seeds | 2 h | not started |
| Implicit benchmark, full grid | 15-20 h, two sessions | not started |
| Wikipedia, full grid | 20 h or more, 256-token inputs | not started, needs a second account |

**Option A — everything.** Roughly 55 GPU-hours. Fits in one week across three accounts if they run
in parallel starting now.

**Option B — tweets and implicit in full; Wikipedia cut to the headline student only.** Roughly
40 hours. Wikipedia's unique contribution is the annotator-agreement data and the disagreement-aware
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

---

## Decision 4. What the paper says if the dynamic weighting turns out to do nothing

This is the largest single risk and it is worth deciding the response before seeing the number, so
the decision is not made under pressure.

Today the per-instance weights sit within 0.03 of uniform, because tau is too large for the spread of
teacher errors (Finding 6). The sweep over tau in {0.05, 0.1, 0.2, 0.5} will resolve it.

**If sharper tau makes D-MTHD beat uniform averaging:** the paper keeps its current central claim and
gains an honest negative-to-positive story about a hyper-parameter that matters more than the
literature suggests. That is publishable in itself.

**If it does not:** two options.
- **4A. Report it and move the contribution.** The paper becomes about committee composition (a
  cross-task specialist committee including an implicit-abuse specialist) and the auxiliary irony
  head, with per-instance weighting reported as a component that did not pay for itself. Smaller
  claim, still a paper, and the negative result is genuinely useful to the field.
- **4B. Drop the weighting from the title and keep it as an ablation.** Same content, less exposure.

**My recommendation: 4A.** A reviewer who sees uniform and dynamic scoring within noise of each other
and no acknowledgement of it will distrust the rest of the paper. Saying it first is worth more than
hiding it.

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
