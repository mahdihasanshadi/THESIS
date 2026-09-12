# Method (draft for the paper)

## 3.1 Problem and notation

| Symbol | Meaning |
|---|---|
| $x_i$, $y_i$ | input text and its majority label, $y_i \in \{1,\dots,C\}$ |
| $s_i \in [0,1]$ | annotator fraction voting "attack" (Wikipedia only) |
| $a_i = 1 - H_b(s_i)/\ln 2$ | annotator agreement; $H_b$ is the binary entropy; $a_i=1$ when unanimous, $0$ at an even split |
| $z_k(x_i) \in \mathbb{R}^C$, $h_k(x_i) \in \mathbb{R}^{d_k}$ | logits and masked-mean-pooled last-layer state of committee teacher $k = 1..K$ (frozen, cached) |
| $z_s(x_i)$, $h_s(x_i) \in \mathbb{R}^{d_s}$ | student logits and pooled state |
| $z_s^{\text{irony}}(x_i) \in \mathbb{R}^2$ | the student's auxiliary irony head |
| $z_{\text{irony}}(x_i) \in \mathbb{R}^2$ | logits of the frozen auxiliary irony teacher (its own label space) |
| $W_k \in \mathbb{R}^{d_k \times d_s}$ | learnable projection from the student width to teacher $k$'s width |
| $T$, $\tau$ | distillation temperature; weight temperature |
| $\alpha, \beta, \gamma, \delta, \kappa$ | loss coefficients; $\kappa$ is the disagreement gain |

Teachers are *task-adapted*: each is initialised from its specialist checkpoint and fine-tuned on
the target training split, so every committee member shares the task's label space. The auxiliary
irony teacher is *not* adapted; it keeps its irony/non-irony label space and never enters the
committee.

## 3.1a The committee, and why one of its members is trained rather than downloaded

The committee is chosen for complementary expertise rather than for size: a general encoder
(BERT-large), a specialist in abuse that says what it means (HateBERT), and a specialist in saying
one thing and meaning another (a Twitter irony model). A committee whose members all know the same
things has nothing to combine, and weighting it per instance cannot help.

One expertise is missing from every public checkpoint and from both established corpora: abuse
carried by implication. The fine-grained cyberbullying corpus files it under a catch-all class and
the Wikipedia corpus records only whether a comment is an attack, so a teacher fine-tuned on either
cannot hold that knowledge, and a weighting scheme cannot route to an expertise no member has. We
therefore build the fourth member: HateBERT fine-tuned on a corpus in which implicit and explicit
abuse are separate labels (Section 4.x), then task-adapted onto the target benchmark like every
other member, its classification head re-initialised for the target label space. Call it the
*implicit specialist*.

This is the paper's central mechanical claim and it is stated as a testable one. On the two
benchmarks that do not label implication, the analogous claim about the *weighting* was tested first
and failed (Section 5.3), for a reason that bears directly on this one: after task adaptation the
committee members agree at Cohen's kappa 0.89 to 0.92, leaving a weighting almost nothing to express
(Section 5.3a). The specialist is a response to that finding rather than an addition made before it.
A committee can only route to expertise that some member has and the others lack, and the point of
training this member on a corpus none of the others has seen is to create exactly that asymmetry. The specialist
forms its own committee, $\mathcal{K}_{\text{spec}} = \mathcal{K}_{\text{homo}} \cup
\{\text{specialist}\}$, so that its effect is a controlled comparison of two full committees over
three seeds rather than a single ablation, and so that every result obtained with the three-teacher
committee remains valid. Two controls accompany it. The first distils from the specialist alone,
which asks whether the committee contributes anything the specialist does not. The second takes the
same student, fine-tunes it on the implicit corpus and then on the target task with no teachers at
all, which asks whether the knowledge came from distillation or simply from the data. If either
control matches the full committee, the corresponding claim is not available to us and the paper
says so.

## 3.2 Per-instance teacher reliability

For each training instance the committee is weighted by how reliable each teacher is on that
instance:

$$
\ell_k(i) = \mathrm{CE}\big(\mathrm{softmax}(z_k(x_i)),\, y_i\big), \qquad
w_k(i) = \frac{\exp(-\ell_k(i)/\tau)}{\sum_{j} \exp(-\ell_j(i)/\tau)} .
$$

When annotator fractions exist, reliability can instead be measured against them
($\ell_k(i) = \mathrm{BCE}(p_k^{\text{attack}}(x_i), s_i)$, "soft reliability"), which rewards a
teacher for being *appropriately uncertain* on contested items rather than for matching the
majority vote.

**Interpretation.** $w_k(i)$ is the posterior probability that teacher $k$ is the right expert for
$x_i$ under a uniform prior over teachers and a likelihood $p(y_i \mid k, x_i)^{1/\tau}$: with
$\tau = 1$ it is exact Bayesian model averaging over the committee on that instance; $\tau > 1$
tempers the likelihood toward the uniform prior (uniform averaging is the $\tau \to \infty$
limit), and $\tau < 1$ sharpens it toward hard selection of the single best teacher. Per-batch
weighting, as in earlier work, replaces $\ell_k(i)$ by its batch mean and therefore cannot
express that different instances call for different experts.

## 3.3 Objective

$$
\bar p(i) = \sum_k w_k(i)\, \mathrm{softmax}\!\big(z_k(x_i)/T\big), \qquad
\mathcal{L}_{\mathrm{KL}}(i) = T^2\, \mathrm{KL}\!\big(\bar p(i)\,\|\,\mathrm{softmax}(z_s(x_i)/T)\big)
$$

$$
\mathcal{L}_{\mathrm{hid}}(i) = \sum_k w_k(i)\, \big\| W_k h_s(x_i) - h_k(x_i) \big\|_2^2, \qquad
\mathcal{L}_{\mathrm{soft}}(i) = \mathrm{BCE}\!\big(\sigma(z_s^{\text{attack}}(x_i)),\, s_i\big)
$$

$$
\mathcal{L}_{\mathrm{irony}}(i) = T^2\, \mathrm{KL}\!\big(\mathrm{softmax}(z_{\text{irony}}(x_i)/T)\,\|\,\mathrm{softmax}(z_s^{\text{irony}}(x_i)/T)\big)
$$

$$
\mathcal{L} = \frac{1}{B}\sum_i \Big[\alpha\, c_i^{\mathrm{KL}}\, \mathcal{L}_{\mathrm{KL}}(i) + \beta\, c_i^{\mathrm{CE}}\, \mathrm{CE}(z_s(x_i), y_i)
 + \gamma\big(\tfrac12 \mathcal{L}_{\mathrm{hid}}(i) + \tfrac12 \mathcal{L}_{\mathrm{soft}}(i)\big) + \delta\, \mathcal{L}_{\mathrm{irony}}(i)\Big]
$$

with $c_i^{\mathrm{CE}} = c_i^{\mathrm{KL}} = 1$ in the standard variant. In the
**disagreement-aware** variant $c_i^{\mathrm{CE}} = a_i$ and $c_i^{\mathrm{KL}} = 1 + \kappa(1 - a_i)$:
where annotators agree the student is anchored to the human label; where they disagree the
label is down-weighted and the committee's graded distribution is trusted more. The soft term is
dropped on datasets without annotator fractions, and the hidden term is dropped in the
"no-hidden" ablation.

## 3.4 Auxiliary irony head

The irony teacher cannot join the committee: its labels are irony versus non-irony, not the task's
classes, so neither $\mathcal{L}_{\mathrm{KL}}$ nor $w_k$ is defined for it. Instead its softened
output on every training text is distilled into a second linear head on the student's pooled
state. The head is discarded at inference; its purpose is to shape the shared representation so
that sarcasm is represented separately from abuse. This is the mechanism behind the sarcasm
claims, and it introduces no sarcasm labels into the bullying corpora.

Separating sarcasm from abuse is the point, and it needs a metric that can see it. Recall on
ironically phrased abuse at a fixed decision threshold cannot distinguish a model that misses
implication from one that reacts to any sarcastic, negative text, and our measurements show the
second is what happens: on the fine-tune-only baseline, mean $p(\text{abusive})$ is 0.693 on
ironic abuse and 0.375 on benign sarcasm, so the two distributions overlap heavily while their means
differ. We therefore report **sarcasm-discrimination AUC**: the ROC-AUC of $1 - p(\text{benign})$
with the ironic-abuse probe as positives and the benign-sarcasm probe as negatives. It is
threshold-free and identically defined on every corpus. The baseline scores 0.776; the
recall-versus-false-positive curve is reported beside it, because a deployment must choose a
threshold even though an evaluation should not.

## 3.5 Algorithm

```
Input: training split D, committee checkpoints {M_k}, irony checkpoint M_irony, student init S
Stage 1  for k in 1..K:  M_k <- fine-tune(M_k, D)             # task adaptation, best validation epoch, frozen
Stage 2  for x_i in D:   cache z_k(x_i), h_k(x_i) for all k;  cache z_irony(x_i)
         release teachers from GPU
Stage 3  for epoch in 1..E:
             for minibatch B in D:
                 z_s, h_s, z_s^irony <- S(x_i) for i in B
                 w_k(i) <- softmax_k(-l_k(i)/tau)               # per-instance reliability (Sec. 3.2)
                 compute L (Sec. 3.3); update S and {W_k} by AdamW
             evaluate macro-F1 on validation; keep best; stop after 2 epochs without improvement
Output: S (projections and irony head discarded)
```

## 3.6 Homogeneous versus heterogeneous

We use "homogeneous" strictly for the *architecture family*: all committee teachers and the
headline student are BERT-lineage encoders (WordPiece or BPE tokenisers, absolute positions,
post-layer-norm). Tokenisers may differ; the pooled-state projection $W_k$ is what makes hidden
transfer possible across widths and vocabularies. Whether homogeneity matters is tested, not
assumed: the same objective is applied to a DeBERTa-v3 student (another transformer family), to a
BiLSTM student (no transformer), and to a committee that includes a DeBERTa-v3 teacher, with and
without the hidden term (Section 5.x).

## 3.7 Does the weighting select, or does it average?

*Written before the grid ran; the answer, reported in Section 5.3, is that it averages. This section
is kept in the method rather than moved to the results because the question it poses is the right one
and the apparatus for asking it is part of the contribution. A reader should meet the question here
and the answer there, not be told the conclusion twice.*

A committee of specialists is only a committee if the weights go somewhere. Because the teachers are
frozen and cached, $w_k(i)$ is a fixed function of the data: it varies across instances and never
across epochs or seeds, so "dynamic" here means instance-adaptive and the paper uses it in no other
sense. Whether it is adaptive in any useful way is an empirical question with a direct answer.

For each teacher we report the *routing contrast*: its mean weight on instances of the implicit class
minus its mean weight on instances of the explicit class, with a percentile bootstrap interval over
2,000 resamples. A contrast whose interval excludes zero is evidence that the weighting selects an
expert. A contrast that spans zero means the committee treats both kinds of abuse alike, in which
case any benefit from the specialist comes from the knowledge it adds and not from selection, and the
paper reports that instead of the claim it would have preferred.

The contrast depends sharply on $\tau$, which is why $\tau$ is swept rather than fixed. At
$\tau = 1$ the observed weights sit within 0.03 of uniform and D-MTHD optimises nearly the same
objective as uniform averaging; the sweep therefore reaches $\tau = 0.05$.
