"""Build the LaTeX thesis from paper/PAPER.md plus the thesis-only material.

    python paper/thesis/build_tex.py

Writes main.tex, frontmatter.tex, chapters/*.tex, appendices/*.tex and references.bib next to this
file. The paper's text is converted rather than retyped, so every number in the thesis is the number
in PAPER.md; what exists only in the thesis (front matter, research questions, preliminaries, the
answers to the research questions, how the study changed, future work, and the tables of data,
sessions, predictions and paired comparisons) is written here or generated from the archive.
"""
import csv
import io
import json
import os
import re
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ARCHIVE = os.environ.get("DMTHD_ARCHIVE", "E:/dmthd-work/kaggle_d738_v7")
MD = io.open(os.path.join(REPO, "paper", "PAPER.md"), encoding="utf-8").read().replace("\r\n", "\n")

TITLE = ("Out-of-Sample Knowledge Distillation for Compact Cyberbullying Detection: "
         "More Unlabelled Text Helps, More Teachers Do Not")

# --------------------------------------------------------------------------------- markdown -> LaTeX
_held = []


def _hold(s):
    _held.append(s)
    return "\x01%d\x02" % (len(_held) - 1)


def _release(s):
    while "\x01" in s:
        s = re.sub("\x01(\\d+)\x02", lambda m: _held[int(m.group(1))], s)
    return s


def ref(num):
    return ("\\ref{ch:%s}" % num) if "." not in num else ("\\ref{sec:%s}" % num)


def inline(s):
    s = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\\)\$", lambda m: _hold("$" + m.group(1) + "$"), s)
    s = re.sub(r"\\cite\{[^}]*\}", lambda m: _hold(m.group(0)), s)
    s = re.sub(r"\[\[(Figure|Table|Chapter|Section|Algorithm|Appendix):([^\]]+)\]\]",
               lambda m: _hold("%s~\\ref{%s}" % (m.group(1), m.group(2))), s)
    s = re.sub(r"\bSections (\d(?:\.\d+)?) (to|and) (\d(?:\.\d+)?)",
               lambda m: _hold("Sections~%s %s~%s" % (ref(m.group(1)), m.group(2), ref(m.group(3)))), s)
    s = re.sub(r"\bSection (\d(?:\.\d+)?)",
               lambda m: _hold(("Section~" if "." in m.group(1) else "Chapter~") + ref(m.group(1))), s)
    s = re.sub(r"github\.com/mahdihasanshadi/THESIS",
               lambda m: _hold("\\url{https://github.com/mahdihasanshadi/THESIS}"), s)
    s = s.replace("+-", _hold("$\\pm$"))
    s = re.sub(r"(^|[\s\[(,/])-(?=\d)", lambda m: m.group(1) + _hold("$-$"), s)
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_")):
        s = s.replace(a, b)
    s = s.replace("~", _hold("$\\sim$"))
    s = s.replace("<", _hold("$<$")).replace(">", _hold("$>$"))
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = re.sub(r'"([^"]+)"', r"``\1''", s)
    return _release(s)


def wrap(s, indent=""):
    return textwrap.fill(s, width=100, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)


def parse_blocks(body):
    lines, out, buf, i = body.split("\n"), [], [], 0

    def flush():
        if buf:
            out.append(list(buf))
            buf.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "$$":
            flush()
            j = i + 1
            while lines[j].strip() != "$$":
                j += 1
            out.append(["$$"] + lines[i + 1:j])
            i = j + 1
            continue
        if not ln.strip():
            flush()
        else:
            buf.append(ln)
        i += 1
    flush()
    return out


def table_tex(block, caption, label, font="\\small"):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block]
    header, body = rows[0], rows[2:]
    n = len(header)
    assert all(len(r) == n for r in body), (label, [len(r) for r in body])
    lines = [" & ".join(inline(c) for c in r) + " \\\\" for r in [header] + body]
    tab = "\n".join(["\\begin{adjustbox}{max width=\\textwidth}",
                     "\\begin{tabular}{%s}" % ("l" + "c" * (n - 1)), "\\toprule", lines[0], "\\midrule"]
                    + lines[1:] + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"])
    return "\n".join(["\\begin{table}[htbp]", "\\centering", font, "\\caption{%s}" % caption,
                      "\\label{%s}" % label, tab, "\\end{table}"])


def list_tex(block):
    ordered = bool(re.match(r"^\d+\. ", block[0]))
    items, cur = [], None
    for ln in block:
        m = re.match(r"^(\d+\.|-) (.*)$", ln)
        if m and not ln.startswith(" "):
            if cur is not None:
                items.append(cur)
            cur = m.group(2)
        else:
            cur += " " + ln.strip()
    items.append(cur)
    env = "enumerate" if ordered else "itemize"
    return "\\begin{%s}\n%s\n\\end{%s}" % (env, "\n".join(wrap("  \\item " + inline(it), "    ") for it in items), env)


def block_tex(block, tables):
    if block[0] == "$$":
        return "\\begin{equation}\n" + "\n".join(block[1:]).strip() + "\n\\end{equation}"
    if block[0].startswith("|"):
        caption, label = tables.pop(0)
        return table_tex(block, caption, label)
    if re.match(r"^\d+\. ", block[0]) or block[0].startswith("- "):
        return list_tex(block)
    return wrap(inline(" ".join(ln.strip() for ln in block)))


def convert(body, tables=(), inserts=(), subs=()):
    for old, new in subs:
        assert body.count(old) == 1, ("substitution not found once", old[:70])
        body = body.replace(old, new)
    tables, used, out = list(tables), set(), []
    for blk in parse_blocks(body):
        raw = " ".join(blk)
        out.append(block_tex(blk, tables))
        for k, (start, code) in enumerate(inserts):
            if raw.startswith(start):
                out.append(code)
                used.add(k)
    assert not tables, ("unused table captions", tables)
    assert len(used) == len(inserts), ("unused inserts", [s for k, (s, _) in enumerate(inserts) if k not in used])
    return "\n\n".join(out)


def sections(md):
    out, key, buf = {}, None, []
    for line in md.split("\n"):
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if m:
            if key is not None:
                out[key] = "\n".join(buf).strip("\n")
            key, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    out[key] = "\n".join(buf).strip("\n")
    return out


SEC = sections(MD)


def _key(prefix):
    ks = [k for k in SEC if k.startswith(prefix)]
    assert len(ks) == 1, (prefix, ks)
    return ks[0]


def body(prefix):
    return SEC[_key(prefix)]


SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor", "of", "on", "or", "the",
         "to", "with", "vs"}


def _cap(part):
    return part[:1].upper() + part[1:] if part[:1].islower() else part


def heading(prefix):
    """Title case in the Chicago manner: articles, conjunctions and prepositions stay lower case unless
    they open the heading or follow a colon; each part of a hyphenated word is capitalised."""
    text = re.sub(r"^(\d+(\.\d+)?\.?|Appendix [A-E]\.) ", "", _key(prefix))
    out, first = [], True
    for w in text.split(" "):
        parts = w.split("-")
        new = []
        for j, p in enumerate(parts):
            bare = p.strip(",:;.").lower()
            if (not first or j) and bare in SMALL:
                new.append(p.lower())
            else:
                new.append(_cap(p))
        out.append("-".join(new))
        first = w.endswith(":")
    return inline(" ".join(out))


def figure(name, caption, label):
    return "\n".join(["\\begin{figure}[htbp]", "\\centering", "\\includegraphics[width=\\textwidth]{figures/%s.pdf}" % name,
                      "\\caption{%s}" % caption, "\\label{%s}" % label, "\\end{figure}"])


def fmt(n):
    return "{:,}".format(int(n))


def write(rel, text):
    path = os.path.join(HERE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text.rstrip("\n") + "\n")
    print("wrote", rel)


# --------------------------------------------------------------------------------- figures
FIG = {
    "schematic": figure("fig_schematic",
                        "In-sample against out-of-sample distillation. The same teacher and student are used in both "
                        "settings; only the text on which the student imitates the teacher differs. In sample, on the "
                        "tweets the teacher was fine-tuned on, its soft labels repeat the gold labels; out of sample, on "
                        "tweets it has never seen, they carry its uncertainty.", "fig:schematic"),
    "tau": figure("fig_tau",
                  "The weighting-temperature sweep on BERT-mini. (a) Even at $\\tau = 0.05$ the largest teacher weight on "
                  "a tweet averages 0.376 against a uniform 0.333. (b) Test macro-F1 of the first seed stays within 0.0007 "
                  "across a hundredfold range of $\\tau$, below fine-tuning at every value.", "fig:tau"),
    "size": figure("fig_size_curve",
                   "The size curve on BERT-mini, one teacher labelling. Filled points: nested subsets of the abuse-domain "
                   "transfer set; hollow points: the same set extended with generic tweets; diamond: 42,013 generic tweets "
                   "alone. Horizontal lines: fine-tuning alone for 6, 13 and 35 epochs, the last two matching the number of "
                   "updates of the 42k and 168k arms. Small dots are single seeds.", "fig:size-curve"),
    "validation": figure("fig_validation",
                         "Validation macro-F1 of BERT-mini against the number of optimisation updates. Fine-tuning alone "
                         "for 35 epochs (gray, three seeds) peaks early and then declines; the 168k transfer arm (blue, three "
                         "seeds) reaches a higher score within the same number of updates.", "fig:validation"),
    "students": figure("fig_students",
                       "Gain over fine-tuning from the same teacher, in sample and with 168,000 unlabelled tweets, on three "
                       "students. Whiskers are paired bootstrap 95\\% intervals.", "fig:students"),
    "tradeoff": figure("fig_tradeoff",
                       "False-positive rate on benign sarcasm against recall on ironic abuse for all 184 evaluated models. "
                       "Every family and objective lies along one trade-off line; the out-of-sample students slide along it "
                       "rather than rising above it.", "fig:tradeoff"),
    "agreement": figure("fig_agreement",
                        "Cohen's kappa between the task-adapted teachers' test predictions. Every pair agrees above "
                        "0.88, and the implicit specialist agrees with HateBERT, the checkpoint it was trained from, "
                        "at 0.962. A per-instance weighting has to work inside what is left.", "fig:agreement"),
    "auc": figure("fig_auc",
                  "Sarcasm-discrimination AUC of every analysed BERT-mini variant, first seed. Every pre-trained variant lies "
                  "between 0.72 and 0.78, in sample or out; only removing pre-training moves the measure.", "fig:auc"),
}

# --------------------------------------------------------------------------------- front matter
ABSTRACT_BLOCKS = parse_blocks(body("Abstract"))
assert ABSTRACT_BLOCKS[-1][0].startswith("**Keywords:**")
ABSTRACT = "\n\n".join(block_tex(b, []) for b in ABSTRACT_BLOCKS[:-1])
KEYWORDS = inline(" ".join(ABSTRACT_BLOCKS[-1]).replace("**Keywords:**", "").strip())
STATEMENTS = parse_blocks(body("Statements"))
ETHICS = inline(" ".join(STATEMENTS[0]).replace("**Ethics.**", "").strip())
assert " ".join(STATEMENTS[0]).startswith("**Ethics.**")

AUTHORS = [("Naheyan Tahmin", "23101001"), ("Mahdi Hasan Shadi", "23101265"),
           ("Tasnim Rahmatullah", "23101269"), ("Arora Paul", "23101439")]

FRONT = r"""
\begin{titlepage}
\centering
\vspace*{0.5cm}
{\LARGE\bfseries <<TITLE>>\par}
\vspace{1.4cm}
{\large by\par}
\vspace{0.8cm}
{\large
<<AUTHORS_TITLE>>\par}
\vspace{1.4cm}
A thesis submitted to the Department of Computer Science and Engineering\\
in partial fulfillment of the requirements for the degree of\\
B.Sc. in Computer Science and Engineering
\vfill
Department of Computer Science and Engineering\\
BRAC University\\
\thesisdate
\vspace{1.2cm}

\copyright\ 2026. BRAC University\\
All rights reserved.
\end{titlepage}

\pagenumbering{roman}
\setcounter{page}{1}

\chapter*{Declaration}
\phantomsection\addcontentsline{toc}{chapter}{Declaration}
It is hereby declared that
\begin{enumerate}
  \item The thesis submitted is our own original work while completing degree at BRAC University.
  \item The thesis does not contain material previously published or written by a third party, except
    where this is appropriately cited.
  \item The thesis does not contain material which has been accepted, or submitted, for any other degree
    or diploma at a university or other institution.
  \item We have acknowledged all main sources of help.
\end{enumerate}

\vspace{0.8cm}
\noindent\textbf{Student's Full Name \& Signature:}

\vspace{1.4cm}
\begin{center}
\begin{tabular}{c@{\hspace{2.2cm}}c}
\rule{5.5cm}{0.4pt} & \rule{5.5cm}{0.4pt}\\
Naheyan Tahmin & Mahdi Hasan Shadi\\
23101001 & 23101265\\[2.2cm]
\rule{5.5cm}{0.4pt} & \rule{5.5cm}{0.4pt}\\
Tasnim Rahmatullah & Arora Paul\\
23101269 & 23101439
\end{tabular}
\end{center}

\chapter*{Approval}
\phantomsection\addcontentsline{toc}{chapter}{Approval}
The thesis titled ``<<TITLE>>'' submitted by
\begin{enumerate}
<<AUTHORS_APPROVAL>>
\end{enumerate}
of \thesissemester{} has been accepted as satisfactory in partial fulfillment of the requirement for the
degree of B.Sc. in Computer Science and Engineering on \defensedate.

\vspace{0.6cm}
\noindent\textbf{Examining Committee:}

% each role beside its signature line, so that the page holds all three with room to sign
\vspace{1.2cm}
\noindent\begin{tabular}{@{}p{4.6cm}l@{}}
Supervisor: & \rule{7cm}{0.4pt}\\
(Member) & Dr. Muhammad Iqbal Hossain\\
 & Associate Professor\\
 & Department of Computer Science and Engineering\\
 & BRAC University\\[1.4cm]
Co-Supervisor: & \rule{7cm}{0.4pt}\\
(Member) & Sheikh Araf Noshin\\
 & Lecturer\\
 & Department of Computer Science and Engineering\\
 & BRAC University\\[1.4cm]
Head of Department: & \rule{7cm}{0.4pt}\\
(Chair) & Dr. Sadia Hamid Kazi\\
 & Associate Professor and Chairperson\\
 & Department of Computer Science and Engineering\\
 & BRAC University
\end{tabular}

\chapter*{Ethics Statement}
\phantomsection\addcontentsline{toc}{chapter}{Ethics Statement}
<<ETHICS>>

\chapter*{Abstract}
\phantomsection\addcontentsline{toc}{chapter}{Abstract}
<<ABSTRACT>>

\vspace{0.4cm}
\noindent\textbf{Keywords:} <<KEYWORDS>>

\chapter*{Acknowledgement}
\phantomsection\addcontentsline{toc}{chapter}{Acknowledgement}
We are grateful to our supervisor, Dr. Muhammad Iqbal Hossain, and our co-supervisor, Sheikh Araf Noshin,
for their guidance and their patience with a thesis that changed direction when the evidence did. We
thank the Department of Computer Science and Engineering at BRAC University for its support, the authors
of the public corpora this work is built on, and Kaggle for the GPU time on which every experiment ran.

\tableofcontents
\clearpage\phantomsection\addcontentsline{toc}{chapter}{List of Figures}
\listoffigures
\clearpage\phantomsection\addcontentsline{toc}{chapter}{List of Tables}
\listoftables

\chapter*{List of Abbreviations}
\phantomsection\addcontentsline{toc}{chapter}{List of Abbreviations}
\begin{tabular}{@{}ll@{}}
AUC & Area under the receiver operating characteristic curve\\
BERT & Bidirectional Encoder Representations from Transformers\\
BiLSTM & Bidirectional long short-term memory network\\
CE & Cross-entropy\\
D-MTHD & Dynamic Multi-Teacher Homogeneous Knowledge Distillation, the method audited here\\
ECE & Expected calibration error\\
F1 & Harmonic mean of precision and recall; macro-F1 averages it over classes\\
FLOPs & Floating-point operations\\
FPR & False-positive rate\\
GPU & Graphics processing unit\\
INT8 & 8-bit integer quantisation\\
KD & Knowledge distillation\\
KL & Kullback--Leibler divergence\\
kNN & $k$ nearest neighbours\\
NLP & Natural language processing\\
RoBERTa & Robustly optimised BERT pre-training approach\\
TF-IDF & Term frequency--inverse document frequency\\
\end{tabular}
"""
FRONT = (FRONT.replace("<<TITLE>>", TITLE)
         .replace("<<AUTHORS_TITLE>>", "\\\\[0.35cm]\n".join("%s\\\\ %s" % a for a in AUTHORS))
         .replace("<<AUTHORS_APPROVAL>>", "\n".join("  \\item %s (%s)" % a for a in AUTHORS))
         .replace("<<ETHICS>>", wrap(ETHICS)).replace("<<ABSTRACT>>", ABSTRACT).replace("<<KEYWORDS>>", KEYWORDS))

# --------------------------------------------------------------------------------- chapter 1
IB = parse_blocks(body("1. Introduction"))
assert " ".join(IB[0]).startswith("Detectors of abusive language")
assert " ".join(IB[1]).startswith("We tested that promise")
assert " ".join(IB[2]).startswith("The second half follows")
assert " ".join(IB[3]).startswith("The gain exists and grows")
assert " ".join(IB[4]).startswith("The knowledge that matters most")
assert " ".join(IB[5]).startswith("The contributions are as follows")
assert " ".join(IB[7]).startswith("We do not claim a new method")
bt = lambda k: block_tex(IB[k], [])  # noqa: E731

CH1 = r"""
\chapter{Introduction}
\label{ch:1}

\section{Background}
\label{sec:1.1}
<<P0>>

This thesis continues the Pre-Thesis II report of February 2026, which proposed Dynamic Multi-Teacher
Homogeneous Knowledge Distillation (D-MTHD) and evaluated it on the Wikipedia Talk personal-attacks corpus
with two teachers and a DistilBERT student. That report found the student at 0.8947 macro-F1 against
teachers at 0.8936 and 0.8904, from one run of each configuration, and named two tasks for this phase:
extending the method to sarcastic and implicit cyberbullying, and testing it against a student trained
without teachers. This thesis does both, on a cleaned fine-grained cyberbullying corpus, and what it
finds changes the method.

\section{Motivation}
\label{sec:1.2}
Moderation at the scale of a social platform depends on detectors cheap enough to run on every post,
and increasingly on the device where the post is written \cite{zhou2019edge, schwartz2020green}.
Compact models make that possible, and distillation is how compact models are usually made
competitive. Two facts make the question worth a thesis. The first is that the abuse detectors miss
most is the abuse that is implied rather than stated: every survey of cyberbullying detection names
sarcastic and indirect abuse as the open case \cite{rosa2019automatic, salawu2020approaches,
emmery2021current}, and implicit hate has needed benchmarks of its own \cite{elsherief2021latent,
ocampo2023indepth}. The second is that the methods proposed for this task have not been given the
controls that would show whether the teachers help at all (Section~\ref{sec:1.3}). A practitioner
choosing how to train a compact detector needs to know whether distillation moves it, where the gain
comes from, and whether any of it reaches implied abuse. Those questions become the four of
Section~\ref{sec:1.4}, the first splitting into an in-sample and an out-of-sample form.

\section{Problem Statement}
\label{sec:1.3}
Three problems meet in this task. The first is deployment: the encoders that detect abuse best are too
large and slow for on-device or high-throughput moderation, and compact students trail them. The second
is implication: the abuse that detectors and annotators miss most is carried by irony, stereotype and
coded reference, and the widely used benchmarks do not label it. The third is evidence: the methods
proposed to close the first gap, committees of teachers weighted per instance, are usually reported
without the control that would show whether the teachers helped at all, which is the same student
trained without a teacher, and without repeated runs or confidence intervals. The Pre-Thesis II report
shared this gap. A distillation method that is not shown to beat that control cannot be said to work, and
a gain that is not measured on implicit abuse cannot be said to address it.

\section{Research Questions and Objectives}
\label{sec:1.4}
The thesis asks four questions.
\begin{description}
  \item[RQ1.] Does multi-teacher distillation with per-instance reliability weighting improve a compact
    cyberbullying detector over the same student fine-tuned without teachers?
  \item[RQ2.] What mechanism explains the answer to RQ1?
  \item[RQ3.] Does distillation on unlabelled text the teachers have never seen improve the student, and
    how does the gain depend on the amount and composition of that text, on the number of optimisation
    updates, and on the student?
  \item[RQ4.] Does any of these interventions improve the student's ability to tell implied abuse from
    harmless sarcasm?
\end{description}
To answer them, the thesis sets four objectives.
\begin{enumerate}
  \item Build a clean benchmark and a controlled grid of five students, three committees, four objectives
    and three seeds, with a no-teacher control and paired bootstrap intervals for every comparison.
  \item Measure the mechanism: what the teachers have memorised, how far they agree, and how much any
    weighting of them could gain.
  \item Construct an unlabelled transfer set screened against every evaluation set, and measure the
    out-of-sample gain against its size and composition and against fine-tuning at matched compute,
    with the predictions written before each run.
  \item Measure the discrimination of implied abuse without a threshold, on probe sets and on a benchmark
    that labels implication.
\end{enumerate}

\section{Approach and Summary of Findings}
\label{sec:1.5}
<<P1>>

<<P2>>

<<P3>>

<<P4>>

\section{Contributions}
\label{sec:1.6}
<<P5>>

<<P6>>

<<P7>>

\section{Thesis Organisation}
\label{sec:1.7}
Chapter~\ref{ch:2} reviews the literature on cyberbullying and implicit abuse, on distilling compact
language models, on multi-teacher distillation, and on evaluation practice. Chapter~\ref{ch:3} describes
the audited method and the out-of-sample setting. Chapter~\ref{ch:4} describes the data, the probe sets,
the transfer set and the protocol. Chapter~\ref{ch:5} reports the results in the order of the argument.
Chapter~\ref{ch:6} answers the research questions, discusses what the findings mean and records how the
study changed. Chapter~\ref{ch:7} states the limitations and future work, and Chapter~\ref{ch:8}
concludes.
"""
for k in range(8):
    CH1 = CH1.replace("<<P%d>>" % k, bt(k))

# --------------------------------------------------------------------------------- chapter 2
CH2_PRELIM = r"""
\section{Preliminaries}
\label{sec:prelim}

\subsection{Knowledge distillation}
A teacher and a student produce logits $z_t(x), z_s(x) \in \mathbb{R}^C$ for a text $x$ and $C$ classes.
Softened distributions $p^{(T)}_t = \mathrm{softmax}(z_t/T)$ and $p^{(T)}_s = \mathrm{softmax}(z_s/T)$
are compared at a temperature $T > 1$, and the student minimises
\begin{equation}
\mathcal{L}_{\mathrm{KD}} = \beta\, \mathrm{CE}\big(\mathrm{softmax}(z_s), y\big)
 + \alpha\, T^2\, \mathrm{KL}\big(p^{(T)}_t \,\|\, p^{(T)}_s\big)
\end{equation}
\cite{hinton2015distilling}. The temperature spreads probability over the classes the teacher considers
wrong, and the factor $T^2$ keeps the gradient of the second term comparable across temperatures. These
\emph{soft labels} carry how the teacher ranks the wrong classes, which a one-hot label cannot: a teacher
that gives a tweet 0.70 for one class and 0.20 for another tells the student that the two are confusable
there. The argument of this thesis turns on when that information exists. If a teacher has fitted a text
so well that it assigns the gold class a probability near one, its soft label and the gold label are the
same thing.

\subsection{Multi-teacher distillation}
With $K$ teachers the target is a mixture $\bar p = \sum_k w_k\, p^{(T)}_k$. The weights can be uniform,
fixed per teacher, or computed per instance from each teacher's confidence or from its loss against the
gold label, so that the student trusts each teacher where it is reliable \cite{you2017learning,
wu2021one, zhang2022confidence}. Intermediate representations can be matched as well, through learned
projections from the student's width to each teacher's \cite{romero2015fitnets, wu2021one}.

\subsection{The transfer set}
The text on which the student imitates the teacher is called the \emph{transfer set}
\cite{hinton2015distilling}. When it is the labelled split the teachers were fine-tuned on, we call the
distillation \emph{in sample}; when it contains text outside every teacher's training split,
\emph{out of sample}. Section~\ref{sec:3.5} defines both precisely.

\subsection{Measures used throughout}
\emph{Macro-F1} is the unweighted mean of the per-class F1 scores, so that small classes count as much
as large ones. \emph{ROC-AUC} is the probability that a randomly chosen positive is ranked above a
randomly chosen negative; 0.5 is chance and 1.0 is perfect, and it needs no decision threshold.
\emph{Cohen's kappa} measures agreement between two classifiers beyond what chance would produce.
\emph{Expected calibration error} measures how far predicted probabilities are from observed accuracy.
A \emph{paired bootstrap} resamples the test set with replacement many times, recomputes the difference
between two systems on each resample, and reports the central 95\% of those differences as an interval;
a difference whose interval excludes zero is unlikely to be an accident of the particular test set
\cite{koehn2004statistical, dror2018hitchhiker}.
"""

CH2_GAP = r"""
\section{Summary and Research Gap}
\label{sec:gap}
Table~\ref{tab:gap} summarises what each line of work establishes, what it leaves open for this thesis,
and where the thesis addresses it.

\begin{table}[htbp]
\centering
\small
\caption{What the literature establishes and what it leaves open.}
\label{tab:gap}
\begin{tabular}{p{3.2cm}p{4.3cm}p{4.3cm}p{2.4cm}}
\toprule
Line of work & Establishes & Leaves open & Addressed in \\
\midrule
Cyberbullying and toxicity detection & Benchmarks and strong fine-tuned encoders & Compact models are
evaluated on aggregate F1 only; implicit abuse is not labelled & Chapter~\ref{ch:4}, Section~\ref{sec:5.9} \\
Implicit hate & Implicit hate needs its own label and benchmark & Whether a compact model can learn it by
distillation & Sections~\ref{sec:5.5} and~\ref{sec:5.9} \\
Distilling compact models & Transfer sets and pre-training matter & Not measured in this domain with the
confounds controlled & Sections~\ref{sec:5.7} and~\ref{sec:5.8} \\
Multi-teacher weighting & Several teachers are claimed to beat one & The no-teacher control, seeds and
intervals; reliability is read in sample & Sections~\ref{sec:5.2} to~\ref{sec:5.6} \\
Evaluation practice & Functional tests, reporting standards, power & A threshold-free measure of
implication for compact detectors & Sections~\ref{sec:4.6} and~\ref{sec:5.9} \\
\bottomrule
\end{tabular}
\end{table}
"""

CH2 = "\n".join([
    "\\chapter{Literature Review}", "\\label{ch:2}",
    "This chapter first sets out the concepts the rest of the thesis relies on, then reviews four lines of "
    "work, and ends with the gap they leave.",
    CH2_PRELIM,
    "\\section{%s}\n\\label{sec:2.1}\n%s" % (heading("2.1 "), convert(body("2.1 "))),
    "\\section{%s}\n\\label{sec:2.2}\n%s" % (heading("2.2 "), convert(body("2.2 "))),
    "\\section{%s}\n\\label{sec:2.3}\n%s" % (heading("2.3 "), convert(body("2.3 "))),
    "\\section{%s}\n\\label{sec:2.4}\n%s" % (heading("2.4 "), convert(body("2.4 "))),
    CH2_GAP,
])

# --------------------------------------------------------------------------------- chapter 3
ALGORITHM = r"""
\section{Training Procedure}
\label{sec:3.7}
Algorithm~\ref{alg:training} gives the procedure shared by every run. In-sample runs use an empty
transfer set; fine-tune-only runs skip the teachers and train on the labelled split with cross-entropy
alone.

\begin{algorithm}[htbp]
\caption{Distillation in and out of sample.}
\label{alg:training}
\begin{algorithmic}[1]
\Require labelled split $\mathcal{D}$; validation split $\mathcal{V}$; unlabelled transfer set
  $\mathcal{U}$, empty in sample; teacher checkpoints $M_1, \dots, M_K$; unadapted irony checkpoint
  $M_{\mathrm{irony}}$; pre-trained student $S$
\For{$k = 1, \dots, K$}
  \State fine-tune $M_k$ on $\mathcal{D}$, keep its best validation epoch, and freeze it \Comment{task adaptation}
\EndFor
\For{each text $x \in \mathcal{D} \cup \mathcal{U}$}
  \State cache $z_k(x)$ and $h_k(x)$ for every teacher $k$, and $z_{\mathrm{irony}}(x)$ from $M_{\mathrm{irony}}$
\EndFor
\If{the run weights out of sample}
  \State cache $h_k(v)$ and the correctness $c_k(v)$ for every $v \in \mathcal{V}$, and build each
    teacher's neighbour index \Comment{for equation (3.7)}
\EndIf
\State set the label mask $m_x = 1$ for $x \in \mathcal{D}$ and $m_x = 0$ for $x \in \mathcal{U}$
\For{epoch $= 1, \dots, E$}
  \For{each minibatch drawn from $\mathcal{D} \cup \mathcal{U}$}
    \State compute $z_s(x)$, $h_s(x)$ and $z^{\mathrm{irony}}_s(x)$ for every $x$ in the minibatch
    \State weight the teachers by $w_k(x)$ of Section~\ref{sec:3.2} in sample, and by
      $w_k^{\mathrm{knn}}(x)$ of Section~\ref{sec:3.5} on every row of an out-of-sample run
    \State compute the loss of Section~\ref{sec:3.3} with the mask $m_x$ and update $S$ and $W_1, \dots, W_K$
  \EndFor
  \State evaluate macro-F1 on the validation split and keep the best epoch
\EndFor
\State \Return $S$, discarding the projections $W_k$ and the irony head
\end{algorithmic}
\end{algorithm}
"""

# thesis-only pointers: the paper has no appendix of training settings to send the reader to
CH3_SUBS = {
    3: [("implemented in PyTorch and Transformers \\cite{paszke2019pytorch, wolf2020transformers}.",
         "implemented in PyTorch and Transformers \\cite{paszke2019pytorch, wolf2020transformers}; the "
         "remaining settings are collected in [[Appendix:app:hyper]].")],
}
CH3 = "\n".join([
    "\\chapter{Methodology}", "\\label{ch:3}",
    wrap("This chapter describes the method the thesis set out to evaluate, D-MTHD, and the change of setting "
         "that turned out to matter. Figure~\\ref{fig:schematic} shows the distinction the thesis turns on: in "
         "both settings the same teacher and the same student are used, and only the text on which the student "
         "imitates the teacher differs. Sections~\\ref{sec:3.1} to~\\ref{sec:3.4} define the committee, the "
         "per-instance weighting, the objective and the specialist teacher exactly as they were audited; "
         "Section~\\ref{sec:3.5} defines in-sample and out-of-sample distillation and their controls; "
         "Section~\\ref{sec:3.6} states what is measured about the weighting; and Section~\\ref{sec:3.7} gives the "
         "training procedure as an algorithm."),
    FIG["schematic"],
] + ["\\section{%s}\n\\label{sec:3.%d}\n%s" % (heading("3.%d " % k), k,
                                             convert(body("3.%d " % k), (), (), CH3_SUBS.get(k, ())))
     for k in range(1, 7)] + [ALGORITHM])

# --------------------------------------------------------------------------------- chapter 4
rep = json.load(open(os.path.join(ARCHIVE, "data", "tweets", "report.json"), encoding="utf-8"))
cc = rep["class_counts"]
CLASSES = ["age", "ethnicity", "gender", "religion", "other_cyberbullying", "not_cyberbullying"]
class_rows = "\n".join("%s & %s & %s & %s & %s \\\\" % (
    c.replace("_", "\\_"), fmt(cc["train"][c]), fmt(cc["val"][c]), fmt(cc["test"][c]),
    fmt(cc["train"][c] + cc["val"][c] + cc["test"][c])) for c in CLASSES)
CLASS_TABLE = r"""
\begin{table}[htbp]
\centering
\small
\caption{Class distribution of the cleaned tweet corpus.}
\label{tab:classes}
\begin{tabular}{lrrrr}
\toprule
Class & Train & Validation & Test & Total \\
\midrule
<<ROWS>>
\midrule
Total & 34,607 & 4,326 & 4,326 & 43,259 \\
\bottomrule
\end{tabular}
\end{table}
""".replace("<<ROWS>>", class_rows)

DATA_TABLE = r"""
\begin{table}[htbp]
\centering
\small
\caption{The data used in this thesis.}
\label{tab:data}
\begin{tabular}{p{4.6cm}p{3.1cm}rrrp{1.7cm}}
\toprule
Set & Role & Train & Validation & Test & Labels \\
\midrule
Cyberbullying tweets, cleaned & primary benchmark & 34,607 & 4,326 & 4,326 & 6 classes \\
Implicit Hate Corpus, single source & implicit benchmark & 16,509 & 2,064 & 2,064 & 3 classes \\
ISHate, held out whole & out-of-domain test & -- & -- & 27,096 & 3 classes \\
Benign sarcasm probe & inference only & -- & -- & 883 & -- \\
Ironic abuse probe & inference only & -- & -- & 1,560 & -- \\
Implicit abuse probe, a subset of it & inference only & -- & -- & 763 & -- \\
Transfer set, abuse-domain base & training, unlabelled & 42,013 & -- & -- & none \\
Transfer set, extended & training, unlabelled & 205,593 & -- & -- & none \\
\bottomrule
\end{tabular}
\end{table}
"""

tb = json.load(open(os.path.join(ARCHIVE, "data", "tweets", "transfer_report.json"), encoding="utf-8"))
tg = json.load(open(os.path.join(ARCHIVE, "data", "tweets", "transfer_big_report.json"), encoding="utf-8"))
NAMES = {"olid": "OLID \\cite{zampieri2019olid}", "hateval": "HatEval \\cite{basile2019hateval}",
         "davidson": "Davidson et al. \\cite{davidson2017automated}",
         "raw_conflicting": "Conflicting-label tweets of the corpus", "sentiment": "TweetEval sentiment",
         "emoji": "TweetEval emoji", "emotion": "TweetEval emotion"}


def _src_rows(r):
    return "\n".join("%s & %s & %s \\\\" % (NAMES[k], fmt(v["rows_in"]), fmt(r["rows_out_by_source"].get(k, 0)))
                     for k, v in r["sources"].items())


def _ov(r, keys):
    return fmt(sum(r["rows_dropped_overlap"].get(k, 0) for k in keys))


SPLITS = ["tweets/train", "tweets/val", "tweets/test"]
TRANSFER_TABLE = r"""
\begin{table}[htbp]
\centering
\small
\caption{Construction of the transfer set. Upper part: rows read and kept per source. Lower part: rows
matching each screen; a row can match more than one screen, so the screens do not sum to the difference.}
\label{tab:transfer}
\begin{tabular}{lrr}
\toprule
Source & Rows read & Rows kept \\
\midrule
\multicolumn{3}{l}{\emph{Abuse-domain base set}} \\
<<BASE>>
Total & <<BASE_IN>> & <<BASE_OUT>> \\
\midrule
\multicolumn{3}{l}{\emph{Generic extension}} \\
<<GEN>>
Total & <<GEN_IN>> & <<GEN_OUT>> \\
\midrule
Extended set, base first & & <<ALL>> \\
\midrule
Screen & Base & Extension \\
\midrule
Shorter than two tokens & <<S1>> & <<S1G>> \\
Duplicate within the set & <<S2>> & <<S2G>> \\
Occurs in a tweet split & <<S3>> & <<S3G>> \\
Occurs in a probe set & <<S4>> & <<S4G>> \\
Occurs in the implicit benchmark's splits & <<S5>> & <<S5G>> \\
Occurs in held-out ISHate & <<S6>> & <<S6G>> \\
Occurs in the base set & -- & <<S7G>> \\
\bottomrule
\end{tabular}
\end{table}
"""
_rep = {"<<BASE>>": _src_rows(tb), "<<BASE_IN>>": fmt(tb["rows_in"]), "<<BASE_OUT>>": fmt(tb["rows_out"]),
        "<<GEN>>": _src_rows(tg), "<<GEN_IN>>": fmt(tg["rows_in"]), "<<GEN_OUT>>": fmt(tg["rows_out"]),
        "<<ALL>>": fmt(tg["rows_written"]),
        "<<S1>>": fmt(tb["rows_in"] - tb["rows_after_min_len"]), "<<S1G>>": fmt(tg["rows_in"] - tg["rows_after_min_len"]),
        "<<S2>>": fmt(tb["rows_dropped_duplicates"]), "<<S2G>>": fmt(tg["rows_dropped_duplicates"]),
        "<<S3>>": _ov(tb, SPLITS), "<<S3G>>": _ov(tg, SPLITS),
        "<<S4>>": fmt(max(v for k, v in tb["rows_dropped_overlap"].items() if k.startswith("probes/"))),
        "<<S4G>>": fmt(max(v for k, v in tg["rows_dropped_overlap"].items() if k.startswith("probes/"))),
        "<<S5>>": _ov(tb, ["implicit/train.csv", "implicit/val.csv", "implicit/test.csv"]),
        "<<S5G>>": _ov(tg, ["implicit/train.csv", "implicit/val.csv", "implicit/test.csv"]),
        "<<S6>>": _ov(tb, ["implicit/test_ood_ishate.csv"]), "<<S6G>>": _ov(tg, ["implicit/test_ood_ishate.csv"]),
        "<<S7G>>": fmt(tg["rows_dropped_overlap"].get("base", 0))}
for k, v in _rep.items():
    TRANSFER_TABLE = TRANSFER_TABLE.replace(k, v)

SESSIONS = r"""
\section{Implementation and Compute}
\label{sec:4.8}
Every student and teacher in this thesis was trained on Kaggle T4 accelerators by one driver script that
resumes finished work from the previous session's output, so that the grid could grow across sessions
of at most twelve hours. Table~\ref{tab:sessions} lists the sessions of the tweet notebook. Two sessions
failed and are reported rather than hidden: the second stopped at teacher caching because the resume
logic had discarded the teacher weights, and the third froze inside a diagnostic cell until Kaggle ended
it; both faults were fixed, and the driver now kills any command that writes nothing for two hours. The
teachers, the implicit specialist and the student pre-trained on the implicit corpus were each trained
once and resumed by every later session. The code, the driver, the notebook and a CPU smoke test that runs
every stage on tiny models before any GPU session are in the repository.

\begin{table}[htbp]
\centering
\small
\caption{The Kaggle sessions of the tweet grid.}
\label{tab:sessions}
\begin{tabular}{clrr}
\toprule
Session & What it added & GPU-hours & Student runs, cumulative \\
\midrule
1 & the in-sample grid on five students and three committees & 8.24 & 120 \\
2 & failed at teacher caching; fixed & -- & 120 \\
3 & froze in a diagnostic cell; fixed & -- & 120 \\
4 & specialist committee, controls, sharpest temperatures & 1.41 & 131 \\
5 & the five out-of-sample arms & 2.11 & 146 \\
6 & the size curve and its first controls & 3.52 & 170 \\
7 & the 35-epoch control and two more students & 3.16 & 179 \\
\midrule
 & total of the completed sessions & 18.44 & 179 \\
\bottomrule
\end{tabular}
\end{table}
"""

CH4 = "\n".join([
    "\\chapter{Datasets and Experimental Setup}", "\\label{ch:4}",
    wrap("Table~\\ref{tab:data} summarises the data. The primary benchmark is a cleaned fine-grained cyberbullying "
         "tweet corpus. A benchmark that labels implication, built from a single corpus, and inference-only probe "
         "sets measure what the tweet corpus cannot. An unlabelled transfer set, screened against all of them, "
         "supplies the out-of-sample text."),
    DATA_TABLE,
    "\\section{%s}\n\\label{sec:4.1}\n%s\n\n%s" % (heading("4.1 "), convert(body("4.1 ")),
                                                  wrap("Table~\\ref{tab:classes} gives the class distribution of the cleaned splits.") + "\n" + CLASS_TABLE),
    "\\section{%s}\n\\label{sec:4.2}\n%s" % (heading("4.2 "), convert(body("4.2 "))),
    "\\section{%s}\n\\label{sec:4.3}\n%s" % (heading("4.3 "), convert(body("4.3 "))),
    "\\section{%s}\n\\label{sec:4.4}\n%s\n\n%s" % (
        heading("4.4 "),
        convert(body("4.4 "), (), (),
                [("is regenerated by the run itself, with the\nsame seed, from the same public sources.",
                  "is regenerated by the run itself, with the same seed, from the same public sources; "
                  "[[Appendix:app:transfer]] says what each file records.")]),
                                                  wrap("Table~\\ref{tab:transfer} records every count.") + "\n" + TRANSFER_TABLE),
    "\\section{%s}\n\\label{sec:4.5}\n%s" % (heading("4.5 "), convert(body("4.5 "))),
    "\\section{%s}\n\\label{sec:4.6}\n%s" % (heading("4.6 "), convert(body("4.6 "))),
    "\\section{%s}\n\\label{sec:4.7}\n%s" % (heading("4.7 "), convert(body("4.7 "))),
    SESSIONS,
])

# --------------------------------------------------------------------------------- chapter 5
R = {
    "5.1 ": dict(subs=[("Three of the original four teachers beat the fine-tune-only headline student",
                        "[[Table:tab:teachers]] reports the teachers after task adaptation. Three of the original "
                        "four beat the fine-tune-only headline student")],
                 tables=[("Teachers after task adaptation, with the classical floor. The training loss is the last "
                          "epoch's cross-entropy on the split the students are distilled on; the probe columns are the "
                          "share of each probe the model assigns to any abusive class, which is its argmax "
                          "decision rather than a threshold on $p(\\text{abusive})$.", "tab:teachers")]),
    "5.2 ": dict(subs=[("Test macro-F1, mean over three seeds; classical floor 0.8798.",
                        "[[Table:tab:main]] gives the in-sample grid: test macro-F1 of every student under every "
                        "objective and committee, against a classical floor of 0.8798.")],
                 tables=[("In-sample distillation: test macro-F1 of the five students under each objective and "
                          "committee, mean over three seeds. The classical floor is 0.8798. Standard deviations "
                          "over seeds run from 0.0002 to 0.0078 and are in the generated table "
                          "\\texttt{paper/tables/main.csv}; the paired intervals are in "
                          "Appendix~\\ref{app:significance}.", "tab:main")]),
    "5.3 ": dict(tables=[("Per-instance weighting minus uniform averaging, with and without the DeBERTa teacher; "
                          "paired bootstrap 95\\% intervals.", "tab:dmthd-uniform"),
                         ("The weighting-temperature sweep on BERT-mini: mean weight of each teacher over the "
                          "training split, and test macro-F1 of the first seed.", "tab:tau")],
                 subs=[("Without the DeBERTa teacher four differences are negative and one is a tie;",
                        "[[Table:tab:dmthd-uniform]] sets the weighting against averaging on every student. Without "
                        "the DeBERTa teacher four differences are negative and one is a tie;"),
                       ("The sweep therefore reaches $\\tau = 0.05$.",
                        "The sweep therefore reaches $\\tau = 0.05$ ([[Table:tab:tau]], [[Figure:fig:tau]]).")],
                 inserts=[("| $\\tau$ |", FIG["tau"])]),
    "5.4 ": dict(subs=[("Scored directly from the teachers' saved test probabilities, before any student is involved:",
                        "[[Table:tab:committee]] scores the teachers' saved test probabilities directly, before any "
                        "student is involved."),
                       ("The teachers are not diverse enough for a weighting to have work to do, and adaptation is what made\nthem so.",
                        "The teachers are not diverse enough for a weighting to have work to do, and adaptation is "
                        "what made them so ([[Table:tab:agreement]], [[Figure:fig:agreement]]).")],
                 inserts=[("| Teacher pair |", FIG["agreement"])],
                 tables=[("Combining the teachers directly on the test set, before any student is trained: macro-F1 "
                          "of each combination rule and the oracle's accuracy.", "tab:committee"),
                         ("Agreement between the task-adapted teachers on the test set.", "tab:agreement")]),
    "5.5 ": dict(subs=[("committee only from 0.9526 to 0.9552. BERT-mini:",
                        "committee only from 0.9526 to 0.9552. [[Table:tab:specialist]] gives the specialist and its "
                        "two controls on the headline student."),
                       ("minus mean weight on the four targeted classes (uniform weight 0.25):",
                        "minus mean weight on the four targeted classes, against a uniform weight of 0.25 "
                        "([[Table:tab:routing]]).")],
                 tables=[("The implicit specialist and its controls on BERT-mini.", "tab:specialist"),
                         ("Routing contrast in the committee with the specialist: mean weight on "
                          "other\\_cyberbullying minus mean weight on the four targeted classes, with 95\\% bootstrap "
                          "intervals. The uniform weight is 0.25.", "tab:routing")]),
    "5.6 ": dict(),
    "5.7 ": dict(subs=[("Five arms\non BERT-mini, three seeds each, beside the in-sample runs of the same objectives; predictions written\nbefore the run (decision D19 in the repository's decision log):",
                        "[[Table:tab:oos]] reports five arms on BERT-mini, three seeds each, beside the in-sample runs "
                        "of the same objectives; the predictions were written before the run and are scored in "
                        "[[Appendix:app:predictions]].")],
                 tables=[("Out-of-sample distillation on BERT-mini with the 42,013-tweet transfer set: the same "
                          "objectives with and without the unlabelled rows, three seeds each. The labelled-split "
                          "column gives each objective's in-sample twin; for the out-of-sample-weighted arm that "
                          "twin is D-MTHD, because the weighting differs only where there are no labels.",
                          "tab:oos")]),
    "5.8 ": dict(tables=[("The size curve and its controls on BERT-mini, one teacher labelling, three seeds per arm. "
                          "Gains are paired bootstrap 95\\% intervals against fine-tuning.", "tab:size-curve"),
                         ("The largest transfer set on three students, three seeds each.", "tab:students")],
                 subs=[("**The curve rises at every step.**",
                        "**The curve rises at every step** ([[Table:tab:size-curve]], [[Figure:fig:size-curve]])."),
                       ("**Optimisation length does not explain the curve.**",
                        "**Optimisation length does not explain the curve** ([[Figure:fig:validation]])."),
                       ("**The endpoint holds on two more students.**",
                        "**The endpoint holds on two more students** ([[Table:tab:students]], [[Figure:fig:students]]).")],
                 inserts=[("| Arm | Transfer rows |", FIG["size"]),
                          ("**Optimisation length does not explain", FIG["validation"]),
                          ("| Student | Fine-tune only |", FIG["students"])]),
    "5.9 ": dict(tables=[("Probe metrics by model family, averaged over all 184 evaluated models, at each model's "
                          "argmax decision. The margin is recall minus false-positive rate.", "tab:families"),
                         ("The implicit benchmark: the compact student, the specialist and the classical floor, "
                          "one seed.", "tab:implicit")],
                 subs=[("**The same failure on a corpus that labels implication.**",
                        "**The same failure on a corpus that labels implication** ([[Table:tab:implicit]])."),
                       ("**Across the grid, methods differ only in how readily they fire.**",
                        "**Across the grid, methods differ only in how readily they fire** ([[Figure:fig:tradeoff]], [[Table:tab:families]])."),
                       ("**Nothing raises the discrimination.**",
                        "**Nothing raises the discrimination** ([[Figure:fig:auc]]).")],
                 inserts=[("**Across the grid, methods differ", FIG["tradeoff"]),
                          ("**Nothing raises the discrimination", FIG["auc"])]),
    "5.10 ": dict(subs=[("Synthetic obfuscation of the abusive test rows, BERT-mini, seed 1, macro-F1:",
                         "[[Table:tab:robustness]] gives macro-F1 under synthetic obfuscation of the abusive test "
                         "rows, on BERT-mini, seed 1."),
                        ("Median of five timed passes after warm-up on an idle machine:",
                         "[[Table:tab:efficiency]] gives the median of five timed passes after warm-up on an idle "
                         "machine.")],
                  tables=[("Macro-F1 under synthetic character-level obfuscation of the abusive test rows: "
                           "letters replaced by digits, adjacent characters swapped, spaces inserted, and all "
                           "three together. BERT-mini, seed 1.", "tab:robustness"),
                          ("Deployment profile of the students and the largest teacher: parameters, latency at "
                           "batch sizes 1 and 32, FLOPs per sequence, model size, and macro-F1 after INT8 dynamic "
                           "quantisation.", "tab:efficiency")]),
}
CH5 = ["\\chapter{Results and Analysis}", "\\label{ch:5}", convert(body("5. Results"))]
for k in ["5.1 ", "5.2 ", "5.3 ", "5.4 ", "5.5 ", "5.6 ", "5.7 ", "5.8 ", "5.9 ", "5.10 "]:
    spec = R[k]
    CH5.append("\\section{%s}\n\\label{sec:%s}\n%s" % (heading(k), k.strip(),
                                                      convert(body(k), spec.get("tables", ()), spec.get("inserts", ()),
                                                              spec.get("subs", ()))))
CH5 = "\n".join(CH5)

# --------------------------------------------------------------------------------- chapter 6
CH6_RQ = r"""
\section{Answers to the Research Questions}
\label{sec:6.1}
\begin{description}
  \item[RQ1. Does multi-teacher distillation with per-instance reliability weighting improve a compact
    cyberbullying detector over the same student fine-tuned without teachers?]
    Not in sample. Across five students, three committees and seven weighting temperatures from 0.05 to 5,
    no distilled student beats the same student fine-tuned without teachers by more than the test set can
    resolve, and per-instance weighting never differs from uniform averaging (Sections~\ref{sec:5.2}
    and~\ref{sec:5.3}). A teacher trained on labelled implication changes nothing either
    (Section~\ref{sec:5.5}).
  \item[RQ2. What mechanism explains the answer to RQ1?] The student imitates its teachers on the one
    split they were themselves fine-tuned on, and they have memorised it. There their soft labels are
    the gold labels, the reliability signal that weights them is saturated, and task adaptation has made them
    agree on more than nine tweets in ten (Sections~\ref{sec:5.4} and~\ref{sec:5.6}).
  \item[RQ3. Does distillation on unlabelled text the teachers have never seen improve the student, and
    how does the gain depend on the amount and composition of that text, on the number of optimisation
    updates, and on the student?] Yes. On BERT-mini the gain grows monotonically with the amount of
    unlabelled text, from +0.0010 at 5,000 tweets to +0.0123 [+0.0035, +0.0211] at 168,000. Generic
    tweets from the same platform work nearly as well as abuse-related ones, fine-tuning for the same
    number of updates gains nothing, and the gain holds on the BiLSTM (+0.0163) and on BERT-small
    (+0.0057, with an interval that includes zero). One teacher is enough: the committee, its weighting
    and hard pseudo-labels add nothing (Sections~\ref{sec:5.7} and~\ref{sec:5.8}).
  \item[RQ4. Does any of these interventions improve the student's ability to tell implied abuse from
    harmless sarcasm?] No. Every pre-trained variant of the student, in sample or out, ranks ironic abuse
    above benign sarcasm with an AUC between 0.73 and 0.78, and none rises above the in-sample band. The
    interventions change how readily the student fires, not what it can tell apart (Section~\ref{sec:5.9}).
\end{description}

\section{What the Findings Mean}
\label{sec:6.2}
"""

CH6_HISTORY = r"""
\section{How the Study Changed}
\label{sec:6.3}
The design of this thesis changed four times, and each change was forced by a measurement rather than
chosen in advance. Every change is recorded, with the evidence that forced it, in the decision log of
the project's repository.
\begin{enumerate}
  \item \textbf{From the Pre-Thesis II method to a controlled grid.} The Pre-Thesis II report distilled two
    teachers into DistilBERT on the Wikipedia corpus and reported the student above its teachers from one
    run, without a student trained without teachers. On the cleaned tweet corpus we added that control,
    three seeds and paired bootstrap intervals, and distillation then raised every student by less than
    the test set can resolve.
  \item \textbf{From dynamic weighting to its mechanism.} The per-instance weights sat close to uniform at
    the default temperature and barely sharpened at a twentieth of it, and the score did not move.
    Measuring the teachers explained why: they had memorised the training split and agreed with one
    another on more than nine tweets in ten.
  \item \textbf{From a specialist teacher to the implication audit.} A teacher trained on labelled
    implication was added to supply the missing expertise. After task adaptation it behaved almost exactly
    like HateBERT, and no student became better at telling implied abuse from harmless sarcasm; the
    threshold-free measure showed that ability fixed by pre-training.
  \item \textbf{From the audit to out-of-sample distillation.} If the teachers' soft labels equal the gold
    labels on the training split, the lever is text the teachers have not fitted. Three runs, each with
    its predictions written down beforehand, then measured the gain, its growth with the amount of text,
    and its survival under the matched-compute and second-student controls.
\end{enumerate}
The title changed with the evidence. The registered title named a dynamic, multi-teacher, homogeneous
and robust framework. The grid measured the first three of those properties and found none of them to
be the source of the gain; the fourth was tested only under synthetic character edits, which this
thesis does not call robustness. The final title names what the evidence supports.
"""

CH6 = "\n".join(["\\chapter{Discussion}", "\\label{ch:6}", CH6_RQ, convert(body("6. Discussion")), CH6_HISTORY])

# --------------------------------------------------------------------------------- chapter 7 and 8
FUTURE = r"""
\section{Future Work}
\label{sec:7.2}
\begin{enumerate}
  \item \textbf{Transfer text that carries implication.} The out-of-sample gain is a task gain because
    nothing in the transfer text marks implication. Composing the transfer set from text in which implied
    abuse is present and labelled, such as the Implicit Hate Corpus used here as a benchmark, is the
    direct test of whether distillation can teach it.
  \item \textbf{A second corpus and the remaining students.} The Wikipedia personal-attacks pipeline is
    built and was not run within the compute available; the size curve on DistilBERT and
    DeBERTa-v3-xsmall was not run either.
  \item \textbf{Where the curve saturates.} The gain was still rising at 168,000 tweets. Larger transfer
    sets would locate its ceiling and its cost.
  \item \textbf{Human validation of the probes.} A 300-item annotation of sarcastic abuse by four
    annotators, with Fleiss' kappa, would replace rule-built probes with verified ones.
  \item \textbf{A teacher that keeps its diversity.} A committee member whose knowledge was not obtained
    by fitting the training split, such as a large language model used zero-shot, is the one committee
    for which per-instance reliability could have something to express.
  \item \textbf{Bengali.} The work was motivated by abuse in Bengali social media, for which no sarcasm
    teacher is available; the out-of-sample recipe needs only unlabelled text and one teacher, which makes
    it the natural first step there.
\end{enumerate}
"""
CH7 = "\n".join(["\\chapter{Limitations and Future Work}", "\\label{ch:7}",
                 "\\section{Limitations}\n\\label{sec:7.1}", convert(body("7. Threats")), FUTURE])
CH8 = "\n".join(["\\chapter{Conclusion}", "\\label{ch:8}", convert(body("8. Conclusion"))])

# --------------------------------------------------------------------------------- appendices
PRED = [
    ("1", "5", "The committee with the transfer set beats fine-tuning by at least 0.010, interval excluding zero",
     "+0.0070 [$-$0.0001, +0.0144]", "failed"),
    ("2", "5", "The committee is a better labeller for the student than one teacher",
     "$-$0.0003 [$-$0.0061, +0.0056]", "failed"),
    ("3", "5", "Out-of-sample reliability weighting adds at most 0.003 over averaging", "+0.0021", "held"),
    ("4", "5", "Hard pseudo-labels gain less than soft labels", "+0.0020 against +0.0070", "held"),
    ("5", "5", "Sarcasm-discrimination AUC stays within 0.756 to 0.777 on every transfer arm",
     "0.727 to 0.765; three arms below", "failed"),
    ("6", "6", "The gain rises with size at fixed composition, and 42k reproduces +0.006 to +0.009",
     "+0.0010, +0.0035, +0.0043, +0.0070", "held"),
    ("7", "6", "42k generic tweets gain at least 0.003 less than 42k abuse-domain tweets, and generic rows "
               "beyond 42k add less per row", "0.0017 less; 0.4 against 1.7 per 10k", "in part"),
    ("8", "6", "Fine-tuning for the 42k arm's updates gains at most 0.002, and the 42k arm beats it by at least "
               "0.004", "+0.0027; +0.0043", "in part"),
    ("rule", "6", "If the 168k arm gains at least 0.012 with the interval clear of zero and still rising, the "
                  "constructive result leads", "+0.0123 [+0.0035, +0.0211], rising", "fired"),
    ("9", "7", "Fine-tuning for the 168k arm's updates gains at most 0.004, and the 168k arm keeps at least "
               "0.006 over it with the interval clear of zero", "$-$0.0013; +0.0136 [+0.0023, +0.0240]", "held"),
    ("10", "7", "BERT-small's 168k arm gains at least 0.005 over fine-tuning and beats its in-sample single "
                "teacher", "+0.0057; +0.0038, interval includes zero", "held"),
    ("11", "7", "The BiLSTM's 168k arm gains at least 0.010 over fine-tuning", "+0.0163 [+0.0063, +0.0271]", "held"),
    ("rule", "7", "If both controls hold, the result is stated for compact students in the plural",
     "both held", "fired"),
]
pred_rows = "\n".join("%s & %s & %s & %s & %s \\\\" % r for r in PRED)
APP_PRED = "\n\n".join([
    "\\chapter{Pre-Registered Predictions and Their Outcomes}", "\\label{app:predictions}",
    convert(body("Appendix D.")),
    wrap("Table~\\ref{tab:predictions} lists every prediction with the number that decided it. The session "
         "column refers to Table~\\ref{tab:sessions}."),
    r"""
{\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{p{0.8cm}p{0.9cm}p{7.0cm}p{3.6cm}p{1.2cm}}
\caption{Every prediction written before the out-of-sample runs, and its outcome.}
\label{tab:predictions}\\
\toprule
No. & Session & Prediction & Measured & Outcome \\
\midrule
\endfirsthead
\toprule
No. & Session & Prediction & Measured & Outcome \\
\midrule
\endhead
\bottomrule
\endfoot
<<ROWS>>
\end{longtable}
}
""".replace("<<ROWS>>", pred_rows)])

APP_HYPER = r"""
\chapter{Hyper-Parameters and Training Settings}
\label{app:hyper}
<<TEXT>>

Table~\ref{tab:hyper} collects them.

\begin{table}[htbp]
\centering
\small
\caption{Training settings of the teachers and the students.}
\label{tab:hyper}
\begin{tabular}{p{3.6cm}p{4.2cm}p{5.8cm}}
\toprule
Setting & Teachers & Students \\
\midrule
Epochs & 5 & 6; 13 and 35 for the matched-steps controls \\
Learning rate & $2 \times 10^{-5}$ & $3 \times 10^{-5}$; $10^{-3}$ for the BiLSTM \\
Batch size & 32 & 32 \\
Schedule & 10\% warm-up, linear decay & 10\% warm-up, linear decay \\
Gradient clipping & 1.0 & 1.0 \\
Model selection & best validation macro-F1 epoch & best validation macro-F1 epoch; early stopping with patience 2, off for the matched-steps controls \\
Maximum length & 128 tokens & 128 tokens \\
Precision & mixed, on GPU & mixed, on GPU \\
Distillation & -- & $\alpha = \beta = 0.4$, $\gamma = 0.2$, $\delta = 0.3$, $T = 4$, $\tau = 1$ \\
Seeds & one run per teacher & 1, 2, 3; seed 1 alone for ablations and sweeps \\
Weight decay & 0.01 (AdamW) & 0.01 (AdamW) \\
\bottomrule
\end{tabular}
\end{table}
""".replace("<<TEXT>>", convert(body("Appendix C.")))

sig_rows = []
with open(os.path.join(REPO, "paper", "tables", "significance.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        sig_rows.append(" & ".join(inline(x) for x in (r["Student"], r["Question"], r["Baseline"], r["Candidate"],
                                                        r["Seeds"], r["Difference"], r["95 per cent interval"],
                                                        r["Interval excludes zero"])) + " \\\\")
APP_SIG = "\n\n".join([
    "\\chapter{All Paired Comparisons}", "\\label{app:significance}",
    convert(body("Appendix A.")),
    wrap("Table~\\ref{tab:significance} lists all %d paired comparisons generated by the significance script, "
         "each a paired bootstrap over the 4,326 test tweets with seeds paired by index and 1,000 resamples per "
         "seed." % len(sig_rows)),
    r"""
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{p{1.5cm}p{2.9cm}p{2.5cm}p{2.6cm}cp{1.0cm}p{1.8cm}c}
\caption{Every paired comparison in the grid: candidate minus baseline test macro-F1.}
\label{tab:significance}\\
\toprule
Student & Question & Baseline & Candidate & Seeds & Difference & 95\% interval & Excl.\ 0 \\
\midrule
\endfirsthead
\toprule
Student & Question & Baseline & Candidate & Seeds & Difference & 95\% interval & Excl.\ 0 \\
\midrule
\endhead
\bottomrule
\endfoot
<<ROWS>>
\end{longtable}
}
""".replace("<<ROWS>>", "\n".join(sig_rows))])

APP_TRANSFER = "\n".join(["\\chapter{The Transfer Set}", "\\label{app:transfer}", convert(body("Appendix B."))])

STAT_REST = "\n\n".join(block_tex(b, []) for b in STATEMENTS[1:]
                        if not " ".join(b).startswith("**Author contributions"))
APP_STATEMENTS = "\n".join(["\\chapter{Data and Reproducibility Statements}", "\\label{app:statements}", STAT_REST])

# --------------------------------------------------------------------------------- main.tex and the bibliography
MAIN = r"""
% Build with pdfLaTeX and BibTeX, for example on Overleaf (set main.tex as the main document) or with
%   pdflatex main && bibtex main && pdflatex main && pdflatex main
% Generated by build_tex.py from paper/PAPER.md; edit PAPER.md or build_tex.py and rebuild rather than
% editing the chapter files by hand, or the next build will overwrite the edit.
\documentclass[12pt,a4paper,oneside]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2.54cm]{geometry}
\usepackage{setspace}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{longtable}
\usepackage{array}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[section]{placeins}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}

% natbib's bibliography heading, listed in the contents like the other unnumbered chapters
\renewcommand{\bibsection}{\chapter*{\bibname}\phantomsection\addcontentsline{toc}{chapter}{\bibname}}

\newcommand{\thesisdate}{October 2026}
\newcommand{\thesissemester}{Fall, 2026}    % CHECK: the semester the department records for this defense
\newcommand{\defensedate}{October 03, 2026}

\setlist{itemsep=2pt,topsep=4pt}

\begin{document}
\input{frontmatter}

\clearpage
\pagenumbering{arabic}
\setcounter{page}{1}
\onehalfspacing
\input{chapters/ch1_introduction}
\input{chapters/ch2_literature}
\input{chapters/ch3_methodology}
\input{chapters/ch4_setup}
\input{chapters/ch5_results}
\input{chapters/ch6_discussion}
\input{chapters/ch7_limitations}
\input{chapters/ch8_conclusion}

\singlespacing
\bibliographystyle{unsrtnat}
\bibliography{references}

\appendix
\input{appendices/a_predictions}
\input{appendices/b_hyperparameters}
\input{appendices/c_transfer}
\input{appendices/d_significance}
\input{appendices/e_statements}
\end{document}
"""

bib = io.open(os.path.join(REPO, "paper", "references.bib"), encoding="utf-8").read()
bib = bib.replace("\u00f6", "{\\\"o}")
bib = re.sub(r"\n\s*note\s*=\s*\{Open access\. Reviewed[^\n]*\}\n", "\n", bib)

# unsrtnat sets titles in sentence case, lowercasing every letter BibTeX is not told to keep: brace the
# words that carry capitals of their own (BERT, HateBERT, SemEval, AI) and the proper nouns.
PROPER_NOUNS = {"English", "Arabic", "Twitter"}


def protect_title(t):
    out, depth, i = [], 0, 0
    while i < len(t):
        c = t[i]
        if c in "{}":
            depth += 1 if c == "{" else -1
            out.append(c)
            i += 1
        elif depth == 0 and c.isalnum():
            j = i
            while j < len(t) and (t[j].isalnum() or t[j] == "'"):
                j += 1
            word = t[i:j]
            keep = any(ch.isupper() for ch in word[1:]) or word in PROPER_NOUNS
            out.append("{%s}" % word if keep else word)
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


bib = re.sub(r"(?m)^(\s*title\s*=\s*\{)(.*)(\},?)\s*$", lambda m: m.group(1) + protect_title(m.group(2)) + m.group(3), bib)
assert all(ord(c) < 128 for c in bib), "non-ASCII left in the bibliography"
assert "Reviewed 21 September" not in bib

write("main.tex", MAIN)
write("frontmatter.tex", FRONT)
write("chapters/ch1_introduction.tex", CH1)
write("chapters/ch2_literature.tex", CH2)
write("chapters/ch3_methodology.tex", CH3)
write("chapters/ch4_setup.tex", CH4)
write("chapters/ch5_results.tex", CH5)
write("chapters/ch6_discussion.tex", CH6)
write("chapters/ch7_limitations.tex", CH7)
write("chapters/ch8_conclusion.tex", CH8)
write("appendices/a_predictions.tex", APP_PRED)
write("appendices/b_hyperparameters.tex", APP_HYPER)
write("appendices/c_transfer.tex", APP_TRANSFER)
write("appendices/d_significance.tex", APP_SIG)
write("appendices/e_statements.tex", APP_STATEMENTS)
write("references.bib", bib)
