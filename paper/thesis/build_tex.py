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


# Short names for the lists of figures and tables; the full caption stays under the float.
SHORT = {
    "fig:schematic": "In-sample against out-of-sample distillation",
    "fig:tau": "The weighting-temperature sweep on BERT-mini",
    "fig:agreement": "Pairwise agreement between the task-adapted teachers",
    "fig:size-curve": "The size curve on BERT-mini",
    "fig:validation": "Validation macro-F1 against optimisation updates",
    "fig:students": "The largest transfer set on three students",
    "fig:auc": "Sarcasm-discrimination AUC of every analysed variant",
    "fig:tradeoff": "False-positive rate against recall, all 184 models",
    "tab:transfer": "Construction of the transfer set",
    "tab:teachers": "Teachers after task adaptation",
    "tab:main": "In-sample distillation on the five students",
    "tab:dmthd-uniform": "Per-instance weighting minus uniform averaging",
    "tab:tau": "Teacher weights and score across the temperature sweep",
    "tab:committee": "Combining the teachers directly on the test set",
    "tab:routing": "Routing contrast in the committee with the specialist",
    "tab:oos": "Out-of-sample distillation on BERT-mini",
    "tab:size-curve": "The size curve and its controls",
    "tab:families": "Probe metrics by model family",
    "tab:robustness": "Macro-F1 under character-level obfuscation",
    "tab:efficiency": "Deployment profile of the students and the largest teacher",
}


def cap_tex(caption, label):
    """\\caption[short]{full} where a short name exists, plain \\caption otherwise."""
    short = SHORT.get(label)
    return "\\caption[%s]{%s}" % (short, caption) if short else "\\caption{%s}" % caption


def table_tex(block, caption, label, font="\\small"):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block]
    header, body = rows[0], rows[2:]
    n = len(header)
    assert all(len(r) == n for r in body), (label, [len(r) for r in body])
    lines = [" & ".join(inline(c) for c in r) + " \\\\" for r in [header] + body]
    tab = "\n".join(["\\begin{adjustbox}{max width=\\textwidth}",
                     "\\begin{tabular}{%s}" % ("l" + "c" * (n - 1)), "\\toprule", lines[0], "\\midrule"]
                    + lines[1:] + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"])
    return "\n".join(["\\begin{table}[htbp]", "\\centering", font, cap_tex(caption, label),
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
                      cap_tex(caption, label), "\\label{%s}" % label, "\\end{figure}"])


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
\begin{tabular}{@{}lp{11.6cm}@{}}
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

This thesis builds that method in full and measures it. Dynamic Multi-Teacher Homogeneous Knowledge
Distillation (D-MTHD) is a committee of task-adapted teachers whose softened outputs are weighted per
instance by how reliable each teacher is on that instance, aligned to the student through learned
projections of their hidden states, with an auxiliary head for irony. We evaluate it on a cleaned
fine-grained cyberbullying corpus, extend it to sarcastic and implicit abuse, and test it against the
control this literature omits, the same student trained with no teacher at all. What the measurements
find changes the method.

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
trained without a teacher, and without repeated runs or confidence intervals. A distillation method that is not shown to beat that control cannot be said to work, and
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

\subsection{The compact student}
A student is \emph{pre-trained} when its weights come from self-supervised training on unlabelled text
before any distillation, and \emph{randomly initialised} when they do not. The distinction carries more
of this thesis's variance than any part of the distillation objective (Section~\ref{sec:5.3}). Students
are named here by parameter count: the BERT-mini and BERT-small checkpoints of Turc et al.
\cite{turc2019wellread} at 11.2M and 28.8M, DistilBERT at 67.0M \cite{sanh2019distilbert},
DeBERTa-v3-xsmall at 70.8M \cite{he2023debertav3}, and a two-layer BiLSTM at 10.4M
\cite{tang2019distilling}. A student is \emph{homogeneous} with its teachers when it shares their
architecture family and \emph{heterogeneous} when it does not, a distinction that matters because the
hidden-state term of Section~\ref{sec:3.3} assumes a state worth projecting.
"""

CH2_CORPORA = r"""
\subsection{What each corpus can support}
The corpora differ in what a result on them is allowed to mean, and the differences decide the design of
Chapter~\ref{ch:4}. The fine-grained cyberbullying corpus \cite{wang2020sosnet} labels the target of the
abuse, age, ethnicity, gender or religion, with two residual classes for abuse that names no group and
for text that is not abusive at all. It is large and its four targeted classes are close to separable by
vocabulary, so an aggregate macro-F1 on it is dominated by classes a lexical model already solves; the
work is in the two residual classes, and Section~\ref{sec:5.8} shows that this is exactly where the
out-of-sample gain lands. Its published version repeats many tweets, some under more than one label, and
the cleaning of Section~\ref{sec:4.1} removes them.

The corpora of Davidson et al. \cite{davidson2017automated}, OLID \cite{zampieri2019olid}, HatEval
\cite{basile2019hateval} and Founta et al. \cite{founta2018large} are the standard sources of
abuse-domain tweets. Their label schemes differ from one another and from ours, which is why this thesis
uses them as text and discards their labels (Section~\ref{sec:4.4}). Borkan et al.
\cite{borkan2019nuanced} show how much a model's apparent quality depends on which subgroups the
evaluation set contains, and Van Hee et al. \cite{vanhee2018automatic} document how duplicate-heavy
corpora inflate held-out scores; both are reasons the splits here are asserted disjoint on normalised
text rather than on identity.

For implication two corpora annotate the distinction directly. The Implicit Hate Corpus
\cite{elsherief2021latent} separates not-hate, explicit hate and implicit hate and adds a six-category
taxonomy of how the implication is carried; ISHate \cite{ocampo2023indepth} separates hate speech into
explicit and subtle layers over a partly overlapping pool of texts. Section~\ref{sec:4.2} reports that
the two disagree on which of their shared hateful texts are implicit almost half the time, and that a
classifier can tell the two corpora apart at 0.91 macro-F1, which is why the benchmark here is built
from one of them and the other is held out whole. ToxiGen \cite{hartvigsen2022toxigen} generates
implicit examples adversarially and HateXplain \cite{mathew2021hatexplain} adds token-level rationales;
neither is used for training here, ToxiGen because its rows reach ISHate and would break the screen.
Sarcasm is supplied by iSarcasm and iSarcasmEval \cite{oprea2020isarcasm, abufarha2022semeval}, whose
labels come from the authors of the posts rather than from third-party annotators, which is what makes
them usable as a benign-sarcasm control.
"""

CH2_DISTIL_LINE = r"""
\subsection{The line of compact BERT students}
Five systems define what is achievable for a compact encoder, and each fixes a different variable.
DistilBERT \cite{sanh2019distilbert} distils during pre-training into a six-layer student and reports
about 97 per cent of BERT-base's language-understanding score at 40 per cent fewer parameters; the
knowledge is general rather than task-specific, and no committee is involved. Patient Knowledge
Distillation \cite{sun2019patient} has the student learn from several of the teacher's intermediate
layers rather than its output alone, which is the ancestor of the hidden-state term audited in
Section~\ref{sec:3.3}. TinyBERT \cite{jiao2020tinybert} distils in two stages, general then
task-specific, and reports about 96.8 per cent of BERT-base on GLUE from a four-layer student; its
task stage augments the training data heavily, so the reported gain mixes the distillation with the
extra text, a confound Section~\ref{sec:5.8} separates deliberately. MiniLM \cite{wang2020minilm}
distils the teacher's self-attention relations instead of its outputs, and MobileBERT
\cite{sun2020mobilebert} redesigns the student around bottleneck blocks and distils from a specially
built teacher; both change the architecture rather than the transfer set, and neither is tested here.

Turc et al. \cite{turc2019wellread} is the result this thesis's grid most nearly reproduces. They
separate pre-training the compact student from distilling into it, and find that the former carries more
of the final score than the recipe used for the latter, while also treating the size and provenance of
the unlabelled transfer data as variables in their own right rather than as a fixed setting. Our
ablation agrees on the first point by a wide margin: removing pre-training costs 0.047 macro-F1, against
at most 0.0014 for removing any component of the distillation objective (Section~\ref{sec:5.3}), and it
is the only in-sample difference in the whole grid whose interval excludes zero. The second point is the
subject of Section~\ref{sec:5.8}.

Tang et al. \cite{tang2019distilling} distil BERT into a single-layer BiLSTM on rule-augmented text and
report a student competitive with far larger models, which is the precedent for the BiLSTM student used
here and, with TinyBERT, for the observation that the recipes reporting the largest gains for small
students are the ones whose transfer set is larger than the labelled data.
"""

CH2_TRANSFER_SECTION = r"""
\section{The Transfer Set, and What Is Known About It}
\label{sec:2.transfer}
The transfer set is the variable this thesis ends up measuring, and it is older than the multi-teacher
literature that displaced it. Buciluă et al. \cite{bucilua2006model} compress an ensemble by using it to
label a large pool of \emph{synthesised, unlabelled} data and training one compact model on the result;
the gain, in their account, comes from having the ensemble's function sampled at many points rather than
from the ensemble itself. Hinton et al. \cite{hinton2015distilling} restate this for soft targets and
say explicitly that the transfer set need not be the labelled training data, and that an unlabelled pool
can be used where one is available. Both statements are about \emph{where the teacher's function is
sampled}, and neither is a statement about how many teachers do the sampling.

Three later results say why the sampling point matters. Stanton et al. \cite{stanton2021does} show that
students frequently fail to match their teacher's function even when they have the capacity to do so,
and that improved generalisation and improved fidelity to the teacher come apart: a student can score
better while agreeing with its teacher no more closely, which is what makes ``the student beat its
teacher'' a weak claim. Cho and Hariharan \cite{cho2019efficacy} show that a larger teacher is not
reliably a better one, because the gap in capacity between teacher and student can itself be the
obstacle. Beyer et al. \cite{beyer2022knowledge} give the account this thesis's results fit most
closely: distillation is function matching, the teacher's function has to be sampled consistently and at
many inputs, and the schedules that work are long. A function is learned where it is sampled; sampling
it only where it already equals the gold label teaches the gold label.

What none of this work provides for the present task is a measurement. The transfer sets in these papers
are augmented copies of the task data \cite{jiao2020tinybert, tang2019distilling}, or unlabelled pools
whose size and composition are fixed rather than varied \cite{turc2019wellread}, and the domain is not
abusive language. Nor is the comparison that would isolate the transfer set, the same student trained
for the same number of updates without it, the comparison those papers are built to make.
Sections~\ref{sec:5.7} and~\ref{sec:5.8} make it: six nested sizes over a 34-fold range, a composition
control at fixed size, two matched-update controls, and three students.
"""

CH2_MULTI_TEACHER = r"""
\subsection{How the weights are computed, and on what data}
The family divides by where each teacher's weight comes from. You et al. \cite{you2017learning} average
the teachers' softened outputs and add a term matching the relative dissimilarity of their intermediate
representations; the combination is fixed rather than per instance. Fukuda et al.
\cite{fukuda2017efficient} avoid combining at all and train each batch against one randomly chosen
teacher, on the argument that an average smooths away exactly the disagreement that carries information.
Liu et al. \cite{liu2020adaptive} weight teachers adaptively and at several representation levels; Du et
al. \cite{du2020agree} combine teachers in gradient space so that conflicting gradients are resolved
rather than averaged; Yuan et al. \cite{yuan2021reinforced} learn a selector by reinforcement learning
that picks which teachers to listen to per instance. Zhang et al. \cite{zhang2022confidence} weight by
each teacher's confidence, and Zou et al. \cite{zou2025dynamic} apply the same idea to semantic parsing.

MT-BERT \cite{wu2021one} is the closest of these to the method audited here: several pre-trained
language-model teachers are co-fine-tuned on the task, their soft labels are weighted per instance by
each teacher's prediction loss, and their hidden states are aligned to the student through learned
projections. D-MTHD is that combination, with the weight read through a softmax at a temperature and an
auxiliary head added for irony (Section~\ref{sec:3.3}); we do not claim the combination as novel.
The heterogeneous student here is trained without the information-flow model of Passalis et al.
\cite{passalis2020heterogeneous}, and Section~\ref{sec:7.1} names that as a confound rather than a
result.

Two assumptions run through all of it. The first is that the teachers collectively hold more than any one
of them \emph{in a way their outputs reveal}, so that a rule reading only their probabilities can
recover it. The second is that reliability scored against the gold label still means something after the
teachers have been fine-tuned on the data the score is read from. Neither assumption is tested in the
papers above, and the reason is structural rather than careless: a weighting scheme is compared against
other weighting schemes, so a saturation that affects all of them equally never appears.
Section~\ref{sec:5.4} tests the first assumption by scoring every combination rule directly on the
teachers' saved probabilities, including a stacked gate fitted by cross-validation, and bounds what any
such rule can gain. Section~\ref{sec:5.6} tests the second by reporting the teachers' training losses on
the split the weights are read from.
"""

CH2_ABUSIVE_SECTION = r"""
\section{Multi-Teacher Distillation Applied to Abusive Language}
\label{sec:2.abusive}
One published system applies this family to the task of this thesis. Prasomphan \cite{prasomphan2025mtkd}
detects cyberbullying in Thai social-media text by distilling three transformer teachers, ThaiBERT,
WangchanBERTa and mT5, each fine-tuned on the task data, into a gradient-boosted tree student. The
teachers' soft probabilities are combined into one distilled distribution by a weighted sum, the student
reads sentence embeddings reduced by principal components, and its objective adds a Kullback--Leibler
term against the combined soft target to the supervised cross-entropy. Accuracies of 92.5, 90.5 and 91.0
per cent are reported on three Thai corpora repurposed from sentiment annotation.

The design is the recipe this thesis audits, in this application, and the differences between what it
reports and what it establishes are the reason the audit exists. The teachers are fine-tuned on the same
data the student is then distilled on, so the soft labels are read where Section~\ref{sec:5.6} shows
them to be saturated. No student trained on the same features without teachers is reported, so the
quantity the reader most needs, what the teachers added, is not available; the comparisons offered are
against other classifiers rather than against the same classifier without distillation. The weights of
the combination are not specified as a function of the data, so the term ``multi-teacher'' covers both a
fixed average and a learned rule. Single runs are reported without seeds or intervals, and accuracy on
corpora repurposed from sentiment labels is a measure the class balance can carry on its own.

The system is nonetheless the right comparison to make, and it raises one question this thesis's grid
does not answer. Its student is not a neural network, so the hidden-state term has nothing to align and
the student cannot inherit anything from pre-training; every claim in Section~\ref{sec:5.10} about
pre-training setting the ceiling is silent about such a student. Whether a tree student on frozen
embeddings behaves like the compact encoders measured here is a question the design of this thesis
supports and does not settle.
"""

CH2_EVALUATION = r"""
\subsection{What a test set of this size can decide}
Card et al. \cite{card2020power} compute the statistical power of common NLP benchmarks and find that
most are too small to detect the differences routinely reported on them, so that a published improvement
is frequently indistinguishable from resampling noise. The consequence for a thesis is a number that has
to be computed before any result is read: on the 4,326-post test set used here a paired bootstrap
interval is about 0.014 wide, so a difference under roughly 0.007 macro-F1 cannot be separated from
zero. Every in-sample distillation effect in this grid is smaller than that, which is a fact about the
measurement and not only about the method, and Section~\ref{sec:7.1} states it as a limitation in both
directions: an absence at this resolution is not a zero.

Dodge et al. \cite{dodge2019show} argue that reported scores are uninterpretable without the search
budget that produced them, since the best of many configurations is a biased estimate of any one of them;
Section~\ref{sec:5.2} reports the best of six per student and says so in the same sentence. Koehn
\cite{koehn2004statistical} and Dror et al. \cite{dror2018hitchhiker} give the paired bootstrap used
throughout, and Guo et al. \cite{guo2017calibration} the calibration error reported for the teachers.
Röttger et al. \cite{rottger2021hatecheck} build functional tests for hate-speech models, held-out sets
constructed to isolate one capability rather than to sample the distribution, with human validation of
every item; the probes of Section~\ref{sec:4.3} are in that spirit and are rule-built rather than
validated, which Section~\ref{sec:7.1} records.

The annotation literature adds the boundary case. Uma et al. \cite{uma2021learning}, Davani et al.
\cite{davani2022dealing}, Plank \cite{plank2022problem} and Peterson et al. \cite{peterson2019human}
argue that disagreement between annotators is signal rather than noise, and that a single majority label
discards it. For this task the texts annotators disagree on are disproportionately the ones whose
hostility is implied, which is the same population the oracle analysis of Section~\ref{sec:5.4} finds
the committee's unreachable headroom to be made of.
"""

CH2_GAP = r"""
\section{Summary and Research Gap}
\label{sec:gap}
Table~\ref{tab:gap} reads the works this chapter has reviewed one at a time: what each does, what it
reports, what limits it for the question of this thesis, and where the thesis takes it up. The pattern
the table makes is the argument of Chapter~\ref{ch:5}. The multi-teacher rows report gains over other
weighting schemes and none reports the comparison against a student trained with no teacher at all. The
transfer-set rows establish that the text a student imitates on is a variable, and none varies it in
this domain with the number of updates held fixed. The implication rows establish that the label exists
and is contested, and none asks whether distillation can move it. The evaluation rows give the
resolution every one of those claims should have been read against.

\begin{center}
{\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{p{2.35cm}p{3.2cm}p{2.4cm}p{3.0cm}p{2.4cm}}
\caption[What the literature establishes and what it leaves open]{What each work establishes, what it
reports, what limits it for this thesis, and where the thesis takes it up.}
\label{tab:gap}\\
\toprule
Work & Method & Reported & Limitation for this task & Taken up in \\
\midrule
\endfirsthead
\toprule
Work & Method & Reported & Limitation for this task & Taken up in \\
\midrule
\endhead
\bottomrule
\endfoot
\multicolumn{5}{l}{\emph{Foundations}} \\*
Buciluă et al.\ \cite{bucilua2006model} & Label a large unlabelled pool with an ensemble and train one
compact model on it & A compact model approaching the ensemble it was trained from & Pre-neural, no soft
targets, and the pool is synthesised rather than in-domain & The claim that the gain comes from
labelling new text, Section~\ref{sec:5.7} \\
Hinton et al.\ \cite{hinton2015distilling} & Match temperature-softened teacher outputs; the transfer
set need not be the labelled data & Ensembles compressed into single models & States that the transfer
set may be unlabelled; does not measure how much of it, or which text & The KL term and its $T^2$
scaling, Section~\ref{sec:3.3} \\
Romero et al.\ \cite{romero2015fitnets} & Regress the student's intermediate layer onto the teacher's
through a learned projection & Thin, deep students trainable where output matching alone fails & Shown
for narrow convolutional nets, not for an encoder that is already pre-trained & The form of
$\mathcal{L}_{\mathrm{hid}}$ and the ablation that measures it, Section~\ref{sec:5.3} \\
\midrule
\multicolumn{5}{l}{\emph{Compact language models}} \\*
Sanh et al.\ \cite{sanh2019distilbert} & Distil during pre-training into a six-layer student & About 97
per cent of BERT-base at 40 per cent fewer parameters & General rather than task-specific; no committee,
no transfer-set variation & The 67.0M student, Section~\ref{sec:4.5} \\
Sun et al.\ \cite{sun2019patient} & The student learns from several intermediate teacher layers & Gains
over output-only distillation & Measured in sample throughout & The hidden-state term,
Section~\ref{sec:3.3} \\
Jiao et al.\ \cite{jiao2020tinybert} & Two-stage distillation, general then task-specific, with heavy
data augmentation & About 96.8 per cent of BERT-base from a four-layer student & The augmentation and
the distillation are varied together & That confound separated, Section~\ref{sec:5.8} \\
Wang et al.\ \cite{wang2020minilm}; Sun et al.\ \cite{sun2020mobilebert} & Distil attention relations;
rebuild the student around bottleneck blocks & Compact students near teacher quality & Architecture-level
changes, orthogonal to the transfer set & Named, not tested, Section~\ref{sec:7.2} \\
Turc et al.\ \cite{turc2019wellread} & Separate pre-training the compact student from distilling into
it & Pre-training the student carries more than the recipe & Not measured for abusive language; the
unlabelled pool is held fixed & Reproduced: pre-training is worth 0.047 here,
Section~\ref{sec:5.3} \\
Tang et al.\ \cite{tang2019distilling} & BERT into a single-layer BiLSTM on rule-augmented text & A
small BiLSTM competitive with far larger models & The gain is confounded with the volume of augmented
text & The BiLSTM student, Sections~\ref{sec:4.5} and~\ref{sec:5.8} \\
\midrule
\multicolumn{5}{l}{\emph{What is known about distillation itself}} \\*
Stanton et al.\ \cite{stanton2021does} & Measure fidelity to the teacher separately from
generalisation & Students often fail to match teachers they have the capacity to match & Vision and
general NLP rather than this task & Why a student scoring above its teacher is a weak claim,
Section~\ref{sec:5.2} \\
Cho and Hariharan \cite{cho2019efficacy} & Vary the teacher's capacity against the student's & A larger
teacher is not reliably a better one & No committee; one teacher at a time & Why a 335M teacher and an
11M student need not pair well, Section~\ref{sec:6.2} \\
Beyer et al.\ \cite{beyer2022knowledge} & Distillation as function matching: consistent views, long
schedules & Large gains from sampling the teacher's function widely & Vision; schedule length and data
are varied together & The account the size curve fits, and the matched-updates control,
Section~\ref{sec:5.8} \\
\midrule
\multicolumn{5}{l}{\emph{Multi-teacher and adaptive weighting}} \\*
You et al.\ \cite{you2017learning} & Average several teachers' soft outputs, plus a dissimilarity term &
A committee beats its members & Fixed combination; no per-instance reliability; no no-teacher control &
The uniform-averaging arm, Section~\ref{sec:5.2} \\
Fukuda et al.\ \cite{fukuda2017efficient} & Train each batch against one randomly chosen teacher &
Avoids averaging away disagreement & Not compared against a student trained without teachers & Bounded
by the combination table, Section~\ref{sec:5.4} \\
Liu et al.\ \cite{liu2020adaptive}; Du et al.\ \cite{du2020agree} & Adaptive multi-level weighting;
combination in gradient space & Improvements over uniform averaging & The weights are read on the data
the teachers were fitted on & The saturation that explains it, Section~\ref{sec:5.6} \\
Yuan et al.\ \cite{yuan2021reinforced} & Reinforcement learning selects teachers per instance & A
learned selector beats fixed rules & Needs a held-out reward; its cost is not reported against the
gain & A stacked gate bounds any selector here, Section~\ref{sec:5.4} \\
Zhang et al.\ \cite{zhang2022confidence}; Zou et al.\ \cite{zou2025dynamic} & Weight teachers per
instance by their confidence & Gains on classification and on parsing & Confidence is saturated on the
split the teachers were fine-tuned on & Measured directly, Sections~\ref{sec:5.3}
and~\ref{sec:5.6} \\
Wu et al.\ \cite{wu2021one} & Co-fine-tune several teachers, weight by prediction loss, align hidden
states through projections & Multi-teacher above single-teacher distillation & Teachers fine-tuned on
the split the student is then distilled on; no no-teacher control & The method audited here,
Section~\ref{sec:3.3} and Sections~\ref{sec:5.2} to~\ref{sec:5.6} \\
Passalis et al.\ \cite{passalis2020heterogeneous} & Model information flow when teacher and student
architectures differ & Heterogeneous transfer improved & Our heterogeneous student is trained without
such a model & Named as a confound, Section~\ref{sec:7.1} \\
\midrule
\multicolumn{5}{l}{\emph{This task}} \\*
Prasomphan \cite{prasomphan2025mtkd} & Three Thai transformer teachers, weighted soft labels, a
gradient-boosted tree student & 92.5, 90.5 and 91.0 per cent accuracy on three Thai corpora & In-sample
teachers; no student without teachers; single runs; labels repurposed from sentiment & The closest
published instance of what is audited here, Section~\ref{sec:2.abusive} \\
Wang et al.\ \cite{wang2020sosnet} & A fine-grained cyberbullying corpus in six classes & Strong
fine-tuned encoders on aggregate F1 & Repeated and conflicting rows; implication is not a label & The
primary benchmark, after cleaning, Section~\ref{sec:4.1} \\
ElSherief et al.\ \cite{elsherief2021latent}; Ocampo et al.\ \cite{ocampo2023indepth} & Annotate
implicit against explicit hate & Models competent on explicit hate lose most of it on implicit & The two
disagree on nearly half of their shared hateful texts & The single-source benchmark and the bound it
carries, Section~\ref{sec:4.2} \\
Caselli et al.\ \cite{caselli2021hatebert}; Barbieri et al.\ \cite{barbieri2020tweeteval} &
Domain-specialised encoders for abuse and for tweets & Better starting points than general BERT & Their
specialisation is spent by task adaptation & The teachers, Sections~\ref{sec:4.5} and~\ref{sec:5.4} \\
\midrule
\multicolumn{5}{l}{\emph{Evaluation practice}} \\*
Röttger et al.\ \cite{rottger2021hatecheck} & Functional tests isolating one capability each, validated
by annotators & Held-out accuracy hides specific failures & Built for hate speech in general, not for
implication against sarcasm & The probes, Section~\ref{sec:4.3}; ours are rule-built,
Section~\ref{sec:7.1} \\
Card et al.\ \cite{card2020power} & Compute the statistical power of common benchmarks & Most are too
small for the differences reported on them & Not applied in this application & The 0.007 resolution
every claim here is read against, Section~\ref{sec:4.7} \\
\end{longtable}
}
\end{center}
"""

CH2 = "\n".join([
    "\\chapter{Literature Review}", "\\label{ch:2}",
    wrap("This chapter sets out the concepts the rest of the thesis relies on, then reads six lines of "
         "work. The first two are the corpora this task is measured on, and the separate literature on "
         "abuse carried by implication. The next two are the line of compact language models, and what "
         "is known about the transfer set, which is the variable this thesis ends up measuring. The "
         "last two are multi-teacher and adaptive weighting, the family the audited method belongs to "
         "together with the one published system that applies it to this task, and evaluation practice, "
         "which supplies the resolution every claim is read against. The chapter ends with a table that "
         "reads each work for its method, its reported result, what limits it here, and where this "
         "thesis takes it up."),
    CH2_PRELIM,
    "\\section{%s}\n\\label{sec:2.1}\n%s" % (heading("2.1 "), convert(body("2.1 "))),
    CH2_CORPORA,
    "\\section{%s}\n\\label{sec:2.2}\n%s" % (heading("2.2 "), convert(body("2.2 "))),
    CH2_DISTIL_LINE,
    CH2_TRANSFER_SECTION,
    "\\section{%s}\n\\label{sec:2.3}\n%s" % (heading("2.3 "), convert(body("2.3 "))),
    CH2_MULTI_TEACHER,
    CH2_ABUSIVE_SECTION,
    "\\section{%s}\n\\label{sec:2.4}\n%s" % (heading("2.4 "), convert(body("2.4 "))),
    CH2_EVALUATION,
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


\textbf{What it costs.} Counting one forward-and-backward pass of the student over one text as the
unit, and writing $c_k$ for the cost of one forward pass of teacher $k$ in the same unit, the
procedure divides into work done once and work done every epoch. Adapting the teachers is
$O\!\left(E_t \sum_k |\mathcal{D}|\, c_k\right)$ for $E_t = 5$ epochs, and caching their outputs over
$\mathcal{D} \cup \mathcal{U}$ is one further pass each, $O\!\left((|\mathcal{D}| + |\mathcal{U}|)
\sum_k c_k\right)$; both are paid once and reused by every student. An out-of-sample run adds the
neighbour search of equation (3.7), which is $O(K |\mathcal{V}| \bar d)$ to embed the validation split
and $O\!\left(K (|\mathcal{D}| + |\mathcal{U}|) |\mathcal{V}| \bar d\right)$ to query it exactly, for a
mean teacher width $\bar d$; with $|\mathcal{V}| = 4{,}326$ this is small beside the caching and is
also done once. Training the student is
$O\!\left(E (|\mathcal{D}| + |\mathcal{U}|)\big(1 + K C + K d_k d_s\big)\right)$: one student pass per
row, plus a $C$-way KL against each teacher and one $d_k \times d_s$ projection each, none of which
involves a teacher forward pass because the logits and states are cached. The teacher terms are
therefore a constant factor per row rather than a factor that grows with the student, and the only
term that grows with the recipe is $|\mathcal{U}|$: the cost is linear in the size of the transfer
set, which is what makes the curve of Section~\ref{sec:5.8} affordable and what sets its price, 18.4
GPU-hours for the whole grid. Inference is $O(1)$ in this unit and unchanged by any of it, since
$W_k$, the irony head and the teachers are all discarded.
"""

# thesis-only pointers: the paper has no appendix of training settings to send the reader to
SYMBOL_TABLE = r"""

Table~\ref{tab:symbols} lists every symbol used in this chapter, with the set it is drawn from and
the value it takes in the runs reported here; Sections~\ref{sec:3.2} to~\ref{sec:3.5} define each one
where it is introduced.

\begin{table}[htbp]
\centering
\footnotesize
\caption[Symbols used in Chapter 3]{Every symbol of this chapter, the set it is drawn from, and the
value or range it takes in the runs of Chapter~\ref{ch:5}.}
\label{tab:symbols}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llll}
\toprule
Symbol & Meaning & Domain & Value or range used \\
\midrule
$x_i$ & a text & strings & tweets, truncated to 128 word pieces \\
$y_i$ & its majority label & $\{1,\dots,C\}$ & one of six classes \\
$C$ & number of classes & $\mathbb{N}$ & 6 on tweets, 3 on the implicit benchmark \\
$K$ & number of teachers & $\mathbb{N}$ & 3 or 4 \\
$B$ & batch size & $\mathbb{N}$ & 32 \\
$E$ & epochs & $\mathbb{N}$ & 6; 13 and 35 in the matched-steps controls \\
$\mathcal{D}$ & labelled training split & set of texts & 34,607 tweets \\
$\mathcal{V}$ & validation split & set of texts & 4,326 tweets \\
$\mathcal{U}$ & unlabelled transfer set & set of texts & empty in sample; 5,000 to 205,593 out \\
$z_k(x_i)$ & teacher $k$'s logits & $\mathbb{R}^C$ & cached once, never updated \\
$h_k(x_i)$ & teacher $k$'s pooled state & $\mathbb{R}^{d_k}$ & masked mean of its last layer \\
$d_k$ & teacher $k$'s width & $\mathbb{N}$ & 768, or 1024 for BERT-large \\
$z_s(x_i)$ & student logits & $\mathbb{R}^C$ & updated every step \\
$h_s(x_i)$ & student pooled state & $\mathbb{R}^{d_s}$ & masked mean, pooled as the teachers are \\
$d_s$ & student width & $\mathbb{N}$ & 256 to 768; 512 for the BiLSTM \\
$z^{\mathrm{irony}}_s(x_i)$ & auxiliary head logits & $\mathbb{R}^2$ & discarded at inference \\
$z_{\mathrm{irony}}(x_i)$ & its target & $\mathbb{R}^2$ & from the irony checkpoint before adaptation \\
$W_k$ & projection student $\to$ teacher $k$ & $\mathbb{R}^{d_k \times d_s}$ & learned, discarded at inference \\
$T$ & distillation temperature & $(0, \infty)$ & 4; swept over $\{1, 2, 8\}$ \\
$\tau$ & weight temperature & $(0, \infty)$ & 1; swept over 0.05 to 5 \\
$\ell_k(i)$ & teacher $k$'s loss on $x_i$ & $[0, \infty)$ & 0.03 to 0.23 on $\mathcal{D}$ (Section~\ref{sec:5.6}) \\
$w_k(i)$ & in-sample weight of teacher $k$ & $(0,1)$, $\sum_k w_k(i) = 1$ & 0.318 to 0.356 on average \\
$\bar p(i)$ & the mixed soft target & simplex in $\mathbb{R}^C$ & near one-hot in sample \\
$m_i$ & label mask & $\{0, 1\}$ & 1 on $\mathcal{D}$, 0 on $\mathcal{U}$ \\
$\alpha, \beta$ & weight of the KL and hard terms & $[0, \infty)$ & 0.4 each; $\alpha$ swept over $\{0.2, 0.6\}$ \\
$\gamma$ & weight of the hidden-state term & $[0, \infty)$ & 0.2; 0 in the ablation \\
$\delta$ & weight of the irony term & $[0, \infty)$ & 0.3; swept over $\{0.1, 0.5\}$ \\
$\mathcal{L}_{\mathrm{KL}}, \mathcal{L}_{\mathrm{hid}}, \mathcal{L}_{\mathrm{irony}}, \mathcal{L}$ & the loss terms and their sum & $[0, \infty)$ & minimised by AdamW \\
$\nu$ & neighbours for the out-of-sample weight & $\mathbb{N}$ & 20 \\
$c_k(v)$ & is teacher $k$ correct on $v$ & $\{0, 1\}$ & measured on $\mathcal{V}$ \\
$r_k(x)$ & teacher $k$'s local accuracy & $(0, 1)$ & Laplace-smoothed, equation (3.7) \\
$w_k^{\mathrm{knn}}(x)$ & out-of-sample weight & $(0,1)$, $\sum_k w_k^{\mathrm{knn}}(x) = 1$ & within 0.03 of uniform \\
$\mathcal{K}_{\mathrm{homo}}, \mathcal{K}_{\mathrm{spec}}$ & the committees & sets of teachers & 3 and 4 members \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""

CH3_SUBS = {
    3: [("implemented in PyTorch and Transformers \\cite{paszke2019pytorch, wolf2020transformers}.",
         "implemented in PyTorch and Transformers \\cite{paszke2019pytorch, wolf2020transformers}; the "
         "remaining settings are collected in [[Appendix:app:hyper]]."),
        ("$m_i \\in \\{0, 1\\}$ a label mask that is 1 on labelled rows and 0 on the unlabelled rows of Section 3.5.",
         "$m_i \\in \\{0, 1\\}$ a label mask that is 1 on labelled rows and 0 on the unlabelled rows of Section 3.5. Every symbol in these five equations "
         "appears in [[Table:tab:symbols]] with the set it is drawn from and the value it takes here.")],
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
] + ["\\section{%s}\n\\label{sec:3.%d}\n%s%s" % (heading("3.%d " % k), k,
                                               convert(body("3.%d " % k), (), (), CH3_SUBS.get(k, ())),
                                               SYMBOL_TABLE if k == 1 else "")
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
\begin{tabular}{p{5.4cm}p{2.6cm}rrrp{1.5cm}}
\toprule
Set & Role & Train & Validation & Test & Labels \\
\midrule
Cyberbullying tweets, cleaned \cite{wang2020sosnet} & primary benchmark & 34,607 & 4,326 & 4,326 & 6 classes \\
Implicit Hate Corpus, single source \cite{elsherief2021latent} & implicit benchmark & 16,509 & 2,064 & 2,064 & 3 classes \\
ISHate, held out whole \cite{ocampo2023indepth} & out-of-domain test & -- & -- & 27,096 & 3 classes \\
Benign sarcasm probe \cite{abufarha2022semeval, oprea2020isarcasm} & inference only & -- & -- & 883 & -- \\
Ironic abuse probe \cite{elsherief2021latent, ocampo2023indepth} & inference only & -- & -- & 1,560 & -- \\
Implicit abuse probe, a subset of it \cite{ocampo2023indepth} & inference only & -- & -- & 763 & -- \\
Transfer set, abuse-domain base \cite{davidson2017automated, zampieri2019olid, basile2019hateval} & training, unlabelled & 42,013 & -- & -- & none \\
Transfer set, extended \cite{barbieri2020tweeteval} & training, unlabelled & 205,593 & -- & -- & none \\
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
\caption[Construction of the transfer set]{Construction of the transfer set. Upper part: rows read and kept per source. Lower part: rows
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
    "5.10 ": dict(tables=[("Probe metrics by model family, averaged over all 184 evaluated models, at each model's "
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
    "5.11 ": dict(subs=[("Synthetic obfuscation of the abusive test rows, BERT-mini, seed 1, macro-F1:",
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
CH5_DECISIONS = r"""
\section{The Research Questions, Decided}
\label{sec:5.12}
The four questions of Section~\ref{sec:1.4} are decided here on the measurements above, under one rule
fixed before any of them was run: a difference counts only if its paired bootstrap interval excludes
zero, and on a test set of 4,326 posts that requires roughly 0.007 macro-F1 (Section~\ref{sec:4.7}).
Table~\ref{tab:rq} states each decision beside the evidence that carries it; the paragraphs below give
the numbers. Chapter~\ref{ch:6} takes up what the decisions mean.

\begin{table}[htbp]
\centering
\footnotesize
\caption[The four research questions and their decisions]{Each research question, the rule that decides
it, the measurement that decides it, and the decision.}
\label{tab:rq}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{p{2.0cm}p{3.5cm}p{5.6cm}p{2.3cm}}
\toprule
Question & Decided by & Measurement & Decision \\
\midrule
RQ1: does the weighted committee beat no teacher at all, in sample? & Any interval excluding zero among
the in-sample distillation-against-fine-tuning comparisons & 27 such comparisons, none excludes zero;
best of six per student is $+0.0012$ to $+0.0071$, five intervals of five containing zero; weighting
minus averaging is $-0.0034$ to $+0.0000$, every interval containing zero & \textbf{No}
(Sections~\ref{sec:5.2}, \ref{sec:5.3}, \ref{sec:5.5}) \\
RQ2: what explains that? & The mechanism has to be measured, not argued & Teachers end training at loss
0.033 to 0.077 on the split the student is distilled on; weights 0.332 / 0.337 / 0.331 against a uniform
0.333, and 0.318 / 0.356 / 0.326 even at $\tau = 0.05$; pairwise $\kappa$ 0.889 to 0.962, all four
teachers right on 83.2 per cent of the test set and wrong together on 4.7 & \textbf{Memorised labels,
saturated weights, spent diversity} (Sections~\ref{sec:5.4}, \ref{sec:5.6}) \\
RQ3: does unlabelled text the teachers never saw help, and how? & Decision D20, written before the run:
the 168k arm gains at least 0.012 with the interval clear of zero and still rising & $+0.0123$ [$+0.0035$,
$+0.0211$] at 168,000 tweets, monotone over six sizes from $+0.0010$ at 5,000; generic text within
$+0.0017$ [$-0.0066$, $+0.0113$] of abuse-domain text at the same size; fine-tuning for the same updates
$-0.0013$ [$-0.0106$, $+0.0072$]; BiLSTM $+0.0163$ [$+0.0063$, $+0.0271$] & \textbf{Yes, and it is the
text} (Sections~\ref{sec:5.7}, \ref{sec:5.8}) \\
RQ4: does any of it improve the reading of implication? & A threshold-free AUC rising above the
in-sample band of 0.756 to 0.777 & 28 in-sample pre-trained variants, mean 0.771, standard deviation
0.004; out-of-sample arms 0.727 to 0.771, three below the band and none above; the randomly initialised
student 0.613; across 184 models recall and false-positive rate correlate at $+0.772$ & \textbf{No}
(Section~\ref{sec:5.10}) \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\textbf{RQ1 is decided against the method.} The grid makes 27 in-sample comparisons of a distilled
student against the same student fine-tuned without teachers, and no interval excludes zero. Taking the
best of six configurations per student, which flatters the method, gives $+0.0012$, $+0.0040$,
$+0.0068$, $+0.0040$ and $+0.0071$ macro-F1, all five intervals containing zero and all five below the
0.007 the test set can resolve. Per-instance weighting against uniform averaging gives $-0.0007$,
$-0.0034$, $+0.0000$, $-0.0029$ and $-0.0029$, and across a hundredfold range of $\tau$ the headline
student moves by 0.0007 and stays under fine-tuning at every value. Adding a teacher trained on labelled
implication changes it by $+0.0005$ [$-0.0059$, $+0.0064$] under averaging and $-0.0005$ [$-0.0069$,
$+0.0055$] under weighting. The answer is no at this resolution, and Section~\ref{sec:7.1} records that
an absence at this resolution is not a zero.

\textbf{RQ2 is decided by three measurements rather than by argument.} Four of the five teachers end
their last epoch at a cross-entropy between 0.033 and 0.077 on the split the weights are read from and
the student is distilled on, so their soft labels carry what the gold labels already carry; the one
teacher that did not memorise the split, DeBERTa at 0.227, is the one whose weight moves, to 0.233 in a
committee of four. The weights are therefore flat at every temperature the sweep reaches. And the
committee has little to route between: pairwise agreement runs from $\kappa = 0.889$ to $0.962$, the
four original teachers are right together on 83.2 per cent of the test set and wrong together on 4.7,
and no combination rule that sees only their probabilities, including a stacked gate fitted by
cross-validation on the test set, improves on their uniform mean by more than 0.005 against an oracle
sitting 4 to 5 accuracy points higher.

\textbf{RQ3 is decided for the data and against the committee.} Decision D20 fixed the rule before the
run: if the largest arm gained at least 0.012 with an interval clear of zero and the curve was still
rising, the constructive result would lead. It gained $+0.0123$ [$+0.0035$, $+0.0211$], the six sizes
are ordered without inversion from $+0.0010$ at 5,000 tweets, and the curve had not flattened. The three
controls hold: generic tweets of the same size come within $+0.0017$ [$-0.0066$, $+0.0113$] of
abuse-domain ones, fine-tuning for the same number of updates gains $-0.0013$ [$-0.0106$, $+0.0072$],
and the gain reappears on a second and third student, $+0.0163$ [$+0.0063$, $+0.0271$] on the BiLSTM and
$+0.0057$ [$-0.0063$, $+0.0171$] on BERT-small. The committee is not part of it: against a single
teacher on the same text it is $-0.0003$ [$-0.0061$, $+0.0056$], its corrected weighting adds $+0.0021$
[$-0.0017$, $+0.0082$], and its hard pseudo-labels gain $+0.0020$ against $+0.0070$ for its soft ones.

\textbf{RQ4 is decided against every intervention in the grid.} Measured without a threshold, 28
in-sample pre-trained variants of the headline student, covering every objective, committee,
temperature, sweep, ablation and control, have a sarcasm-discrimination AUC of mean 0.771 and standard
deviation 0.004. The out-of-sample arms run from 0.727 to 0.771: three of five fall below that band and
none rises above it, which is also where a pre-registered prediction failed as written. The only
intervention that moves the measure is removing pre-training, which drops it to 0.613. Across all 184
evaluated models recall on ironic abuse and the false-positive rate on benign sarcasm correlate at
$+0.772$ with a margin of $0.357 \pm 0.046$, so what the interventions change is where a model sits on
one trade-off line, not which line it is on. On a corpus that labels implication the same students rank
implied abuse against benign text at 0.600 to 0.609 AUC, against 0.820 for the same architecture trained
on that corpus.
"""

CH5_PER_CLASS = r"""
\section{Where the Gain Lands: The Classes, and the Confusion Between Them}
\label{sec:5.9}
A macro-F1 of 0.8516 is an average over six classes that are not equally hard, so the number alone
does not say what the unlabelled text taught. This section takes the arms of the two sections above
apart class by class, from the saved test predictions of every seed
(\texttt{scripts/per\_class.py}). Table~\ref{tab:per-class} gives the per-class F1 and
Table~\ref{tab:confusion} the confusion that moves.

\begin{table}[htbp]
\centering
\small
\caption[Per-class test F1 across the arms]{Per-class test F1, mean over three seeds, for the arms of
Sections~\ref{sec:5.7} and~\ref{sec:5.8}. The first four columns are the classes that name a target,
the last two the residual classes.}
\label{tab:per-class}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccccc}
\toprule
Arm & age & ethnicity & gender & religion & other\_cb & not\_cb \\
\midrule
Fine-tune only & 0.963 & 0.942 & 0.864 & 0.917 & 0.707 & 0.644 \\
Fine-tune, 35 epochs & 0.962 & 0.944 & 0.865 & 0.920 & 0.693 & 0.643 \\
Single teacher, in sample & 0.964 & 0.946 & 0.865 & 0.922 & 0.702 & 0.644 \\
$+$ 42k transfer rows & 0.965 & 0.947 & 0.871 & 0.921 & 0.714 & 0.660 \\
$+$ 168k transfer rows & 0.969 & 0.946 & 0.880 & 0.924 & \textbf{0.721} & \textbf{0.671} \\
\midrule
BiLSTM, fine-tune only & 0.977 & 0.982 & 0.897 & 0.950 & 0.719 & 0.690 \\
BiLSTM, $+$ 168k transfer rows & 0.981 & 0.983 & 0.913 & 0.955 & \textbf{0.747} & \textbf{0.734} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\textbf{The benchmark is two tasks.} Fine-tuning alone already scores 0.864 to 0.963 on the four
classes that name a target, and 0.644 and 0.707 on the two that do not. The spread is a property of
the labels rather than of the method: a tweet that attacks a religion contains the words of that
religion, while a tweet sorted into other\_cyberbullying or not\_cyberbullying is sorted on whether
it is abusive at all. Every method in this thesis inherits that split, and it is the reason aggregate
macro-F1 is a blunt instrument here. Four sixths of it is settled before any teacher is involved.

\textbf{The out-of-sample gain is concentrated where the labels are hardest.} Against fine-tuning the
168k arm moves the four targeted classes by $+0.005$, $+0.005$, $+0.016$ and $+0.007$, and the two
residual classes by $+0.014$ and $+0.027$. Those two are two sixths of the macro average and carry 55
per cent of its change; on the BiLSTM they carry 74 per cent of a larger change, $+0.028$ and
$+0.044$. The 42k arm has the same shape at half the size, $+0.007$ and $+0.017$. The recipe does not
raise the score uniformly. It works on the boundary the labelled split draws worst, which is the
boundary between abuse that names no group and text that is not abuse at all.

\textbf{Neither in-sample distillation nor more optimisation does this.} Distilling from the same
teacher on the labelled split alone moves the four targeted classes by $+0.001$ to $+0.005$ and the
two residual classes by $+0.000$ and $-0.004$: what little it gains, it gains where the task was
already solved. Fine-tuning for the 168k arm's number of updates is worse on the class that matters,
$-0.013$ on other\_cyberbullying, while gaining $+0.002$ and $+0.003$ on ethnicity and religion.
Both controls move the easy classes and leave the hard ones alone, which is the per-class form of the
null of Section~\ref{sec:5.2} and of the matched-steps control of Section~\ref{sec:5.8}.

\begin{table}[htbp]
\centering
\small
\caption[Confusion between the two residual classes]{Confusion between the two residual classes, mean
counts over three seeds. The test split holds 583 other\_cyberbullying and 617 not\_cyberbullying
rows, so these two errors are most of what either class loses.}
\label{tab:confusion}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lcccc}
\toprule
Arm & other $\rightarrow$ not & not $\rightarrow$ other & other correct & not correct \\
\midrule
Fine-tune only & 106 & 126 & 419 & 401 \\
Fine-tune, 35 epochs & 115 & 132 & 415 & 411 \\
Single teacher, in sample & 107 & 132 & 422 & 401 \\
$+$ 42k transfer rows & 98 & 124 & 432 & 411 \\
$+$ 168k transfer rows & 101 & 117 & 433 & \textbf{421} \\
\midrule
BiLSTM, fine-tune only & 110 & 117 & 426 & 424 \\
BiLSTM, $+$ 168k transfer rows & 109 & \textbf{94} & 438 & \textbf{463} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\textbf{What changes is one direction of one confusion.} Fine-tuning sends 126 of the 617
not\_cyberbullying rows into other\_cyberbullying and 106 of the 583 other\_cyberbullying rows the
other way; between them those two errors are most of what either class loses. Adding 168,000
unlabelled tweets takes the first from 126 to 117 and raises the rows called correctly from 401 to
421, while the reverse error moves from 106 to 101. On the BiLSTM the same intervention takes the
first error from 117 to 94 and the correct count from 424 to 463, a shift of 39 rows in a class of
617. The student has become less willing to call an ordinary tweet abusive, and it learned that on
text where no label said so.

That is the same behaviour Section~\ref{sec:5.10} measures on the probes and reports as a cost.
Firing less readily is an improvement when the tweet is not abusive and a loss when it is ironically
abusive, so the per-class table and the probe trade-off are two views of one change rather than two
findings. It also says where the ceiling of this recipe is. The gain arrives as a few dozen rows
moved across one boundary, and 101 and 117 rows are still on the wrong side of it.
"""

CH5 = ["\\chapter{Results and Analysis}", "\\label{ch:5}", convert(body("5. Results"), (), (), [
    ("transferred (5.10), and what survives for practice (5.11).",
     "where that gain lands class by class (5.9), what never transferred (5.10), what survives for "
     "practice (5.11), and the decision each research question comes to on this evidence "
     "(5.12).")])]
for k in ["5.1 ", "5.2 ", "5.3 ", "5.4 ", "5.5 ", "5.6 ", "5.7 ", "5.8 "]:
    spec = R[k]
    CH5.append("\\section{%s}\n\\label{sec:%s}\n%s" % (heading(k), k.strip(),
                                                      convert(body(k), spec.get("tables", ()), spec.get("inserts", ()),
                                                              spec.get("subs", ()))))
CH5.append(CH5_PER_CLASS)
for k in ["5.10 ", "5.11 "]:
    spec = R[k]
    CH5.append("\\section{%s}\n\\label{sec:%s}\n%s" % (heading(k), k.strip(),
                                                      convert(body(k), spec.get("tables", ()), spec.get("inserts", ()),
                                                              spec.get("subs", ()))))
CH5.append(CH5_DECISIONS)
CH5 = "\n".join(CH5)

# --------------------------------------------------------------------------------- chapter 6
CH6_RQ = r"""
\section{Answers to the Research Questions}
\label{sec:6.1}
Section~\ref{sec:5.12} decides the four questions on the measurements and records the numbers that
decide them. This section states each answer and what follows from it.
\begin{description}
  \item[RQ1. Does multi-teacher distillation with per-instance reliability weighting improve a compact
    cyberbullying detector over the same student fine-tuned without teachers?]
    Not in sample, at a resolution of 0.007 macro-F1. What this licenses is narrow and worth stating
    precisely: across every student, committee and temperature in the grid, a practitioner who ran the
    committee and a practitioner who ran neither would not be able to tell their students apart on this
    test set. It does not license the claim that the effect is zero.
  \item[RQ2. What mechanism explains the answer to RQ1?] The student imitates its teachers on the one
    split they were themselves fine-tuned on, and they have memorised it, so the soft labels repeat the
    gold labels and the signal meant to weight them has nothing left to express. The mechanism is a
    property of where the teachers are read, not of this implementation, and it applies to any scheme in
    the error-weighted family that reads reliability on the training split.
  \item[RQ3. Does distillation on unlabelled text the teachers have never seen improve the student, and
    how does the gain depend on the amount and composition of that text, on the number of optimisation
    updates, and on the student?] Yes for the text, no for the committee. The gain grows with how much
    unlabelled text the student sees, survives a control that spends the same number of updates without
    it, holds on three students of two families, and comes from ordinary text of the platform rather
    than from abuse-related text. One teacher supplies all of it.
  \item[RQ4. Does any of these interventions improve the student's ability to tell implied abuse from
    harmless sarcasm?] No. Measured without a threshold, the ability is set by pre-training, and the
    interventions move only how readily the student fires. A method reporting recall at a fixed cut can
    therefore show progress on implication while having none, which is why the measure here is an area
    and not a decision.
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
  \item \textbf{From the method as proposed to a controlled grid.} The method was to be evaluated the
    way this literature evaluates it, by setting a distilled student against its teachers. We added the
    baseline those comparisons leave out, the same student trained with no teacher, along with three
    seeds and paired bootstrap intervals, and distillation then raised every student by less than the
    test set can resolve.
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
The title changed with the evidence. The title this work began under named a dynamic, multi-teacher,
homogeneous and robust framework. The grid measured the first three of those properties and found none of them to
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
% the department's reference sets its body single-spaced, 14.5 pt on 12 pt
\singlespacing
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
