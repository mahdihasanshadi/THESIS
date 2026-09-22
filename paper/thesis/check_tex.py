"""Static checks on the generated LaTeX, for machines without a TeX installation.

    python paper/thesis/check_tex.py

Catches the mistakes that stop a pdfLaTeX build or corrupt it silently: unbalanced braces and
environments, an odd number of inline-math dollars in a paragraph, special characters left unescaped in
text, a percent sign that would comment out the rest of a line, references and citations without
targets, missing figure files, and Markdown that was not converted. It is not a substitute for a
compile; it makes the first compile likely to succeed.
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, "main.tex"), os.path.join(HERE, "frontmatter.tex")] + \
    sorted(glob.glob(os.path.join(HERE, "chapters", "*.tex"))) + sorted(glob.glob(os.path.join(HERE, "appendices", "*.tex")))
MATH_ENVS = ("equation", "equation*", "align", "align*")
TAB_ENVS = ("tabular", "longtable", "tabularx")
problems = []


def strip_comments(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


labels, refs, cites, graphics = set(), [], [], []
for path in FILES:
    rel = os.path.relpath(path, HERE)
    raw = io.open(path, encoding="utf-8").read()
    if "\x01" in raw or "\x02" in raw:
        problems.append((rel, 0, "unreleased placeholder"))
    lines = raw.split("\n")
    text = "\n".join(strip_comments(ln) for ln in lines)
    for n, ln in enumerate(lines, 1):
        if re.search(r"\d%", ln) and not re.search(r"\\%", ln):
            problems.append((rel, n, "percent after a digit comments out the rest of the line"))
        for pat, what in ((r"\*\*", "Markdown bold"), (r"(?<!`)`(?!`)", "Markdown backtick"), (r"\[\[", "unconverted token"),
                          (r"^\s*\|", "Markdown table row"), (r"^#+ ", "Markdown heading")):
            if re.search(pat, strip_comments(ln)):
                problems.append((rel, n, what))
    # braces
    depth = 0
    for n, ln in enumerate(text.split("\n"), 1):
        s = re.sub(r"\\[{}]", "", ln)
        for ch in s:
            depth += ch == "{"
            depth -= ch == "}"
            if depth < 0:
                problems.append((rel, n, "closing brace without an opening one"))
                depth = 0
    if depth:
        problems.append((rel, 0, "%d unclosed brace(s)" % depth))
    # environments
    stack = []
    for n, ln in enumerate(text.split("\n"), 1):
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", ln):
            if kind == "begin":
                stack.append((env, n))
            elif not stack or stack[-1][0] != env:
                problems.append((rel, n, "\\end{%s} does not match %s" % (env, stack[-1] if stack else "nothing")))
            else:
                stack.pop()
    for env, n in stack:
        problems.append((rel, n, "\\begin{%s} never closed" % env))
    # per-paragraph math and special characters in text mode
    body = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % "|".join(re.escape(e) for e in MATH_ENVS), " ", text, flags=re.S)
    body = re.sub(r"\\\[.*?\\\]", " ", body, flags=re.S)
    for para in re.split(r"\n\s*\n", body):
        dollars = len(re.findall(r"(?<!\\)\$", para))
        if dollars % 2:
            problems.append((rel, 0, "odd number of $ in paragraph starting: " + para.strip()[:70]))
        nomath = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", para, flags=re.S)
        nomath = re.sub(r"\\(ref|label|cite|url|includegraphics|href|input)\{[^}]*\}", " ", nomath)
        nomath = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", " ", nomath)
        for ch, what in (("_", "underscore"), ("^", "caret"), ("#", "hash")):
            for m in re.finditer(r"(?<!\\)" + re.escape(ch), nomath):
                problems.append((rel, 0, "unescaped %s near: %s" % (what, nomath[max(0, m.start() - 30):m.start() + 30].strip())))
    # ampersands outside tables
    outside = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % "|".join(TAB_ENVS), " ", text, flags=re.S)
    outside = re.sub(r"\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\}", " ", outside, flags=re.S)
    outside = re.sub(r"\\usepackage\[[^\]]*\]", " ", outside)
    for m in re.finditer(r"(?<!\\)&", outside):
        problems.append((rel, 0, "ampersand outside a table near: " + outside[max(0, m.start() - 40):m.start() + 20]))
    labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
    refs += [(rel, r) for r in re.findall(r"\\ref\{([^}]+)\}", text)]
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", text):
        cites += [(rel, k.strip()) for k in group.split(",")]
    graphics += [(rel, g) for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)]

for rel, r in refs:
    if r not in labels:
        problems.append((rel, 0, "\\ref{%s} has no \\label" % r))
bib = io.open(os.path.join(HERE, "references.bib"), encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
for rel, k in cites:
    if k not in keys:
        problems.append((rel, 0, "\\cite{%s} is not in references.bib" % k))
for rel, g in graphics:
    if not os.path.exists(os.path.join(HERE, g)):
        problems.append((rel, 0, "figure file missing: " + g))
if any(ord(c) > 127 for c in bib):
    problems.append(("references.bib", 0, "non-ASCII character"))

for p in problems:
    print("%-34s line %-4s %s" % p)
cited = {k for _, k in cites}
print("\n%d files, %d labels, %d references, %d citations of %d distinct keys (%d in the bibliography), %d figures"
      % (len(FILES), len(labels), len(refs), len(cites), len(cited), len(keys), len(graphics)))
print("uncited bibliography entries:", sorted(keys - cited) or "none")
print("CHECK PASSED" if not problems else "%d PROBLEMS" % len(problems))
sys.exit(1 if problems else 0)
