"""Render the thesis to PDF on a machine with no TeX installation.

    python paper/thesis/render_pdf.py [--out DIR] [--name FILE.pdf] [--browser PATH]

Reads main.tex and every file it inputs, converts them to one HTML page and prints it with headless
Edge or Chrome: equations as MathML, figures as vector SVG converted from figures/*.pdf, numbered
chapters, sections, figures, tables, equations and algorithms, resolved cross-references, natbib's
sorted and compressed numeric citations, and an unsrtnat-style bibliography from references.bib, with
BibTeX's sentence-casing of titles. A first print finds the page of every heading, figure and table;
the next fills the contents, the lists of figures and tables and the page numbers (roman in the front
matter, arabic from Chapter 1), repeated until the pages stop moving; the last step stamps the page
numbers and adds bookmarks and page labels.

It understands the LaTeX that build_tex.py writes, not LaTeX in general, and warns about anything it
does not know. The typeset thesis is still the pdfLaTeX build of main.tex; this is the draft to read and
circulate when no TeX is at hand. Needs pypdf, PyMuPDF and reportlab.
"""
import argparse
import html
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = re.compile(r"\\([a-zA-Z]+\*?|.)", re.S)
NUM = re.compile(r"\d+(?:\.\d+)?|\.\d+")
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/microsoft-edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
LETTER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WARNINGS = []


def warn(msg):
    if msg not in WARNINGS:
        WARNINGS.append(msg)


def esc(s):
    return html.escape(s, quote=False)


def slug(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def match_brace(s, i, open_="{", close="}"):
    """Index of the bracket closing the one at s[i]; escaped braces do not count."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced %s at: %s" % (open_, s[i:i + 60]))


def skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def read_group(s, i):
    """Argument at s[i:] (after spaces): a braced group or a single token. Returns (text, next index)."""
    i = skip_ws(s, i)
    if i >= len(s):
        return "", i
    if s[i] == "{":
        j = match_brace(s, i)
        return s[i + 1:j], j + 1
    if s[i] == "\\":
        m = CMD.match(s, i)
        return m.group(0), m.end()
    return s[i], i + 1


def read_opt(s, i):
    """Optional [argument] at s[i:], braces respected. Returns (text or None, next index)."""
    j = skip_ws(s, i)
    if j < len(s) and s[j] == "[":
        depth = 0
        k = j + 1
        while k < len(s):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            elif s[k] == "]" and depth == 0:
                return s[j + 1:k], k + 1
            k += 1
    return None, i


def env_end(s, i, env):
    """For content starting at s[i] inside \\begin{env}: (content end, index after \\end{env})."""
    pat = re.compile(r"\\(begin|end)\{" + re.escape(env) + r"\}")
    depth = 1
    for m in pat.finditer(s, i):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    raise ValueError("\\begin{%s} never closed" % env)


def math_end(s, i):
    """Index of the $ closing the inline math that opens at s[i]."""
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "$":
            return j
        j += 1
    raise ValueError("unclosed $ at: " + s[i:i + 60])


def strip_comments(src):
    out = []
    for line in src.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut] + "\x00")
    # a comment removes the line end and the next line's leading spaces, as in TeX
    return re.sub(r"\x00\n[ \t]*", "", "\n".join(out)).replace("\x00", "")


def tex_len(s):
    """A TeX length as CSS: 5.5cm, 0.4pt, 0.8\\textwidth."""
    s = s.strip()
    m = re.fullmatch(r"([\d.]*)\s*\\(textwidth|linewidth|columnwidth)", s)
    if m:
        return "%g%%" % (100 * float(m.group(1) or 1))
    m = re.fullmatch(r"(-?[\d.]+)\s*(cm|mm|in|pt|em|ex|bp)", s)
    if m:
        unit = {"bp": "pt"}.get(m.group(2), m.group(2))
        return m.group(1) + unit
    warn("length not understood: " + s)
    return "0"


def roman(n):
    out = ""
    for v, r in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                 (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += r
            n -= v
    return out


# ------------------------------------------------------------------------------------------------ MathML
GREEK = dict(alpha="\u03b1", beta="\u03b2", gamma="\u03b3", delta="\u03b4", epsilon="\u03f5", varepsilon="\u03b5",
             zeta="\u03b6", eta="\u03b7", theta="\u03b8", vartheta="\u03d1", iota="\u03b9", kappa="\u03ba",
             lambda_="\u03bb", mu="\u03bc", nu="\u03bd", xi="\u03be", pi="\u03c0", rho="\u03c1", sigma="\u03c3",
             tau="\u03c4", upsilon="\u03c5", phi="\u03d5", varphi="\u03c6", chi="\u03c7", psi="\u03c8", omega="\u03c9",
             Gamma="\u0393", Delta="\u0394", Theta="\u0398", Lambda="\u039b", Xi="\u039e", Pi="\u03a0",
             Sigma="\u03a3", Upsilon="\u03a5", Phi="\u03a6", Psi="\u03a8", Omega="\u03a9")
GREEK["lambda"] = GREEK.pop("lambda_")
MATH_MO = {"in": "\u2208", "notin": "\u2209", "cup": "\u222a", "cap": "\u2229", "to": "\u2192",
           "rightarrow": "\u2192", "leftarrow": "\u2190", "Rightarrow": "\u21d2", "mid": "\u2223", "pm": "\u00b1",
           "mp": "\u2213", "times": "\u00d7", "cdot": "\u22c5", "sim": "\u223c", "approx": "\u2248",
           "le": "\u2264", "leq": "\u2264", "ge": "\u2265", "geq": "\u2265", "neq": "\u2260", "ne": "\u2260",
           "equiv": "\u2261", "propto": "\u221d", "dots": "\u2026", "ldots": "\u2026", "cdots": "\u22ef",
           "subset": "\u2282", "subseteq": "\u2286", "setminus": "\u2216", "ast": "\u2217", "circ": "\u2218",
           "langle": "\u27e8", "rangle": "\u27e9", "lVert": "\u2016", "rVert": "\u2016", "Vert": "\u2016",
           "vert": "|", "forall": "\u2200", "exists": "\u2203", "mapsto": "\u21a6", "colon": ":"}
MATH_MI = {"infty": "\u221e", "ell": "\u2113", "top": "\u22a4", "emptyset": "\u2205", "partial": "\u2202",
           "nabla": "\u2207"}
FUNCS = {"exp", "log", "ln", "max", "min", "arg", "sup", "inf", "lim", "sin", "cos", "tan", "det", "Pr"}
LARGEOPS = {"sum": "\u2211", "prod": "\u220f", "int": "\u222b", "bigcup": "\u22c3", "bigcap": "\u22c2"}
MSPACE = {",": "0.1667em", ":": "0.2222em", ">": "0.2222em", ";": "0.2778em", "!": "-0.1667em", " ": "0.25em",
          "quad": "1em", "qquad": "2em", "enspace": "0.5em", "thinspace": "0.1667em"}
BIGSIZE = {"big": "1.2em", "Big": "1.62em", "bigg": "2.05em", "Bigg": "2.47em"}
# U+203E rather than the macron: at text size Chromium sets a macron accent into the letter it covers
ACCENTS = {"bar": "\u203e", "overline": "\u203e", "hat": "^", "widehat": "^", "tilde": "~", "widetilde": "~",
           "vec": "\u2192", "dot": "\u02d9", "ddot": "\u00a8"}
OPEN = set("([{\u27e8")
FENCES = set("()[]{}|\u2016\u27e8\u27e9")
BINREL = set("+\u2212=<>\u2264\u2265\u2260\u2248\u223c\u2208\u222a\u2229\u2192\u00d7\u00b1\u22c5,;:/\u2223")


def script_letter(c):
    special = dict(B="\u212c", E="\u2130", F="\u2131", H="\u210b", I="\u2110", L="\u2112", M="\u2133",
                   R="\u211b", e="\u212f", g="\u210a", o="\u2134")
    if c in special:
        return special[c]
    if "A" <= c <= "Z":
        return chr(0x1D49C + ord(c) - 65)
    if "a" <= c <= "z":
        return chr(0x1D4B6 + ord(c) - 97)
    return c


def double_struck(c):
    special = dict(C="\u2102", H="\u210d", N="\u2115", P="\u2119", Q="\u211a", R="\u211d", Z="\u2124")
    if c in special:
        return special[c]
    if "A" <= c <= "Z":
        return chr(0x1D538 + ord(c) - 65)
    return c


def mo(c, **attrs):
    c = {"-": "\u2212", "*": "\u2217"}.get(c, c)
    if c in FENCES or c == "/":
        attrs.setdefault("stretchy", "false")
        tight(attrs)
    return ["mo", c, attrs]


def tight(attrs):
    """TeX puts no space around delimiters or a slash; MathML's operator dictionary does, outside prefix form."""
    attrs.setdefault("lspace", "0")
    attrs.setdefault("rspace", "0")
    return attrs


class Math:
    """LaTeX math to MathML, for the subset the thesis uses."""

    def __init__(self, src, text_fn):
        self.s, self.i, self.text_fn = src, 0, text_fn

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def expr(self, closer=None):
        nodes = []
        while True:
            self.ws()
            if self.i >= len(self.s):
                if closer:
                    warn("unclosed group in math: " + self.s)
                break
            c = self.s[self.i]
            if closer and c == closer:
                self.i += 1
                break
            if c in "^_":
                self.i += 1
                arg = self.script()
                base = nodes.pop() if nodes else ["row", []]
                nodes.append(self.attach(base, c, arg))
                continue
            node = self.atom()
            if node is not None:
                nodes.append(node)
        prev = None
        for n in nodes:  # a sign after an opening fence, an operator or nothing is unary, as in TeX
            if n[0] == "mo" and n[1] in ("\u2212", "+", "\u00b1") and (
                    prev is None or (prev[0] == "mo" and (prev[1] in OPEN or prev[1] in BINREL))):
                n[2]["form"] = "prefix"
            if n[0] != "mspace":
                prev = n
        return nodes

    @staticmethod
    def attach(base, kind, arg):
        if base[0] == "sub" and kind == "^":
            return ["subsup", base[1], base[2], arg]
        if base[0] == "sup" and kind == "_":
            return ["subsup", base[1], arg, base[2]]
        if base[0] == "under" and kind == "^":
            return ["underover", base[1], base[2], arg]
        if base[0] == "overop" and kind == "_":
            return ["underover", base[1], arg, base[2]]
        if base[0] == "mo" and base[2].get("largeop"):
            return ["under", base, arg] if kind == "_" else ["overop", base, arg]
        return ["sub", base, arg] if kind == "_" else ["sup", base, arg]

    def script(self):
        self.ws()
        c = self.s[self.i]
        if c == "{":
            self.i += 1
            return ["row", self.expr("}")]
        if c == "\\":
            return self.atom()
        self.i += 1
        if c.isdigit():
            return ["mn", c]
        if c.isalpha():
            return ["mi", c, {}]
        return mo(c)

    def raw(self):
        self.ws()
        if self.i < len(self.s) and self.s[self.i] == "{":
            j = match_brace(self.s, self.i)
            out = self.s[self.i + 1:j]
            self.i = j + 1
            return out
        c = self.s[self.i]
        self.i += 1
        return c

    def arg(self):
        self.ws()
        if self.s[self.i] == "{":
            self.i += 1
            return ["row", self.expr("}")]
        return self.atom()

    def delim(self):
        self.ws()
        if self.s[self.i] == "\\":
            m = CMD.match(self.s, self.i)
            self.i = m.end()
            name = m.group(1)
            return {"|": "\u2016", "{": "{", "}": "}", "langle": "\u27e8", "rangle": "\u27e9", "lVert": "\u2016",
                    "rVert": "\u2016", "Vert": "\u2016", "vert": "|", "lvert": "|", "rvert": "|"}.get(name, name)
        c = self.s[self.i]
        self.i += 1
        return c

    def atom(self):
        s, c = self.s, self.s[self.i]
        if c == "{":
            self.i += 1
            return ["row", self.expr("}")]
        if c == "}":
            self.i += 1
            warn("stray } in math: " + s)
            return None
        if c == "\\":
            m = CMD.match(s, self.i)
            self.i = m.end()
            return self.command(m.group(1))
        if c.isdigit() or (c == "." and self.i + 1 < len(s) and s[self.i + 1].isdigit()):
            m = NUM.match(s, self.i)
            self.i = m.end()
            return ["mn", m.group(0)]
        self.i += 1
        if c.isalpha():
            return ["mi", c, {}]
        if c == "~":
            return ["mspace", "0.25em"]
        if c == "'":
            return ["mo", "\u2032", {}]
        if c == "&":
            return None
        return mo(c)

    def command(self, name):
        if name in GREEK:
            return ["mi", GREEK[name], {"mathvariant": "normal"} if name[0].isupper() else {}]
        if name in ("mathcal", "mathscr"):
            return ["mi", "".join(script_letter(c) for c in self.raw() if not c.isspace()), {}]
        if name == "mathbb":
            return ["mi", "".join(double_struck(c) for c in self.raw() if not c.isspace()), {}]
        if name in ("mathrm", "operatorname", "mathup", "mathsf"):
            raw = self.raw().strip()
            if re.fullmatch(r"[A-Za-z0-9]+", raw):
                return ["mi", raw, {"mathvariant": "normal"} if len(raw) == 1 else {}]
            return ["row", Math(raw, self.text_fn).expr()]
        if name == "mathit":
            return ["mi", self.raw().strip(), {}]
        if name in ("text", "textrm", "mbox", "textnormal", "textit", "textup"):
            return ["mtext", self.text_fn(self.raw())]
        if name in ("frac", "dfrac", "tfrac"):
            return ["frac", self.arg(), self.arg()]
        if name in LARGEOPS:
            return ["mo", LARGEOPS[name], {"largeop": "true", "movablelimits": "true"}]
        if name in FUNCS:
            return ["mi", name, {}]
        base = name[:-1] if name[-1:] in "lrm" and name[:-1] in BIGSIZE else name
        if base in BIGSIZE:
            size = BIGSIZE[base]
            return ["mo", self.delim(), tight({"minsize": size, "maxsize": size, "stretchy": "true", "symmetric": "true"})]
        if name in ("left", "right"):
            d = self.delim()
            return None if d == "." else ["mo", d, tight({"stretchy": "true", "symmetric": "true"})]
        if name in MSPACE:
            return ["mspace", MSPACE[name]]
        if name == "|":
            return ["mo", "\u2016", tight({"stretchy": "false"})]
        if name in ("{", "}"):
            return ["mo", name, tight({"stretchy": "false"})]
        if name in ("%", "&", "#", "_", "$"):
            return ["mo", name, {}]
        if name in MATH_MO:
            ch = MATH_MO[name]
            return ["mo", ch, tight({"stretchy": "false"}) if ch in FENCES else {}]
        if name in MATH_MI:
            return ["mi", MATH_MI[name], {}]
        if name in ACCENTS:
            return ["over", self.arg(), ["mo", ACCENTS[name], {"stretchy": "false"}]]
        warn("unknown math command \\" + name)
        return ["mi", name, {"mathvariant": "normal"}]

    @classmethod
    def ser(cls, n):
        k = n[0]
        if k in ("mi", "mo"):
            attrs = "".join(' %s="%s"' % kv for kv in sorted(n[2].items()))
            return "<%s%s>%s</%s>" % (k, attrs, esc(n[1]), k)
        if k == "mn":
            return "<mn>%s</mn>" % esc(n[1])
        if k == "mtext":
            return "<mtext>%s</mtext>" % n[1]
        if k == "mspace":
            return '<mspace width="%s"></mspace>' % n[1]
        if k == "row":
            inner = "".join(cls.ser(c) for c in n[1])
            return inner if len(n[1]) == 1 else "<mrow>%s</mrow>" % inner
        if k in ("sub", "sup"):
            return "<m%s>%s%s</m%s>" % (k, cls.ser(n[1]), cls.ser(n[2]), k)
        if k == "subsup":
            return "<msubsup>%s%s%s</msubsup>" % (cls.ser(n[1]), cls.ser(n[2]), cls.ser(n[3]))
        if k == "under":
            return "<munder>%s%s</munder>" % (cls.ser(n[1]), cls.ser(n[2]))
        if k == "overop":
            return "<mover>%s%s</mover>" % (cls.ser(n[1]), cls.ser(n[2]))
        if k == "underover":
            return "<munderover>%s%s%s</munderover>" % (cls.ser(n[1]), cls.ser(n[2]), cls.ser(n[3]))
        if k == "over":
            return '<mover accent="true">%s%s</mover>' % (cls.ser(n[1]), cls.ser(n[2]))
        if k == "frac":
            return "<mfrac>%s%s</mfrac>" % (cls.ser(n[1]), cls.ser(n[2]))
        raise ValueError(k)

    @classmethod
    def render(cls, src, text_fn, display=False):
        nodes = cls(src, text_fn).expr()
        body = cls.ser(["row", nodes]) if nodes else ""
        if display:
            return '<math display="block">%s</math>' % body
        return "<math>%s</math>" % body


# ------------------------------------------------------------------------------------------------ BibTeX
ACCENT_COMBINING = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304",
                    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
                    "r": "\u030a", "d": "\u0323", "b": "\u0331"}
SPECIAL_LETTERS = {"o": "\u00f8", "O": "\u00d8", "ss": "\u00df", "aa": "\u00e5", "AA": "\u00c5", "ae": "\u00e6",
                   "AE": "\u00c6", "oe": "\u0153", "OE": "\u0152", "l": "\u0142", "L": "\u0141", "i": "\u0131"}


def bib_plain(s):
    """A BibTeX field value as HTML text: accents resolved, braces dropped."""
    def accent(m):
        base = m.group(2) or m.group(3)
        base = {"\\i": "i", "\\j": "j"}.get(base, base)
        return unicodedata.normalize("NFC", base + ACCENT_COMBINING[m.group(1)])
    s = re.sub(r"\\([\'`^\"~=.])\s*(?:\{(\\?[A-Za-z])\}|(\\?[A-Za-z]))", accent, s)
    s = re.sub(r"\\([uvHckrdb])(?:\{(\\?[A-Za-z])\}|\s+(\\?[A-Za-z]))", accent, s)
    s = re.sub(r"\\(ss|aa|AA|ae|AE|oe|OE|o|O|l|L|i)(?![A-Za-z])\s*", lambda m: SPECIAL_LETTERS[m.group(1)], s)
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#").replace("\\$", "$")
    s = s.replace("---", "\u2014").replace("--", "\u2013").replace("~", "\u00a0")
    s = re.sub(r"\\(emph|textit|textbf|textrm|textsc|mbox)\s*", "", s)
    if "\\" in s:
        warn("LaTeX left in a bibliography field: " + s)
    return esc(s.replace("{", "").replace("}", ""))


def sentence_case(t):
    """BibTeX's change.case$ with "t": lowercase all but the first letter, letters after ': ' and braced text."""
    out, depth, prev_colon = [], 0, False
    for i, c in enumerate(t):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and i > 0 and not (prev_colon and t[i - 1].isspace()):
            c = c.lower()
        out.append(c)
        if depth == 0 and c not in "{}":
            if c == ":":
                prev_colon = True
            elif not c.isspace():
                prev_colon = False
    return "".join(out)


def parse_bib(src):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", src):
        kind, key = m.group(1).lower(), m.group(2)
        i = m.end()
        fields = {}
        while True:
            i = skip_ws(src, i)
            if i >= len(src) or src[i] == "}":
                break
            fm = re.compile(r"([A-Za-z]+)\s*=\s*").match(src, i)
            if not fm:
                break
            name, i = fm.group(1).lower(), fm.end()
            if src[i] == "{":
                j = match_brace(src, i)
                value, i = src[i + 1:j], j + 1
            elif src[i] == '"':
                j = src.index('"', i + 1)
                value, i = src[i + 1:j], j + 1
            else:
                vm = re.compile(r"[^,}\s]+").match(src, i)
                value, i = vm.group(0), vm.end()
            fields[name] = re.sub(r"\s+", " ", value.strip())
            i = skip_ws(src, i)
            if i < len(src) and src[i] == ",":
                i += 1
        entries[key] = (kind, fields)
    return entries


def split_top(s, sep_re):
    """Split s on a regex, only where no brace is open."""
    parts, depth, last = [], 0, 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0:
            m = sep_re.match(s, i)
            if m:
                parts.append(s[last:i])
                last = i = m.end()
                continue
        i += 1
    parts.append(s[last:])
    return [p.strip() for p in parts]


def fmt_names(field):
    names = []
    for n in split_top(field, re.compile(r"\s+and\s+")):
        if n == "others":
            names.append(None)
            continue
        parts = split_top(n, re.compile(r","))
        if len(parts) == 1:
            names.append(bib_plain(parts[0]))
        elif len(parts) == 2:
            names.append(bib_plain(parts[1] + " " + parts[0]))
        else:
            names.append(bib_plain(parts[2] + " " + parts[0] + ", " + parts[1]))
    if names and names[-1] is None:
        return ", ".join(names[:-1]) + " et\u00a0al."
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def breakable(s):
    return re.sub(r"([/._-])", r"\1<wbr>", esc(s))


def fmt_entry(kind, f):
    """One reference in the style of unsrtnat.bst."""
    blocks = []
    if "author" in f:
        blocks.append(fmt_names(f["author"]))
    if "title" in f:
        blocks.append(bib_plain(sentence_case(f["title"])))
    pages = f.get("pages", "")
    pages_txt = bib_plain(re.sub(r"(?<!-)-(?!-)", "--", pages))
    year = f.get("year", "")
    if kind == "article":
        vnp = f.get("volume", "")
        if f.get("number"):
            vnp += "(%s)" % f["number"]
        if pages:
            vnp = vnp + ":" + pages_txt if vnp else ("pages " if re.search(r"[-,+]", pages) else "page ") + pages_txt
        blocks.append(", ".join(x for x in ["<i>%s</i>" % bib_plain(f.get("journal", "")), vnp, year] if x))
    elif kind in ("inproceedings", "incollection", "conference"):
        parts = ["In <i>%s</i>" % bib_plain(f.get("booktitle", ""))]
        if f.get("volume"):
            parts.append("volume %s" % f["volume"])
        if pages:
            parts.append(("pages " if re.search(r"[-,+]", pages) else "page ") + pages_txt)
        for extra in ("organization", "publisher", "address"):
            if f.get(extra):
                parts.append(bib_plain(f[extra]))
        parts.append(year)
        blocks.append(", ".join(p for p in parts if p))
    else:
        blocks.append(", ".join(x for x in [bib_plain(f.get("howpublished", "")), year] if x))
    if f.get("doi"):
        blocks.append('doi: <a href="https://doi.org/%s">%s</a>' % (esc(f["doi"]), breakable(f["doi"])))
    if f.get("url"):
        blocks.append('URL <a class="url" href="%s">%s</a>' % (esc(f["url"]), breakable(f["url"])))
    if f.get("note"):
        blocks.append(bib_plain(f["note"]))
    out = ""
    for b in blocks:
        if not b:
            continue
        if out:
            out += " " if re.search(r"[.?!]$", re.sub(r"<[^>]+>", "", out)) else ". "
        out += b
    if not re.search(r"[.?!]$", re.sub(r"<[^>]+>", "", out)):
        out += "."
    return out


# ------------------------------------------------------------------------------------------------ LaTeX
FONT_SIZES = {"tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge"}
BLOCK_ENVS = r"longtable|table|figure|tabular|enumerate|itemize|description|center|equation|algorithm"


class Thesis:
    def __init__(self, macros, bib, bib_in_toc):
        self.macros, self.bib, self.bib_in_toc = macros, bib, bib_in_toc
        self.chapter = self.section = self.subsection = 0
        self.fig = self.tab = self.eq = self.alg = 0
        self.appendix = False
        self.labels = {}
        self.target = (None, None)
        self.toc, self.lof, self.lot = [], [], []
        self.cites = []
        self.anchors = []
        self.pending = []
        self.phantom = None
        self.tabcolsep = "6pt"
        self.figures = []
        self.seq = 0

    # ---- numbering
    def chapnum(self):
        return LETTER[self.chapter - 1] if self.appendix else str(self.chapter)

    def anchor(self, name):
        name = slug(name)
        if name in self.anchors:
            self.seq += 1
            name = "%s-%d" % (name, self.seq)
        self.anchors.append(name)
        return name

    def inject(self, block):
        """Put pending anchors inside the next block, so they land on the page the block starts on."""
        if not self.pending:
            return block
        spans = "".join('<span class="anchor" id="%s"></span>' % a for a in self.pending)
        self.pending = []
        j = block.index(">") + 1
        return block[:j] + spans + block[j:]

    # ---- inline text
    def inline(self, s):
        out, closers = [], []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == "\\":
                m = CMD.match(s, i)
                name, i = m.group(1), m.end()
                if name[:1].isalpha() and not name.endswith("*"):
                    while i < n and s[i] in " \t":
                        i += 1
                    if i < n and s[i] == "\n":
                        i += 1
                i = self.inline_command(name, s, i, out, closers)
                continue
            if c == "{":
                j = match_brace(s, i)
                out.append(self.inline(s[i + 1:j]))
                i = j + 1
                continue
            if c == "}":
                i += 1
                continue
            if c == "$":
                j = math_end(s, i)
                out.append(Math.render(s[i + 1:j], self.inline))
                i = j + 1
                continue
            if c == "~":
                out.append("\u00a0")
            elif s.startswith("---", i):
                out.append("\u2014")
                i += 2
            elif s.startswith("--", i):
                out.append("\u2013")
                i += 1
            elif s.startswith("``", i):
                out.append("\u201c")
                i += 1
            elif s.startswith("''", i):
                out.append("\u201d")
                i += 1
            elif c == "`":
                out.append("\u2018")
            elif c == "'":
                out.append("\u2019")
            elif c == "\n":
                out.append(" ")
            elif c == "&":
                warn("& outside a table: " + s[max(0, i - 30):i + 30])
                out.append("&amp;")
            else:
                out.append(esc(c))
            i += 1
        out.extend(reversed(closers))
        return "".join(out)

    def inline_command(self, name, s, i, out, closers):
        simple = {"textbf": ("<b>", "</b>"), "textit": ("<i>", "</i>"), "emph": ("<em>", "</em>"),
                  "texttt": ("<code>", "</code>"), "textsc": ('<span class="sc">', "</span>"),
                  "textrm": ("", ""), "textsf": ("", ""), "mbox": ('<span class="nobr">', "</span>"),
                  "underline": ("<u>", "</u>"), "textnormal": ("", "")}
        chars = {"%": "%", "&": "&amp;", "#": "#", "_": "_", "$": "$", "{": "{", "}": "}", " ": " ",
                 ",": "\u202f", "@": "", "/": "", "-": "\u00ad", "copyright": "\u00a9", "dots": "\u2026",
                 "ldots": "\u2026", "textellipsis": "\u2026", "textendash": "\u2013", "textemdash": "\u2014",
                 "S": "\u00a7", "P": "\u00b6", "textbackslash": "\\", "ss": "\u00df", "LaTeX": "LaTeX", "TeX": "TeX"}
        ignore = {"noindent", "centering", "phantomsection", "par", "relax", "protect", "raggedright", "nobreak",
                  "FloatBarrier", "sloppy", "indent", "null"}
        if name in simple:
            arg, i = read_group(s, i)
            a, b = simple[name]
            out.append(a + self.inline(arg) + b)
        elif name in chars:
            out.append(chars[name])
        elif name in self.macros:
            out.append(self.inline(self.macros[name]))
        elif name == "\\":
            opt, j = read_opt(s, i)
            if opt is not None:
                out.append('<span class="vsp" style="height:%s"></span>' % tex_len(opt))
                i = j
            else:
                out.append("<br>")
        elif name == "newline":
            out.append("<br>")
        elif name in ("cite", "citep", "citet"):
            _, i = read_opt(s, i)
            _, i = read_opt(s, i)
            keys, i = read_group(s, i)
            out.append(self.cite([k.strip() for k in keys.split(",") if k.strip()], name))
        elif name in ("ref", "eqref", "autoref"):
            key, i = read_group(s, i)
            out.append("\ue000%s\ue001" % key.strip())
        elif name == "label":
            key, i = read_group(s, i)
            self.label(key)
        elif name in ("url", "href"):
            url, i = read_group(s, i)
            text = url
            if name == "href":
                text, i = read_group(s, i)
                text = self.inline(text)
            else:
                text = breakable(url)
            out.append('<a class="url" href="%s">%s</a>' % (esc(url), text))
        elif name in ("bfseries", "itshape", "ttfamily", "em"):
            tag = {"bfseries": "b", "itshape": "i", "ttfamily": "code", "em": "em"}[name]
            out.append("<%s>" % tag)
            closers.append("</%s>" % tag)
        elif name in FONT_SIZES:
            out.append('<span class="fs-%s">' % name)
            closers.append("</span>")
        elif name == "rule":
            _, i = read_opt(s, i)
            w, i = read_group(s, i)
            h, i = read_group(s, i)
            out.append('<span class="rule" style="width:%s;border-top-width:%s"></span>' % (tex_len(w), tex_len(h)))
        elif name in ("hspace", "hspace*"):
            w, i = read_group(s, i)
            out.append('<span class="hsp" style="width:%s"></span>' % tex_len(w))
        elif name in ("vspace", "vspace*"):
            h, i = read_group(s, i)
            out.append('<span class="vsp" style="height:%s"></span>' % tex_len(h))
        elif name in ("hfill", "hfil", "quad", "qquad"):
            out.append({"quad": "\u2003", "qquad": "\u2003\u2003"}.get(name, " "))
        elif name == "footnote":
            arg, i = read_group(s, i)
            warn("footnote rendered inline: " + arg[:50])
            out.append(" (" + self.inline(arg) + ")")
        elif name in ignore:
            pass
        elif name == "addcontentsline":
            _, i = read_group(s, i)
            _, i = read_group(s, i)
            _, i = read_group(s, i)
            warn("\\addcontentsline inside a paragraph ignored")
        else:
            warn("unknown text command \\" + name)
        return i

    def cite(self, keys, name):
        nums = []
        for k in keys:
            if k not in self.bib:
                warn("citation without a bibliography entry: " + k)
                continue
            if k not in self.cites:
                self.cites.append(k)
            nums.append(self.cites.index(k) + 1)
        nums = sorted(set(nums))
        groups = []
        for x in nums:
            if groups and x == groups[-1][-1] + 1:
                groups[-1].append(x)
            else:
                groups.append([x])
        link = lambda x: '<a href="#bib-%s">%d</a>' % (slug(self.cites[x - 1]), x)
        parts = []
        for g in groups:
            if len(g) >= 3:
                parts.append(link(g[0]) + "\u2013" + link(g[-1]))
            else:
                parts.extend(link(x) for x in g)
        text = '<span class="cite">[%s]</span>' % ", ".join(parts)
        if name == "citet":
            warn("\\citet rendered as a numeric citation")
        return text

    def label(self, key):
        key = key.strip()
        if key in self.labels:
            warn("label defined twice: " + key)
        if self.target[1] is None:
            warn("label with nothing to refer to: " + key)
        self.labels[key] = self.target

    # ---- blocks
    def blocks(self, src, align=None, top=False):
        out, para = [], []
        state = {"noindent": False, "cont": False, "after_heading": False, "align": align}

        def flush():
            text = "".join(para).strip()
            para.clear()
            if not text:
                return
            body = self.inline(text)
            if not re.sub(r"<span class=\"(anchor|vsp)\"[^>]*></span>|\s", "", body):
                if "vsp" in body:
                    out.append(self.inject('<div class="vgap">%s</div>' % body))
                return
            cls = []
            if state["noindent"] or state["cont"] or state["after_heading"]:
                cls.append("noindent")
            if state["align"] == "center":
                cls.append("center")
            out.append(self.inject('<p%s>%s</p>' % (' class="%s"' % " ".join(cls) if cls else "", body)))
            state["noindent"] = state["cont"] = state["after_heading"] = False

        def emit(block, after_heading=False, cont=False):
            flush()
            out.append(self.inject(block))
            state["after_heading"] = after_heading or (state["after_heading"] and not block.startswith("<h"))
            state["cont"] = cont

        i, n = 0, len(src)
        while i < n:
            c = src[i]
            if c == "\n":
                j = i + 1
                while j < n and src[j] in " \t":
                    j += 1
                if j < n and src[j] == "\n":
                    flush()
                    state["cont"] = False
                    while j < n and src[j] in " \t\n":
                        j += 1
                    i = j
                    continue
                para.append(" ")
                i += 1
                continue
            if c == "$":
                j = math_end(src, i)
                para.append(src[i:j + 1])
                i = j + 1
                continue
            if c == "{" and not "".join(para).strip():
                j = match_brace(src, i)
                inner = src[i + 1:j]
                if re.search(r"\\begin\{(%s)\}" % BLOCK_ENVS, inner):
                    flush()
                    out.append(self.block_group(inner))
                    i = j + 1
                    continue
                para.append(src[i:j + 1])
                i = j + 1
                continue
            if c != "\\":
                para.append(c)
                i += 1
                continue
            m = CMD.match(src, i)
            name, j = m.group(1), m.end()
            base = name.rstrip("*")
            star = name.endswith("*")
            if base in ("chapter", "section", "subsection"):
                short, j = read_opt(src, j)
                title, j = read_group(src, j)
                emit(self.heading(base, star, title), after_heading=True)
                i = j
            elif name == "begin":
                env, j = read_group(src, j)
                cs = j
                ce, after = env_end(src, cs, env)
                i = self.environment(env, src[cs:ce], out, para, flush, emit, state)
                i = after
            elif name in ("vspace", "vspace*"):
                h, j = read_group(src, j)
                flush()
                out.append(self.inject('<div class="vgap" style="height:%s"></div>' % tex_len(h)))
                i = j
            elif name == "vfill":
                flush()
                out.append('<div class="vfill"></div>')
                i = j
            elif name in ("clearpage", "newpage", "cleardoublepage"):
                flush()
                out.append('<div class="pagebreak"></div>')
                i = j
            elif name == "par":
                flush()
                i = j
            elif name == "noindent":
                if not "".join(para).strip():
                    state["noindent"] = True
                    j = skip_ws(src, j)
                i = j
            elif name == "centering":
                state["align"] = "center"
                i = j
            elif name == "label":
                key, j = read_group(src, j)
                self.label(key)
                i = j
            elif name == "phantomsection":
                a = self.anchor("ph-%d" % (len(self.anchors) + 1))
                self.phantom = a
                if out and out[-1].startswith("<h1") and not "".join(para).strip():
                    k = out[-1].index(">") + 1
                    out[-1] = out[-1][:k] + '<span class="anchor" id="%s"></span>' % a + out[-1][k:]
                else:
                    self.pending.append(a)
                i = j
            elif name == "addcontentsline":
                where, j = read_group(src, j)
                level, j = read_group(src, j)
                title, j = read_group(src, j)
                anchor = self.phantom or self.target[1]
                if where == "toc":
                    self.toc.append(({"chapter": 0, "section": 1, "subsection": 2}[level], None, self.inline(title), anchor))
                self.phantom = None
                i = j
            elif name in ("tableofcontents", "listoffigures", "listoftables"):
                flush()
                key = {"tableofcontents": "toc", "listoffigures": "lof", "listoftables": "lot"}[name]
                title = {"toc": "Contents", "lof": "List of Figures", "lot": "List of Tables"}[key]
                a = self.anchor("list-" + key)
                out.append(self.inject('<h1 class="chapter star" id="%s"><span class="chap-title">%s</span></h1>'
                                       % (a, title)))
                out.append("<!--LIST:%s-->" % key)
                i = j
            elif name == "pagenumbering":
                style, j = read_group(src, j)
                flush()
                self.pending.append(self.anchor("pn-" + style))
                i = j
            elif name == "setcounter":
                _, j = read_group(src, j)
                _, j = read_group(src, j)
                i = j
            elif name in ("onehalfspacing", "singlespacing", "doublespacing"):
                flush()
                if top:
                    out.append('</div><div class="sp-%s">' % name.replace("spacing", ""))
                else:
                    warn("\\%s below the top level ignored" % name)
                i = j
            elif name == "bibliographystyle":
                _, j = read_group(src, j)
                i = j
            elif name == "bibliography":
                _, j = read_group(src, j)
                flush()
                a = self.anchor("bibliography")
                if self.bib_in_toc:
                    self.toc.append((0, None, "Bibliography", a))
                out.append(self.inject('<h1 class="chapter star" id="%s"><span class="chap-title">Bibliography</span></h1>'
                                       % a))
                out.append("<!--BIBLIOGRAPHY-->")
                i = j
            elif name == "appendix":
                flush()
                self.appendix = True
                self.chapter = 0
                i = j
            elif name == "setlength":
                what, j = read_group(src, j)
                val, j = read_group(src, j)
                if what.strip() == "\\tabcolsep":
                    self.tabcolsep = tex_len(val)
                i = j
            else:
                para.append(src[i:j])
                i = j
        flush()
        return "\n".join(out)

    def block_group(self, inner):
        """{\\footnotesize \\setlength{\\tabcolsep}{4pt} \\begin{longtable}...}: declarations scope the tables."""
        saved = self.tabcolsep
        size = None
        m = re.match(r"\s*\\(%s)\b" % "|".join(FONT_SIZES), inner)
        if m:
            size = m.group(1)
            inner = inner[m.end():]
        body = self.blocks(inner)
        self.tabcolsep = saved
        return '<div class="fs-%s">%s</div>' % (size, body) if size else body

    def heading(self, level, star, title_tex):
        title = self.inline(title_tex)
        if level == "chapter":
            if star:
                a = self.anchor("chap-" + re.sub(r"<[^>]+>", "", title).lower())
                self.target = (None, a)
                return '<h1 class="chapter star" id="%s"><span class="chap-title">%s</span></h1>' % (a, title)
            self.chapter += 1
            self.section = self.subsection = self.fig = self.tab = self.eq = 0
            num = self.chapnum()
            a = self.anchor(("app-" if self.appendix else "ch-") + num)
            self.target = (num, a)
            self.toc.append((0, num, title, a))
            for lst in (self.lof, self.lot):
                if lst and lst[-1] != "gap":
                    lst.append("gap")
            word = "Appendix" if self.appendix else "Chapter"
            return ('<h1 class="chapter" id="%s"><span class="chap-label">%s %s</span>'
                    '<span class="chap-title">%s</span></h1>' % (a, word, num, title))
        if level == "section":
            if star:
                return '<h2 class="section">%s</h2>' % title
            self.section += 1
            self.subsection = 0
            num = "%s.%d" % (self.chapnum(), self.section)
            a = self.anchor("sec-" + num)
            self.target = (num, a)
            self.toc.append((1, num, title, a))
            return ('<h2 class="section" id="%s"><span class="secnum">%s</span><span class="sectitle">%s</span></h2>'
                    % (a, num, title))
        self.subsection += 1
        num = "%s.%d.%d" % (self.chapnum(), self.section, self.subsection)
        a = self.anchor("sec-" + num)
        self.target = (num, a)
        self.toc.append((2, num, title, a))
        return ('<h3 class="subsection" id="%s"><span class="secnum">%s</span><span class="sectitle">%s</span></h3>'
                % (a, num, title))

    # ---- environments
    def environment(self, env, content, out, para, flush, emit, state):
        if env in ("equation", "equation*"):
            emit(self.equation(content, env == "equation"), cont=True)
        elif env == "figure":
            emit(self.figure(content))
        elif env == "table":
            emit(self.table(content))
        elif env == "longtable":
            emit(self.longtable(content))
        elif env == "tabular":
            spec, rest = read_group(content, 0)
            emit('<div class="plain-tabular%s">%s</div>' % (" center" if state["align"] == "center" else "",
                                                            self.tabular(spec, content[rest:], "plain")))
        elif env in ("enumerate", "itemize"):
            items = self.items(content)
            tag = "ol" if env == "enumerate" else "ul"
            emit("<%s class=\"list\">%s</%s>" % (tag, "".join("<li>%s</li>" % self.inline(b) for _, b in items), tag),
                 cont=True)
        elif env == "description":
            items = self.items(content)
            emit('<div class="desc">%s</div>' % "".join(
                '<div class="desc-item"><span class="desc-label">%s</span> %s</div>'
                % (self.inline(lab or ""), self.inline(b)) for lab, b in items), cont=True)
        elif env == "center":
            emit('<div class="center">%s</div>' % self.blocks(content, align="center"))
        elif env == "titlepage":
            emit('<section class="titlepage">%s</section>' % self.blocks(content, align="center"))
        elif env == "algorithm":
            emit(self.algorithm(content))
        elif env == "adjustbox":
            _, rest = read_group(content, 0)
            emit(self.blocks(content[rest:]))
        else:
            warn("unknown environment " + env)
            para.append(content)
        return 0

    @staticmethod
    def items(content):
        parts = []
        for m in re.finditer(r"\\item\b", content):
            parts.append(m.start())
        items = []
        for k, start in enumerate(parts):
            end = parts[k + 1] if k + 1 < len(parts) else len(content)
            body = content[start + 5:end]
            label, j = read_opt(body, 0)
            items.append((label, body[j:].strip() if label is not None else body.strip()))
        return items

    def equation(self, content, numbered):
        labels = re.findall(r"\\label\{([^}]*)\}", content)
        content = re.sub(r"\\label\{[^}]*\}", "", content)
        num = ""
        a = None
        if numbered:
            self.eq += 1
            num = "(%s.%d)" % (self.chapnum(), self.eq)
            a = self.anchor("eq-%s-%d" % (self.chapnum(), self.eq))
            self.target = (num.strip("()"), a)
        for lab in labels:
            self.label(lab)
        return ('<div class="equation"%s><div class="eq-body">%s</div><div class="eq-num">%s</div></div>'
                % (' id="%s"' % a if a else "", Math.render(content.strip(), self.inline, display=True), num))

    def caption_of(self, content):
        m = re.search(r"\\caption\s*(\[)?", content)
        if not m:
            return None, content
        i = m.end() - (1 if m.group(1) else 0)
        _, i = read_opt(content, i)
        text, j = read_group(content, i)
        return text, content[:m.start()] + content[j:]

    def figure(self, content):
        cap, rest = self.caption_of(content)
        self.fig += 1
        num = "%s.%d" % (self.chapnum(), self.fig)
        a = self.anchor("fig-" + num)
        self.target = (num, a)
        for lab in re.findall(r"\\label\{([^}]*)\}", rest):
            self.label(lab)
        imgs = []
        for m in re.finditer(r"\\includegraphics\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}", rest):
            width = "100%"
            for opt in (m.group(1) or "").split(","):
                k, _, v = opt.partition("=")
                if k.strip() == "width":
                    width = tex_len(v)
            path = m.group(2).strip()
            self.figures.append(path)
            svg = "figures/" + os.path.splitext(os.path.basename(path))[0] + ".svg"
            imgs.append('<img src="%s" style="width:%s" alt="">' % (svg, width))
        cap_html = self.inline(cap or "")
        self.lof.append((num, cap_html, a))
        return ('<figure class="float" id="%s">%s<figcaption><span class="caplabel">Figure %s:</span> %s'
                '</figcaption></figure>' % (a, "".join(imgs), num, cap_html))

    def table(self, content):
        cap, rest = self.caption_of(content)
        self.tab += 1
        num = "%s.%d" % (self.chapnum(), self.tab)
        a = self.anchor("tab-" + num)
        self.target = (num, a)
        for lab in re.findall(r"\\label\{([^}]*)\}", rest):
            self.label(lab)
        size = next((s for s in ("scriptsize", "footnotesize", "small") if re.search(r"\\%s\b" % s, rest)), None)
        m = re.search(r"\\begin\{tabular\}", rest)
        spec, j = read_group(rest, m.end())
        ce, _ = env_end(rest, j, "tabular")
        cap_html = self.inline(cap or "")
        self.lot.append((num, cap_html, a))
        return ('<div class="float tablefloat" id="%s"><div class="caption"><span class="caplabel">Table %s:</span> %s</div>'
                '<div class="fit%s">%s</div></div>'
                % (a, num, cap_html, " fs-" + size if size else "", self.tabular(spec, rest[j:ce], "booktabs")))

    def longtable(self, content):
        spec, j = read_group(content, 0)
        body = content[j:]
        parts = re.split(r"\\(endfirsthead|endhead|endfoot|endlastfoot)\b", body)
        before = {parts[k]: parts[k - 1] for k in range(1, len(parts), 2)}  # each marker ends the part before it
        first = before.get("endfirsthead", before.get("endhead", parts[0]))
        cap, first = self.caption_of(first)
        self.tab += 1
        num = "%s.%d" % (self.chapnum(), self.tab)
        a = self.anchor("tab-" + num)
        self.target = (num, a)
        for lab in re.findall(r"\\label\{([^}]*)\}", first):
            self.label(lab)
        first = re.sub(r"\\label\{[^}]*\}", "", first)
        first = re.sub(r"^\s*\\\\", "", first)
        rows = parts[-1] if len(parts) > 1 else ""
        foot = before.get("endfoot", "")
        cap_html = self.inline(cap or "")
        self.lot.append((num, cap_html, a))
        table = self.tabular(spec, first + rows + foot, "booktabs", long=True)
        return ('<div class="longtable-wrap" id="%s"><div class="caption"><span class="caplabel">Table %s:</span> %s</div>'
                '%s</div>' % (a, num, cap_html, table))

    def tabular(self, spec, body, style, long=False):
        cols = self.colspec(spec)
        items = []
        for row, extra in self.rows(body):
            while True:
                m = re.match(r"\s*\\(toprule|midrule|bottomrule|hline|cmidrule(?:\([^)]*\))?\{[^}]*\})", row)
                if not m:
                    break
                kind = m.group(1)
                items.append(("rule", "mid" if kind.startswith("cmidrule") or kind == "hline" else kind[:-4]))
                row = row[m.end():]
            if row.strip():
                items.append(("row", self.cells(row), extra))
        head_end = -1
        if style == "booktabs":
            mids = [k for k, it in enumerate(items) if it == ("rule", "mid")]
            if mids:
                head_end = mids[0]
        html_rows = []
        pending = None
        for k, it in enumerate(items):
            if it[0] == "rule":
                pending = it[1]
                if html_rows and pending in ("mid", "bottom"):
                    html_rows[-1][1].append("r-pre" if pending == "mid" else "r-bottom")
                continue
            cls = []
            if pending in ("top", "mid"):
                cls.append("r-" + pending)
            pending = None
            html_rows.append((k < head_end, cls, it[1], it[2]))
        out = []
        for is_head, cls, cells, extra in html_rows:
            tag = "th" if is_head else "td"
            tds, col = [], 0
            for cell in cells:
                span, align, content = 1, None, cell
                mm = re.match(r"\s*\\multicolumn\s*\{(\d+)\}", cell)
                if mm:
                    span = int(mm.group(1))
                    cspec, j = read_group(cell, mm.end())
                    content, _ = read_group(cell, j)
                    align = self.colspec(cspec)[0]["align"]
                c = cols[col] if col < len(cols) else {"align": "l", "width": None, "l": None, "r": None}
                style_ = []
                if c.get("width") and span == 1:
                    style_.append("width:%s" % c["width"])
                lp = c.get("l") if col > 0 or c.get("l") is not None else None
                rp = cols[min(col + span - 1, len(cols) - 1)].get("r") if cols else None
                if lp is not None:
                    style_.append("padding-left:%s" % lp)
                if rp is not None:
                    style_.append("padding-right:%s" % rp)
                if extra:
                    style_.append("padding-bottom:%s" % tex_len(extra))
                attrs = ' class="al-%s"' % (align or c["align"])
                if span > 1:
                    attrs += ' colspan="%d"' % span
                if style_:
                    attrs += ' style="%s"' % ";".join(style_)
                tds.append("<%s%s>%s</%s>" % (tag, attrs, self.inline(content.strip()), tag))
                col += span
            out.append((is_head, '<tr%s>%s</tr>' % (' class="%s"' % " ".join(cls) if cls else "", "".join(tds))))
        head = "".join(r for h, r in out if h)
        rows = "".join(r for h, r in out if not h)
        cls = "tab %s%s" % (style, " long" if long else "")
        return '<table class="%s" style="--tcs:%s">%s<tbody>%s</tbody></table>' % (
            cls, self.tabcolsep, "<thead>%s</thead>" % head if head else "", rows)

    @staticmethod
    def colspec(spec):
        cols, pending, i = [], None, 0
        while i < len(spec):
            c = spec[i]
            if c in "lcr":
                cols.append({"align": c, "width": None, "l": pending, "r": None})
                pending = None
                i += 1
            elif c in "pmb":
                w, i = read_group(spec, i + 1)
                cols.append({"align": "p", "width": tex_len(w), "l": pending, "r": None})
                pending = None
            elif c == "@":
                g, i = read_group(spec, i + 1)
                hm = re.search(r"\\hspace\*?\{([^}]*)\}", g)
                gap = tex_len(hm.group(1)) if hm else "0"
                if cols:
                    cols[-1]["r"] = gap
                pending = "0"
            elif c in "<>":
                _, i = read_group(spec, i + 1)
            elif c == "*":
                count, i = read_group(spec, i + 1)
                sub, i = read_group(spec, i)
                spec = spec[:i] + sub * int(count) + spec[i:]
            else:
                i += 1
        return cols

    @staticmethod
    def rows(body):
        rows, depth, start, i = [], 0, 0, 0
        while i < len(body):
            c = body[i]
            if c == "\\":
                if body.startswith("\\\\", i):
                    content = body[start:i]
                    i += 2
                    opt, j = read_opt(body, i)
                    if opt is not None:
                        i = j
                    rows.append((content, opt))
                    start = i
                    continue
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "$":
                i = math_end(body, i)
            i += 1
        if body[start:].strip():
            rows.append((body[start:], None))
        return rows

    @staticmethod
    def cells(row):
        cells, depth, start, i = [], 0, 0, 0
        while i < len(row):
            c = row[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "$":
                i = math_end(row, i)
            elif c == "&" and depth == 0:
                cells.append(row[start:i])
                start = i + 1
            i += 1
        cells.append(row[start:])
        return cells

    def algorithm(self, content):
        cap, rest = self.caption_of(content)
        self.alg += 1
        num = str(self.alg)
        a = self.anchor("alg-" + num)
        self.target = (num, a)
        for lab in re.findall(r"\\label\{([^}]*)\}", rest):
            self.label(lab)
        m = re.search(r"\\begin\{algorithmic\}", rest)
        opt, j = read_opt(rest, m.end())
        ce, _ = env_end(rest, j, "algorithmic")
        numbered = opt is not None and opt.strip() not in ("", "0")
        return ('<div class="float algorithm" id="%s"><div class="alg-cap"><b>Algorithm %s</b> %s</div>'
                '<div class="alg-body">%s</div></div>' % (a, num, self.inline(cap or ""),
                                                          self.algorithmic(rest[j:ce], numbered)))

    def algorithmic(self, body, numbered):
        kw = r"Require|Ensure|State|Statex|For|ForAll|EndFor|While|EndWhile|If|ElsIf|Else|EndIf|Loop|EndLoop|Repeat|Until"
        pieces = re.split(r"\\(%s)\b" % kw, body)
        lines, depth, n = [], 0, 0
        for k in range(1, len(pieces), 2):
            cmd, text = pieces[k], pieces[k + 1]
            arg = ""
            if cmd in ("For", "ForAll", "While", "If", "ElsIf", "Until"):
                arg, j = read_group(text, 0)
                text = text[j:]
            comment = ""
            cm = re.search(r"\\Comment\s*\{", text)
            if cm:
                ctext, j = read_group(text, cm.end() - 1)
                comment = ctext
                text = text[:cm.start()] + text[j:]
            text = re.sub(r"\\Return\b", r"\\textbf{return}", text)
            html_ = self.inline(text.strip())
            arg_html = self.inline(arg.strip())
            if cmd in ("EndFor", "EndWhile", "EndIf", "EndLoop", "Else", "ElsIf", "Until"):
                depth = max(0, depth - 1)
            words = {"Require": "<b>Require:</b> " + html_, "Ensure": "<b>Ensure:</b> " + html_,
                     "State": html_, "Statex": html_,
                     "For": "<b>for</b> %s <b>do</b>" % arg_html, "ForAll": "<b>for all</b> %s <b>do</b>" % arg_html,
                     "EndFor": "<b>end for</b>", "While": "<b>while</b> %s <b>do</b>" % arg_html,
                     "EndWhile": "<b>end while</b>", "If": "<b>if</b> %s <b>then</b>" % arg_html,
                     "ElsIf": "<b>else if</b> %s <b>then</b>" % arg_html, "Else": "<b>else</b>",
                     "EndIf": "<b>end if</b>", "Loop": "<b>loop</b>", "EndLoop": "<b>end loop</b>",
                     "Repeat": "<b>repeat</b>", "Until": "<b>until</b> " + arg_html}
            line_no = ""
            if numbered and cmd not in ("Require", "Ensure", "Statex"):
                n += 1
                line_no = '<span class="alg-no">%d:</span>' % n
            ind = 0 if cmd in ("Require", "Ensure") else depth
            com = '<span class="alg-comment">\u25b7 %s</span>' % self.inline(comment) if comment else ""
            lines.append('<div class="alg-line" style="--depth:%d">%s%s%s</div>' % (ind, line_no, com, words[cmd]))
            if cmd in ("For", "ForAll", "While", "If", "ElsIf", "Else", "Loop", "Repeat"):
                depth += 1
        return "".join(lines)

    # ---- lists and bibliography
    def entry(self, cls, num, title, anchor, pages, dots=True):
        pg = pages.get(anchor, "000")
        numspan = '<span class="num">%s</span>' % num if num else ""
        return ('<div class="toc-entry %s%s%s">%s<span class="t"><a href="#%s">%s</a></span><span class="pg">%s</span></div>'
                % (cls, " dots" if dots else "", "" if num else " nonum", numspan, anchor, title, pg))

    def list_html(self, key, pages):
        if key == "toc":
            return '<div class="toc">%s</div>' % "".join(
                self.entry("l%d" % lvl, num, title, a, pages, dots=lvl > 0) for lvl, num, title, a in self.toc)
        rows = []
        for it in (self.lof if key == "lof" else self.lot):
            if it == "gap":
                rows.append('<div class="toc-gap"></div>')
            else:
                rows.append(self.entry("l1", it[0], it[1], it[2], pages))
        return '<div class="toc">%s</div>' % "".join(rows)

    def bibliography_html(self):
        if not self.cites:
            return ""
        width = len("[%d]" % len(self.cites)) * 0.55 + 0.2
        rows = []
        for k, key in enumerate(self.cites, 1):
            kind, fields = self.bib[key]
            rows.append('<div class="bib-entry" id="bib-%s"><span class="bib-label">[%d]</span><div class="bib-text">%s</div></div>'
                        % (slug(key), k, fmt_entry(kind, fields)))
        return '<div class="bibliography" style="--bibw:%.2fem">%s</div>' % (width, "".join(rows))


# ------------------------------------------------------------------------------------------------ page
CSS = r"""
@page { size: A4; margin: 25.4mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; width: 159.2mm; font-family: Cambria, "Cambria Math", Georgia, serif; font-size: 11pt; color: #000;
  background: #fff; text-align: justify; hyphens: auto; font-kerning: normal; text-rendering: geometricPrecision; }
math { font-family: "Cambria Math", math; }
a { color: inherit; text-decoration: none; }
code { font-family: Consolas, "Courier New", monospace; font-size: 0.9em; overflow-wrap: anywhere; hyphens: none; }
.nobr { white-space: nowrap; }
.sc { font-variant: small-caps; }
p { margin: 0; text-indent: 1.5em; orphans: 2; widows: 2; }
p.noindent { text-indent: 0; }
p.center, .center p { text-align: center; text-indent: 0; }
.sp-single { line-height: 1.22; }
.sp-onehalf { line-height: 1.5; }
.sp-onehalf .toc, .sp-onehalf table.tab, .sp-onehalf .caption, .sp-onehalf figcaption { line-height: 1.25; }
.vgap { display: block; }
.vsp { display: block; }
.hsp { display: inline-block; }
.vfill { flex: 1 1 auto; }
.pagebreak { break-after: page; height: 0; }
.rule { display: inline-block; height: 0; border-top: 0.4pt solid #000; vertical-align: baseline; }
.anchor { display: inline; }
.fs-tiny { font-size: 0.5em; } .fs-scriptsize { font-size: 0.667em; } .fs-footnotesize { font-size: 0.833em; }
.fs-small { font-size: 0.913em; } .fs-normalsize { font-size: 1em; } .fs-large { font-size: 1.2em; }
.fs-Large { font-size: 1.44em; } .fs-LARGE { font-size: 1.728em; } .fs-huge { font-size: 2.074em; } .fs-Huge { font-size: 2.488em; }

h1.chapter { break-before: page; break-after: avoid; margin: 4.2em 0 3.3em; font-size: 1em; font-weight: bold;
  text-align: left; hyphens: manual; line-height: 1.15; }
h1.chapter .chap-label { display: block; font-size: 2.07em; margin-bottom: 0.8em; }
h1.chapter .chap-title { display: block; font-size: 2.07em; }
/* the number hangs, a quad before the title, and the title wraps under itself, as LaTeX's \@hangfrom does */
h2.section, h3.subsection { display: flex; align-items: baseline; font-weight: bold; break-after: avoid;
  break-inside: avoid; text-align: left; hyphens: manual; line-height: 1.25; }
h2.section { font-size: 1.36em; margin: 1.15em 0 0.55em; }
h3.subsection { font-size: 1.17em; margin: 1em 0 0.4em; }
h2 .secnum, h3 .secnum { flex: none; margin-right: 1em; }

.titlepage { height: 245mm; display: flex; flex-direction: column; text-align: center; break-after: page; line-height: 1.3; }
.titlepage p { text-indent: 0; text-align: center; }
.titlepage > * { flex-shrink: 0; }

figure.float { margin: 1.2em 0; break-inside: avoid; text-align: center; }
figure.float img { display: block; margin: 0 auto; }
.caption, figcaption { font-size: 0.913em; line-height: 1.3; display: table; margin: 0.55em auto 0; text-align: justify;
  text-indent: 0; hyphens: auto; }
.tablefloat .caption, .longtable-wrap .caption { margin: 0 auto 0.5em; }
.caplabel { font-weight: bold; }
.float.tablefloat { margin: 1.2em 0; break-inside: avoid; }
.fit { width: 100%; }
.fit > table { margin: 0 auto; }
table.tab { border-collapse: collapse; font-variant-numeric: lining-nums tabular-nums; line-height: 1.25;
  text-align: left; hyphens: manual; }
table.tab td, table.tab th { padding: 0.13em var(--tcs, 6pt); vertical-align: top; font-weight: normal; }
table.tab th { vertical-align: bottom; }
table.tab .al-l { text-align: left; } table.tab .al-c { text-align: center; } table.tab .al-r { text-align: right; }
table.tab .al-p { text-align: left; }
table.booktabs td.al-l, table.booktabs td.al-c, table.booktabs td.al-r { white-space: nowrap; }
table.plain td { white-space: nowrap; }
table.booktabs th.al-l, table.booktabs th.al-c, table.booktabs th.al-r { white-space: normal; }
tr.r-top > * { border-top: 0.08em solid #000; padding-top: calc(0.65ex + 0.13em); }
tr.r-mid > * { border-top: 0.05em solid #000; padding-top: calc(0.65ex + 0.13em); }
tr.r-pre > * { padding-bottom: calc(0.4ex + 0.13em); }
tr.r-bottom > * { border-bottom: 0.08em solid #000; padding-bottom: calc(0.4ex + 0.13em); }
table.long tr { break-inside: avoid; }
table.long thead { display: table-header-group; }
.longtable-wrap { margin: 1em 0; }
.longtable-wrap > table { margin: 0 auto; }
.plain-tabular { margin: 0.3em 0; }
.plain-tabular.center > table, .center .plain-tabular > table { margin: 0 auto; }
table.plain td { padding: 0.05em var(--tcs, 6pt); vertical-align: baseline; }

.equation { display: grid; grid-template-columns: 3.4em 1fr 3.4em; align-items: center; margin: 0.75em 0;
  break-inside: avoid; text-indent: 0; }
.equation .eq-body { grid-column: 2; text-align: center; }
.equation .eq-num { grid-column: 3; text-align: right; }

ol.list, ul.list { margin: 0.35em 0; padding-left: 2.5em; }
ol.list li, ul.list li { margin: 0.15em 0; padding-left: 0.2em; }
.desc { margin: 0.35em 0; }
.desc-item { padding-left: 2.5em; text-indent: -2em; margin: 0.15em 0; }
.desc-label { font-weight: bold; }

.algorithm { margin: 1.2em 0; break-inside: avoid; border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000;
  text-align: left; hyphens: manual; }
.alg-cap { border-bottom: 0.4pt solid #000; padding: 0.25em 0; }
.alg-body { padding: 0.3em 0 0.45em; line-height: 1.38; }
.alg-line { position: relative; padding-left: calc(2.3em + var(--depth) * 1.5em); }
.alg-no { position: absolute; left: 0; width: 1.7em; text-align: right; font-size: 0.83em; line-height: 1.66; }
.alg-comment { float: right; margin-left: 1em; }

.toc { line-height: 1.25; text-align: left; hyphens: manual; }
.toc-entry { position: relative; padding-right: 2.6em; margin: 0.08em 0; }
.toc-entry .num { display: inline-block; text-indent: 0; background: #fff; }
.toc-entry .t { background: #fff; padding-right: 0.3em; }
.toc-entry .pg { position: absolute; right: 0; bottom: 0; background: #fff; padding-left: 0.3em; text-indent: 0; }
.toc-entry.l0 { font-weight: bold; margin-top: 0.7em; padding-left: 1.6em; text-indent: -1.6em; }
.toc-entry.l0 .num { width: 1.6em; }
.toc-entry.l0.nonum { padding-left: 0; text-indent: 0; }
.toc-entry.l1 { padding-left: 3.9em; text-indent: -2.4em; }
.toc-entry.l1 .num { width: 2.4em; }
.toc-entry.l2 { padding-left: 7.2em; text-indent: -3.3em; }
.toc-entry.l2 .num { width: 3.3em; }
.toc-entry.dots { background-image: radial-gradient(circle at 50% 74%, #000 0.055em, transparent 0.075em);
  background-size: 0.5em 1.25em; background-repeat: repeat-x; background-position: left bottom;
  background-clip: content-box; background-origin: content-box; }
.toc-gap { height: 0.7em; }

.bibliography { line-height: 1.22; }
.bib-entry { display: grid; grid-template-columns: var(--bibw) 1fr; column-gap: 0.55em; margin: 0 0 0.45em; break-inside: avoid; }
.bib-label { text-align: right; }
.bib-text { text-align: justify; hyphens: auto; }
.url { font-family: Consolas, "Courier New", monospace; font-size: 0.9em; }
"""

JS = r"""
<script>
(function () {
  // adjustbox{max width=\textwidth}: shrink a table that is still wider than the text once its headers wrap
  document.querySelectorAll('.fit').forEach(function (box) {
    var t = box.firstElementChild;
    if (!t) return;
    var need = t.getBoundingClientRect().width, avail = box.getBoundingClientRect().width;
    if (need > avail + 0.5) { t.style.zoom = (avail / need).toFixed(4); t.setAttribute('data-zoom', t.style.zoom); }
  });
  // an equation whose formula is wider than the space between the equation-number columns
  document.querySelectorAll('.eq-body').forEach(function (b) {
    var m = b.firstElementChild;
    if (!m) return;
    var r = document.createRange();
    r.selectNodeContents(m);
    var w = r.getBoundingClientRect().width, line = b.parentElement.getBoundingClientRect().width;
    b.setAttribute('data-width', (w / line).toFixed(3));
    if (w > b.getBoundingClientRect().width + 0.5) b.setAttribute('data-overfull', (w / line).toFixed(3));
  });
})();
</script>
"""


def glue_math(content):
    """TeX cannot break between a formula and a character touching it ("[$-$0.0057,"); a browser can."""
    return re.sub(r"([^\s>]*)(<math>.*?</math>)([^\s<]*)",
                  lambda m: '<span class="nobr">%s</span>' % m.group(0) if (m.group(1) or m.group(3)) else m.group(0),
                  content)


def page(title, body, anchors):
    links = "".join('<a href="#%s"></a>' % a for a in anchors)
    return ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><title>%s</title><style>%s</style></head>'
            '<body><div class="sp-single">%s</div><div style="display:none">%s</div>%s</body></html>'
            % (esc(title), CSS, body, links, JS))


def load(path):
    return io.open(path, encoding="utf-8").read()


def expand_inputs(src, base):
    def repl(m):
        name = m.group(1).strip()
        path = os.path.join(base, name if name.endswith(".tex") else name + ".tex")
        return expand_inputs(strip_comments(load(path)), base)
    return re.sub(r"\\input\{([^}]*)\}", repl, src)


def print_pdf(browser, html_path, pdf_path):
    profile = tempfile.mkdtemp()
    try:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--disable-extensions", "--user-data-dir=" + profile, "--no-pdf-header-footer",
               "--print-to-pdf=" + pdf_path, "file:///" + html_path.replace("\\", "/")]
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if not os.path.exists(pdf_path):
            sys.exit("printing failed (%s): %s" % (r.returncode, r.stderr[-1500:]))
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def dump_dom(browser, html_path):
    profile = tempfile.mkdtemp()
    try:
        r = subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--user-data-dir=" + profile,
                            "--dump-dom", "file:///" + html_path.replace("\\", "/")],
                           capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace")
        return r.stdout
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def destinations(pdf_path):
    """Named destination -> (page index, distance of its top below the top of the text area, in pt)."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    height = float(reader.pages[0].mediabox.height)
    out = {}
    for name, dest in reader.named_destinations.items():
        top = dest.get("/Top")
        # Chromium writes the destination relative to the page area inside the margins
        out[name.lstrip("/")] = (reader.get_destination_page_number(dest), None if top is None else height - float(top))
    return out, len(reader.pages)


FLOAT_LINE = re.compile(r'^<(?:figure class="float"|div class="float (?:tablefloat|algorithm)") id="([^"]+)"', re.M)
FLOWS_PAST = ("<p>", "<p ", '<div class="equation"', '<ol class="list"', '<ul class="list"', '<div class="desc"')


def apply_moves(content, moves):
    """Place each deferred float after the next moves[id] blocks of running text, never past a heading."""
    if not moves:
        return content
    lines = content.split("\n")
    for fid, k in moves.items():
        idx = next(i for i, ln in enumerate(lines) if (FLOAT_LINE.match(ln) or [None, None])[1] == fid)
        line = lines.pop(idx)
        j = idx
        for _ in range(k):
            if j < len(lines) and lines[j].startswith(FLOWS_PAST):
                j += 1
        lines.insert(j, line)
    return "\n".join(lines)


def floats_to_defer(pdf_path, dests, float_ids, margin_pt=72.0, min_gap_pt=45.0):
    """Floats that open a page while the page before ends in a gap a float left because it did not fit."""
    import fitz
    doc = fitz.open(pdf_path)
    out = []
    for name, (p, y) in dests.items():
        if name not in float_ids or p == 0 or y is None or y > 6:
            continue
        prev = doc[p - 1]
        # Chromium paints every page white first, and white is not ink
        white = lambda c: c is None or all(x > 0.98 for x in c)
        ink = [d["rect"].y1 for d in prev.get_drawings() if not (white(d.get("fill")) and white(d.get("color")))]
        bottom = max([b[3] for b in prev.get_text("blocks")] + ink + [0.0])
        if prev.rect.height - margin_pt - bottom > min_gap_pt:
            out.append(name)
    doc.close()
    return out


def labels_for(dests, npages):
    r0, a0 = (dests.get(k, (None,))[0] for k in ("pn-roman", "pn-arabic"))
    labels = []
    for p in range(npages):
        if a0 is not None and p >= a0:
            labels.append(str(p - a0 + 1))
        elif r0 is not None and p >= r0:
            labels.append(roman(p - r0 + 1))
        else:
            labels.append(None)
    return labels


def finalize(raw_pdf, out_pdf, labels, thesis, dests, title, authors):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import Fit
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font = "Times-Roman"
    for path, idx in ((r"C:\Windows\Fonts\cambria.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", None)):
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("PageFont", path, subfontIndex=idx) if idx is not None else TTFont("PageFont", path))
                font = "PageFont"
                break
            except Exception:
                pass
    overlay_path = raw_pdf + ".numbers.pdf"
    c = canvas.Canvas(overlay_path, pagesize=A4)
    w, _ = A4
    for lab in labels:
        if lab:
            c.setFont(font, 11)
            c.drawCentredString(w / 2, 42, lab)
        c.showPage()
    c.save()
    overlay = PdfReader(overlay_path)
    writer = PdfWriter(clone_from=raw_pdf)
    for i, pg in enumerate(writer.pages):
        pg.merge_page(overlay.pages[i])
        pg.compress_content_streams()
    # page labels as the reader sees them: none on the title page, then roman, then arabic
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        kind = lambda x: None if x is None else ("D" if x.isdigit() else "r")
        if i == len(labels) or kind(labels[i]) != kind(labels[start]):
            runs.append((start, i - 1, kind(labels[start])))
            start = i
    for a, b, kind in runs:
        if kind is None:
            writer.set_page_label(a, b, prefix="Title")
        else:
            writer.set_page_label(a, b, style="/" + kind, start=1)
    parents = {}
    for level, num, ttl, anchor in thesis.toc:
        if anchor not in dests:
            continue
        text = re.sub(r"<[^>]+>", "", ttl)
        text = html.unescape(("%s  %s" % (num, text)) if num else text)
        parent = parents.get(level - 1) if level > 0 else None
        item = writer.add_outline_item(text, dests[anchor][0], parent=parent, fit=Fit.fit())
        parents[level] = item
        for deeper in [k for k in parents if k > level]:
            del parents[deeper]
    writer.page_mode = "/UseOutlines"
    writer.add_metadata({"/Title": title, "/Author": authors, "/Subject": "B.Sc. thesis, BRAC University (draft)",
                         "/Creator": "paper/thesis/render_pdf.py"})
    merged = raw_pdf + ".merged.pdf"
    with open(merged, "wb") as f:
        writer.write(f)
    os.remove(overlay_path)
    # merging leaves the replaced content streams behind as orphans: drop unused objects and recompress
    import fitz
    doc = fitz.open(merged)
    doc.save(out_pdf, garbage=4, deflate=True, deflate_fonts=True, clean=True)
    doc.close()
    os.remove(merged)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "build"), help="directory for the HTML, figures and PDF")
    ap.add_argument("--name", default="thesis.pdf", help="file name of the PDF")
    ap.add_argument("--browser", default=None, help="path to Edge or Chrome")
    args = ap.parse_args()
    browser = args.browser or next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("no Edge or Chrome found; pass --browser")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    main_tex = strip_comments(load(os.path.join(HERE, "main.tex")))
    macros = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", main_tex)}
    bib_in_toc = bool(re.search(r"\\renewcommand\{\\bibsection\}.*addcontentsline\{toc\}", main_tex))
    body = main_tex[main_tex.index("\\begin{document}") + len("\\begin{document}"):main_tex.index("\\end{document}")]
    body = expand_inputs(body, HERE)
    bib = parse_bib(load(os.path.join(HERE, "references.bib")))
    thesis = Thesis(macros, bib, bib_in_toc)
    content = thesis.blocks(body, top=True)
    content = content.replace("<!--BIBLIOGRAPHY-->", thesis.bibliography_html())

    def resolve(m):
        key = m.group(1)
        if key not in thesis.labels:
            warn("reference to an undefined label: " + key)
            return "<b>??</b>"
        num, a = thesis.labels[key]
        return '<a href="#%s">%s</a>' % (a, num)
    content = glue_math(re.sub("\ue000(.*?)\ue001", resolve, content))

    import fitz
    for path in sorted(set(thesis.figures)):
        src = os.path.join(HERE, path)
        dst = os.path.join(out, "figures", os.path.splitext(os.path.basename(path))[0] + ".svg")
        if not os.path.exists(src):
            warn("figure missing: " + path)
            continue
        doc = fitz.open(src)
        io.open(dst, "w", encoding="utf-8").write(doc[0].get_svg_image(text_as_path=True))
        doc.close()

    title = re.sub(r"\s+", " ", re.search(r"\\LARGE\\bfseries\s*(.*?)\\par", body, re.S).group(1)).strip()
    authors = ", ".join(re.findall(r"^([A-Z][A-Za-z. ]+)\\\\\s*\d{8}", body, re.M)) or "BRAC University"
    html_path = os.path.join(out, "thesis.html")
    raw_pdf = os.path.join(out, "thesis.raw.pdf")
    # print until the page numbers stop moving; between prints, defer any float that left a gap, as LaTeX
    # would have floated it to the next page and let the text run on
    pages, labels, dests, moves = {}, [], {}, {}
    float_ids = set(FLOAT_LINE.findall(content))
    for attempt in range(12):
        current = apply_moves(content, moves)
        filled = current
        for key in ("toc", "lof", "lot"):
            filled = filled.replace("<!--LIST:%s-->" % key, thesis.list_html(key, pages))
        io.open(html_path, "w", encoding="utf-8").write(page(title, filled, thesis.anchors))
        print_pdf(browser, html_path, raw_pdf)
        dests, npages = destinations(raw_pdf)
        labels = labels_for(dests, npages)
        new_pages = {a: labels[p] or "" for a, (p, _) in dests.items()}
        defer = [f for f in floats_to_defer(raw_pdf, dests, float_ids)
                 if moves.get(f, 0) < 4 and apply_moves(content, dict(moves, **{f: moves.get(f, 0) + 1})) != current]
        for f in defer:
            moves[f] = moves.get(f, 0) + 1
        print("pass %d: %d pages%s" % (attempt + 1, npages, "; deferring " + ", ".join(defer) if defer else ""))
        if not defer and new_pages == pages:
            break
        pages = new_pages
    dom = dump_dom(browser, html_path)
    zooms = re.findall(r'data-zoom="([\d.]+)"', dom)
    overfull = re.findall(r'data-overfull="([\d.]+)"', dom)
    widths = re.findall(r'data-width="([\d.]+)"', dom)
    pdf_path = os.path.join(out, args.name)
    finalize(raw_pdf, pdf_path, labels, thesis, dests, title, authors)
    os.remove(raw_pdf)
    print("floats deferred past running text: %s" % (", ".join("%s by %d" % kv for kv in sorted(moves.items())) or "none"))
    print("tables shrunk to fit: %s" % (", ".join(zooms) or "none"))
    print("equation widths as a fraction of the line: %s; wider than the space between the numbers: %s"
          % (", ".join(widths) or "none", ", ".join(overfull) or "none"))
    print("%d chapters, %d figures, %d tables, %d equations, %d citations of %d references"
          % (sum(1 for e in thesis.toc if e[0] == 0 and e[1]), sum(1 for x in thesis.lof if x != "gap"),
             sum(1 for x in thesis.lot if x != "gap"), sum(1 for a in thesis.anchors if a.startswith("eq-")),
             len(thesis.cites), len(thesis.cites)))
    for w in WARNINGS:
        print("warning:", w)
    print("%d pages -> %s (%.0f KB)" % (len(labels), pdf_path, os.path.getsize(pdf_path) / 1024))


if __name__ == "__main__":
    main()
