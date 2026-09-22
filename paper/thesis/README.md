# The thesis in LaTeX

`main.tex` is the thesis in the BRAC University format of the Pre-Thesis II report: title page,
declaration, approval, ethics statement, abstract, acknowledgement, contents, lists of figures, tables
and abbreviations, eight chapters, bibliography and five appendices.

## Compile

On Overleaf: New Project, Upload Project, choose the zip of this folder; set the compiler to pdfLaTeX
and the main document to `main.tex`. Locally, with any TeX distribution:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Change the text

The chapters are generated, so edit the sources and rebuild rather than editing `chapters/*.tex`:

- `paper/PAPER.md` holds the text shared with the journal version of the paper.
- `build_tex.py` holds what exists only in the thesis: front matter, research questions, preliminaries,
  answers to the research questions, how the study changed, future work, and the tables of data,
  sessions, predictions and paired comparisons.
- `make_figures.py` draws every figure from the final Kaggle archive and the generated tables.

```
python paper/thesis/make_figures.py --archive E:/dmthd-work/kaggle_d738_v7
python paper/thesis/build_tex.py
python paper/thesis/check_tex.py
```

`check_tex.py` runs the static checks that make the first compile likely to succeed: balanced braces
and environments, escaped special characters, every reference, citation and figure present.

## Check before submission

- The semester on the approval page, `\thesissemester` in `main.tex`, is set to Fall, 2026.
- The designations of the examining committee are copied from the Pre-Thesis II report.
- The acknowledgement is a draft for the authors to make their own.
