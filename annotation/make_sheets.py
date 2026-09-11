"""Split the 300-item annotation sheet into one workbook per annotator, so four people can work in
parallel without editing the same file.

    python annotation/make_sheets.py --sheet annotation/sarcbully_sheet.csv --out annotation/sheets

Each annotator gets `annotator_<id>.xlsx`: the same 300 rows in the same order, a `label` column
restricted to the four allowed values by a drop-down (so a typo cannot break the agreement
computation), a `notes` column, frozen header, and a second worksheet with the guideline summary.
A CSV twin is written for anyone who prefers Google Sheets or a plain editor.

When everyone is finished, put the files back in one folder and run:

    python annotation/merge_sheets.py --sheets annotation/sheets --sheet annotation/sarcbully_sheet.csv
    python annotation/compute_kappa.py --sheet annotation/sarcbully_sheet_filled.csv --out probes
"""
import argparse
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

LABELS = ["benign_sarcasm", "sarcastic_abuse", "not_sarcastic", "unsure"]
GUIDE = [
    ("What you are deciding", ""),
    ("", "Every row is a tweet the source dataset marked as sarcastic. For each one, decide whether the"),
    ("", "sarcasm is harmless or whether it carries an attack on a person or a group."),
    ("", ""),
    ("benign_sarcasm", "Sarcastic or ironic, but it attacks nobody: self-mockery, jokes about situations,"),
    ("", "complaints about weather, exams, companies, politics in general."),
    ("sarcastic_abuse", "The sarcasm carries an attack on a person or a group: insult, contempt, demeaning"),
    ("", "stereotype, threat, harassment. The attack may be indirect; ask whether the target would"),
    ("", "reasonably feel attacked."),
    ("not_sarcastic", "Not sarcastic at all; the source label was wrong."),
    ("unsure", "You genuinely cannot decide after reading twice. Use sparingly."),
    ("", ""),
    ("Rules", ""),
    ("1", "Judge the tweet on its own text. Do not search for context or guess who wrote it."),
    ("2", "Profanity alone is not abuse: \"this exam was f***ing wonderful\" is benign_sarcasm."),
    ("3", "Naming a public figure is not automatically abuse. Mocking a decision is benign; calling"),
    ("", "the person subhuman, or attacking their group, is abuse."),
    ("4", "Abuse toward a group counts the same as abuse toward a person."),
    ("5", "Quoting someone else's abuse to mock it is benign_sarcasm, unless the mockery itself attacks."),
    ("6", "@user placeholders and hashtags are part of the text; a hashtag can carry the attack."),
    ("", ""),
    ("Working alone", "Do not discuss items with the others until everyone has finished. Disagreement is"),
    ("", "measured, not a mistake; it is what the agreement statistic reports."),
    ("Time", "About 300 tweets at 20 seconds each is under two hours."),
    ("Content warning", "Some tweets contain slurs and harassment. Take breaks; there is no time limit."),
]


def build_workbook(df, annotator, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "annotation"
    header = ["id", "text", "label", "notes"]
    ws.append(header)
    for c, name in enumerate(header, 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="0B6E6B")
        cell.alignment = Alignment(vertical="center")
    for r in df.itertuples():
        ws.append([r.id, r.text, "", ""])
    ws.freeze_panes = "A2"
    widths = {"A": 8, "B": 100, "C": 20, "D": 30}
    for col, wdt in widths.items():
        ws.column_dimensions[col].width = wdt
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True, showDropDown=False)
    dv.error = "Pick one of: " + ", ".join(LABELS)
    dv.errorTitle = "Not an allowed label"
    dv.prompt = "benign_sarcasm / sarcastic_abuse / not_sarcastic / unsure"
    dv.promptTitle = "Label"
    ws.add_data_validation(dv)
    dv.add(f"C2:C{len(df) + 1}")

    gs = wb.create_sheet("guideline")
    gs.append([f"Annotator {annotator}", "Fill only the `label` column on the annotation sheet."])
    gs.append([])
    for a, b in GUIDE:
        gs.append([a, b])
    gs.column_dimensions["A"].width = 22
    gs.column_dimensions["B"].width = 110
    for row in gs.iter_rows(min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    for c in ("A1", "A13"):
        gs[c].font = Font(bold=True)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="annotation/sarcbully_sheet.csv")
    ap.add_argument("--out", default="annotation/sheets")
    ap.add_argument("--annotators", default="A1,A2,A3,A4")
    args = ap.parse_args()
    df = pd.read_csv(args.sheet, dtype=str).fillna("")
    os.makedirs(args.out, exist_ok=True)
    for a in args.annotators.split(","):
        xlsx = os.path.join(args.out, f"annotator_{a}.xlsx")
        build_workbook(df, a, xlsx)
        csv = os.path.join(args.out, f"annotator_{a}.csv")
        out = df[["id", "text"]].copy()
        out["label"] = ""
        out["notes"] = ""
        out.to_csv(csv, index=False, encoding="utf-8")
        print(f"{a}: {len(df)} rows -> {xlsx} and {csv}")
    print(f"\nHand one workbook to each person. Allowed labels: {', '.join(LABELS)}.")


if __name__ == "__main__":
    main()
