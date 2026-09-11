"""Paper-ready tables from a finished runs/ tree. Emits LaTeX (booktabs) and CSV side by side, and
skips any table whose data is not there yet, so it can be run while experiments are still going.

    python -m dmthd.tables --runs runs/tweets --data data/tweets --out paper/tables/tweets

Produces, when the data exists:
  main        student x mode, mean +- std over seeds, with probe metrics and parameters
  teachers    each teacher, with the classical floor for reference
  ablations   D-MTHD minus one component, on the headline student
  homogeneity student family x hidden-state term (the measured claim about "homogeneous")
  committee   homogeneous vs heterogeneous teacher committee, same students
  efficiency  parameters, FLOPs, latency, INT8 deployment
  robustness  clean vs obfuscated, and cross-dataset transfer
  implicit    the table the paper's claim is argued on: F1 on the hard class, sarcasm-discrimination
              AUC, and the false-positive rate on benign sarcasm that buys it
  sweeps      validation macro-F1 per hyper-parameter value
Numbers come only from results.json / eval_*.json / bench_*.csv; nothing is typed by hand.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

MODE_LABEL = {"ft": "Fine-tune only", "skd": "Single-teacher KD", "uniform": "Uniform multi-teacher",
              "dmthd": "D-MTHD", "dmthd_dis": "D-MTHD + disagreement", "ft_hetero": "Fine-tune only",
              "skd_hetero": "Single-teacher KD (het.)", "uniform_hetero": "Uniform multi-teacher (het.)",
              "dmthd_hetero": "D-MTHD (heterogeneous committee)",
              "uniform_spec": "Uniform multi-teacher + implicit specialist",
              "dmthd_spec": "D-MTHD + implicit specialist"}
STUDENT_LABEL = {"bert-mini": "BERT-mini", "bert-small": "BERT-small", "distilbert": "DistilBERT",
                 "deberta-xsmall": "DeBERTa-v3-xsmall", "bilstm": "BiLSTM", "tiny": "BERT-tiny"}
ABLATION_LABEL = {"ablation_no_dynamic": "uniform weights instead of per-instance",
                  "ablation_no_hidden": "no hidden-state term", "ablation_no_aux": "no auxiliary irony head",
                  "ablation_per_batch": "per-batch instead of per-instance weights",
                  "ablation_from_scratch": "randomly initialised student",
                  "ablation_spec_only": "implicit specialist alone, no committee",
                  "ablation_no_spec": "committee without the implicit specialist",
                  "ablation_implicit_pretrain": "student pre-trained on the implicit corpus, no distillation"}
# The class whose F1 the implicit claim is argued on, per label scheme seen in a results.json.
HARD_CLASS = ("implicit_hate", "other_cyberbullying")


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(runs):
    """One row per finished run directory, with test metrics and probe metrics where present."""
    rows = []
    for res in glob.glob(os.path.join(runs, "**", "results.json"), recursive=True):
        d = os.path.dirname(res)
        r = _load(res) or {}
        t = r.get("test", {})
        rel = os.path.relpath(d, runs).replace("\\", "/").split("/")
        is_teacher = rel[0] == "teachers"
        row = {"dir": d, "student": "teacher" if is_teacher else rel[0],
               "mode": rel[1] if is_teacher else (rel[1] if len(rel) > 2 else ""),
               "seed": r.get("seed"), "params": r.get("params"), "macro_f1": t.get("macro_f1"),
               "accuracy": t.get("accuracy"), "ece": t.get("ece"), "roc_auc": t.get("roc_auc"),
               "pr_auc": t.get("pr_auc"), "train_time_s": r.get("train_time_s"), "tag": r.get("tag", "")}
        if is_teacher:
            row["mode"], row["student"] = "teacher", rel[1]
        ev = _load(os.path.join(d, "eval_test.json")) or {}
        row["benign_fpr"] = ev.get("benign_sarcasm_fpr")
        row["ironic_recall"] = ev.get("ironic_abuse_recall")
        # F1 on whichever class actually holds abuse-by-implication in this corpus
        for cls in HARD_CLASS:
            if cls in (t.get("per_class_f1") or {}):
                row["hard_class"], row["hard_class_f1"] = cls, t["per_class_f1"][cls]
                break
        # the threshold-free sarcasm metric, and the operating point it implies
        ia = _load(os.path.join(d, "implicit_analysis", "implicit_analysis.json")) or {}
        row["sarcasm_auc"] = ia.get("sarcasm_discrimination_auc")
        op = ia.get("operating_point_at_0.5") or {}
        row["ironic_recall_at_half"], row["benign_fpr_at_half"] = op.get("ironic_recall"), op.get("benign_fpr")
        safe = ia.get("threshold_for_fpr_10pct") or {}
        row["ironic_recall_at_fpr10"] = safe.get("ironic_recall")
        for v in ("leet", "swap", "space", "mixed"):
            o = _load(os.path.join(d, f"eval_test_obf_{v}.json"))
            if o:
                row[f"obf_{v}"] = o.get("macro_f1")
        for other in ("wikipedia", "tweets", "implicit"):
            tr = _load(os.path.join(d, f"transfer_{other}.json"))
            if tr:
                row[f"transfer_{other}"] = tr.get("binary_macro_f1")
                row[f"transfer_{other}_auc"] = tr.get("roc_auc")
                # the collapsed score hides the case the paper is about, so keep the focused one too
                foc = tr.get("implicit_hate") or {}
                row["transfer_implicit_focus_recall"] = foc.get("recall")
                row["transfer_implicit_focus_auc"] = foc.get("roc_auc")
        q = _load(os.path.join(d, "quantize_eval.json"))
        if q:
            row["int8_macro_f1"] = q["int8"]["macro_f1"]
            row["int8_size_mb"] = q["int8"]["size_MB"]
            row["fp32_size_mb"] = q["fp32"]["size_MB"]
            row["int8_lat_b1"] = q["int8"].get("latency_ms_per_sample_b1")
            row["fp32_lat_b1"] = q["fp32"].get("latency_ms_per_sample_b1")
        rows.append(row)
    return pd.DataFrame(rows)


def agg(df, keys=("student", "mode")):
    """Mean and standard deviation over seeds, keeping the metrics the paper reports."""
    metrics = [c for c in ("macro_f1", "accuracy", "ece", "roc_auc", "pr_auc", "benign_fpr", "ironic_recall",
                           "obf_mixed", "transfer_wikipedia", "transfer_tweets", "transfer_implicit",
                           "hard_class_f1", "sarcasm_auc", "ironic_recall_at_half", "benign_fpr_at_half",
                           "ironic_recall_at_fpr10", "transfer_implicit_focus_recall") if c in df.columns]
    g = df.groupby(list(keys), dropna=False)
    out = g.agg(n=("seed", "count"), params=("params", "first"), **{f"{m}_mean": (m, "mean") for m in metrics},
                **{f"{m}_std": (m, "std") for m in metrics}).reset_index()
    return out


def fmt(mean, std=None, nd=4):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def _is_numeric_col(values):
    """A column is right-aligned only when every non-empty cell starts with a digit or a sign."""
    for v in values:
        t = str(v).strip().lstrip("$").replace("\\", "")
        if t in ("", "--"):
            continue
        if not re.match(r"^[-+0-9.]", t):
            return False
    return True


def latex(df, caption, label, colfmt=None, note=None):
    cols = list(df.columns)
    colfmt = colfmt or "".join("r" if _is_numeric_col(df[c]) else "l" for c in cols)
    head = " & ".join(c.replace("_", " ") for c in cols) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in df.values)
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}",
           f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule", head, "\\midrule", body, "\\bottomrule",
           "\\end{tabular}"]
    if note:
        out.append(f"\\begin{{tablenotes}}\\footnotesize {note} \\end{{tablenotes}}")
    out.append("\\end{table}")
    return "\n".join(out)


def write(df, out_dir, name, caption, label, note=None):
    if df.empty:
        print(f"  {name}: no data yet, skipped")
        return
    os.makedirs(out_dir, exist_ok=True)
    df = df.where(pd.notna(df), "--")
    df = df.replace({"nan": "--", "None": "--"})
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(latex(df, caption, label, note=note))
    print(f"  {name}: {len(df)} rows -> {name}.tex / {name}.csv")


def floor_row(runs):
    """The TF-IDF + logistic-regression floor, if it was run."""
    for p in glob.glob(os.path.join(runs, "**", "tfidf_lr", "results.json"), recursive=True):
        r = _load(p) or {}
        return r.get("test", {})
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", default=None, help="prepared split directory, for the dataset table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline", default="bert-mini")
    args = ap.parse_args()
    df = collect(args.runs)
    if df.empty:
        raise SystemExit(f"no results.json under {args.runs}")
    os.makedirs(args.out, exist_ok=True)
    print(f"collected {len(df)} runs from {args.runs}")

    # ---- teachers ----
    t = df[df["mode"] == "teacher"]
    if not t.empty:
        rows = [{"Model": STUDENT_LABEL.get(r.student, r.student), "Params (M)": f"{(r.params or 0) / 1e6:.1f}" if r.params else "--",
                 "Macro-F1": fmt(r.macro_f1), "Accuracy": fmt(r.accuracy), "ECE": fmt(r.ece, nd=3),
                 "Benign-sarcasm FPR": fmt(r.benign_fpr, nd=3), "Ironic-abuse recall": fmt(r.ironic_recall, nd=3)}
                for r in t.itertuples()]
        fl = floor_row(args.runs)
        if fl:
            rows.append({"Model": "TF-IDF + logistic regression", "Params (M)": "--", "Macro-F1": fmt(fl.get("macro_f1")),
                         "Accuracy": fmt(fl.get("accuracy")), "ECE": fmt(fl.get("ece"), nd=3),
                         "Benign-sarcasm FPR": "--", "Ironic-abuse recall": "--"})
        write(pd.DataFrame(rows), args.out, "teachers", "Task-adapted teachers and the classical floor.", "teachers")

    s = df[df["mode"] != "teacher"].copy()
    s["is_ablation"] = s["mode"].str.startswith("ablation_")
    s["is_sweep"] = s["mode"].str.startswith("sweep_")

    # ---- main table ----
    main_modes = ["ft", "skd", "uniform", "dmthd", "uniform_spec", "dmthd_spec", "dmthd_dis"]
    m = agg(s[s["mode"].isin(main_modes)])
    if not m.empty:
        m["order"] = m["mode"].map({k: i for i, k in enumerate(main_modes)})
        m = m.sort_values(["student", "order"])
        rows = [{"Student": STUDENT_LABEL.get(r.student, r.student), "Method": MODE_LABEL.get(r.mode, r.mode),
                 "Seeds": int(r.n), "Params (M)": f"{(r.params or 0) / 1e6:.1f}" if r.params else "--",
                 "Macro-F1": fmt(r.macro_f1_mean, r.macro_f1_std), "Accuracy": fmt(r.accuracy_mean, r.accuracy_std),
                 "Benign-sarcasm FPR $\\downarrow$": fmt(getattr(r, "benign_fpr_mean", None), getattr(r, "benign_fpr_std", None), 3),
                 "Ironic-abuse recall $\\uparrow$": fmt(getattr(r, "ironic_recall_mean", None), getattr(r, "ironic_recall_std", None), 3)}
                for r in m.itertuples()]
        write(pd.DataFrame(rows), args.out, "main",
              "Main results. Mean $\\pm$ standard deviation over seeds; probe metrics are inference-only.", "main")

    # ---- the implicit table: the one the paper's claim is argued on ----
    # Macro-F1 can improve while the hard class gets worse, and recall at a fixed threshold can
    # improve while the model simply fires more often. Both failure modes are visible here and
    # nowhere else in the paper.
    imp_modes = main_modes + ["ablation_spec_only", "ablation_no_spec", "ablation_implicit_pretrain"]
    im = agg(s[s["mode"].isin(imp_modes) & (s["student"] == args.headline)])
    if not im.empty and "sarcasm_auc_mean" in im.columns and im["sarcasm_auc_mean"].notna().any():
        im["order"] = im["mode"].map({k: i for i, k in enumerate(imp_modes)})
        im = im.sort_values("order")
        hard = next(iter(df.get("hard_class", pd.Series(dtype=object)).dropna().unique()), "hard class")
        rows = [{"Method": MODE_LABEL.get(r.mode, ABLATION_LABEL.get(r.mode, r.mode)), "Seeds": int(r.n),
                 "Macro-F1": fmt(r.macro_f1_mean, r.macro_f1_std),
                 f"F1 on {hard.replace('_', ' ')}": fmt(getattr(r, "hard_class_f1_mean", None),
                                                        getattr(r, "hard_class_f1_std", None), 3),
                 "Sarcasm-discrimination AUC": fmt(getattr(r, "sarcasm_auc_mean", None),
                                                   getattr(r, "sarcasm_auc_std", None), 3),
                 "Ironic recall at 0.5": fmt(getattr(r, "ironic_recall_at_half_mean", None), None, 3),
                 "Benign FPR at 0.5": fmt(getattr(r, "benign_fpr_at_half_mean", None), None, 3),
                 "Ironic recall at FPR 0.10": fmt(getattr(r, "ironic_recall_at_fpr10_mean", None), None, 3)}
                for r in im.itertuples()]
        write(pd.DataFrame(rows), args.out, "implicit",
              "Detection of abuse by implication. Sarcasm-discrimination AUC ranks the ironic-abuse "
              "probe against the benign-sarcasm probe and is threshold-free, so it separates a model "
              "that cannot see implication from one that sees it but cannot tell it from harmless "
              "sarcasm. The last three rows are the controls: the specialist without a committee, the "
              "committee without the specialist, and the same student pre-trained on the implicit "
              "corpus instead of distilled from it.", "implicit")

    # ---- routing: does the weighting select an expert, or average over the committee? ----
    rt = _load(os.path.join(args.runs, "weight_routing", "weight_routing.json"))
    if rt:
        pos, neg = rt["contrast_groups"]["implicit_like"], rt["contrast_groups"]["explicit_like"]
        rows = []
        for tau, entry in rt["by_tau"].items():
            for teacher, c in entry["routing_contrast"].items():
                if c:
                    rows.append({"tau": tau, "Teacher": teacher,
                                 f"weight on {pos.replace('_', ' ')}": fmt(entry["mean_weight"][teacher].get(pos), None, 3),
                                 f"weight on {neg.replace('_', ' ')}": fmt(entry["mean_weight"][teacher].get(neg), None, 3),
                                 "difference": f"{c['delta']:+.4f}",
                                 "95 per cent interval": f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"})
        if rows:
            write(pd.DataFrame(rows), args.out, "routing",
                  "Where the per-instance weights go. A difference whose interval excludes zero means "
                  "the weighting selects a teacher for abuse by implication rather than averaging over "
                  "the committee.", "routing")

    # ---- ablations ----
    a = agg(s[s["is_ablation"]])
    if not a.empty:
        base = agg(s[(s["mode"] == "dmthd") & (s["student"] == args.headline)])
        rows = []
        if not base.empty:
            b = base.iloc[0]
            rows.append({"Configuration": "D-MTHD (full)", "Macro-F1": fmt(b.macro_f1_mean, b.macro_f1_std),
                         "$\\Delta$": "--", "Benign-sarcasm FPR": fmt(getattr(b, "benign_fpr_mean", None), None, 3)})
        ref = base.iloc[0].macro_f1_mean if not base.empty else np.nan
        for r in a[a["student"] == args.headline].itertuples():
            rows.append({"Configuration": ABLATION_LABEL.get(r.mode, r.mode),
                         "Macro-F1": fmt(r.macro_f1_mean, r.macro_f1_std),
                         "$\\Delta$": "--" if np.isnan(ref) else f"{r.macro_f1_mean - ref:+.4f}",
                         "Benign-sarcasm FPR": fmt(getattr(r, "benign_fpr_mean", None), None, 3)})
        write(pd.DataFrame(rows), args.out, "ablations",
              f"Ablations on {STUDENT_LABEL.get(args.headline, args.headline)}; $\\Delta$ is relative to full D-MTHD.", "ablations")

    # ---- homogeneity: student family x hidden-state term ----
    hom = []
    for student in s["student"].unique():
        full = agg(s[(s["student"] == student) & (s["mode"] == "dmthd")])
        noh = agg(s[(s["student"] == student) & (s["mode"] == "ablation_no_hidden")])
        if full.empty:
            continue
        f0 = full.iloc[0]
        row = {"Student": STUDENT_LABEL.get(student, student),
               "Family": "BERT-lineage" if student in ("bert-mini", "bert-small", "distilbert", "tiny") else
                         ("DeBERTa-v3" if "deberta" in student else "BiLSTM (not a transformer)"),
               "D-MTHD": fmt(f0.macro_f1_mean, f0.macro_f1_std)}
        if not noh.empty:
            n0 = noh.iloc[0]
            row["without hidden term"] = fmt(n0.macro_f1_mean, n0.macro_f1_std)
            row["gain from hidden term"] = f"{f0.macro_f1_mean - n0.macro_f1_mean:+.4f}"
        hom.append(row)
    if hom and any("gain from hidden term" in r for r in hom):
        write(pd.DataFrame(hom), args.out, "homogeneity",
              "Does architectural homogeneity matter? Gain from the hidden-state alignment term by student family.",
              "homogeneity")

    # ---- committee: homogeneous vs heterogeneous ----
    comm = []
    for student in s["student"].unique():
        h = agg(s[(s["student"] == student) & (s["mode"] == "dmthd")])
        e = agg(s[(s["student"] == student) & (s["mode"] == "dmthd_hetero")])
        if h.empty or e.empty:
            continue
        h0, e0 = h.iloc[0], e.iloc[0]
        comm.append({"Student": STUDENT_LABEL.get(student, student),
                     "Homogeneous committee": fmt(h0.macro_f1_mean, h0.macro_f1_std),
                     "With a DeBERTa teacher": fmt(e0.macro_f1_mean, e0.macro_f1_std),
                     "$\\Delta$": f"{e0.macro_f1_mean - h0.macro_f1_mean:+.4f}"})
    write(pd.DataFrame(comm), args.out, "committee",
          "Homogeneous versus heterogeneous teacher committee, same students and objective.", "committee")

    # ---- efficiency ----
    eff = []
    bench = {}
    for b in glob.glob(os.path.join(args.runs, "bench_*.csv")):
        dev = os.path.basename(b).replace("bench_", "").replace(".csv", "")
        bench[dev] = pd.read_csv(b)
    for student in s["student"].unique():
        d = s[(s["student"] == student) & (s["mode"] == "dmthd")]
        if d.empty:
            continue
        r = d.iloc[0]
        row = {"Student": STUDENT_LABEL.get(student, student),
               "Params (M)": f"{(r.params or 0) / 1e6:.1f}" if r.params else "--"}
        for dev, bdf in bench.items():
            mask = bdf["model"].astype(str).str.replace("\\", "/", regex=False).str.contains(f"/{student}/")
            if mask.any():
                b0 = bdf[mask].iloc[0]
                for c in ("latency_ms_per_sample_b1", "latency_ms_per_sample_b32"):
                    if c in b0 and not pd.isna(b0[c]):
                        row[f"{dev} {c.split('_')[-1]} (ms)"] = f"{b0[c]:.2f}"
                if "flops_G_per_seq" in b0 and not pd.isna(b0["flops_G_per_seq"]):
                    row["GFLOPs/seq"] = f"{b0['flops_G_per_seq']:.2f}"
        if "int8_macro_f1" in d.columns and not pd.isna(r.get("int8_macro_f1", np.nan)):
            row["fp32 size (MB)"] = f"{r.fp32_size_mb:.0f}"
            row["INT8 size (MB)"] = f"{r.int8_size_mb:.0f}"
            row["INT8 macro-F1"] = f"{r.int8_macro_f1:.4f}"
        eff.append(row)
    write(pd.DataFrame(eff), args.out, "efficiency",
          "Deployment profile of the distilled students (first seed).", "efficiency")

    # ---- robustness ----
    rob = []
    for student in s["student"].unique():
        for mode in ("ft", "dmthd"):
            d = s[(s["student"] == student) & (s["mode"] == mode)]
            if d.empty:
                continue
            r = d.iloc[0]
            row = {"Student": STUDENT_LABEL.get(student, student), "Method": MODE_LABEL.get(mode, mode),
                   "Clean": fmt(r.macro_f1)}
            for v in ("leet", "swap", "space", "mixed"):
                c = f"obf_{v}"
                if c in d.columns and not pd.isna(r.get(c, np.nan)):
                    row[v.capitalize()] = f"{r[c]:.4f}"
            for other in ("wikipedia", "tweets"):
                c = f"transfer_{other}"
                if c in d.columns and not pd.isna(r.get(c, np.nan)):
                    row[f"transfer to {other}"] = f"{r[c]:.4f}"
            if len(row) > 3:
                rob.append(row)
    write(pd.DataFrame(rob), args.out, "robustness",
          "Robustness: obfuscated test variants and cross-dataset transfer (first seed).", "robustness")

    # ---- sweeps ----
    sw = s[s["is_sweep"]].copy()
    if not sw.empty:
        sw["param"] = sw["mode"].str.replace("sweep_", "", regex=False).str.rsplit("_", n=1).str[0]
        sw["value"] = sw["mode"].str.rsplit("_", n=1).str[-1]
        rows = [{"Hyper-parameter": r.param, "Value": r.value, "Macro-F1": fmt(r.macro_f1)}
                for r in sw.sort_values(["param", "value"]).itertuples()]
        write(pd.DataFrame(rows), args.out, "sweeps",
              "Hyper-parameter sweeps on the headline student (one seed, selected on validation).", "sweeps")

    # ---- dataset table ----
    if args.data:
        rep = _load(os.path.join(args.data, "report.json"))
        if rep:
            sp = rep.get("split", {})
            rows = [{"Item": k.replace("_", " "), "Value": (json.dumps(v) if isinstance(v, dict) else str(v))}
                    for k, v in rep.items() if k not in ("class_counts", "source")]
            if sp:
                rows.append({"Item": "train / val / test", "Value": f"{sp.get('train')} / {sp.get('val')} / {sp.get('test')}"})
            write(pd.DataFrame(rows), args.out, "dataset", "Corpus preparation counts.", "dataset")

    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    print(f"raw collection -> {os.path.join(args.out, 'all_runs.csv')}")


if __name__ == "__main__":
    main()
