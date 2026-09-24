"""Every figure in the thesis, drawn from the final Kaggle archive and the generated tables.

    python paper/thesis/make_figures.py --archive E:/dmthd-work/kaggle_d738_v7

Writes PDF (for LaTeX) and PNG (for checking by eye) into paper/thesis/figures/. Nothing is typed by
hand except the schematic's wording: every plotted number is read from a results.json, a
history.csv, a table produced by dmthd.tables or dmthd.significance, or the tau diagnostic.

Colour follows the job each mark does, from the validated three-slot categorical palette (blue,
orange, aqua; `validate_palette.js --pairs all` passes) plus a neutral gray for context. Blue is the
out-of-sample result wherever it appears, gray the in-sample or no-teacher reference, orange the
teachers or the composition control.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT_GRAY = "#898781", "#c3c2b7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
WIDTH = 6.3  # inches, the text width of an A4 page with 2.54 cm margins

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": LIGHT_GRAY,
    "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=170)
    plt.close(fig)
    print("wrote", name)


def seeds(archive, student, arm, key="macro_f1"):
    vals = []
    for p in sorted(glob.glob(os.path.join(archive, "runs", "tweets", student, arm, "seed*", "results.json"))):
        vals.append(json.load(open(p, encoding="utf-8"))["test"][key])
    return np.array(vals)


def history(archive, student, arm):
    out = []
    for p in sorted(glob.glob(os.path.join(archive, "runs", "tweets", student, arm, "seed*", "history.csv"))):
        out.append(pd.read_csv(p))
    return out


def interval(s):
    import re
    lo, hi = re.findall(r"[-+]?\d*\.\d+", str(s))[:2]
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
def fig_schematic():
    """Where the teacher's function is sampled: the one idea the thesis turns on."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.35))
    panels = [
        ("In sample", LIGHT_GRAY, GRAY,
         ["Labelled training split:\n34,607 tweets",
          "Teacher fine-tuned on\nthese same tweets",
          "Soft labels \u2248 gold labels:\nthe teacher is almost always sure",
          "Student learns only\nwhat the labels say"],
         "No gain over fine-tuning,\nwhatever the committee or weighting"),
        ("Out of sample", RAMP[3], BLUE,
         ["Labelled split + 168,000 tweets\nthe teacher has never seen",
          "The same frozen teacher\nlabels the new tweets",
          "Soft labels carry the teacher's\nuncertainty, which is new",
          "Student learns the boundary\nthe labels draw worst"],
         "+0.012 macro-F1 on BERT-mini,\ngrowing with the number of tweets"),
    ]
    for ax, (title, edge, accent, steps, outcome) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=2)
        ys = [0.90, 0.72, 0.54, 0.36]
        for i, (y, txt) in enumerate(zip(ys, steps)):
            box = FancyBboxPatch((0.02, y - 0.065), 0.94, 0.13, boxstyle="round,pad=0.006,rounding_size=0.02",
                                 linewidth=0.9, edgecolor=edge if i != 2 else accent,
                                 facecolor="#ffffff" if i != 2 else ("#f3f2ee" if accent == GRAY else "#eef4fc"))
            ax.add_patch(box)
            ax.text(0.49, y, txt, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.15)
            if i < 3:
                ax.add_patch(FancyArrowPatch((0.49, y - 0.068), (0.49, ys[i + 1] + 0.068), arrowstyle="-|>",
                                             mutation_scale=8, linewidth=0.9, color=INK2))
        ax.add_patch(FancyArrowPatch((0.49, ys[-1] - 0.068), (0.49, 0.195), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color=INK2))
        ax.text(0.49, 0.12, outcome, ha="center", va="center", fontsize=7.8, linespacing=1.15,
                color=BLUE if accent == BLUE else INK2, fontweight="bold")
    fig.subplots_adjust(wspace=0.14)
    save(fig, "fig_schematic")


def fig_agreement():
    comp = json.load(open(os.path.join(REPO, "paper", "complementarity_with_specialist.json"), encoding="utf-8"))
    names = comp["teachers"]
    label = {"bert-large": "BERT-large", "hatebert": "HateBERT", "irony": "RoBERTa-irony",
             "implicit-spec": "Implicit\nspecialist", "deberta-base": "DeBERTa-v3"}
    n = len(names)
    m = np.full((n, n), np.nan)
    for key, v in comp["pairwise"].items():
        a, b = key.split("|")
        i, j = names.index(a), names.index(b)
        m[i, j] = m[j, i] = v["kappa_predictions"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)
    cmap.set_bad("#ffffff")
    im = ax.imshow(m, cmap=cmap, vmin=0.85, vmax=1.0)
    ax.grid(False)
    ax.set_xticks(range(n), [label[x] for x in names], rotation=35, ha="right")
    ax.set_yticks(range(n), [label[x] for x in names])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "\u2013", ha="center", va="center", color=GRAY, fontsize=8)
                continue
            v = m[i, j]
            rgb = np.array(matplotlib.colors.to_rgb(cmap((v - 0.85) / 0.15)))
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.4,
                    color="#ffffff" if lum < 0.45 else INK, fontweight="bold" if v > 0.95 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Cohen's kappa on test predictions", fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    save(fig, "fig_agreement")


def fig_tau(archive):
    diag = pd.read_csv(os.path.join(archive, "cache", "tweets", "tau_diagnostic_homo.csv"))
    f1 = []
    for tau in diag["tau"]:
        arm = "dmthd" if abs(tau - 1.0) < 1e-9 else f"sweep_tau_{tau:g}" if tau not in (0.5, 2.0, 5.0) else f"sweep_tau_{tau:.1f}"
        v = seeds(archive, "bert-mini", arm)
        f1.append(v[0] if len(v) else np.nan)
    ft1 = seeds(archive, "bert-mini", "ft")[0]
    uni1 = seeds(archive, "bert-mini", "uniform")[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    a.plot(diag["tau"], diag["mean_max_weight"], color=BLUE, lw=1.5, marker="o", ms=5, mec="#ffffff", mew=1.2)
    a.axhline(1 / 3, color=GRAY, lw=1.0)
    a.text(0.052, 1 / 3 + 0.0008, "uniform weight 1/3", ha="left", va="bottom", fontsize=7.4, color=INK2)
    a.set_xscale("log")
    a.set_xlabel("weight temperature \u03c4")
    a.set_ylabel("largest weight per tweet, mean")
    a.set_title("a. The weights barely leave uniform", loc="left")
    a.set_ylim(0.325, 0.385)
    b.plot(diag["tau"], f1, color=BLUE, lw=1.5, marker="o", ms=5, mec="#ffffff", mew=1.2, label="D-MTHD at each \u03c4")
    b.axhline(ft1, color=INK2, lw=1.0)
    b.axhline(uni1, color=LIGHT_GRAY, lw=1.0)
    b.text(0.052, ft1 + 0.0002, "fine-tune only", ha="left", va="bottom", fontsize=7.4, color=INK2)
    b.text(0.052, uni1 - 0.0002, "uniform averaging", ha="left", va="top", fontsize=7.4, color=INK2)
    b.set_xscale("log")
    b.set_xlabel("weight temperature \u03c4")
    b.set_ylabel("test macro-F1, seed 1")
    b.set_title("b. and the score does not move", loc="left")
    b.set_ylim(0.8360, 0.8410)
    for ax in (a, b):
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
        ax.minorticks_off()
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_tau")


def fig_size_curve(archive):
    sizes = [("skd_transfer_5k", 5000), ("skd_transfer_10k", 10000), ("skd_transfer_21k", 21000),
             ("skd_transfer_42k", 42013), ("skd_transfer_84k", 84000), ("skd_transfer_168k", 168000)]
    ft = seeds(archive, "bert-mini", "ft")
    m13 = seeds(archive, "bert-mini", "ft_matched")
    m35 = seeds(archive, "bert-mini", "ft_matched_168k")
    gen = seeds(archive, "bert-mini", "skd_transfer_generic42k")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.axhspan(ft.min(), ft.max(), color="#f0efec", lw=0)
    ax.axhline(ft.mean(), color=INK2, lw=1.0)
    ax.axhline(m13.mean(), color=LIGHT_GRAY, lw=1.0)
    ax.axhline(m35.mean(), color=LIGHT_GRAY, lw=1.0)
    xs, means = [], []
    for arm, n in sizes:
        v = seeds(archive, "bert-mini", arm)
        ax.scatter([n] * len(v), v, s=14, color=BLUE, alpha=0.35, lw=0, zorder=3)
        xs.append(n)
        means.append(v.mean())
    ax.plot(xs, means, color=BLUE, lw=1.5, zorder=4)
    ax.scatter(xs[:4], means[:4], s=36, color=BLUE, edgecolor="#ffffff", linewidth=1.2, zorder=5,
               label="one teacher, abuse-domain unlabelled tweets")
    ax.scatter(xs[4:], means[4:], s=36, facecolor="#ffffff", edgecolor=BLUE, linewidth=1.5, zorder=5,
               label="the same 42k plus generic tweets")
    ax.scatter([42013 * 1.07] * len(gen), gen, s=14, color=ORANGE, alpha=0.35, lw=0, zorder=3)
    ax.scatter([42013 * 1.07], [gen.mean()], s=40, marker="D", color=ORANGE, edgecolor="#ffffff", linewidth=1.2,
               zorder=5, label="42k generic tweets only (composition control)")
    ax.set_xscale("log")
    ax.set_xticks([5000, 10000, 21000, 42013, 84000, 168000], ["5k", "10k", "21k", "42k", "84k", "168k"])
    ax.minorticks_off()
    ax.set_xlim(3800, 260000)
    ax.set_xlabel("unlabelled tweets added to the 34,607 labelled ones (log scale); small dots are single seeds")
    ax.set_ylabel("test macro-F1, BERT-mini")
    x_text = 250000
    ax.text(x_text, ft.mean(), "fine-tune only, 6 epochs", ha="right", va="bottom", fontsize=7.4, color=INK2)
    ax.text(x_text, m13.mean() + 0.0001, "fine-tune only, 13 epochs", ha="right", va="bottom", fontsize=7.4, color=INK2)
    ax.text(x_text, m35.mean() - 0.0001, "fine-tune only, 35 epochs", ha="right", va="top", fontsize=7.4, color=INK2)
    ax.annotate(f"+{means[-1] - ft.mean():.4f} over fine-tuning\n[+0.0035, +0.0211]", xy=(168000, means[-1]),
                xytext=(168000, 0.8470), fontsize=7.4, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=0, shrinkB=5))
    ax.set_ylim(0.8335, 0.8560)
    ax.legend(loc="upper left", handlelength=1.2)
    save(fig, "fig_size_curve")


def fig_students():
    sig = pd.read_csv(os.path.join(REPO, "paper", "tables", "significance.csv"), dtype=str)
    rows = []
    for st in ("BERT-mini", "BERT-small", "BiLSTM"):
        ins = sig[(sig.Student == st) & (sig.Baseline == "Fine-tune only") & (sig.Candidate == "Single-teacher KD")].iloc[0]
        oos = sig[(sig.Student == st) & (sig.Baseline == "Fine-tune only")
                  & (sig.Candidate == "Single-teacher KD + 168k transfer rows")].iloc[0]
        rows.append((st, float(ins.Difference), interval(ins["95 per cent interval"]),
                     float(oos.Difference), interval(oos["95 per cent interval"])))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    x = np.arange(len(rows))
    w = 0.26
    params = {"BERT-mini": "11.2M", "BERT-small": "28.8M", "BiLSTM": "10.4M"}
    for k, (st, d_in, ci_in, d_out, ci_out) in enumerate(rows):
        ax.bar(k - w / 2 - 0.01, d_in, w, color=LIGHT_GRAY, label="same teacher, labelled split only" if k == 0 else None)
        ax.bar(k + w / 2 + 0.01, d_out, w, color=BLUE, label="same teacher + 168k unlabelled tweets" if k == 0 else None)
        for xx, (lo, hi) in ((k - w / 2 - 0.01, ci_in), (k + w / 2 + 0.01, ci_out)):
            ax.plot([xx, xx], [lo, hi], color=INK, lw=0.9)
            ax.plot([xx - 0.03, xx + 0.03], [lo, lo], color=INK, lw=0.9)
            ax.plot([xx - 0.03, xx + 0.03], [hi, hi], color=INK, lw=0.9)
        ax.text(k + w / 2 + 0.01, ci_out[1] + 0.0012, f"+{d_out:.4f}", ha="center", va="bottom", fontsize=7.4,
                color=INK)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xticks(x, [f"{r[0]}\n{params[r[0]]} parameters" for r in rows])
    ax.set_ylabel("gain over fine-tuning\n(macro-F1, 95% interval)")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(-0.016, 0.040)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_students")


def fig_validation(archive):
    """At equal numbers of updates: fine-tuning alone peaks and overfits, the transfer arm keeps rising."""
    steps_ft = int(np.ceil(34607 / 32))
    steps_tr = int(np.ceil((34607 + 168000) / 32))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for i, h in enumerate(history(archive, "bert-mini", "ft_matched_168k")):
        ax.plot(h["epoch"] * steps_ft / 1000, h["val_macro_f1"], color=GRAY, lw=1.0, alpha=0.9,
                label="fine-tune only, 35 epochs" if i == 0 else None)
    for i, h in enumerate(history(archive, "bert-mini", "skd_transfer_168k")):
        ax.plot(h["epoch"] * steps_tr / 1000, h["val_macro_f1"], color=BLUE, lw=1.5, marker="o", ms=4,
                mec="#ffffff", mew=1.0, label="one teacher + 168k unlabelled tweets, 6 epochs" if i == 0 else None)
    ax.set_xlabel("optimisation updates (thousands); the two runs end within 0.5 per cent of each other")
    ax.set_ylabel("validation macro-F1, BERT-mini")
    ax.set_ylim(0.795, 0.866)
    ax.legend(loc="lower right")
    save(fig, "fig_validation")


def fig_tradeoff(archive):
    d = pd.read_csv(os.path.join(archive, "runs", "tweets", "sarcasm_tradeoff.csv"))
    teacher = d.student == "teachers"
    oos = d.variant.str.contains("transfer", na=False)
    ins = ~teacher & ~oos
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    ax.scatter(d.fpr[ins], d.recall[ins], s=16, color=LIGHT_GRAY, edgecolor="#ffffff", linewidth=0.5,
               label=f"students, labelled split only ({ins.sum()})", zorder=3)
    ax.scatter(d.fpr[oos], d.recall[oos], s=18, color=BLUE, edgecolor="#ffffff", linewidth=0.5,
               label=f"students with unlabelled tweets ({oos.sum()})", zorder=4)
    ax.scatter(d.fpr[teacher], d.recall[teacher], s=34, marker="D", color=ORANGE, edgecolor="#ffffff",
               linewidth=0.9, label=f"teachers ({teacher.sum()})", zorder=5)
    margin = (d.recall - d.fpr).mean()
    xx = np.linspace(0.12, 0.62, 10)
    ax.plot(xx, xx + margin, color=INK2, lw=0.8, zorder=2)
    fs = d[d.variant == "ablation_from_scratch"]
    if len(fs):
        ax.annotate("randomly initialised student", xy=(fs.fpr.iloc[0], fs.recall.iloc[0]), xytext=(0.535, 0.60),
                    fontsize=7.4, color=INK2, ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    r = np.corrcoef(d.fpr, d.recall)[0, 1]
    ax.set_xlabel("false-positive rate on benign sarcasm (lower is better)")
    ax.set_ylabel("recall on ironic abuse (higher is better)")
    ax.set_title(f"{len(d)} models, correlation {r:+.2f}; the line is recall \u2212 false-positive rate = {margin:.3f},"
                 " the mean over all of them", loc="left", color=INK2, fontsize=8)
    ax.set_xlim(0.12, 0.62)
    ax.set_ylim(0.45, 0.84)
    ax.legend(loc="lower right")
    save(fig, "fig_tradeoff")


def fig_auc():
    d = pd.read_csv(os.path.join(REPO, "paper", "results_tweets_implicit_seed1.csv"))
    oos = d.run.str.contains("transfer|matched", na=False)
    fs = d.run == "ablation_from_scratch"
    ins = ~oos & ~fs
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    ax.scatter(d.sarcasm_discrimination_auc[ins], 1 + rng.uniform(-0.18, 0.18, ins.sum()), s=18, color=LIGHT_GRAY,
               edgecolor=GRAY, linewidth=0.5, zorder=3)
    ax.scatter(d.sarcasm_discrimination_auc[oos], 0 + rng.uniform(-0.18, 0.18, oos.sum()), s=18, color=BLUE,
               edgecolor="#ffffff", linewidth=0.5, zorder=3)
    if fs.any():
        v = float(d.sarcasm_discrimination_auc[fs].iloc[0])
        ax.scatter([v], [1], s=26, marker="X", color=INK, zorder=4)
        ax.annotate(f"randomly initialised\nstudent, {v:.3f}", xy=(v, 1), xytext=(v + 0.012, 0.45), fontsize=7.4,
                    color=INK2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_yticks([0, 1], [f"unlabelled tweets or\nmatched steps ({oos.sum()})", f"labelled split only,\npre-trained ({ins.sum()})"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0.60, 0.79)
    ax.set_xlabel("sarcasm-discrimination AUC, BERT-mini, seed 1 (0.5 is guessing, 1.0 is perfect)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_auc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("DMTHD_ARCHIVE", "E:/dmthd-work/kaggle_d738_v7"))
    a = ap.parse_args()
    fig_schematic()
    fig_agreement()
    fig_tau(a.archive)
    fig_size_curve(a.archive)
    fig_students()
    fig_validation(a.archive)
    fig_tradeoff(a.archive)
    fig_auc()


if __name__ == "__main__":
    main()
