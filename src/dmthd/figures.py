"""Paper figures from the analysis CSV/JSON files. Vector PDF plus PNG at 300 dpi.

    python -m dmthd.figures pareto        --csv runs/tweets/pareto.csv --out paper/figures/pareto_tweets
    python -m dmthd.figures weights       --csv runs/tweets/weights.csv --out paper/figures/weights_tweets
    python -m dmthd.figures bands         --csv runs/wikipedia/agreement_bands.csv --out paper/figures/bands_wikipedia
    python -m dmthd.figures complementarity --json runs/tweets/complementarity.json --out paper/figures/complementarity_tweets
    python -m dmthd.figures robustness    --runs runs/tweets --out paper/figures/robustness_tweets

Style: one column wide (3.4 in), serif fonts, colour-blind-safe palette, no chart junk.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


def pareto(csv, out):
    d = pd.read_csv(csv)
    lat = [c for c in d.columns if c.startswith("latency_ms_per_sample_b1")][0]
    d = d.dropna(subset=[lat])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    modes = {m: i for i, m in enumerate(sorted(d["mode"].unique()))}
    for m, i in modes.items():
        s = d[d["mode"] == m]
        ax.scatter(s[lat], s["macro_f1"], s=18 + 2.5 * np.sqrt(s["params_M"].fillna(1)), color=PALETTE[i % len(PALETTE)], label=m, alpha=0.85, edgecolor="white", linewidth=0.4)
        for _, r in s.iterrows():
            ax.annotate(r["dir_tag"], (r[lat], r["macro_f1"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Latency, ms per sample, batch 1")
    ax.set_ylabel("Macro-F1")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _save(fig, out)


def weights(csv, out):
    d = pd.read_csv(csv)
    wcols = [c for c in d.columns if c.startswith("w_")]
    g = d.groupby("epoch")[wcols].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, c in enumerate(wcols):
        m, s = g[(c, "mean")], g[(c, "std")].fillna(0)
        ax.plot(g.index, m, marker="o", ms=3, lw=1.2, color=PALETTE[i], label=c[2:])
        ax.fill_between(g.index, m - s, m + s, color=PALETTE[i], alpha=0.15, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean teacher weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6)
    _save(fig, out)


def bands(csv, out):
    d = pd.read_csv(csv)
    order = ["unanimous_0.0-0.1", "clear_0.1-0.2", "ambiguous_0.2-0.8", "clear_0.8-0.9", "unanimous_0.9-1.0"]
    d["band"] = pd.Categorical(d["band"], [b for b in order if b in set(d["band"])], ordered=True)
    runs = list(d["run"].unique())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    width = 0.8 / max(len(runs), 1)
    for i, r in enumerate(runs):
        s = d[d["run"] == r].sort_values("band")
        x = np.arange(len(s)) + i * width
        ax.bar(x, s["macro_f1"], width, color=PALETTE[i], label=os.path.basename(os.path.dirname(os.path.dirname(r))) + "/" + os.path.basename(os.path.dirname(r)))
    ax.set_xticks(np.arange(len(order)) + width * (len(runs) - 1) / 2)
    ax.set_xticklabels([b.replace("_", "\n") for b in order], fontsize=6)
    ax.set_ylabel("Macro-F1")
    ax.legend(fontsize=6)
    _save(fig, out)


def complementarity(js, out):
    r = json.load(open(js))
    names = r["teachers"]
    n = len(names)
    M = np.ones((n, n))
    for k, v in r["pairwise"].items():
        a, b = k.split("|")
        i, j = names.index(a), names.index(b)
        M[i, j] = M[j, i] = v["error_overlap"]
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.6 else "black")
    ax.set_title(f"Error overlap; oracle acc. {r['oracle_accuracy']:.3f}", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)


def robustness(runs, out):
    rows = []
    for d in glob.glob(os.path.join(runs, "*", "*", "seed*")):
        clean = os.path.join(d, "eval_test.json")
        if not os.path.exists(clean):
            continue
        base = json.load(open(clean))["macro_f1"]
        for v in ("leet", "swap", "space", "mixed"):
            p = os.path.join(d, f"eval_test_obf_{v}.json")
            if os.path.exists(p):
                rows.append({"student": d.replace("\\", "/").split("/")[-3], "mode": d.replace("\\", "/").split("/")[-2], "variant": v,
                             "clean": base, "obf": json.load(open(p))["macro_f1"]})
    if not rows:
        print("no obfuscation evaluations found")
        return
    df = pd.DataFrame(rows)
    df["drop"] = df["clean"] - df["obf"]
    piv = df.pivot_table(index="variant", columns=["student", "mode"], values="drop")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    piv.plot.bar(ax=ax, color=PALETTE[:piv.shape[1]], width=0.8)
    ax.set_ylabel("Macro-F1 drop vs clean test")
    ax.set_xlabel("")
    ax.legend(fontsize=5.5, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out)
    df.to_csv(out + ".csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pareto", "weights", "bands"):
        s = sub.add_parser(name); s.add_argument("--csv", required=True); s.add_argument("--out", required=True)
    c = sub.add_parser("complementarity"); c.add_argument("--json", required=True); c.add_argument("--out", required=True)
    r = sub.add_parser("robustness"); r.add_argument("--runs", required=True); r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"pareto": lambda: pareto(a.csv, a.out), "weights": lambda: weights(a.csv, a.out), "bands": lambda: bands(a.csv, a.out),
     "complementarity": lambda: complementarity(a.json, a.out), "robustness": lambda: robustness(a.runs, a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
