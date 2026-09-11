"""Why does the detector miss indirect abuse? Three measurements that separate the plausible causes.

    python -m dmthd.implicit_analysis --model_dir runs/tweets/bert-mini/ft/seed1 \
        --test data/tweets/test.csv --probes probes --out runs/tweets/implicit_analysis

1. Lexical dependence. Split the ironic-abuse probe into items that contain an explicit profanity
   marker and items that do not, and report recall on each. If recall collapses on the clean half,
   the model is a profanity detector wearing a classifier's clothes, and no amount of distillation
   from teachers trained on the same data will fix it.
2. Operating point. Recall on ironic abuse and false-positive rate on benign sarcasm across decision
   thresholds, so the paper can report a curve instead of one arbitrary 0.5 cut.
3. Where the errors are on the benchmark itself: the confusion between `other_cyberbullying` (the
   catch-all that holds most indirect abuse) and `not_cyberbullying`.

The profanity list is a coarse lexical proxy, deliberately made of common intensifiers and insults
rather than slurs; it is used only to split the probe, never as a feature.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from .evaluate import make_loader, predict_probs
from .models import load_classifier, load_tokenizer
from .utils import get_device, label_names, map_labels, not_bullying_index, save_json

# The classes where indirect abuse hides, per scheme: the first is the one whose lexical
# dependence is broken out, the rest are reported for their confusion with it.
HARD_CLASSES = {"six": ("other_cyberbullying", "not_cyberbullying"),
                "five": ("not_cyberbullying",),
                "binary": ("cyberbullying", "not_cyberbullying"),
                "implicit3": ("implicit_hate", "not_hate", "explicit_hate")}

PROFANITY = r"\b(f+u+c+k\w*|sh[i1]t\w*|b[i1]tch\w*|a+s+s+h+o+l+e\w*|d[i1]ck\w*|cunt\w*|wh[o0]re\w*|slut\w*|"\
            r"bastard\w*|idiot\w*|stupid\w*|moron\w*|dumb\w*|retard\w*|trash|garbage|scum|filth\w*|"\
            r"kill\s+your\w*|die|hate\s+you|loser\w*|pathetic|disgusting|ugly)\b"


def has_profanity(s):
    return bool(re.search(PROFANITY, str(s), flags=re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = get_device()
    names = label_names(args.scheme)
    nb = not_bullying_index(args.scheme)
    tok = load_tokenizer(args.model_dir)
    model = load_classifier(args.model_dir, len(names)).to(device).eval()
    fn = lambda ii, am: model(input_ids=ii, attention_mask=am).logits

    def probs_for(df):
        return predict_probs(fn, make_loader(df, tok, args.max_len, args.batch, False), device)

    res = {"model_dir": args.model_dir}
    os.makedirs(args.out, exist_ok=True)

    # ---- 1. lexical dependence on the ironic-abuse probe ----
    pos = pd.read_csv(os.path.join(args.probes, "ironic_abuse.csv"))
    p_pos = probs_for(pos)
    pos["p_bullying"] = 1.0 - p_pos[:, nb]
    pos["caught"] = (p_pos.argmax(1) != nb)
    pos["explicit"] = pos["text"].map(has_profanity)
    lex = {}
    for flag, label in ((True, "with_profanity"), (False, "without_profanity")):
        sub = pos[pos["explicit"] == flag]
        lex[label] = {"n": int(len(sub)), "recall": float(sub["caught"].mean()) if len(sub) else None,
                      "mean_p_bullying": float(sub["p_bullying"].mean()) if len(sub) else None}
    res["ironic_abuse_lexical_split"] = lex

    imp_path = os.path.join(args.probes, "implicit_abuse.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path)
        p_imp = probs_for(imp)
        imp["caught"] = (p_imp.argmax(1) != nb)
        imp["explicit"] = imp["text"].map(has_profanity)
        res["implicit_abuse_lexical_split"] = {
            lab: {"n": int((imp["explicit"] == f).sum()),
                  "recall": float(imp[imp["explicit"] == f]["caught"].mean()) if (imp["explicit"] == f).any() else None}
            for f, lab in ((True, "with_profanity"), (False, "without_profanity"))}

    # ---- 2. operating point ----
    neg_path = os.path.join(args.probes, "benign_sarcasm_screened.csv")
    neg = pd.read_csv(neg_path if os.path.exists(neg_path) else os.path.join(args.probes, "benign_sarcasm.csv"))
    p_neg = 1.0 - probs_for(neg)[:, nb]
    # The metric that actually matches the paper's claim: can the model tell indirect abuse from
    # harmless sarcasm at all? Rank every probe item by p(bullying); ironic abuse are the positives,
    # benign sarcasm the negatives. This is threshold-free, so it separates "cannot see it" from
    # "sees it but cannot distinguish it".
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_disc = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s_disc = np.r_[pos["p_bullying"].values, p_neg]
    res["sarcasm_discrimination_auc"] = float(roc_auc_score(y_disc, s_disc))
    res["sarcasm_discrimination_ap"] = float(average_precision_score(y_disc, s_disc))
    res["mean_p_bullying"] = {"ironic_abuse": float(pos["p_bullying"].mean()), "benign_sarcasm": float(p_neg.mean())}
    curve = []
    for th in np.arange(0.1, 0.95, 0.05):
        curve.append({"threshold": round(float(th), 2),
                      "ironic_recall": float((pos["p_bullying"] >= th).mean()),
                      "benign_fpr": float((p_neg >= th).mean())})
    pd.DataFrame(curve).to_csv(os.path.join(args.out, "operating_point.csv"), index=False)
    res["operating_point_at_0.5"] = next(c for c in curve if abs(c["threshold"] - 0.5) < 1e-6)
    # the threshold that keeps false positives at or under 10 per cent
    ok = [c for c in curve if c["benign_fpr"] <= 0.10]
    res["threshold_for_fpr_10pct"] = max(ok, key=lambda c: c["ironic_recall"]) if ok else None

    # ---- 3. where the benchmark errors are ----
    te = map_labels(pd.read_csv(args.test), args.label_col, args.scheme)
    p_te = probs_for(te)
    te["pred"] = p_te.argmax(1)
    cm = pd.crosstab(te["label"].map(dict(enumerate(names))), te["pred"].map(dict(enumerate(names))),
                     rownames=["true"], colnames=["pred"])
    cm.to_csv(os.path.join(args.out, "confusion.csv"))
    res["confusion"] = cm.to_dict()
    for cls in HARD_CLASSES.get(args.scheme, ()):
        if cls in names:
            i = names.index(cls)
            sub = te[te["label"] == i]
            if len(sub):
                res[f"{cls}_recall"] = float((sub["pred"] == i).mean())
                worst = sub[sub["pred"] != i]["pred"].map(dict(enumerate(names))).value_counts()
                res[f"{cls}_confused_with"] = worst.head(3).to_dict()
    # does the benchmark's own hard class depend on profanity too?
    hard = next((c for c in HARD_CLASSES.get(args.scheme, ()) if c in names), None)
    if hard:
        i = names.index(hard)
        sub = te[te["label"] == i].copy()
        sub["explicit"] = sub["text"].map(has_profanity)
        res[f"{hard}_lexical_split"] = {
            lab: {"n": int((sub["explicit"] == f).sum()),
                  "recall": float((sub[sub["explicit"] == f]["pred"] == i).mean()) if (sub["explicit"] == f).any() else None}
            for f, lab in ((True, "with_profanity"), (False, "without_profanity"))}

    save_json(res, os.path.join(args.out, "implicit_analysis.json"))
    for k, v in res.items():
        if k != "confusion":
            print(f"{k}: {v}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
