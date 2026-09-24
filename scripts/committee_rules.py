"""What the committee is worth before any student sees it, and what the student could receive.

    python scripts/committee_rules.py --runs runs/tweets --out paper/tables

Reads the teachers' saved test probabilities and scores every way of combining them that needs no gold
label at inference: the uniform mean, confidence weighting, entropy weighting, taking the most confident
teacher, and a stacked logistic-regression gate fitted over the test set by five-fold cross-validation,
which is as favourable a test of learned routing as can be made without touching the training data. The
oracle that picks a correct teacher whenever one exists bounds what any routing rule could reach.

Table 5.5 of the thesis is this file's output. If no combination beats the best single teacher by more
than the test set can resolve, no student distilled from that committee can be expected to gain from it.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

TAGS = ["bert-large", "hatebert", "irony", "implicit-spec", "deberta-base"]
COMMITTEES = [("Homogeneous (3)", ["bert-large", "hatebert", "irony"]),
              ("+ implicit specialist (4)", ["bert-large", "hatebert", "irony", "implicit-spec"]),
              ("+ DeBERTa (4)", ["bert-large", "hatebert", "irony", "deberta-base"]),
              ("All five", TAGS)]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(runs):
    probs, y = {}, None
    for t in TAGS:
        d = os.path.join(runs, "teachers", t)
        p = np.load(os.path.join(d, "test_probs.npy")).astype(np.float64)
        if not np.allclose(p.sum(1), 1, atol=1e-3):
            p = softmax(p)
        probs[t] = p
        labels = np.load(os.path.join(d, "test_labels.npy"))
        assert y is None or (y == labels).all(), "teachers were scored on different test rows"
        y = labels
    return probs, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs/<dataset>, holding teachers/<tag>/test_probs.npy")
    ap.add_argument("--out", default=None, help="directory for committee_rules.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probs, y = load(a.runs)
    f1 = lambda pred: f1_score(y, pred, average="macro")
    print("test rows: %d" % len(y))
    for t in TAGS:
        print("  %-14s macro-F1 %.4f  accuracy %.4f  mean top probability %.3f"
              % (t, f1(probs[t].argmax(1)), accuracy_score(y, probs[t].argmax(1)), probs[t].max(1).mean()))

    rows = []
    for name, tags in COMMITTEES:
        stack = np.stack([probs[t] for t in tags], 0)                       # [K, N, C]
        best_single = max(f1(probs[t].argmax(1)) for t in tags)
        uniform = stack.mean(0)
        conf = stack.max(-1)                                                # [K, N]
        confidence = (stack * (conf / conf.sum(0, keepdims=True))[..., None]).sum(0)
        ent = -(stack * np.log(stack + 1e-12)).sum(-1)                      # [K, N]
        w_ent = np.exp(-ent) / np.exp(-ent).sum(0, keepdims=True)
        entropy_w = (stack * w_ent[..., None]).sum(0)
        most_conf = stack[conf.argmax(0), np.arange(stack.shape[1])]
        correct = np.stack([probs[t].argmax(1) == y for t in tags], 0)
        oracle_acc = float(correct.any(0).mean())
        # where the committee is not unanimous: how often some teacher is right, and the uniform mean
        disagree = (stack.argmax(-1) != stack.argmax(-1)[0]).any(0)
        print("  %-26s disagreement on %.3f of the test set; there some teacher is right on %.3f and the "
              "uniform mean on %.3f" % (name, disagree.mean(), correct.any(0)[disagree].mean(),
                                        (uniform.argmax(1) == y)[disagree].mean()))
        # stacked gate: logistic regression on the concatenated teacher probabilities, five-fold CV
        x = np.concatenate([probs[t] for t in tags], 1)
        gate = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=a.seed).split(x, y):
            gate[te] = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(x[tr], y[tr]).predict(x[te])
        rows.append({"Committee": name, "Best single": "%.4f" % best_single,
                     "Uniform mean": "%.4f" % f1(uniform.argmax(1)),
                     "Confidence-weighted": "%.4f" % f1(confidence.argmax(1)),
                     "Entropy-weighted": "%.4f" % f1(entropy_w.argmax(1)),
                     "Most confident teacher": "%.4f" % f1(most_conf.argmax(1)),
                     "Stacked gate (5-fold CV)": "%.4f" % f1(gate),
                     "Oracle accuracy": "%.4f" % oracle_acc})
    t = pd.DataFrame(rows)
    print()
    print(t.to_string(index=False))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        t.to_csv(os.path.join(a.out, "committee_rules.csv"), index=False)
        print("\nwrote", os.path.join(a.out, "committee_rules.csv"))


if __name__ == "__main__":
    main()
