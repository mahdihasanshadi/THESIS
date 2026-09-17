"""What is the committee itself worth, before any student sees it?

Reads the five teachers' saved test probabilities from the Kaggle v4 tree and scores every way of
combining them that needs no gold label at inference (uniform mean, log-mean, confidence-weighted,
most-confident-teacher), the oracle, and a stacked gate fitted by cross-validation over the test set
(logistic regression on the concatenated teacher probabilities). If no combination beats the best
single teacher, no student distilled from the committee can be expected to gain from it either.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

T = r"E:\dmthd-work\kaggle_d738_v4\runs\tweets\teachers"
TAGS = ["bert-large", "hatebert", "irony", "implicit-spec", "deberta-base"]
COMMITTEES = {"homo": TAGS[:3], "spec": TAGS[:4], "hetero": TAGS[:3] + ["deberta-base"], "all five": TAGS}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


probs, y = {}, None
for t in TAGS:
    p = np.load(os.path.join(T, t, "test_probs.npy")).astype(np.float64)
    if not np.allclose(p.sum(1), 1, atol=1e-3):
        p = softmax(p)
    probs[t] = p
    yy = np.load(os.path.join(T, t, "test_labels.npy"))
    assert y is None or (y == yy).all()
    y = yy
n, k = y.shape[0], probs[TAGS[0]].shape[1]
f1 = lambda pred: f1_score(y, pred, average="macro")
print(f"test n={n}, classes={k}")
for t in TAGS:
    p = probs[t]
    print(f"  {t:<14} macro-F1 {f1(p.argmax(1)):.4f}  acc {accuracy_score(y, p.argmax(1)):.4f}  "
          f"mean max-prob {p.max(1).mean():.3f}  share of items with max-prob>0.99: {(p.max(1) > 0.99).mean():.3f}")

print("\ncombinations (no gold label needed at inference):")
for name, tags in COMMITTEES.items():
    P = np.stack([probs[t] for t in tags])                       # K x n x C
    uni = P.mean(0)
    logm = np.exp(np.log(P + 1e-12).mean(0))
    conf = P.max(2)                                               # K x n
    ent = -(P * np.log(P + 1e-12)).sum(2)                         # K x n
    w_conf = conf / conf.sum(0, keepdims=True)
    w_ent = np.exp(-ent) / np.exp(-ent).sum(0, keepdims=True)
    pick = P[conf.argmax(0), np.arange(n)]                        # most confident teacher decides
    correct = (P.argmax(2) == y)                                  # K x n
    oracle = correct.any(0)
    best_single = max(f1(probs[t].argmax(1)) for t in tags)
    print(f"  {name:<9} best single {best_single:.4f} | uniform mean {f1(uni.argmax(1)):.4f} | log-mean {f1(logm.argmax(1)):.4f} "
          f"| confidence-weighted {f1((w_conf[:, :, None] * P).sum(0).argmax(1)):.4f} | entropy-weighted {f1((w_ent[:, :, None] * P).sum(0).argmax(1)):.4f} "
          f"| most-confident picks {f1(pick.argmax(1)):.4f} | oracle acc {oracle.mean():.4f}")
    # where the teachers disagree, how often is the uniform mean right, and how often is at least one teacher right?
    dis = (P.argmax(2) != P.argmax(2)[0]).any(0)
    print(f"            disagreement on {dis.mean():.3f} of items; there: uniform right {(uni.argmax(1) == y)[dis].mean():.3f}, "
          f"some teacher right {oracle[dis].mean():.3f}, best single right {(probs[tags[0]].argmax(1) == y)[dis].mean():.3f}")

print("\nstacked gate, 5-fold CV over the test set (an estimate of what learned routing could reach):")
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for name, tags in COMMITTEES.items():
    X = np.concatenate([probs[t] for t in tags], 1)
    pred = np.zeros(n, dtype=int)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    print(f"  {name:<9} stacked macro-F1 {f1(pred):.4f}")
