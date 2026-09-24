# -*- coding: utf-8 -*-
"""Per-class analysis of where the out-of-sample gain lands, from the saved test predictions.

Reads test_probs.npy and test_labels.npy for the arms compared in Section 5.8 and reports, per
class, F1 averaged over the three seeds, the change against fine-tuning, and the confusion between
the two residual classes that Section 5.7 says the gain lives in. Nothing here is transcribed: the
numbers the thesis prints come from this script.

    python scripts/per_class.py [--archive E:/dmthd-work/kaggle_d738_v7]
"""
import argparse
import glob
import json
import os

import numpy as np

# the label encoder sorts the class names, which is the order the saved predictions use;
# asserted against the split's class counts below rather than assumed
CLASSES = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]
TEST_COUNTS = [793, 784, 753, 617, 583, 796]


def arm(archive, student, name):
    """(labels, predictions per seed) for one arm, or None when it was not run."""
    out = []
    for p in sorted(glob.glob(os.path.join(archive, "runs", "tweets", student, name, "seed*"))):
        f, g = os.path.join(p, "test_probs.npy"), os.path.join(p, "test_labels.npy")
        if os.path.exists(f) and os.path.exists(g):
            out.append((np.load(g), np.load(f).argmax(1)))
    return out or None


def f1_per_class(y, p, n=6):
    out = []
    for c in range(n):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def mean_f1(runs):
    return np.stack([f1_per_class(y, p) for y, p in runs]).mean(0)


def confusion(runs, a, b):
    """Mean count of true-a predicted-b over the seeds."""
    return float(np.mean([int(((y == a) & (p == b)).sum()) for y, p in runs]))


def write_csv(path, header, rows):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("DMTHD_ARCHIVE", "E:/dmthd-work/kaggle_d738_v7"))
    ap.add_argument("--tables", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "paper", "tables"))
    a = ap.parse_args()

    plan = [("bert-mini", "ft", "Fine-tune only"),
            ("bert-mini", "ft_matched_168k", "Fine-tune, 35 epochs"),
            ("bert-mini", "skd", "Single teacher, in sample"),
            ("bert-mini", "skd_transfer_42k", "+ 42k transfer rows"),
            ("bert-mini", "skd_transfer_168k", "+ 168k transfer rows"),
            ("bilstm", "ft", "BiLSTM, fine-tune only"),
            ("bilstm", "skd_transfer_168k", "BiLSTM, + 168k transfer rows")]

    rows, base = [], {}
    for student, name, label in plan:
        runs = arm(a.archive, student, name)
        if runs is None:
            print("missing:", student, name)
            continue
        counts = np.bincount(runs[0][0], minlength=6).tolist()
        assert counts == TEST_COUNTS, ("label order is not what CLASSES says", student, name, counts)
        f1 = mean_f1(runs)
        if name == "ft":
            base[student] = f1
        rows.append((student, label, len(runs), f1, runs))

    w = max(len(r[1]) for r in rows)
    print("PER-CLASS TEST F1, MEAN OVER SEEDS")
    print("-" * (w + 8 + 9 * len(CLASSES)))
    print(" " * (w + 8) + "".join("%9s" % c[:8] for c in CLASSES))
    for student, label, n, f1, _ in rows:
        print("%-*s  n=%d  " % (w, label, n) + "".join("%9.3f" % v for v in f1))
    print()
    print("CHANGE AGAINST THE SAME STUDENT FINE-TUNED WITHOUT TEACHERS")
    print("-" * (w + 8 + 9 * len(CLASSES)))
    print(" " * (w + 8) + "".join("%9s" % c[:8] for c in CLASSES))
    for student, label, n, f1, _ in rows:
        if label.endswith("fine-tune only") or label == "Fine-tune only":
            continue
        d = f1 - base[student]
        print("%-*s  n=%d  " % (w, label, n) + "".join("%+9.3f" % v for v in d))

    print()
    print("THE TWO RESIDUAL CLASSES, MEAN COUNTS OVER SEEDS (test split has 583 and 617)")
    print("-" * 78)
    o, nc = CLASSES.index("other_cyberbullying"), CLASSES.index("not_cyberbullying")
    print("%-*s  %10s %10s %10s %10s" % (w, "", "other->not", "not->other", "other ok", "not ok"))
    for student, label, n, f1, runs in rows:
        print("%-*s  %10.1f %10.1f %10.1f %10.1f"
              % (w, label, confusion(runs, o, nc), confusion(runs, nc, o),
                 confusion(runs, o, o), confusion(runs, nc, nc)))

    print()
    print("SHARE OF THE MACRO-F1 CHANGE CARRIED BY THE TWO RESIDUAL CLASSES")
    print("-" * 78)
    for student, label, n, f1, _ in rows:
        if label == "Fine-tune only" or label == "BiLSTM, fine-tune only":
            continue
        d = f1 - base[student]
        share = (d[o] + d[nc]) / d.sum() if abs(d.sum()) > 1e-9 else float("nan")
        print("%-*s  macro delta %+0.4f, residual classes %+0.4f (%.0f per cent)"
              % (w, label, d.mean(), (d[o] + d[nc]) / 6, 100 * share))

    os.makedirs(a.tables, exist_ok=True)
    write_csv(os.path.join(a.tables, "per_class.csv"), ["Arm", "Seeds"] + CLASSES,
              [[label, n] + ["%.3f" % v for v in f1] for _, label, n, f1, _ in rows])
    conf = []
    for student, label, n, f1, runs in rows:
        d = f1 - base[student]
        share = (d[o] + d[nc]) / d.sum() if abs(d.sum()) > 1e-9 else float("nan")
        conf.append([label, "%.0f" % confusion(runs, o, nc), "%.0f" % confusion(runs, nc, o),
                     "%.0f" % confusion(runs, o, o), "%.0f" % confusion(runs, nc, nc),
                     "%+.4f" % d.mean(), "" if label.endswith("fine-tune only") or label == "Fine-tune only"
                     else "%.0f" % (100 * share)])
    write_csv(os.path.join(a.tables, "per_class_confusion.csv"),
              ["Arm", "other->not", "not->other", "other correct", "not correct",
               "macro delta", "share from the two residual classes, per cent"], conf)


if __name__ == "__main__":
    main()
