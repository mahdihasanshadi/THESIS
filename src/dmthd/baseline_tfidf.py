"""Classical floor: TF-IDF (word 1-2 grams + character 3-5 grams) with logistic regression.
Runs on CPU in about a minute and writes the same results.json layout as the neural runs,
so it appears in the aggregate table.

    python -m dmthd.baseline_tfidf --data_dir data/tweets --out_dir runs/tweets/tfidf_lr/seed1
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_metrics
from .utils import Timer, ensure_dir, label_names, map_labels, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    names = label_names(args.scheme)
    tr = map_labels(pd.read_csv(os.path.join(args.data_dir, "train.csv")), args.label_col, args.scheme)
    va = map_labels(pd.read_csv(os.path.join(args.data_dir, "val.csv")), args.label_col, args.scheme)
    te = map_labels(pd.read_csv(os.path.join(args.data_dir, "test.csv")), args.label_col, args.scheme)

    timer = Timer()
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300000, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(tr["text"]), char.fit_transform(tr["text"])]).tocsr()
    Xva = hstack([word.transform(va["text"]), char.transform(va["text"])]).tocsr()
    Xte = hstack([word.transform(te["text"]), char.transform(te["text"])]).tocsr()

    clf = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced", random_state=args.seed)
    clf.fit(Xtr, tr["label"].values)
    val_metrics = compute_metrics(va["label"].values, clf.predict_proba(Xva), names)
    probs = clf.predict_proba(Xte)
    res = {"student": "tfidf_lr", "model_name": "tfidf_lr", "mode": "tfidf", "tag": "", "scheme": args.scheme,
           "seed": args.seed, "C": args.C, "features": int(Xtr.shape[1]), "params": int(Xtr.shape[1] * len(names)),
           "best_val_macro_f1": val_metrics["macro_f1"], "train_time_s": timer.elapsed(),
           "test": compute_metrics(te["label"].values, probs, names)}
    ensure_dir(args.out_dir)
    np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), te["label"].values)
    save_json(res, os.path.join(args.out_dir, "results.json"))
    print("VAL:", val_metrics["macro_f1"], "TEST:", res["test"])


if __name__ == "__main__":
    main()
