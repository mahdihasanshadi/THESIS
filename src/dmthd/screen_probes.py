"""Cut the manual screening of the benign-sarcasm probe set down to a short list.

The probe set must contain sarcastic tweets that are NOT abusive. Rather than a person reading
all of them, a classical bullying classifier trained on the tweet corpus flags the ones it is
confident are abusive; a person reviews only the flagged rows. The unflagged rows are kept as-is.

    python -m dmthd.screen_probes --data_dir E:/dmthd-work/data/tweets --probes probes --threshold 0.8

Writes probes/benign_sarcasm_review.csv (flagged rows, for a person to check) and
probes/benign_sarcasm_screened.csv (all rows with p_bullying < threshold), plus counts.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils import label_names, map_labels, not_bullying_index, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    tr = map_labels(pd.read_csv(os.path.join(args.data_dir, "train.csv")), "label_name", "six")
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300000, sublinear_tf=True)
    X = hstack([word.fit_transform(tr["text"]), char.fit_transform(tr["text"])]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=1).fit(X, tr["label"].values)

    probe = pd.read_csv(os.path.join(args.probes, "benign_sarcasm.csv"))
    Xp = hstack([word.transform(probe["text"]), char.transform(probe["text"])]).tocsr()
    probs = clf.predict_proba(Xp)
    nb = not_bullying_index("six")
    probe["p_bullying"] = 1.0 - probs[:, nb]
    probe["pred_class"] = [label_names("six")[i] for i in probs.argmax(1)]
    flagged = probe[probe["p_bullying"] >= args.threshold].sort_values("p_bullying", ascending=False)
    kept = probe[probe["p_bullying"] < args.threshold]
    flagged.to_csv(os.path.join(args.probes, "benign_sarcasm_review.csv"), index=False)
    kept.drop(columns=["p_bullying", "pred_class"]).to_csv(os.path.join(args.probes, "benign_sarcasm_screened.csv"), index=False)
    report = {"rows": int(len(probe)), "threshold": args.threshold, "flagged_for_review": int(len(flagged)),
              "kept_unflagged": int(len(kept)), "mean_p_bullying": float(probe["p_bullying"].mean()),
              "note": "a person reads benign_sarcasm_review.csv and deletes the rows that are truly abusive; "
                      "the survivors are appended back to benign_sarcasm_screened.csv"}
    save_json(report, os.path.join(args.probes, "screening_report.json"))
    print(report)


if __name__ == "__main__":
    main()
