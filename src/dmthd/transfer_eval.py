"""Cross-dataset transfer: apply a model trained on one corpus to the other corpus's test set,
collapsing both sides to bullying-vs-not.

  tweets model -> Wikipedia test : prediction is bullying if the argmax is any class except
                                   not_cyberbullying; the Wikipedia label is already binary.
  Wikipedia model -> tweets test  : the tweet label is collapsed to bullying-vs-not; the model's
                                   class 1 is bullying.

    python -m dmthd.transfer_eval --model_dir runs/tweets/bert-mini/dmthd/seed1 --model_scheme six \
        --csv data/wikipedia/test.csv --csv_scheme binary --csv_label_col label --max_len 256
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from .evaluate import make_loader, predict_probs
from .models import load_classifier, load_tokenizer
from .utils import get_device, label_names, map_labels, not_bullying_index, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_scheme", required=True, choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--csv_scheme", required=True, choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--csv_label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--focus_class", default=None,
                    help="a class of the target corpus to report separately, e.g. implicit_hate. "
                         "Collapsed transfer hides exactly the case the paper is about: a model can "
                         "score well overall by catching explicit abuse and still miss every "
                         "implication, so that subset is scored on its own against the benign class.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    names = label_names(args.model_scheme)
    tok = load_tokenizer(args.model_dir)
    model = load_classifier(args.model_dir, len(names)).to(device).eval()
    df = map_labels(pd.read_csv(args.csv), args.csv_label_col, args.csv_scheme)
    y_bin = (df["label"].values != not_bullying_index(args.csv_scheme)).astype(int)
    probs = predict_probs(lambda ii, am: model(input_ids=ii, attention_mask=am).logits,
                          make_loader(df, tok, args.max_len, args.batch, False), device)
    nb = not_bullying_index(args.model_scheme)
    p_bully = 1.0 - probs[:, nb]
    pred = (probs.argmax(1) != nb).astype(int)
    res = {"model_dir": args.model_dir, "model_scheme": args.model_scheme, "target_csv": args.csv, "n": int(len(df)),
           "binary_macro_f1": float(f1_score(y_bin, pred, average="macro")),
           "bullying_f1": float(f1_score(y_bin, pred)), "roc_auc": float(roc_auc_score(y_bin, p_bully)),
           "pr_auc": float(average_precision_score(y_bin, p_bully)), "predicted_bullying_rate": float(pred.mean()),
           "true_bullying_rate": float(y_bin.mean())}
    if args.focus_class:
        target = label_names(args.csv_scheme)
        if args.focus_class not in target:
            raise SystemExit(f"--focus_class {args.focus_class} is not a class of {args.csv_scheme}: {target}")
        fi, bi = target.index(args.focus_class), not_bullying_index(args.csv_scheme)
        m = np.isin(df["label"].values, [fi, bi])
        yf = (df["label"].values[m] == fi).astype(int)
        res[args.focus_class] = {
            "n_positive": int(yf.sum()), "n_negative": int((1 - yf).sum()),
            "recall": float(pred[m][yf == 1].mean()),
            "false_positive_rate": float(pred[m][yf == 0].mean()),
            "roc_auc": float(roc_auc_score(yf, p_bully[m])),
            "mean_p_abusive": {"positive": float(p_bully[m][yf == 1].mean()),
                               "negative": float(p_bully[m][yf == 0].mean())}}
    save_json(res, args.out or os.path.join(args.model_dir, "transfer_" + os.path.basename(os.path.dirname(args.csv)) + ".json"))
    print(res)


if __name__ == "__main__":
    main()
