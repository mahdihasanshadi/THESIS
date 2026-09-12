"""Stage 2: run every frozen teacher once over the training split and cache what the
student needs: logits [N, C] and masked-mean-pooled last-layer states [N, d_k], indexed by
training-row order. Also caches the auxiliary irony teacher's logits (its own label space).

    python -m dmthd.cache_teachers --data_dir data/tweets --out cache/tweets \
        --teachers runs/tweets/teachers/bert-large runs/tweets/teachers/hatebert runs/tweets/teachers/irony \
        --aux_model cardiffnlp/twitter-roberta-base-irony
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .evaluate import make_loader
from .models import encode_batch, load_classifier, load_tokenizer
from .utils import ensure_dir, get_device, label_names, map_labels, save_json, split_fingerprint


@torch.no_grad()
def run_teacher(model_dir_or_name, num_labels, df, max_len, batch, device):
    tok = load_tokenizer(model_dir_or_name)
    model = load_classifier(model_dir_or_name, num_labels).to(device).eval()
    loader = make_loader(df, tok, max_len, batch, False)
    logits, pooled = [], []
    for b in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg, po = encode_batch(model, b["input_ids"].to(device), b["attention_mask"].to(device))
        logits.append(lg.float().cpu().numpy())
        pooled.append(po.float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(logits, 0), np.concatenate(pooled, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teachers", nargs="*", default=[], help="task-adapted teacher directories")
    ap.add_argument("--aux_model", default=None, help="frozen irony model in its own label space")
    ap.add_argument("--aux_labels", type=int, default=2)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    C = len(label_names(args.scheme))
    df = map_labels(pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv")), args.label_col, args.scheme)
    if args.limit:
        df = df.head(args.limit)
    ensure_dir(args.out)
    # A teacher directory that resumed from a previous session can arrive complete except for the
    # weights, because the resume drops checkpoints it believes are finished with. Loading it then
    # fails deep inside the model loader with a message about the hub, several minutes of GPU time
    # after the run started. Say plainly what is missing instead.
    WEIGHTS = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
               "pytorch_model.bin.index.json", "bilstm.pt")
    bare = [t for t in args.teachers
            if os.path.isdir(t) and not any(os.path.exists(os.path.join(t, w)) for w in WEIGHTS)]
    if bare:
        raise SystemExit(
            "these teacher directories have no model weights, only metadata: " + ", ".join(bare) +
            ". A resumed session leaves finished checkpoints behind to save disk; re-run with "
            "RESUME_WEIGHTS=all, or resume from a session whose output still holds the teacher "
            "weights, or retrain the teachers.")
    meta = {"split": args.split, "n": int(len(df)), "scheme": args.scheme, "teachers": [], "aux": None,
            "fingerprint": split_fingerprint(df["text"].tolist(), df["label"].tolist())}
    for tdir in args.teachers:
        tag = os.path.basename(os.path.normpath(tdir))
        logits, pooled = run_teacher(tdir, C, df, args.max_len, args.batch, device)
        np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), logits=logits.astype(np.float16),
                            pooled=pooled.astype(np.float16))
        acc = float((logits.argmax(1) == df["label"].values).mean())
        meta["teachers"].append({"tag": tag, "dir": tdir, "dim": int(pooled.shape[1]), "train_acc": acc})
        print(f"cached {tag}: logits {logits.shape} pooled {pooled.shape} train-acc {acc:.4f}", flush=True)
    if args.aux_model:
        logits, _ = run_teacher(args.aux_model, args.aux_labels, df, args.max_len, args.batch, device)
        np.savez_compressed(os.path.join(args.out, "aux_irony.npz"), logits=logits.astype(np.float16))
        meta["aux"] = {"model": args.aux_model, "labels": args.aux_labels,
                       "positive_rate": float((logits.argmax(1) == 1).mean())}
        print(f"cached aux irony logits {logits.shape}; predicted-irony rate {meta['aux']['positive_rate']:.3f}")
    save_json(meta, os.path.join(args.out, "meta.json"))


if __name__ == "__main__":
    main()
