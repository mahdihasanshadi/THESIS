"""Stage 1: task-adapt a teacher checkpoint on a dataset's training split.

    python -m dmthd.train_teacher --model_name bert-large-uncased --data_dir data/tweets \
        --out_dir runs/tweets/teachers/bert-large --epochs 5 --lr 2e-5 --batch 32 --fp16 --grad_ckpt

Keeps the epoch with the best validation macro-F1, then reports test metrics to results.json.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from .evaluate import compute_metrics, make_loader, predict_probs
from .models import load_classifier, load_tokenizer, uses_amp
from .utils import Timer, ensure_dir, get_device, label_names, map_labels, save_json, set_seed


def class_weight_tensor(labels, num_labels, device):
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    w = counts.sum() / (num_labels * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tokenizer_name", default=None,
                    help="tokenizer to use when the checkpoint ships none (e.g. bert-base-uncased for prajjwal1/* models)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary", "implicit3"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--soft_col", default="soft_label", help="optional annotator-fraction column (Wikipedia)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--class_weighted", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N training rows")
    ap.add_argument("--no_resume", action="store_true", help="ignore an existing ckpt_last.pt and start over")
    ap.add_argument("--keep_ckpt", action="store_true", help="keep ckpt_last.pt after a successful run (testing)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CPU only)"), flush=True)
    names = label_names(args.scheme)
    C = len(names)
    tr = map_labels(pd.read_csv(os.path.join(args.data_dir, "train.csv")), args.label_col, args.scheme)
    va = map_labels(pd.read_csv(os.path.join(args.data_dir, "val.csv")), args.label_col, args.scheme)
    te = map_labels(pd.read_csv(os.path.join(args.data_dir, "test.csv")), args.label_col, args.scheme)
    if args.limit:
        tr, va, te = tr.head(args.limit), va.head(max(50, args.limit // 5)), te.head(max(50, args.limit // 5))

    tok = load_tokenizer(args.tokenizer_name or args.model_name)
    model = load_classifier(args.model_name, C).to(device)
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_loader = make_loader(tr, tok, args.max_len, args.batch, True, soft_col=args.soft_col)
    val_loader = make_loader(va, tok, args.max_len, args.batch * 2, False)
    test_loader = make_loader(te, tok, args.max_len, args.batch * 2, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    use_amp = uses_amp(args.model_name, device, args.fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    cw = class_weight_tensor(tr["label"].values, C, device) if args.class_weighted else None

    ensure_dir(args.out_dir)
    best_f1, history, timer = -1.0, [], Timer()
    start_epoch, elapsed_before = 1, 0.0
    ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best_f1, history, start_epoch, elapsed_before = ck["best_f1"], ck["history"], ck["epoch"] + 1, ck["elapsed_s"]
        print(f"resumed from checkpoint after epoch {ck['epoch']} (best val macro-F1 so far {best_f1:.4f})", flush=True)
    logits_fn = lambda ii, am: model(input_ids=ii, attention_mask=am).logits
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        run_loss, n, nan_batches = 0.0, 0, 0
        for batch in train_loader:
            ii, am, y = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ii, attention_mask=am).logits
                loss = F.cross_entropy(logits.float(), y, weight=cw)
                if "soft" in batch and C == 2:
                    p = torch.softmax(logits.float(), -1)[:, 1].clamp(1e-6, 1 - 1e-6)
                    loss = loss + F.binary_cross_entropy(p, batch["soft"].to(device))
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        model.eval()
        val_metrics = compute_metrics(va["label"].values, predict_probs(logits_fn, val_loader, device), names)
        history.append({"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_macro_f1": val_metrics["macro_f1"],
                        "val_acc": val_metrics["accuracy"], "elapsed_s": elapsed_before + timer.elapsed()})
        print(f"epoch {epoch}: loss {run_loss / max(n, 1):.4f}  val macro-F1 {val_metrics['macro_f1']:.4f}" + (f"  (skipped {nan_batches} non-finite batches)" if nan_batches else ""), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save_pretrained(args.out_dir)
            tok.save_pretrained(args.out_dir)
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_f1": best_f1, "history": history,
                    "elapsed_s": elapsed_before + timer.elapsed()}, ckpt_path)
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    if os.path.exists(ckpt_path) and not args.keep_ckpt:
        os.remove(ckpt_path)

    best = load_classifier(args.out_dir, C).to(device).eval()
    best_fn = lambda ii, am: best(input_ids=ii, attention_mask=am).logits
    probs = predict_probs(best_fn, test_loader, device)
    res = {"model_name": args.model_name, "scheme": args.scheme, "seed": args.seed, "epochs": args.epochs,
           "lr": args.lr, "batch": args.batch, "max_len": args.max_len, "best_val_macro_f1": best_f1,
           "train_time_s": timer.elapsed(), "test": compute_metrics(te["label"].values, probs, names)}
    np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), te["label"].values)
    save_json(res, os.path.join(args.out_dir, "results.json"))
    print("TEST:", res["test"])


if __name__ == "__main__":
    main()
