"""Stage 3: train a compact student. Four modes share one script so every baseline is the
same code path with different flags.

  ft       fine-tune only (the control)            --mode ft
  skd      single-teacher KD                       --mode skd --teachers <one cache tag>
  uniform  multi-teacher, equal weights            --mode uniform --teachers a b c
  dmthd    multi-teacher, dynamic weights          --mode dmthd --teachers a b c [--aux --delta 0.3]

Ablations: --per_batch (instead of per-instance), --no_hidden, --from_scratch.
Disagreement-aware variant (datasets with an annotator-fraction column): --disagreement --kappa 1.0
[--reliability soft].
Students: any HF encoder name, or `bilstm` for the non-transformer heterogeneous student.

    python -m dmthd.train_student --student google/bert_uncased_L-4_H-256_A-4 --data_dir data/tweets \
        --cache cache/tweets --teachers bert-large hatebert irony --mode dmthd --aux --delta 0.3 \
        --seed 1 --out_dir runs/tweets/bert-mini/dmthd/seed1
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup

from .evaluate import compute_metrics, make_loader, predict_probs
from .losses import agreement_from_soft, dmthd_loss
from .models import Student, count_params, is_bilstm, load_classifier, load_tokenizer, uses_amp
from .utils import Timer, ensure_dir, get_device, label_names, load_json, map_labels, save_json, set_seed, split_fingerprint


def load_caches(cache_dir, tags, train_df=None):
    meta = load_json(os.path.join(cache_dir, "meta.json"))
    if train_df is not None and meta.get("fingerprint"):
        fp = split_fingerprint(train_df["text"].tolist(), train_df["label"].tolist())
        if fp != meta["fingerprint"]:
            raise SystemExit(
                f"cache/split mismatch: {cache_dir} was built on a different training split "
                f"(cache {meta['fingerprint']}, current {fp}). Teacher logits are indexed by row "
                f"position, so training on this pairing would be silently wrong. Rebuild the cache "
                f"with dmthd.cache_teachers on the current split.")
    metas = {t["tag"]: t for t in meta["teachers"]}
    logits, pooled, dims = [], [], []
    for tag in tags:
        z = np.load(os.path.join(cache_dir, f"{tag}.npz"))
        logits.append(torch.from_numpy(z["logits"].astype(np.float32)))
        pooled.append(torch.from_numpy(z["pooled"]))  # float16 on CPU; cast per batch
        dims.append(metas[tag]["dim"])
    return torch.stack(logits, 0), pooled, dims  # [K, N, C], list of [N, d_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="dmthd", choices=["ft", "skd", "uniform", "dmthd"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--teachers", nargs="*", default=[], help="cache tags, e.g. bert-large hatebert irony")
    ap.add_argument("--aux", action="store_true", help="add the auxiliary irony head (needs aux_irony.npz)")
    ap.add_argument("--scheme", default="six", choices=["six", "five", "binary"])
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--soft_col", default="soft_label")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--per_batch", action="store_true")
    ap.add_argument("--no_hidden", action="store_true")
    ap.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--disagreement", action="store_true", help="scale CE by annotator agreement and KL by disagreement")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--reliability", default="hard", choices=["hard", "soft"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-5 for transformers, 1e-3 for the BiLSTM")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="free-text label stored in results.json (e.g. ablation name)")
    ap.add_argument("--no_resume", action="store_true", help="ignore an existing ckpt_last.pt and start over")
    ap.add_argument("--keep_ckpt", action="store_true", help="keep ckpt_last.pt after a successful run (testing)")
    args = ap.parse_args()

    if args.mode != "ft" and not (args.cache and args.teachers):
        raise SystemExit("--mode skd/uniform/dmthd needs --cache and --teachers")
    if args.mode == "skd" and len(args.teachers) != 1:
        raise SystemExit("--mode skd takes exactly one teacher")
    lr = args.lr or (1e-3 if is_bilstm(args.student) else 3e-5)

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
    has_soft = args.soft_col in tr.columns and C == 2
    if args.disagreement and not has_soft:
        raise SystemExit("--disagreement needs a binary task with an annotator-fraction column")

    t_logits, t_pooled, dims = (None, None, [])
    aux_logits = None
    if args.mode != "ft":
        t_logits, t_pooled, dims = load_caches(args.cache, args.teachers, train_df=tr)
        assert t_logits.shape[1] >= len(tr), "cache shorter than training split: rebuild the cache on this split"
        if args.aux:
            aux_logits = torch.from_numpy(np.load(os.path.join(args.cache, "aux_irony.npz"))["logits"].astype(np.float32))

    tok = load_tokenizer(args.student)
    use_hidden = (args.mode != "ft") and not args.no_hidden
    student = Student(args.student, C, teacher_dims=dims if use_hidden else (), aux_labels=2 if aux_logits is not None else 0,
                      from_scratch=args.from_scratch).to(device)
    train_loader = make_loader(tr, tok, args.max_len, args.batch, True, soft_col=args.soft_col)
    val_loader = make_loader(va, tok, args.max_len, args.batch * 2, False)
    test_loader = make_loader(te, tok, args.max_len, args.batch * 2, False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    use_amp = uses_amp(args.student, device, args.fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ensure_dir(args.out_dir)
    logits_fn = lambda ii, am: student(ii, am)[0]
    best_f1, bad_epochs, history, timer = -1.0, 0, [], Timer()
    start_epoch, elapsed_before, stopped = 1, 0.0, False
    ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu")
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best_f1, bad_epochs, history = ck["best_f1"], ck["bad_epochs"], ck["history"]
        start_epoch, elapsed_before = ck["epoch"] + 1, ck["elapsed_s"]
        print(f"resumed from checkpoint after epoch {ck['epoch']} (best val macro-F1 so far {best_f1:.4f})", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        sums, n, wsum, nan_batches = {}, 0, None, 0
        for batch in train_loader:
            ii, am = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            y, idx = batch["labels"].to(device), batch["idx"]
            kw = {}
            if t_logits is not None:
                kw["teacher_logits"] = t_logits[:, idx].to(device)
                if use_hidden:
                    kw["teacher_pooled"] = [p[idx].to(device).float() for p in t_pooled]
                    kw["projections"] = student.proj
                if aux_logits is not None:
                    kw["aux_teacher_logits"] = aux_logits[idx].to(device)
            if has_soft and "soft" in batch:
                soft = batch["soft"].to(device)
                kw["soft_targets"] = soft
                if args.disagreement:
                    kw["agreement"] = agreement_from_soft(soft)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, pooled, aux = student(ii, am)
                loss, parts, w = dmthd_loss(
                    logits.float(), y, T=args.T, tau=args.tau, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                    per_instance=not args.per_batch, uniform=(args.mode == "uniform"),
                    student_pooled=pooled.float(), use_hidden=use_hidden,
                    aux_logits=None if aux is None else aux.float(), delta=args.delta,
                    kappa=args.kappa, reliability=args.reliability, **kw)
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            bs = len(y)
            sums["total"] = sums.get("total", 0.0) + loss.item() * bs
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach() if hasattr(v, 'detach') else v) * bs
            if w is not None:
                wsum = w.cpu() * bs if wsum is None else wsum + w.cpu() * bs
            n += bs
        student.eval()
        vm = compute_metrics(va["label"].values, predict_probs(logits_fn, val_loader, device), names)
        row = {"epoch": epoch, "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"], "elapsed_s": elapsed_before + timer.elapsed(), "nan_batches": nan_batches}
        row.update({f"loss_{k}": v / max(n, 1) for k, v in sums.items()})
        if wsum is not None:
            for k, tag in enumerate(args.teachers):
                row[f"w_{tag}"] = float(wsum[k] / max(n, 1))
        history.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, bad_epochs = vm["macro_f1"], 0
            student.save(args.out_dir, tok)
        else:
            bad_epochs += 1
            stopped = bad_epochs >= args.patience
        # power-cut insurance: everything needed to continue from the next epoch
        torch.save({"epoch": epoch, "student": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_f1": best_f1, "bad_epochs": bad_epochs, "history": history,
                    "elapsed_s": elapsed_before + timer.elapsed()}, ckpt_path)
        if stopped:
            print(f"early stop at epoch {epoch}")
            break
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    if os.path.exists(ckpt_path) and not args.keep_ckpt:
        os.remove(ckpt_path)

    best = load_classifier(args.out_dir, C).to(device).eval()
    probs = predict_probs(lambda ii, am: best(input_ids=ii, attention_mask=am).logits, test_loader, device)
    res = {"student": args.student, "mode": args.mode, "tag": args.tag, "teachers": args.teachers, "aux": bool(aux_logits is not None),
           "per_instance": not args.per_batch, "hidden": use_hidden, "from_scratch": args.from_scratch,
           "disagreement": args.disagreement, "kappa": args.kappa if args.disagreement else None, "reliability": args.reliability,
           "scheme": args.scheme, "seed": args.seed, "T": args.T, "tau": args.tau, "alpha": args.alpha, "beta": args.beta,
           "gamma": args.gamma, "delta": args.delta if aux_logits is not None else 0.0, "lr": lr, "epochs_run": len(history),
           "best_val_macro_f1": best_f1, "params": count_params(best), "train_time_s": elapsed_before + timer.elapsed(),
           "test": compute_metrics(te["label"].values, probs, names)}
    np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), te["label"].values)
    save_json(res, os.path.join(args.out_dir, "results.json"))
    print("TEST:", res["test"])


if __name__ == "__main__":
    main()
