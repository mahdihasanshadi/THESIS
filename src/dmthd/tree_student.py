"""A non-neural student of the kind the multi-teacher papers with the largest reported gains use.

    python -m dmthd.tree_student --data_dir data/tweets --cache cache/tweets \
        --out_dir runs/tweets/xgb/uniform/seed1 --teachers bert-large hatebert irony --mode uniform --seed 1

Frozen sentence embeddings, reduced by PCA, classified by gradient-boosted trees, exactly the recipe
of Prasomphan (2025): sentence embeddings, PCA, XGBoost, soft targets blended with the gold labels at
alpha 0.7 / beta 0.3 and temperature 3. The arms differ in one thing only, where the target comes
from: the gold labels alone (`ft`, the control that paper omits), one teacher's softened outputs
blended with the gold labels (`skd`), the committee's (`uniform`), the committee's alone (`soft`), or
the committee's argmax as hard labels (`pseudo`). Features, hyper-parameters, split and seeds are held
equal, so a difference between arms is the label and nothing else.

The student cannot learn a representation: the encoder is frozen and the trees see a fixed vector, so
this is the setting in which soft labels have the most room to act, and the one our neural grid does
not cover. Decision D22 states the predictions before it runs.

Embeddings are cached per split and model under --emb_cache, so the arms and seeds share one pass.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd

from .evaluate import compute_metrics
from .utils import ensure_dir, get_device, label_names, load_json, map_labels, save_json, set_seed, split_fingerprint

MODES = ("ft", "skd", "uniform", "soft", "pseudo")


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed(texts, model_name, max_len, batch, cache_path=None):
    """Sentence embeddings from a frozen encoder: mean pooling over tokens, L2 normalised, as
    sentence-transformers does for the MiniLM models."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)["x"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                h = model(**enc).last_hidden_state
            v = mean_pool(h.float(), enc["attention_mask"])
            out.append(torch.nn.functional.normalize(v, dim=1).cpu().numpy())
            if i % (batch * 50) == 0:
                print(f"  embedded {i + len(out[-1])}/{len(texts)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    x = np.concatenate(out, 0).astype(np.float32)
    if cache_path:
        ensure_dir(os.path.dirname(cache_path))
        np.savez_compressed(cache_path, x=x)
    return x


def softmax(z, T=1.0):
    z = z.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def teacher_soft(cache_dir, tags, train_df, n, T):
    """The committee's softened outputs on the training split, averaged uniformly, [N, C]."""
    meta = load_json(os.path.join(cache_dir, "meta.json"))
    fp = split_fingerprint(train_df["text"].tolist(), train_df["label"].tolist())
    if meta.get("fingerprint") and meta["fingerprint"] != fp:
        raise SystemExit(
            f"cache/split mismatch: {cache_dir} was built on a different training split "
            f"(cache {meta['fingerprint']}, current {fp}). Teacher logits are indexed by row position, "
            f"so training on this pairing would be silently wrong.")
    probs = []
    for tag in tags:
        z = np.load(os.path.join(cache_dir, f"{tag}.npz"))["logits"].astype(np.float32)[:n]
        if len(z) != n:
            raise SystemExit(f"teacher {tag} has {len(z)} cached rows for {n} training rows")
        probs.append(softmax(z, T))
    return np.mean(probs, 0)


def targets(mode, y, C, soft, alpha, beta):
    onehot = np.eye(C, dtype=np.float32)[y]
    if mode == "ft":
        return onehot
    if mode == "pseudo":
        return np.eye(C, dtype=np.float32)[soft.argmax(1)]
    if mode == "soft":
        return soft
    return alpha * onehot + beta * soft          # skd and uniform differ only in how many teachers


def fit_trees(xtr, ytr, xva, yva, xte, seed, rounds, depth, lr, checkpoints):
    """Gradient-boosted trees on the target matrix. Returns (val macro-F1 curve, best rounds, test
    scores [N, C]). The number of rounds is chosen on validation macro-F1, as every neural arm keeps
    its best validation epoch."""
    from sklearn.metrics import f1_score
    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
    if XGBRegressor is not None:
        device = "cuda" if os.environ.get("TREE_DEVICE", "auto") != "cpu" and _cuda() else "cpu"
        model = XGBRegressor(n_estimators=rounds, max_depth=depth, learning_rate=lr, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", device=device,
                             random_state=seed, n_jobs=os.cpu_count() or 4, verbosity=0)
        model.fit(xtr, ytr)
        curve = [(k, float(f1_score(yva, model.predict(xva, iteration_range=(0, k)).argmax(1), average="macro")))
                 for k in checkpoints if k <= rounds]
        best = max(curve, key=lambda kv: kv[1])[0]
        return curve, best, model.predict(xte, iteration_range=(0, best))
    # no XGBoost on this machine: the same model family from scikit-learn, one regressor per class.
    # It cannot predict at an earlier round, so each budget is a separate fit and the grid is coarser.
    from sklearn.ensemble import HistGradientBoostingRegressor
    curve, best_scores, best, best_score = [], None, None, -1.0
    for k in sorted({c for c in checkpoints if c <= rounds} | {rounds})[::max(1, len(checkpoints) // 4)]:
        va, te = [], []
        for c in range(ytr.shape[1]):
            m = HistGradientBoostingRegressor(max_iter=k, learning_rate=lr, max_depth=depth,
                                              early_stopping=False, random_state=seed)
            m.fit(xtr, ytr[:, c])
            va.append(m.predict(xva))
            te.append(m.predict(xte))
        score = float(f1_score(yva, np.stack(va, 1).argmax(1), average="macro"))
        curve.append((k, score))
        if score > best_score:
            best, best_score, best_scores = k, score, np.stack(te, 1)
    return curve, best, best_scores


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def to_probs(scores):
    p = np.clip(scores, 0, None) + 1e-9
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="uniform", choices=MODES)
    ap.add_argument("--cache", default=None, help="teacher logit cache from dmthd.cache_teachers")
    ap.add_argument("--teachers", nargs="*", default=[], help="cache tags; the first one alone is used by skd")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--emb_cache", default=None, help="directory for the cached embeddings")
    ap.add_argument("--scheme", default="six")
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the gold labels")
    ap.add_argument("--beta", type=float, default=0.3, help="weight on the teachers' soft labels")
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    names = label_names(args.scheme)
    C = len(names)
    splits = {}
    for split in ("train", "val", "test"):
        df = map_labels(pd.read_csv(os.path.join(args.data_dir, f"{split}.csv")), args.label_col, args.scheme)
        splits[split] = df.head(args.limit) if args.limit else df
    slug = re.sub(r"[^a-z0-9]+", "-", args.embedder.lower()).strip("-")

    t0 = time.time()
    x = {}
    for split, df in splits.items():
        path = os.path.join(args.emb_cache, f"{split}_{slug}.npz") if args.emb_cache else None
        if args.limit and path:
            path = path.replace(".npz", f"_limit{args.limit}.npz")
        x[split] = embed(df["text"].tolist(), args.embedder, args.max_len, args.batch, path)
    embed_s = time.time() - t0

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(args.pca, x["train"].shape[1], len(x["train"])), random_state=0).fit(x["train"])
    xtr, xva, xte = (pca.transform(x[s]).astype(np.float32) for s in ("train", "val", "test"))
    ytr, yva, yte = (splits[s]["label"].values for s in ("train", "val", "test"))

    soft = None
    tags = args.teachers[:1] if args.mode == "skd" else list(args.teachers)
    if args.mode != "ft":
        if not (args.cache and tags):
            raise SystemExit(f"mode {args.mode} needs --cache and --teachers")
        soft = teacher_soft(args.cache, tags, splits["train"], len(ytr), args.T)
    ytr_target = targets(args.mode, ytr, C, soft, args.alpha, args.beta)

    t1 = time.time()
    checkpoints = [k for k in range(50, args.rounds + 1, 50)]
    curve, best, scores = fit_trees(xtr, ytr_target, xva, yva, xte, args.seed, args.rounds, args.depth,
                                    args.lr, checkpoints)
    train_s = time.time() - t1
    probs = to_probs(scores)
    res = {"student": f"trees:{args.embedder}", "mode": args.mode, "tag": args.tag,
           "teachers": tags if args.mode != "ft" else [], "scheme": args.scheme, "seed": args.seed,
           "T": args.T, "alpha": args.alpha if args.mode in ("skd", "uniform") else None,
           "beta": args.beta if args.mode in ("skd", "uniform") else None, "lr": args.lr,
           "embedder": args.embedder, "pca_dim": int(xtr.shape[1]), "depth": args.depth,
           "rounds_run": int(best), "rounds_budget": args.rounds, "params": None,
           "best_val_macro_f1": max(s for _, s in curve), "val_curve": curve,
           "embed_time_s": round(embed_s, 1), "train_time_s": round(train_s, 1),
           "test": compute_metrics(yte, probs, names)}
    ensure_dir(args.out_dir)
    np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), yte)
    pd.DataFrame(curve, columns=["rounds", "val_macro_f1"]).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    save_json(res, os.path.join(args.out_dir, "results.json"))
    print(f"{args.mode} seed {args.seed}: {best} rounds, val {res['best_val_macro_f1']:.4f}, "
          f"test macro-F1 {res['test']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
