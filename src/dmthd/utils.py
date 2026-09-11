import json
import os
import random
import re
import time

import numpy as np

# Label schemes. Order of SIX matches label_info.json from the Phase-3 runs.
SIX = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]
FIVE = ["age", "ethnicity", "gender", "not_cyberbullying", "religion"]
BINARY = ["not_cyberbullying", "cyberbullying"]
# The implicit benchmark separates abuse-by-implication from abuse that says it outright, which is
# the distinction neither of the other two corpora labels.
IMPLICIT3 = ["not_hate", "explicit_hate", "implicit_hate"]

SCHEMES = {"six": SIX, "five": FIVE, "binary": BINARY, "implicit3": IMPLICIT3}
SCHEME_CHOICES = list(SCHEMES)
# The benign class of each scheme: the one the probes count as "did not fire".
BENIGN = {"six": "not_cyberbullying", "five": "not_cyberbullying", "binary": "not_cyberbullying",
          "implicit3": "not_hate"}


def label_names(scheme: str):
    return SCHEMES[scheme]


def not_bullying_index(scheme: str) -> int:
    """Index of the benign class under each scheme (used by the sarcasm probes)."""
    return label_names(scheme).index(BENIGN[scheme])


def map_labels(df, label_col: str, scheme: str):
    """Return a copy of df restricted to the scheme, with an integer `label` column.

    six    : all six classes as they are.
    five   : drop other_cyberbullying (the class the literature treats as noisy).
    binary : not_cyberbullying -> 0, everything else -> 1. If the column is already
             numeric (Wikipedia export), it is used as-is.
    """
    df = df.copy()
    names = label_names(scheme)
    if scheme == "binary" and np.issubdtype(df[label_col].dtype, np.number):
        df["label"] = df[label_col].astype(int)
        return df
    if scheme == "binary":
        df["label"] = (df[label_col].astype(str) != "not_cyberbullying").astype(int)
        return df
    df = df[df[label_col].isin(names)].copy()
    df["label"] = df[label_col].map({n: i for i, n in enumerate(names)}).astype(int)
    return df


def clean_text(t: str) -> str:
    """Training text: keep content, collapse whitespace only."""
    return re.sub(r"\s+", " ", str(t)).strip()


def normalize_for_matching(t: str) -> str:
    """Matching key for de-duplication and leakage checks: lower-case, collapse whitespace,
    strip the '@user' placeholders and URLs so trivially re-posted tweets collide."""
    t = str(t).lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def split_fingerprint(texts, labels=None) -> str:
    """Stable hash of a training split's row order. Teacher caches are indexed by position, so a
    cache may only be used with the split it was built on; this is what the check compares."""
    import hashlib
    h = hashlib.sha256()
    for i, t in enumerate(texts):
        h.update(str(t).encode("utf-8", "ignore"))
        if labels is not None:
            h.update(b"|" + str(labels[i]).encode())
        h.update(b"\n")
    return h.hexdigest()[:32]
