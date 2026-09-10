"""De-duplication, conflicting-label removal, stratified splitting and leakage assertions.

The published cyberbullying tweet corpus contains repeated tweets, some with different
labels. Reviewers know this. Everything here is deterministic and reports its counts.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import clean_text, normalize_for_matching

KEY = "_key"


def add_keys(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].map(clean_text)
    df[KEY] = df[text_col].map(normalize_for_matching)
    return df


def dedup_and_split(df: pd.DataFrame, text_col: str, label_col: str, seed: int = 42,
                    val_frac: float = 0.10, test_frac: float = 0.10, min_tokens: int = 2):
    """Returns (train, val, test, report). Rules, in order:
    1. drop rows shorter than `min_tokens` tokens after cleaning;
    2. drop every text that appears with more than one label (all its copies);
    3. drop exact duplicates on the matching key, keeping the first;
    4. stratified split; assert the three splits share no key.
    """
    report = {"rows_in": int(len(df))}
    df = add_keys(df, text_col)
    df = df[df[KEY].str.split().str.len() >= min_tokens]
    report["rows_after_min_len"] = int(len(df))

    n_labels = df.groupby(KEY)[label_col].nunique()
    conflict = set(n_labels[n_labels > 1].index)
    report["texts_with_conflicting_labels"] = int(len(conflict))
    report["rows_dropped_conflicting"] = int(df[KEY].isin(conflict).sum())
    df = df[~df[KEY].isin(conflict)]

    before = len(df)
    df = df.drop_duplicates(KEY, keep="first")
    report["rows_dropped_duplicates"] = int(before - len(df))
    report["rows_out"] = int(len(df))

    trainval, test = train_test_split(df, test_size=test_frac, stratify=df[label_col], random_state=seed)
    val_rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(trainval, test_size=val_rel, stratify=trainval[label_col], random_state=seed)
    assert_disjoint({"train": train, "val": val, "test": test})

    report["split"] = {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))}
    report["class_counts"] = {name: {str(k): int(v) for k, v in d[label_col].value_counts().items()}
                              for name, d in [("train", train), ("val", val), ("test", test)]}
    out = []
    for d in (train, val, test):
        d = d.drop(columns=[KEY]).reset_index(drop=True)
        out.append(d)
    return out[0], out[1], out[2], report


def assert_disjoint(splits: dict, key: str = KEY):
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(splits[names[i]][key]) & set(splits[names[j]][key])
            assert not inter, f"LEAK: {len(inter)} texts shared between {names[i]} and {names[j]}"


def overlap_count(df_a: pd.DataFrame, df_b: pd.DataFrame, text_col_a: str, text_col_b: str) -> int:
    """How many texts of df_a also occur in df_b (matching key). Use it to check a test set
    against any corpus a teacher checkpoint was trained on."""
    a = set(df_a[text_col_a].map(normalize_for_matching))
    b = set(df_b[text_col_b].map(normalize_for_matching))
    return len(a & b)
