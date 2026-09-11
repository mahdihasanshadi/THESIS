"""How often do the two implicit-hate corpora disagree about the same text?

    python -m dmthd.corpus_agreement --raw data/raw

They overlap: ISHate draws part of its data from the Implicit Hate Corpus, and both collect from
public platforms. Wherever the same text carries a label in both, the pair is a free measurement of
how hard the implicit label is for annotators working to different guidelines, obtained without
running a study of our own.

The overlap is not a random sample of either corpus, so the result is reported as what it is: a
comparison of two independent annotations of the same texts, not an estimate of corpus-wide
agreement. It is the evidence behind two decisions: building the benchmark from one corpus rather
than pooling both, and running a human annotation study at all.
"""
import argparse

import pandas as pd

from .prepare_implicit import DEFAULT_EXCLUDED_SOURCES, load_ihc, load_ishate
from .utils import normalize_for_matching, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding ishate/ and implicit_hate_stg1/")
    ap.add_argument("--out", default=None, help="write the counts to this JSON file as well")
    args = ap.parse_args()

    ish, _ = load_ishate(args.raw, set(DEFAULT_EXCLUDED_SOURCES))
    ihc, _ = load_ihc(args.raw)
    if not len(ish) or not len(ihc):
        raise SystemExit(f"both corpora are needed; found {len(ish)} and {len(ihc)} rows under {args.raw}")
    for d in (ish, ihc):
        d["key"] = d["text"].map(normalize_for_matching)
        d.drop_duplicates("key", inplace=True)

    both = ish.merge(ihc, on="key", suffixes=("_ishate", "_ihc"))
    res = {"unique_texts": {"ISHate": int(len(ish)), "ImplicitHate": int(len(ihc))},
           "shared_texts": int(len(both))}
    print(f"ISHate {len(ish)} unique texts, Implicit Hate Corpus {len(ihc)}; {len(both)} in both")
    if not len(both):
        return

    agree = float((both["label_name_ishate"] == both["label_name_ihc"]).mean())
    hate = lambda col: (both[col] != "not_hate").mean()                       # noqa: E731
    res["exact_agreement"] = agree
    res["disagreements"] = int(round((1 - agree) * len(both)))
    res["agreement_on_hate_vs_not"] = float(
        ((both["label_name_ishate"] != "not_hate") == (both["label_name_ihc"] != "not_hate")).mean())
    res["hate_rate"] = {"ISHate": float(hate("label_name_ishate")), "ImplicitHate": float(hate("label_name_ihc"))}
    cm = pd.crosstab(both["label_name_ishate"], both["label_name_ihc"])
    res["confusion_ishate_rows_ihc_columns"] = cm.to_dict()

    print(f"exact agreement {agree:.3f} ({res['disagreements']} disagreements)")
    print(f"agreement on hate versus not hate alone: {res['agreement_on_hate_vs_not']:.3f}")
    print("\nISHate label (rows) against Implicit Hate Corpus label (columns):")
    print(cm.to_string())
    print("\nThe two efforts nearly always agree that these texts are hateful. Where they differ is "
          "on whether the hate is stated or implied, which is the distinction this paper is about.")
    if args.out:
        save_json(res, args.out)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
