# Corpus manifests

None of the three corpora may be re-hosted: each carries its own licence and none of them is ours to
redistribute. A paper in that position usually claims reproducibility on the strength of a build
script, which nobody can check without rebuilding everything and hoping the result matches.

These files close that gap. Each manifest holds one row per example: the split it landed in, its
label, its source corpus, and a SHA-1 of the normalised text. No text, so nothing is redistributed.
It is enough to confirm, row by row, that a rebuild is identical to ours, and to find exactly which
rows differ when it is not.

| Corpus | Rows | Built by |
|---|---|---|
| `tweets` | 43,259 | `python -m dmthd.prepare_tweets` |
| `wikipedia` | 114,253 | `python -m dmthd.prepare_wikipedia --download` |
| `implicit` | 20,637 | `python -m dmthd.prepare_implicit --probes probes --download` |

The implicit manifest covers the train, validation and test splits, which come from the Implicit Hate
Corpus alone. The 27,096-row ISHate out-of-domain set is written by the same command and is not in
the manifest, because nothing is trained on it and its provenance is a single unmodified source.

To check a rebuild:

```bash
python -m dmthd.manifest --data_dir data/implicit --out /tmp/mine.csv.gz \
    --verify paper/manifests/implicit_manifest.csv.gz
```

It prints `MATCH` and exits 0, or prints how many rows differ in each direction and exits 1.

The `_meta.json` beside each manifest records the split fingerprints. Teacher logit caches are
indexed by row position, so a cache may only be paired with the split it was built on; the
fingerprint is what that check compares, and it is published here so the pairing can be verified
from outside.
