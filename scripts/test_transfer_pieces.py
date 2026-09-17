"""Unit checks for the out-of-sample distillation pieces, CPU only, no corpus and no Hub.

    python scripts/test_transfer_pieces.py

1. dmthd_loss: label_mask removes unlabelled rows from the hard-label term and nothing else; a weight
   override replaces the computed weights exactly where it is not NaN.
2. knn_reliability.local_reliability: a teacher that is right on one cluster of validation texts and
   wrong on another is trusted on transfer texts near the first cluster and not near the second.
3. prepare_transfer: the conflicting-label tweets are recovered from a raw corpus, texts that occur in a
   split or a probe are dropped, and the output carries no label column.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dmthd.knn_reliability import local_reliability, weights_from_reliability   # noqa: E402
from dmthd.losses import dmthd_loss, teacher_weights                            # noqa: E402

checks = []


def check(name, got, want=True):
    checks.append((name, got == want, got, want))


# ---- 1. the loss
torch.manual_seed(0)
B, K, C = 8, 3, 6
s_logits = torch.randn(B, C, requires_grad=True)
t_logits = torch.randn(K, B, C)
y = torch.randint(0, C, (B,))
mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
full, parts_full, _ = dmthd_loss(s_logits, y, teacher_logits=t_logits, use_hidden=False)
masked, parts_masked, _ = dmthd_loss(s_logits, y, teacher_logits=t_logits, use_hidden=False, label_mask=mask)
ce_first_half = torch.nn.functional.cross_entropy(s_logits[:4], y[:4])
check("label_mask: the hard-label term averages over labelled rows only", torch.allclose(parts_masked["ce"], ce_first_half))
check("label_mask: the teacher term is unchanged", torch.allclose(parts_masked["kl"], parts_full["kl"]))
none, parts_none, _ = dmthd_loss(s_logits, y, teacher_logits=t_logits, use_hidden=False, label_mask=torch.zeros(B))
check("label_mask: an all-unlabelled batch has a zero hard-label term and a finite loss",
      float(parts_none["ce"]) == 0.0 and torch.isfinite(none).item())
over = torch.full((B, K), float("nan"))
over[4:] = torch.tensor([0.7, 0.2, 0.1])
w = teacher_weights(t_logits, y, 1.0, override=over)
w_plain = teacher_weights(t_logits, y, 1.0)
check("override: NaN rows keep the computed weights", torch.allclose(w[:4], w_plain[:4]))
check("override: other rows take the override", torch.allclose(w[4:], torch.tensor([0.7, 0.2, 0.1]).expand(4, K)))
check("override: rows sum to one", torch.allclose(w.sum(1), torch.ones(B)))
_, _, mean_w = dmthd_loss(s_logits, y, teacher_logits=t_logits, use_hidden=False, weights=over)
check("dmthd_loss passes the override through", mean_w.shape == (K,) and abs(float(mean_w.sum()) - 1.0) < 1e-5)

# ---- 2. local reliability
rng = np.random.default_rng(0)
d = 16
c1, c2 = rng.normal(size=d), rng.normal(size=d)
val = np.concatenate([c1 + 0.05 * rng.normal(size=(100, d)), c2 + 0.05 * rng.normal(size=(100, d))])
correct = np.r_[np.ones(100), np.zeros(100)]                       # right near c1, wrong near c2
query = np.stack([c1 + 0.05 * rng.normal(size=d), c2 + 0.05 * rng.normal(size=d)])
r = local_reliability(val, correct, query, k=20)
check("local reliability is high near the cluster the teacher gets right", r[0] > 0.9)
check("and low near the cluster it gets wrong", r[1] < 0.1)
w2 = weights_from_reliability(np.array([[0.9, 0.3], [0.5, 0.5]]), tau=1.0)
check("weights are proportional to local accuracy at tau = 1", np.allclose(w2[0], [0.75, 0.25]) and np.allclose(w2[1], [0.5, 0.5]))

# ---- 3. the transfer-set builder, offline (no Hub sources), on a synthetic raw corpus
tmp = tempfile.mkdtemp()
raw = os.path.join(tmp, "raw.csv")
rows = [("this tweet is fine and long enough", "not_cyberbullying"),
        ("you people are all the same honestly", "ethnicity"),
        ("you people are all the same honestly", "religion"),        # conflicting: becomes transfer text
        ("nobody asked you grandpa go to sleep", "age"),
        ("nobody asked you grandpa go to sleep", "gender"),           # conflicting, but also in a probe
        ("what a lovely day for everyone", "not_cyberbullying"),
        ("what a lovely day for everyone", "not_cyberbullying")]      # duplicate, not conflicting
pd.DataFrame(rows, columns=["tweet_text", "cyberbullying_type"]).to_csv(raw, index=False)
data_dir, probes = os.path.join(tmp, "tweets"), os.path.join(tmp, "probes")
os.makedirs(data_dir), os.makedirs(probes)
pd.DataFrame({"text": ["this tweet is fine and long enough"], "label_name": ["not_cyberbullying"], "label": [3]}).to_csv(os.path.join(data_dir, "train.csv"), index=False)
pd.DataFrame({"text": ["Nobody asked you grandpa, go to sleep!"], "source": ["x"], "split": ["y"]}).to_csv(os.path.join(probes, "ironic_abuse.csv"), index=False)
out = os.path.join(data_dir, "transfer.csv")
r = subprocess.run([sys.executable, "-m", "dmthd.prepare_transfer", "--data_dir", data_dir, "--out", out, "--probes", probes,
                    "--raw", raw, "--sources", ""], env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
                   capture_output=True, text=True)
check("prepare_transfer runs offline on the raw corpus alone", r.returncode == 0)
if r.returncode == 0:
    tf = pd.read_csv(out)
    check("the transfer set is unlabelled", "label" not in tf.columns and "label_name" not in tf.columns)
    check("a conflicting-label tweet becomes transfer text", (tf["text"] == "you people are all the same honestly").sum() == 1)
    check("a conflicting-label tweet that is also a probe text is dropped", not tf["text"].str.contains("grandpa").any())
    check("split texts and non-conflicting duplicates are not transfer text", len(tf) == 1)
else:
    print(r.stdout[-800:], r.stderr[-800:])

ok = True
for name, good, got, want in checks:
    ok &= bool(good)
    print(("PASS " if good else "FAIL ") + name + ("" if good else f" (got {got!r}, wanted {want!r})"))
print("TRANSFER PIECES TEST " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
