"""Regression test for the resume bug that cost a Kaggle session.

The driver drops model weights when resuming a previous session, keeping them only where a run still
needs its checkpoint. Teachers were caught by that rule once the probe stage had written them an
eval_test.json, so a resumed session held four teachers as metadata with no weights. Nothing noticed
until a new teacher joined the committee, which invalidates the logit cache and rebuilds it from
exactly those directories.

Self-contained: builds a synthetic runs tree, no corpora and no GPU.

    python scripts/test_resume_weights.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_driver():
    spec = importlib.util.spec_from_file_location("rb", os.path.join(ROOT, "kaggle", "run_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rb"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_tree(src):
    """A finished session: teacher and student both probe-evaluated, plus one student still pending."""
    cases = {
        "teachers/bert-large": ["results.json", "eval_test.json", "config.json", "model.safetensors"],
        "bert-mini/ft/seed1": ["results.json", "eval_test.json", "config.json", "model.safetensors"],
        "bert-mini/ft/seed2": ["results.json", "config.json", "model.safetensors"],
    }
    for rel, files in cases.items():
        d = os.path.join(src, *rel.split("/"))
        os.makedirs(d, exist_ok=True)
        for f in files:
            open(os.path.join(d, f), "w").write("{}" if f.endswith(".json") else "x")


def has_weights(d):
    return os.path.exists(os.path.join(d, "model.safetensors"))


def main():
    rb = load_driver()
    base = tempfile.mkdtemp()
    src, dst = os.path.join(base, "src"), os.path.join(base, "dst")
    make_tree(src)
    rb.Bench._copy_run_tree(rb.Bench.__new__(rb.Bench), src, dst)

    checks = [
        ("a probe-evaluated teacher keeps its weights", has_weights(os.path.join(dst, "teachers", "bert-large")), True),
        ("a probe-evaluated student drops its weights", has_weights(os.path.join(dst, "bert-mini", "ft", "seed1")), False),
        ("a pending student keeps its weights", has_weights(os.path.join(dst, "bert-mini", "ft", "seed2")), True),
    ]

    # and the backstop: caching must refuse a weightless teacher by name, not fail inside the loader
    data = os.path.join(base, "data")
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, "train.csv"), "w", encoding="utf-8") as fh:
        print("text,label_name", file=fh)
        for name in ("age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"):
            print(f"some text about {name},{name}", file=fh)
    bare = os.path.join(base, "bare-teacher")
    os.makedirs(bare, exist_ok=True)
    json.dump({}, open(os.path.join(bare, "config.json"), "w"))
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "dmthd.cache_teachers", "--data_dir", data,
                        "--out", os.path.join(base, "cache"), "--teachers", bare],
                       env=env, capture_output=True, text=True, cwd=ROOT)
    said = "no model weights" in (r.stdout + r.stderr)
    checks.append(("caching names a weightless teacher instead of failing in the loader",
                   r.returncode != 0 and said, True))

    ok = True
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(("PASS " if good else "FAIL ") + name)
    print("RESUME WEIGHTS TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
