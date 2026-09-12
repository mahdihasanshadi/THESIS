"""Does mixed precision cost the narrow students accuracy? Measure it before quoting their numbers.

    python scripts/fp16_check.py --data_dir /kaggle/working/data/tweets --out /kaggle/working/runs/fp16_check

The same configuration produced two very different results:

    BERT-mini, fine-tune only, 3 seeds, CPU, fp32   test macro-F1 0.8779 / 0.8770 / 0.8760
    BERT-mini, fine-tune only, 3 seeds, T4,  fp16   test macro-F1 0.8393 (mean over the same seeds)

Same script, same split (34,607 / 4,326 / 4,326, seed 42), same six epochs, same learning rate. The
gap is 3.8 points, forty times the seed-to-seed spread, and it puts the student four points *below*
a TF-IDF baseline, which a fine-tuned transformer should not be. The remaining difference is `--fp16`
and the hardware it runs on, and the pattern across students points the same way: the two narrow
BERT students and the BiLSTM land below the classical floor while the two wide models land above it.

This trains the affected students with and without mixed precision on the same seeds and prints the
difference. It is cheap, a few minutes per run on a T4, and until it has been run no number for
BERT-mini or BERT-small in this project should be quoted, because a training defect and a scientific
result are not the same thing and this one is currently indistinguishable from either.
"""
import argparse
import json
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
STUDENTS = {"bert-mini": "google/bert_uncased_L-4_H-256_A-4",
            "bert-small": "google/bert_uncased_L-4_H-512_A-8",
            "distilbert": "distilbert-base-uncased"}


def run(model, out, seed, fp16, data_dir, epochs, scheme, label_col, max_len, limit):
    if os.path.exists(os.path.join(out, "results.json")):
        return
    cmd = [sys.executable, "-m", "dmthd.train_student", "--student", model, "--data_dir", data_dir,
           "--out_dir", out, "--seed", str(seed), "--batch", "32", "--epochs", str(epochs),
           "--mode", "ft", "--scheme", scheme, "--label_col", label_col, "--max_len", str(max_len),
           "--tag", "fp16" if fp16 else "fp32"]
    if fp16:
        cmd.append("--fp16")
    if limit:
        cmd += ["--limit", str(limit)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, env={**os.environ, "PYTHONPATH": SRC}, check=False)


def score(d):
    try:
        r = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
        return r["test"]["macro_f1"], r.get("best_val_macro_f1")
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--students", nargs="*", default=["bert-mini", "bert-small"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--scheme", default="six")
    ap.add_argument("--label_col", default="label_name")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    for tag in a.students:
        for seed in a.seeds:
            for fp16 in (True, False):
                run(STUDENTS[tag], os.path.join(a.out, tag, "fp16" if fp16 else "fp32", f"seed{seed}"),
                    seed, fp16, a.data_dir, a.epochs, a.scheme, a.label_col, a.max_len, a.limit)

    print("\n" + "=" * 78)
    print(f"{'student':<12}{'precision':<11}{'seed':<6}{'val macro-F1':>14}{'test macro-F1':>15}")
    summary = {}
    for tag in a.students:
        for prec in ("fp32", "fp16"):
            tests = []
            for seed in a.seeds:
                t, v = score(os.path.join(a.out, tag, prec, f"seed{seed}"))
                if t is None:
                    continue
                tests.append(t)
                print(f"{tag:<12}{prec:<11}{seed:<6}{(v or float('nan')):>14.4f}{t:>15.4f}")
            if tests:
                summary[(tag, prec)] = sum(tests) / len(tests)
    print("=" * 78)
    for tag in a.students:
        a32, a16 = summary.get((tag, "fp32")), summary.get((tag, "fp16"))
        if a32 is None or a16 is None:
            continue
        print(f"{tag}: fp32 {a32:.4f}, fp16 {a16:.4f}, mixed precision costs {a32 - a16:+.4f}")
    print("\nIf the difference is within a few thousandths, mixed precision is exonerated and the\n"
          "low scores are real. If it is a point or more, every fp16 number for that student has to\n"
          "be retrained before it goes anywhere near the paper.")


if __name__ == "__main__":
    main()
