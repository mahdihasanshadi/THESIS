# D-MTHD Phase 3 — codebase and runbook

Code for the two-week plan: data fix, teacher adaptation, teacher caching, student
distillation (fine-tune only / single-teacher / uniform / D-MTHD), sarcasm probes, an
efficiency benchmark reviewers can trust, and result aggregation with bootstrap intervals.

Every entry point is `python -m dmthd.<name> --help`. Every run writes a `results.json`,
so the Kaggle driver can be killed and restarted and it resumes where it stopped.

**What the project is about.** Detecting abuse that is carried by implication rather than stated
outright, with a model small enough to deploy. Two documents carry the reasoning: `LOG.md` is the
chronological lab notebook, one entry per step with the numbers copied from the result files;
`DECISIONS.md` is the reasoning record, with every keyword defined, every decision paired with what
it rejected and why, findings numbered so the paper can cite them, and an explicit list of what the
evidence does and does not yet license. Read `DECISIONS.md` first.

## Three benchmarks

| Name | Task | Rows (train/val/test) | Classical floor | Why it is here |
|---|---|---|---|---|
| `tweets` | six-class cyberbullying | 34,607 / 4,326 / 4,326 | 0.8798 macro-F1 | the primary benchmark; indirect abuse hides inside `other_cyberbullying` |
| `wikipedia` | binary personal attack, with annotator fractions | 68,750 / 22,782 / 22,721 | 0.8759 | a second domain and the only one with per-annotator agreement |
| `implicit` | not_hate / explicit_hate / implicit_hate | 37,744 / 4,718 / 4,719 | 0.6809, and only **0.4530** on implicit_hate against 0.7474 on explicit_hate | the only corpus where implication is a label, so the only one on which the paper's claim can be tested rather than asserted |

## Where things run (no local GPU)

| Work | Where | Why |
|---|---|---|
| Data fix, probes, statistics, figures, writing | this laptop (16 cores, CPU) | no GPU needed |
| BERT-tiny / BERT-mini students | this laptop, CPU, overnight | 11M-parameter models train in minutes per epoch on 16 cores |
| Teachers (BERT-large, HateBERT, irony), BERT-small, DistilBERT, Wikipedia runs | Kaggle (T4 x2 or P100, 30 GPU-h per account per week, sessions to 12 h) and Colab | the only real GPU need |
| Overflow | Colab Pro (about USD 10/month) or a RunPod / Vast.ai rental (RTX 3090 at about USD 0.3/h) | if Kaggle queues stall; the whole plan is 60–80 GPU-hours |

Four Kaggle accounts (one per team member) give 120 GPU-hours a week, more than the plan needs.
The tweet corpus is already a Kaggle dataset (`andrewmvd/cyberbullying-classification`), so on
Kaggle it is attached, not downloaded.

**Disk on this laptop:** C: has under 7 GB free. Everything large lives on E:
`E:\dmthd-work\{data,cache,runs,hf-cache}`. Keep only the code in this OneDrive folder;
never put runs or caches here or OneDrive will sync gigabytes.

## Local setup (done on 10 Sep)

```powershell
$env:PYTHONPATH = "C:\Users\Sadi\OneDrive\Desktop\dmthd-p3\src"
$env:HF_HOME    = "E:\dmthd-work\hf-cache"      # model downloads go to E:
pip install -r requirements.txt                 # torch CPU wheel already installed
```

Corpus: downloaded from the Hugging Face mirror `mattematics/cyberbullying`
(47,692 rows, identical columns to the Kaggle file) to `E:\dmthd-work\data\raw\cyberbullying_tweets.csv`.

## Pipeline

```powershell
# 1. Day-1 data fix: de-duplicate, drop conflicting labels, stratified 80/10/10, leakage assertions
python -m dmthd.prepare_tweets --raw E:\dmthd-work\data\raw\cyberbullying_tweets.csv --out E:\dmthd-work\data\tweets
#    -> train.csv val.csv test.csv label_info.json report.json (every removal count for the paper)

# 2. Teachers (GPU): task-adapt each checkpoint on the training split, keep best validation epoch
python -m dmthd.train_teacher --model_name bert-large-uncased --data_dir data/tweets --out_dir runs/tweets/teachers/bert-large --epochs 5 --lr 2e-5 --batch 32 --fp16 --grad_ckpt
python -m dmthd.train_teacher --model_name GroNLP/hateBERT   --data_dir data/tweets --out_dir runs/tweets/teachers/hatebert   --epochs 5 --lr 2e-5 --batch 32 --fp16
python -m dmthd.train_teacher --model_name cardiffnlp/twitter-roberta-base-irony --data_dir data/tweets --out_dir runs/tweets/teachers/irony --epochs 5 --lr 2e-5 --batch 32 --fp16

# 3. Cache teacher logits + pooled states once; also the frozen irony teacher's own logits
python -m dmthd.cache_teachers --data_dir data/tweets --out cache/tweets --teachers runs/tweets/teachers/bert-large runs/tweets/teachers/hatebert runs/tweets/teachers/irony --aux_model cardiffnlp/twitter-roberta-base-irony

# 4. Students: the four modes are one script with different flags (seeds 1 2 3)
S=google/bert_uncased_L-4_H-256_A-4
python -m dmthd.train_student --student $S --data_dir data/tweets --mode ft      --out_dir runs/tweets/bert-mini/ft/seed1      --seed 1
python -m dmthd.train_student --student $S --data_dir data/tweets --mode skd     --cache cache/tweets --teachers bert-large --out_dir runs/tweets/bert-mini/skd/seed1 --seed 1
python -m dmthd.train_student --student $S --data_dir data/tweets --mode uniform --cache cache/tweets --teachers bert-large hatebert irony --out_dir runs/tweets/bert-mini/uniform/seed1 --seed 1
python -m dmthd.train_student --student $S --data_dir data/tweets --mode dmthd   --cache cache/tweets --teachers bert-large hatebert irony --aux --delta 0.3 --out_dir runs/tweets/bert-mini/dmthd/seed1 --seed 1
#    ablations: --per_batch   --no_hidden   (drop --aux)   --from_scratch   --mode uniform

# 5. Sarcasm probes and per-class metrics for any saved model directory
python -m dmthd.evaluate --model_dir runs/tweets/bert-mini/dmthd/seed1 --csv data/tweets/test.csv --scheme six --probe_neg data/probes/benign_sarcasm.csv --probe_pos data/probes/ironic_abuse.csv

# 6. Efficiency: warm-up, 5 repeats, median; run on Kaggle (--device cuda) and here (--device cpu)
python -m dmthd.bench --model_dirs runs/tweets/bert-mini/dmthd/seed1 runs/tweets/teachers/bert-large --csv data/tweets/test.csv --device cpu --out runs/tweets/bench_cpu.csv

# 7. Tables and paired bootstrap intervals
python -m dmthd.aggregate --runs runs/tweets --out runs/tweets/summary.csv
python -m dmthd.aggregate --compare runs/tweets/bert-mini/ft runs/tweets/bert-mini/dmthd
```

`--limit N` on the training scripts uses only N rows: use it for debugging.
`--scheme five` drops `other_cyberbullying`; `--scheme binary` collapses to bullying / not.
For Wikipedia, export the Phase-2 splits as `text,label_name,label,soft_label` CSVs and add
`--scheme binary --label_col label`; the soft-label BCE term switches on automatically.

## Smoke tests (CPU, before every Kaggle run)

Two of them, and both matter. The first covers the library, the second covers the driver, which is
where the expensive bugs have actually been: a stage-wiring mistake costs a whole GPU session.

```powershell
python scripts\smoke_cpu.py --raw E:\dmthd-work\data\raw\cyberbullying_tweets.csv --root E:\dmthd-work\smoke
python scripts\smoke_driver.py --raw E:\dmthd-work\data\raw\cyberbullying_tweets.csv --implicit_raw E:\dmthd-work\data\raw --root E:\dmthd-work\smoke_driver
```

The driver test makes four passes: the full grid, every teacher collapsing, the implicit benchmark
end to end, and the implicit specialist being re-headed from three classes to six.

## The implicit benchmark and its specialist teacher

```powershell
# build the corpus (downloads ISHate and Implicit Hate Corpus stage 1 from the Hub)
python -m dmthd.prepare_implicit --raw E:\dmthd-work\data\raw --out E:\dmthd-work\data\implicit --probes probes --download

# run the whole benchmark; the specialist is off here, where it would duplicate the task teacher
SPECIALIST=0 python kaggle/run_benchmark.py --dataset implicit --stage all

# on the tweet benchmark the specialist joins the committee automatically
python kaggle/run_benchmark.py --dataset tweets --stage all --raw ...\cyberbullying_tweets.csv
```

Two measurements specific to the implicit claim:

```powershell
# does the model miss implication, or see it and fail to tell it from harmless sarcasm?
python -m dmthd.implicit_analysis --model_dir runs/tweets/bert-mini/ft/seed1 --test data/tweets/test.csv --probes probes --out runs/tweets/implicit_analysis

# does the per-instance weighting route implication to the specialist, or average over everyone?
python -m dmthd.weight_routing --cache cache/tweets --data_dir data/tweets --scheme six --out runs/tweets/weight_routing
```

Notebooks: `kaggle/dmthd_implicit.ipynb` first if you want the specialist trained once and reused,
then `kaggle/dmthd_tweets.ipynb` with the implicit output attached.

## Kaggle in three steps

1. The code lives at https://github.com/mahdihasanshadi/THESIS. Kaggle can clone it only if the repository is public, or if you attach the code zip (E:\dmthd-work\dmthd-p3-code.zip) as a Kaggle dataset instead. The simplest route is `kaggle/dmthd_tweets.ipynb`: on Kaggle choose Create, Import Notebook, and upload that file.
2. New notebook → Settings: Accelerator **GPU T4 x2**, Internet **on** → Add data: `andrewmvd/cyberbullying-classification`.
3. One cell, then **Save Version → Save & Run All (Commit)** so it keeps running for up to 12 h after you close the tab:

```python
!git clone https://github.com/mahdihasanshadi/THESIS.git dmthd-p3 && cd dmthd-p3 && pip install -q -r requirements.txt
%cd dmthd-p3
!ROOT=/kaggle/working PYTHONPATH=src python kaggle/run_tweets.py --stage all \
    --raw /kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv
```

Environment overrides for the driver: `TEACHERS`, `STUDENTS`, `SEEDS`, `MODES`, `ROOT`, `GPU`.
Split work across accounts by setting `STUDENTS` differently on each (e.g. one account runs
`distilbert-base-uncased:distilbert`, another `google/bert_uncased_L-4_H-512_A-8:bert-small`).
After the teachers stage finishes once, publish `runs/tweets/teachers` and `cache/tweets` as a
Kaggle dataset so the other accounts skip straight to students. Download `runs/` at the end;
`aggregate` works on any merged `runs/` tree.

## Output layout

```
runs/tweets/teachers/<tag>/           model + results.json + history.csv + test_probs.npy
runs/tweets/<student>/<mode>/seed<k>/ model + dmthd_heads.pt + results.json + history.csv (loss parts, mean teacher weights per epoch)
runs/tweets/<student>/ablation_<tag>/seed<k>/
runs/tweets/summary.csv               mean ± std over seeds
runs/tweets/bench_{cuda,cpu}.csv      efficiency table
cache/tweets/<tag>.npz                teacher logits + pooled states, aux_irony.npz, meta.json
data/tweets/report.json               de-duplication counts for the paper
```

## Day-1 checklist mapping

| Plan item | Command |
|---|---|
| Rebuild splits, removal counts, leakage assertions | `prepare_tweets` (done 10 Sep; see `report.json`) |
| Explain the 0.94 vs 0.86 validation gap | compare `val_macro_f1` in `history.csv` with `results.json` test on the clean split |
| Corrected latency benchmark | `bench` |
| Teacher adaptation script | `train_teacher` |
| Fine-tune-only baselines | `train_student --mode ft` |
