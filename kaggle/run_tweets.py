"""Backward-compatible alias: the tweet benchmark now runs through run_benchmark.py.

    PYTHONPATH=src python kaggle/run_tweets.py --stage all --raw <cyberbullying_tweets.csv>
"""
import os
import runpy
import sys

if "--dataset" not in sys.argv:
    sys.argv.insert(1, "--dataset")
    sys.argv.insert(2, "tweets")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_benchmark.py"), run_name="__main__")
