"""Shared paths for the 2026-09 re-analysis scripts.

Importing this module chdirs into data/review_2026_09/, where every review
script reads and writes its intermediate files (all untracked, like the rest
of data/), and puts scripts/ on sys.path so the repo's analysis modules
import. Outputs of the pre-fix (timezone-bug) pipeline live in
data/superseded_buggy_tz/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "review_2026_09")
SUPERSEDED = os.path.join(DATA, "superseded_buggy_tz")

os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
