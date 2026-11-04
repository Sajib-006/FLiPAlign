"""Puts the code folders of this repository on the Python import path.

Every script starts by importing this file, so it can be run from the repository root as
    python3 <folder>/<script>.py
Data and result paths in the scripts (data/, results/, landmarks/) are relative to the repository root, so run the
scripts from there.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ("flipalign", "stacklabel", "runs_placement", "runs_stacks", "runs_baselines", "checks")
for _d in reversed(FOLDERS):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(1, _p)
