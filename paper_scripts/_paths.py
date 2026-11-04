"""Shared paths for the paper scripts.

RESULTS  root of the saved results (env RESULTS, default: ../results of this repo)
         placement files: RESULTS/placement, stack files: RESULTS/stack, other files: RESULTS/*.json
OUT      where tables and figures are written (env OUT, default: paper_scripts/out)
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.environ.get("RESULTS", os.path.join(HERE, "..", "results")))
PLACE = os.path.join(RES, "placement")
STACK = os.path.join(RES, "stack")
OUT = os.path.abspath(os.environ.get("OUT", os.path.join(HERE, "out")))
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
