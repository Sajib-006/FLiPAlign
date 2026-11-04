"""Pairs of neighbouring sections in the six stacks, for FLiPAlign placement (2026-10-22).
Pair k (0..41): stack k // 7, A = section (k % 7) + 1, B = section k % 7 of that stack. Positions: Allen reconstruction
(truth). Labels on each side: own 12 clusters -> 6 regions (the rule used for all Allen pairs). A is moved by a random
rotation, flip (50%) and shift (up to 150 units), seed 9100 + k, as if mounted freely.
load_stack_pair(k) returns the tuple of real_fine.load plus (A_true, M, s0) to score and compose the answer."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import os
import numpy as np
from smoothalign import rot, FLIP
from bench_gt import UNIT
from bench_domains import neighbourhood_domains
import stack_jlp as J

_P = None


def load_stack_pair(k):
    global _P
    if _P is None: _P = J.sections()
    p, secs = _P; st, j = divmod(k, 7); stack = secs[st * J.KS:(st + 1) * J.KS]
    SA = J.load_section(p, stack[j + 1], secs.index(stack[j + 1])); SB = J.load_section(p, stack[j], secs.index(stack[j]))
    At, B = SA["xy"] / UNIT, SB["xy"] / UNIT
    la = neighbourhood_domains(At, SA["cl"], K=6, seed=secs.index(stack[j + 1]))
    lb = neighbourhood_domains(B, SB["cl"], K=6, seed=secs.index(stack[j]))
    rng = np.random.default_rng(9100 + k); th = rng.uniform(0, 2 * np.pi); fl = int(rng.integers(2)); fl = 0 if os.environ.get("FLIP_KNOWN") else fl; s0 = rng.uniform(-150, 150, 2)
    M = rot(th) @ (FLIP if fl else np.eye(2))
    A = (At - At.mean(0)) @ M.T + At.mean(0) + s0
    dummy = np.zeros((len(A), 1), np.float32)
    return (f"stack {st} pair {j}", A, la, dummy, B, lb, np.zeros((len(B), 1), np.float32), 1.0), At
