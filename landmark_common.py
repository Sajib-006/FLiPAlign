"""Real pairs with clicked landmarks (2026-10-16): data loaders, truth fit and landmark error, shared by the
tool runner and the FLiPAlign evaluation. Frames and units as in the analysis: A = Visium, B = Xenium or Visium HD,
unit 9.44 um; placement P(a) = R(deg) F^flip (a - mean A) + mean B + t. Landmarks: landmark_work/clean/pairN_landmarks_um.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from smoothalign import rot, FLIP

UNIT = 9.44
HOME = os.path.dirname(os.path.abspath(__file__))
NAMES = {1: "breast Rep1", 2: "breast Rep2", 3: "colon P2CRC", 4: "colon P3NAT", 5: "colon P5NAT", 6: "lung HD1", 7: "lung HD2"}


def load_pair(k):
    """-> dict A, la, XA, genes, B, lb, XB  (A, B in units)"""
    if k in (6, 7):
        from lung_load import load_lung
        name, A, la, XA, B, lb, XB, rad = load_lung(k)
        return dict(name=name, A=np.asarray(A, float), la=np.asarray(la), XA=XA, genes=[f"g{i}" for i in range(XA.shape[1])], B=np.asarray(B, float), lb=np.asarray(lb), XB=XB)
    if k in (1, 2):
        import run_compare as RC
        name, A, la, XA, genes, B, lb, XB, rad = list(RC.pairs())[k]
        return dict(name=name, A=np.asarray(A, float), la=np.asarray(la), XA=XA, genes=list(genes), B=np.asarray(B, float), lb=np.asarray(lb), XB=XB)
    import real_fine as RF
    name, A, la, XA, B, lb, XB, rad = RF.load(k)
    return dict(name=name, A=np.asarray(A, float), la=np.asarray(la), XA=XA, genes=[f"g{i}" for i in range(XA.shape[1])], B=np.asarray(B, float), lb=np.asarray(lb), XB=XB)


def landmarks(k, root=None):
    d = pd.read_csv(f"{root or HOME}/landmark_work/clean/pair{k}_landmarks_um.csv")
    return d[["ax_um", "ay_um"]].values / UNIT, d[["bx_um", "by_um"]].values / UNIT


def rigid_fit(LA, LB):
    """proper rotation R and shift tau with LB ~ LA R^T + tau (units)"""
    ca, cb = LA.mean(0), LB.mean(0); U, S, Vt = np.linalg.svd((LA - ca).T @ (LB - cb)); R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca


def truth(k, A, B, root=None):
    LA, LB = landmarks(k, root); R, tau = rigid_fit(LA, LB)
    resid = np.linalg.norm(LA @ R.T + tau - LB, axis=1) * UNIT
    shift_true = R @ A.mean(0) + tau - B.mean(0)               # where A's centre lands, relative to B's centre
    return dict(R=R, tau=tau, shift_true=shift_true, angle_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360),
                floor_um=float(np.median(resid)), n=len(LA))


def lm_error(LA, LB, Abar, Bbar, deg, fl, t):
    """per-landmark error (um) of placement (deg, fl, t) for landmarks LA given the centres of A and B used by the method"""
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    P = (LA - Abar) @ M.T + Bbar + np.asarray(t, float)
    return np.linalg.norm(P - LB, axis=1) * UNIT
