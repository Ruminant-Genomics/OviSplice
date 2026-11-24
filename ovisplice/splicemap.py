# -*- coding: utf-8 -*-
"""
Distance-free Ruminant SpliceMap module (AbSplice paradigm, zero physical-distance features).

Features extracted from the nearest active empirical splice site within +-500 bp of each
candidate variant in target tissue t:
  1. has_active_500_t : binary indicator of an empirically active splice site within +-500 bp
  2. psi_ref_t        : empirical reference splice-site usage in target tissue t
  3. psi_het_t        : 4 * psi_ref_t * (1 - psi_ref_t), capturing intermediate splicing susceptibility
  4. psi_spec_t       : |psi_ref_t - psi_pan|, capturing tissue-specific baseline divergence
"""
from __future__ import annotations

import numpy as np


def build_splicemap_index(
    cid_arr: np.ndarray,
    pos_arr: np.ndarray,
    strand_arr: np.ndarray,
    u_mat: np.ndarray,
    valid_mat: np.ndarray,
) -> dict:
    """Build sorted per-(chromosome, strand) empirical tissue usage lookup index."""
    idx = {}
    for c in np.unique(cid_arr):
        for st in (-1, 1):
            m = (cid_arr == c) & (strand_arr == st)
            if not np.any(m):
                continue
            ps = pos_arr[m]
            o = np.argsort(ps, kind="mergesort")
            ms = np.flatnonzero(m)[o]
            idx[(int(c), int(st))] = (pos_arr[ms], u_mat[ms], valid_mat[ms])
    return idx


def query_pure_splicemap(
    sm_idx: dict,
    cids: np.ndarray,
    poss: np.ndarray,
    strands: np.ndarray,
    tids: np.ndarray,
    pan_col: int | None = 0,
) -> np.ndarray:
    """Query the 4 strictly distance-free empirical SpliceMap tissue priors."""
    n = len(cids)
    feats = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        ent = sm_idx.get((int(cids[i]), int(strands[i])))
        ti = int(tids[i])
        if ent is None or ti < 0:
            continue
        ps, um, vm = ent
        pv = int(poss[i])
        j0 = int(np.searchsorted(ps, pv - 500, side="left"))
        j1 = int(np.searchsorted(ps, pv + 500, side="right"))
        if j1 <= j0:
            continue
        act = vm[j0:j1, ti] > 0
        if not np.any(act):
            continue
        ps_a = ps[j0:j1][act]
        u_a = um[j0:j1, ti][act]
        k = int(np.argmin(np.abs(ps_a - pv)))
        psi_t = float(np.clip(u_a[k], 0.0, 1.0))
        if pan_col is not None:
            psi_p = float(np.clip(um[j0:j1, pan_col][act][k], 0.0, 1.0))
        else:
            psi_p = float(np.nanmean(np.clip(um[j0:j1][act][k], 0.0, 1.0)))
        feats[i, 0] = 1.0
        feats[i, 1] = psi_t
        feats[i, 2] = 4.0 * psi_t * (1.0 - psi_t)
        feats[i, 3] = abs(psi_t - psi_p)
    return feats


def fit_cv_splicemap(
    dna_scores: np.ndarray | None,
    sm_feats: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c_reg: float = 0.5,
) -> np.ndarray:
    """
    Evaluate SpliceMap-ONLY (when dna_scores is None) or DNA + Ruminant SpliceMap
    under strict leave-chromosome-fold-out L2 logistic regression.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if dna_scores is None:
        X = sm_feats.astype(np.float64)
    else:
        xd = np.log1p(100.0 * np.clip(np.nan_to_num(dna_scores, nan=0.0), 0.0, 2.0))[:, None]
        X = np.hstack([xd, sm_feats, xd * sm_feats[:, 1:3]]).astype(np.float64)

    oof = np.zeros(len(labels), dtype=np.float64)
    for f in np.unique(folds):
        tr, te = (folds != f), (folds == f)
        sc = StandardScaler()
        clf = LogisticRegression(C=c_reg, max_iter=500, class_weight="balanced")
        clf.fit(sc.fit_transform(X[tr]), labels[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof
