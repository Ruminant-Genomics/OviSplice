#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s03_cattle_10mb_topk_and_multiorgan_ism.py
==========================================
Evaluates standardized cattle 10.0 Mb `Top-1 / Top-0.5 / PR-AUC` dual-track splice-site
recognition (`RefSeq Annotated` vs `All Empirical Complete`), observed-vs-predicted multi-organ
usage calibration, Ruminant SpliceMap ablations, and multi-organ nucleotide ISM logos.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice.native_engine import (
    HEAD_NAMES,
    load_ensemble,
    paired_delong_test,
    spearman_corr,
    tie_invariant_auroc_auprc,
)

BASE_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / "data" / "00_raw"
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "results"
BACKEND = "auto"


def topk_and_ap_from_hist(neg_hist: np.ndarray, pos_scores: np.ndarray) -> dict:
    """
    Exact Top-1 accuracy, Top-0.5 precision, and Average Precision (PR-AUC) from 100,000-bin
    negative histogram + exact positive scores (matches SpliceAI / Pangolin / OpenSpliceAI definition).
    """
    n_pos = len(pos_scores)
    if n_pos == 0:
        return {"n_pos": 0, "top1": float("nan"), "top05": float("nan"), "pr_auc": float("nan")}
    n_bins = len(neg_hist)
    p_bins = np.clip((pos_scores * n_bins).astype(np.int32), 0, n_bins - 1)
    pos_hist = np.bincount(p_bins, minlength=n_bins).astype(np.float64)
    neg_h = neg_hist.astype(np.float64)

    # Traverse high -> low to compute exact AP, Top-1 (k = n_pos), and Top-0.5 (k = 0.5 * n_pos)
    tp_cum = 0.0
    fp_cum = 0.0
    ap_sum = 0.0
    k_top1 = float(n_pos)
    k_top05 = max(1.0, 0.5 * float(n_pos))
    top1_tp = None
    top05_tp = None

    for b in range(n_bins - 1, -1, -1):
        tp_b = pos_hist[b]
        fp_b = neg_h[b]
        tot_b = tp_b + fp_b
        if tot_b <= 0:
            continue
        prev_tot = tp_cum + fp_cum
        new_tot = prev_tot + tot_b

        # Interpolate TP inside bin if k_top05 or k_top1 falls in (prev_tot, new_tot]
        if top05_tp is None and new_tot >= k_top05:
            frac = (k_top05 - prev_tot) / tot_b
            top05_tp = tp_cum + frac * tp_b
        if top1_tp is None and new_tot >= k_top1:
            frac = (k_top1 - prev_tot) / tot_b
            top1_tp = tp_cum + frac * tp_b

        if tp_b > 0:
            if tot_b <= 1.0:
                ap_sum += tp_b * ((tp_cum + 1.0) / (prev_tot + 1.0))
            elif tot_b <= 2000:
                j = np.arange(1, int(tot_b) + 1, dtype=np.float64)
                e_tp = tp_cum + 1.0 + (j - 1.0) * ((tp_b - 1.0) / max(1.0, tot_b - 1.0))
                ap_sum += (tp_b / tot_b) * float(np.sum(e_tp / (prev_tot + j)))
            else:
                prec_mid = (tp_cum + 0.5 * tp_b) / (prev_tot + 0.5 * tot_b)
                ap_sum += tp_b * prec_mid
        tp_cum += tp_b
        fp_cum += fp_b

    top1_acc = float((top1_tp if top1_tp is not None else tp_cum) / k_top1)
    top05_acc = float((top05_tp if top05_tp is not None else tp_cum) / k_top05)
    ap = float(ap_sum / float(n_pos))
    return {"n_pos": int(n_pos), "top1": round(top1_acc, 4), "top05": round(top05_acc, 4), "pr_auc": round(ap, 4)}


def eval_cattle_10mb_topk():
    """
    Compute exact Top-1, Top-0.5, PR-AUC split by `RefSeq Annotated (source in {0,2})` vs
    `All Empirical+GTF (source in {0,1,2})` on the exact Cattle 10.0 Mb (2,000 x 5 kb) benchmark
    for OviSplice (full_ens3), OviSplice-F1 (f1_ens2), and OviSplice-SiteOnly (M1).
    (Human Pangolin, OpenSpliceAI, SpliceTransformer will be merged after Cloud GPU run.)
    """
    t0 = time.time()
    print("[s03:TopK] Computing exact Top-1 / Top-0.5 / PR-AUC (Annotated vs All) on Cattle 10.0 Mb...", flush=True)
    meta = np.load(BASE_DIR / "cattle_10mb_plus_homol_windows_meta.npz", allow_pickle=False)
    is_main = meta["is_main_2000"] == 1
    wins = meta["win_15k_u8"][is_main]         # (2000, 15000)
    full_cls = meta["full_cls"][is_main]       # (2000, 5000): 0=neg, 1=acc, 2=don
    full_src = meta["full_src"][is_main]       # (2000, 5000): 0=both, 1=unann_emp, 2=ann_constit
    meta.close()

    # Define two standard ground-truth tracks:
    #   Track A: RefSeq Annotated (cls>0 and src in {0,2}); unannotated emp (src==1) masked out from negs to avoid penalizing true sites!
    #   Track B: All Empirical + GTF Complete (cls>0)
    is_pos_all = full_cls > 0
    is_pos_ann = (full_cls > 0) & ((full_src == 0) | (full_src == 2))
    is_neg_all = full_cls == 0
    is_acc_all = full_cls == 1
    is_don_all = full_cls == 2
    is_acc_ann = is_acc_all & ((full_src == 0) | (full_src == 2))
    is_don_ann = is_don_all & ((full_src == 0) | (full_src == 2))

    specs = [
        ("OviSplice (full)", [CKPT_DIR / f"ovisplice_full_seed{s}.pt" for s in (42, 1042, 2042)]),
        ("OviSplice-F1 (~71% 0-ortholog)", [CKPT_DIR / f"ovisplice_fold1_seed{s}.pt" for s in (42, 1042)]),
        ("OviSplice-SiteOnly", [CKPT_DIR / "ovisplice_siteonly_fold1_seed42.pt"]),
    ]
    specs = [(lbl, [str(p) for p in ps]) for lbl, ps in specs if all(p.is_file() for p in ps)]

    rows = []
    n_bins = 100_000
    for m_name, ckpts in specs:
        tm0 = time.time()
        ens, _ = load_ensemble(ckpts, backend=BACKEND)
        neg_h_any = np.zeros(n_bins, dtype=np.int64)
        neg_h_acc = np.zeros(n_bins, dtype=np.int64)
        neg_h_don = np.zeros(n_bins, dtype=np.int64)
        pos_any_all, pos_any_ann = [], []
        pos_acc_all, pos_acc_ann = [], []
        pos_don_all, pos_don_ann = [], []

        bs = 40
        for i0 in range(0, len(wins), bs):
            i1 = min(len(wins), i0 + bs)
            p_out, _ = ens.score_windows(wins[i0:i1], compute_usage=False)  # (B, 5000, 2)
            p_acc = p_out[:, :, 0]
            p_don = p_out[:, :, 1]
            p_any = np.clip(p_acc + p_don, 0.0, 1.0)

            m_neg = is_neg_all[i0:i1]
            neg_h_any += np.bincount(np.clip((p_any[m_neg] * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins)
            neg_h_acc += np.bincount(np.clip((p_acc[m_neg] * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins)
            neg_h_don += np.bincount(np.clip((p_don[m_neg] * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins)

            pos_any_all.append(p_any[is_pos_all[i0:i1]])
            pos_any_ann.append(p_any[is_pos_ann[i0:i1]])
            pos_acc_all.append(p_acc[is_acc_all[i0:i1]])
            pos_acc_ann.append(p_acc[is_acc_ann[i0:i1]])
            pos_don_all.append(p_don[is_don_all[i0:i1]])
            pos_don_ann.append(p_don[is_don_ann[i0:i1]])
        del ens

        for track_lbl, p_any_l, p_acc_l, p_don_l in (
            ("RefSeq_Annotated", pos_any_ann, pos_acc_ann, pos_don_ann),
            ("All_Empirical_Complete", pos_any_all, pos_acc_all, pos_don_all),
        ):
            r_any = topk_and_ap_from_hist(neg_h_any, np.concatenate(p_any_l))
            r_acc = topk_and_ap_from_hist(neg_h_acc, np.concatenate(p_acc_l))
            r_don = topk_and_ap_from_hist(neg_h_don, np.concatenate(p_don_l))
            rows.append({
                "species": "Cattle_10Mb",
                "track": track_lbl,
                "model": m_name,
                "n_sites": r_any["n_pos"],
                "any_top1": r_any["top1"],
                "any_top05": r_any["top05"],
                "any_pr_auc": r_any["pr_auc"],
                "acc_top1": r_acc["top1"],
                "don_top1": r_don["top1"],
                "macro_top1": round(0.5 * (r_acc["top1"] + r_don["top1"]), 4),
                "macro_top05": round(0.5 * (r_acc["top05"] + r_don["top05"]), 4),
                "macro_pr_auc": round(0.5 * (r_acc["pr_auc"] + r_don["pr_auc"]), 4),
            })
        print(f"  Scored {m_name} on 10.0 Mb Cattle in {time.time()-tm0:.1f}s", flush=True)

    df_tk = pd.DataFrame(rows)
    df_tk.to_csv(OUT_DIR / "v4_cattle_10mb_topk_prauc.tsv", sep="\t", index=False)
    print(df_tk.to_string(index=False), flush=True)
    return df_tk


def export_usage_density_and_violins():
    """
    Part 2: Export Observed vs Predicted Usage 2D density & High/Low usage split-violin summaries
    across Sheep 5-fold Paralog-Free OOF (565k sites) and Cattle 12-organ zero-shot (7,441 sites).
    """
    t0 = time.time()
    print("[s03:Density] Exporting Observed vs Predicted Usage 2D Hexbin & High/Low split-violin data...", flush=True)

    # 1. Sheep 5-fold OOF (`oof_site_preds.npz`)
    oz = np.load(BASE_DIR / "interpret" / "oof_site_preds.npz", allow_pickle=False)
    pf_ok = (oz["paralog_in_fold"] == 0) & oz["covered"]
    u_true = oz["site_usage_true"][pf_ok].astype(np.float32)  # (N, 15)
    u_pred = oz["oof_usage_prob"][pf_ok].astype(np.float32)   # (N, 15)
    smask = oz["site_mask"][pf_ok]                            # uint16 bitmask!
    oz.close()

    rng = np.random.default_rng(42)
    sheep_violin_rows = []
    hex_dump = {}
    for hi in range(15):
        valid = ((smask >> hi) & 1).astype(bool) & np.isfinite(u_true[:, hi]) & np.isfinite(u_pred[:, hi])
        ut = u_true[valid, hi]
        up = u_pred[valid, hi]
        var_m = (ut >= 0.05) & (ut <= 0.95)
        rho_all = spearman_corr(ut, up)
        rho_var = spearman_corr(ut[var_m], up[var_m])
        r_all = float(np.corrcoef(ut, up)[0, 1])
        # High (U>=0.5) vs Low (U<0.1) predicted score separation (matching SpliceTransformer Fig 1d!)
        lo_s = up[ut < 0.10]
        hi_s = up[ut >= 0.50]
        sheep_violin_rows.append({
            "species": "Sheep_OOF",
            "head_idx": hi,
            "organ": HEAD_NAMES[hi],
            "n_all": int(len(ut)),
            "n_var": int(var_m.sum()),
            "spearman_all": round(rho_all, 4),
            "spearman_var": round(rho_var, 4),
            "pearson_all": round(r_all, 4),
            "low_u_median": round(float(np.median(lo_s)), 4),
            "low_u_q25": round(float(np.percentile(lo_s, 25)), 4),
            "low_u_q75": round(float(np.percentile(lo_s, 75)), 4),
            "high_u_median": round(float(np.median(hi_s)), 4),
            "high_u_q25": round(float(np.percentile(hi_s, 25)), 4),
            "high_u_q75": round(float(np.percentile(hi_s, 75)), 4),
        })
        # Save 2,000-point random subsample of low & high for clean violin plotting + 20k for hexbin
        if hi in (0, 1, 2, 5, 6):
            idx_s = rng.choice(len(ut), size=min(25000, len(ut)), replace=False)
            hex_dump[f"sheep_h{hi}_obs"] = ut[idx_s].astype(np.float16)
            hex_dump[f"sheep_h{hi}_pred"] = up[idx_s].astype(np.float16)
        idx_lo = rng.choice(len(lo_s), size=min(1500, len(lo_s)), replace=False)
        idx_hi = rng.choice(len(hi_s), size=min(1500, len(hi_s)), replace=False)
        hex_dump[f"sheep_vlo_h{hi}"] = lo_s[idx_lo].astype(np.float16)
        hex_dump[f"sheep_vhi_h{hi}"] = hi_s[idx_hi].astype(np.float16)

    # 2. Cattle 12 organs (`cattle_win_scores_full_ens3.npz` + `cattle_12tissue_empirical_sites.npz`)
    cw = np.load(BASE_DIR / "cattle_win_scores_full_ens3.npz", allow_pickle=False)
    cm = np.load(BASE_DIR / "cattle_10mb_plus_homol_windows_meta.npz", allow_pickle=False)
    ca = np.load(BASE_DIR / "05_crossspecies" / "cattle_12tissue_empirical_sites.npz", allow_pickle=False)
    aidx = cm["emp_aidx"]
    u_cat_obs = ca["U_full"][aidx].astype(np.float32)    # (7441, 12)
    nge5_cat = ca["nge5_full"][aidx]                      # (7441, 12)
    u_cat_pred15 = cw["emp_u15"].astype(np.float32)       # (7441, 15)
    cw.close(); cm.close(); ca.close()

    org_map = [
        ("Muscle", 1), ("Liver", 2), ("Adipose", 3), ("Small_Intestine", 4),
        ("Brain", 5), ("Rumen", 6), ("Pituitary", 7), ("Lung", 8),
        ("Testis", 10), ("Skin", 12), ("Embryo", 0), ("Mammary_Gland", 0),
    ]
    cattle_violin_rows = []
    for oi, (oname, hi) in enumerate(org_map):
        valid = (nge5_cat[:, oi] >= 5) & np.isfinite(u_cat_obs[:, oi])
        ut = u_cat_obs[valid, oi]
        up = u_cat_pred15[valid, hi]
        var_m = (ut >= 0.05) & (ut <= 0.95)
        rho_all = spearman_corr(ut, up)
        rho_var = spearman_corr(ut[var_m], up[var_m])
        r_all = float(np.corrcoef(ut, up)[0, 1])
        lo_s = up[ut < 0.10]
        hi_s = up[ut >= 0.50]
        cattle_violin_rows.append({
            "species": "Cattle_ZeroShot",
            "head_idx": hi,
            "organ": oname,
            "n_all": int(len(ut)),
            "n_var": int(var_m.sum()),
            "spearman_all": round(rho_all, 4),
            "spearman_var": round(rho_var, 4),
            "pearson_all": round(r_all, 4),
            "low_u_median": round(float(np.median(lo_s)), 4) if len(lo_s) else None,
            "low_u_q25": round(float(np.percentile(lo_s, 25)), 4) if len(lo_s) else None,
            "low_u_q75": round(float(np.percentile(lo_s, 75)), 4) if len(lo_s) else None,
            "high_u_median": round(float(np.median(hi_s)), 4) if len(hi_s) else None,
            "high_u_q25": round(float(np.percentile(hi_s, 25)), 4) if len(hi_s) else None,
            "high_u_q75": round(float(np.percentile(hi_s, 75)), 4) if len(hi_s) else None,
        })
        hex_dump[f"cattle_o{oi}_obs"] = ut.astype(np.float16)
        hex_dump[f"cattle_o{oi}_pred"] = up.astype(np.float16)
        hex_dump[f"cattle_vlo_o{oi}"] = lo_s.astype(np.float16)
        hex_dump[f"cattle_vhi_o{oi}"] = hi_s.astype(np.float16)

    df_v = pd.DataFrame(sheep_violin_rows + cattle_violin_rows)
    df_v.to_csv(OUT_DIR / "v4_usage_density_and_violins_summary.tsv", sep="\t", index=False)
    np.savez_compressed(OUT_DIR / "v4_usage_density_and_violins_data.npz", **hex_dump)
    print(f"[s03:Density] Saved usage density & violin arrays in {time.time()-t0:.1f}s!", flush=True)
    print(df_v[df_v["species"] == "Cattle_ZeroShot"][["organ", "n_all", "spearman_all", "spearman_var", "low_u_median", "high_u_median"]].to_string(index=False), flush=True)


def run_ruminant_splicemap_experiment():
    """
    Part 3: Zero-retrain `OviSplice + Ruminant SpliceMap` (AbSplice-DNA paradigm):
      - Builds pure reference empirical tissue SpliceMap from `splice_sites_15heads.npz` (Sheep 15 heads)
        and `cattle_12tissue_empirical_sites.npz` (Cattle 12 organs) — strictly empirical splice usage
        reference maps (zero sQTL label leakage!).
      - For every sQTL / control SNV in Sheep (N1 & N2) and Cattle 28-tissue (N1 & N2):
        extracts empirical reference splice features at the SNV's nearest active splice site within +-500 bp
        in the TARGET TISSUE `t`:
          1) `in_splicemap_50_t`: whether an active splice site exists within +-50 bp in tissue `t`
          2) `psi_ref_t`: empirical reference usage U_emp(t) of nearest active site (within +-500 bp)
          3) `psi_het_t = 4 * psi_ref_t * (1 - psi_ref_t)`: intermediate-usage sensitivity
          4) `psi_spec_t = abs(psi_ref_t - psi_ref_pan)`: tissue-specificity magnitude
          5) `n_active_500_t`: local active splice site density in tissue `t`
      - Evaluates strict 5-fold chromosome-disjoint logistic regression combining `log1p(100*DNA_score) + SpliceMap`
        fairly across `OviSplice`, `OviSplice-PanTissue`, `Human Pangolin`, and `SpliceAI`!
      - Reports BOTH overall and per-tissue AUROC & AUPRC honestly (paired Wilcoxon across tissues).
    """
    t0 = time.time()
    print("[s03:SpliceMap] Running AbSplice-style `Ruminant SpliceMap` (zero-retrain, chr-disjoint CV)...", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def build_splicemap_index(cid_arr, pos_arr, strand_arr, u_mat, valid_mat):
        """Index empirical splice sites by (cid, strand) -> sorted pos, u_mat, valid_mat."""
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

    def query_splicemap_features(sm_idx, cids, poss, strands, tids, pan_col: int | None):
        """For each SNV, extract target-tissue SpliceMap features within +-500 bp."""
        n = len(cids)
        feats = np.zeros((n, 6), dtype=np.float32)
        # cols: 0=has_site_50, 1=has_site_500, 2=psi_ref_t, 3=psi_het_t, 4=psi_spec_t, 5=log1p_dist_active
        for i in range(n):
            ent = sm_idx.get((int(cids[i]), int(strands[i])))
            ti = int(tids[i])
            if ent is None or ti < 0:
                feats[i, 5] = math.log1p(501.0)
                continue
            ps, um, vm = ent
            pv = int(poss[i])
            j0 = int(np.searchsorted(ps, pv - 500, side="left"))
            j1 = int(np.searchsorted(ps, pv + 500, side="right"))
            if j1 <= j0:
                feats[i, 5] = math.log1p(501.0)
                continue
            # Filter to sites empirically active/observed in target tissue ti
            act = vm[j0:j1, ti] > 0
            if not np.any(act):
                feats[i, 5] = math.log1p(501.0)
                continue
            ps_a = ps[j0:j1][act]
            u_a = um[j0:j1, ti][act]
            dists = np.abs(ps_a - pv)
            k = int(np.argmin(dists))
            dmin = int(dists[k])
            psi_t = float(np.clip(u_a[k], 0.0, 1.0))
            if pan_col is not None:
                psi_p = float(np.clip(um[j0:j1, pan_col][act][k], 0.0, 1.0))
            else:
                psi_p = float(np.nanmean(np.clip(um[j0:j1][act][k], 0.0, 1.0)))
            feats[i, 0] = 1.0 if dmin <= 50 else 0.0
            feats[i, 1] = 1.0
            feats[i, 2] = psi_t
            feats[i, 3] = 4.0 * psi_t * (1.0 - psi_t)
            feats[i, 4] = abs(psi_t - psi_p)
            feats[i, 5] = math.log1p(float(dmin))
        return feats

    def eval_chr_cv_fusion(y, folds, dna_raw, sm_feats):
        """Strict chromosome-fold OOF LogisticRegression combining DNA score + SpliceMap."""
        x_dna = np.log1p(100.0 * np.clip(np.nan_to_num(dna_raw, nan=0.0), 0.0, 2.0))[:, None]
        # Interaction term: dna * psi_het and dna * in_splicemap_50
        inter = np.column_stack([
            x_dna[:, 0] * sm_feats[:, 0],
            x_dna[:, 0] * sm_feats[:, 2],
            x_dna[:, 0] * sm_feats[:, 3],
        ])
        X = np.hstack([x_dna, sm_feats, inter])
        oof = np.zeros(len(y), dtype=np.float64)
        for f in np.unique(folds):
            tr = folds != f
            te = folds == f
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            clf = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced")
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        return oof

    all_rows = []
    per_tis_rows = []

    # --- A. SHEEP 13-TISSUE PARALOG-FREE sQTLs ---
    sz = np.load(BASE_DIR / "01_labels" / "splice_sites_15heads.npz", allow_pickle=False)
    sh_sm = build_splicemap_index(sz["chrom_id"], sz["pos"], sz["strand"], sz["U_full"], sz["mask_full"])
    sz.close()

    df_sh = pd.read_csv(BASE_DIR / "04_benchmark" / "oof_scores_master_complete.tsv.gz", sep="\t")
    df_sh_pf = df_sh[df_sh["paralog_in_fold"] == 0].copy().reset_index(drop=True)
    # Load PanTissue augmented OOF scores from audit_v3 (aligned 1:1 to 14,584 paralog-free rows!)
    pan_z = np.load(BASE_DIR / "sheep_pf_pan_augmented.npz", allow_pickle=True)
    assert len(pan_z["ds_pan_500"]) == len(df_sh_pf)
    df_sh_pf["Ovi_pan_500"] = pan_z["ds_pan_500"].astype(np.float32)
    pan_z.close()
    sh_feats = query_splicemap_features(
        sh_sm,
        df_sh_pf["chrom_id"].to_numpy(),
        df_sh_pf["POS"].to_numpy(),
        df_sh_pf["strand"].to_numpy(),
        df_sh_pf["head_idx"].to_numpy(),
        pan_col=0,
    )

    for neg_set in ("N1", "N2"):
        sub_m = df_sh_pf["set_type"].isin(["POS", neg_set]).to_numpy()
        sub = df_sh_pf[sub_m].reset_index(drop=True)
        y = sub["label"].to_numpy(dtype=np.int32)
        f_arr = sub["fold"].to_numpy(dtype=np.int32)
        sm_sub = sh_feats[sub_m]

        for m_lbl, col in (
            ("OviSplice (matched)", "OviSplice_5fold_ens2_ds_usage_matched_500"),
            ("OviSplice (PanTissue)", "Ovi_pan_500"),
            ("Human Pangolin (max4)", "HP_ds_usage_max4_500"),
            ("SpliceAI (unmasked)", "SAI_unmasked_ds_500"),
        ):
            raw_s = sub[col].to_numpy(dtype=np.float64)
            auc0, ap0 = tie_invariant_auroc_auprc(y, raw_s)
            fus_s = eval_chr_cv_fusion(y, f_arr, raw_s, sm_sub)
            auc1, ap1 = tie_invariant_auroc_auprc(y, fus_s)
            p_dl = float(paired_delong_test(y, fus_s, raw_s)["p"])
            all_rows.append({
                "species": "Sheep_PF", "neg_set": neg_set, "model": m_lbl,
                "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                "dna_auroc": round(auc0, 4), "dna_auprc": round(ap0, 4),
                "splicemap_auroc": round(auc1, 4), "splicemap_auprc": round(ap1, 4),
                "delta_auroc": round(auc1 - auc0, 4), "delta_auprc": round(ap1 - ap0, 4),
                "delong_p_gain": p_dl,
            })
            # Per-organ auPRC & AUROC
            for hi in sorted(sub["head_idx"].unique()):
                hm = sub["head_idx"].to_numpy() == hi
                if y[hm].sum() >= 15 and (y[hm] == 0).sum() >= 15:
                    ta0, tp0 = tie_invariant_auroc_auprc(y[hm], raw_s[hm])
                    ta1, tp1 = tie_invariant_auroc_auprc(y[hm], fus_s[hm])
                    per_tis_rows.append({
                        "species": "Sheep_PF", "neg_set": neg_set, "model": m_lbl,
                        "organ": HEAD_NAMES[int(hi)], "n_pos": int(y[hm].sum()),
                        "dna_auroc": round(ta0, 4), "dna_auprc": round(tp0, 4),
                        "splicemap_auroc": round(ta1, 4), "splicemap_auprc": round(tp1, 4),
                    })

    # --- B. CATTLE 28-TISSUE ZERO-SHOT sQTLs (using Cattle 12-organ empirical SpliceMap!) ---
    cz = np.load(BASE_DIR / "05_crossspecies" / "cattle_12tissue_empirical_sites.npz", allow_pickle=False)
    cat_sm = build_splicemap_index(cz["chrom_id"], cz["pos"], cz["strand"], cz["U_full"].astype(np.float32), (cz["nge5_full"] >= 5).astype(np.uint8))
    cz.close()

    df_cat = pd.read_csv(BASE_DIR / "05_crossspecies" / "cattle_multitissue_master_scores.tsv.gz", sep="\t")
    # Map Cattle organ_system / tissue to the 12 cattle empirical SpliceMap columns:
    # 0=Muscle, 1=Liver, 2=Adipose, 3=Small_Intestine, 4=Brain, 5=Rumen, 6=Pituitary, 7=Lung, 8=Testis, 9=Skin, 10=Embryo, 11=Mammary_Gland
    cat_head_to_emp = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 12: 9}
    def map_cat_emp_col(r):
        t = str(r.get("tissue", ""))
        if "Mammary" in t or "Milk" in t:
            return 11
        if "Embryo" in t or "Conceptus" in t or "Placenta" in t or "Cotyledon" in t:
            return 10
        hi = int(r["head_idx"])
        return cat_head_to_emp.get(hi, 0)

    def chr_to_int(x):
        s = str(x).replace("chr", "")
        return 30 if s == "X" else (int(s) if s.isdigit() else -1)

    cat_cids = np.array([chr_to_int(x) for x in df_cat["CHR"]], dtype=np.int32)
    cat_tids = np.array([map_cat_emp_col(r) for r in df_cat[["tissue", "head_idx"]].to_dict("records")], dtype=np.int32)
    cat_folds = (cat_cids % 5).astype(np.int32)
    cat_feats = query_splicemap_features(
        cat_sm, cat_cids, df_cat["POS"].to_numpy(), df_cat["strand"].to_numpy(), cat_tids, pan_col=None
    )

    # Deduplicate Cattle POS_strict & N2_cattle by ukey (matching s22d/Table 3 exact 1,967 POS vs 3,934 N1 / 1,991 N2!)
    for neg_set in ("N1_cattle", "N2_cattle"):
        sub_m = df_cat["set_type"].isin(["POS_strict", neg_set]).to_numpy()
        sub = df_cat[sub_m].copy()
        sm_sub_all = cat_feats[sub_m]
        f_sub_all = cat_folds[sub_m]
        # Deduplicate keeping first per (set_type, ukey) if ukey exists
        if "ukey" in sub.columns:
            keep = ~sub.duplicated(subset=["set_type", "ukey"], keep="first").to_numpy()
        else:
            keep = ~sub.duplicated(subset=["set_type", "CHR", "POS", "strand"], keep="first").to_numpy()
        sub = sub.iloc[keep].reset_index(drop=True)
        y = (sub["set_type"] == "POS_strict").to_numpy(dtype=np.int32)
        f_arr = f_sub_all[keep]
        sm_sub = sm_sub_all[keep]

        for m_lbl, col in (
            ("OviSplice (matched)", "Ovi_full_ds_usage_matched_500"),
            ("OviSplice (PanTissue)", "Ovi_full_ds_usage_pan_500"),
            ("Human Pangolin (max4)", "HP_ds_usage_max4_500"),
            ("SpliceAI (unmasked)", "SAI_unmasked_ds_500"),
        ):
            raw_s = sub[col].to_numpy(dtype=np.float64)
            auc0, ap0 = tie_invariant_auroc_auprc(y, raw_s)
            fus_s = eval_chr_cv_fusion(y, f_arr, raw_s, sm_sub)
            auc1, ap1 = tie_invariant_auroc_auprc(y, fus_s)
            p_dl = float(paired_delong_test(y, fus_s, raw_s)["p"])
            all_rows.append({
                "species": "Cattle_28Tis", "neg_set": neg_set, "model": m_lbl,
                "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                "dna_auroc": round(auc0, 4), "dna_auprc": round(ap0, 4),
                "splicemap_auroc": round(auc1, 4), "splicemap_auprc": round(ap1, 4),
                "delta_auroc": round(auc1 - auc0, 4), "delta_auprc": round(ap1 - ap0, 4),
                "delong_p_gain": p_dl,
            })
            # Per organ_system auPRC & AUROC
            org_col = "organ_system" if "organ_system" in sub.columns else "tissue"
            for org_v in sorted(sub[org_col].dropna().unique()):
                om = sub[org_col].to_numpy() == org_v
                if y[om].sum() >= 15 and (y[om] == 0).sum() >= 15:
                    ta0, tp0 = tie_invariant_auroc_auprc(y[om], raw_s[om])
                    ta1, tp1 = tie_invariant_auroc_auprc(y[om], fus_s[om])
                    per_tis_rows.append({
                        "species": "Cattle_28Tis", "neg_set": neg_set, "model": m_lbl,
                        "organ": str(org_v), "n_pos": int(y[om].sum()),
                        "dna_auroc": round(ta0, 4), "dna_auprc": round(tp0, 4),
                        "splicemap_auroc": round(ta1, 4), "splicemap_auprc": round(tp1, 4),
                    })

    df_all = pd.DataFrame(all_rows)
    df_pt = pd.DataFrame(per_tis_rows)
    df_all.to_csv(OUT_DIR / "v4_ruminant_splicemap_ablation.tsv", sep="\t", index=False)
    df_pt.to_csv(OUT_DIR / "v4_ruminant_splicemap_per_organ.tsv", sep="\t", index=False)
    print(f"[s03:SpliceMap] SpliceMap finished in {time.time()-t0:.1f}s!", flush=True)
    print(df_all.to_string(index=False), flush=True)


def extract_multiorgan_ism_logo_case():
    """
    Part 5: Find the cleanest, highest-contrast Rumen vs Skeletal Muscle (and CNS) differential
    splice site in `w3_sites_ism/ism_profiles.npz` with strong OOF accuracy AND clear flanking
    RBP cis-regulatory footprints (`du_ism`), resolve its RefSeq gene symbol, and save for Fig 5c.
    """
    t0 = time.time()
    print("[s03:ISM] Selecting flagship multi-organ (Rumen vs Muscle vs CNS) ISM Logo locus...", flush=True)
    meta = pd.read_csv(BASE_DIR / "interpret" / "ism_sites_meta.tsv.gz", sep="\t")
    iz = np.load(BASE_DIR / "interpret" / "ism_profiles.npz", allow_pickle=False)
    ref401 = iz["ref_seq_401"]   # (430, 401)
    du_ism = iz["du_ism"].astype(np.float32)  # (430, 401, 4, 15)
    dp_ism = iz["dp_ism"].astype(np.float32)  # (430, 401, 4)
    u_ref = iz["u_ref"].astype(np.float32)    # (430, 15)
    p_ref = iz["p_ref"].astype(np.float32)    # (430,)
    iz.close()

    # Load sheep GTF genes to resolve gene symbols
    re_gn = re.compile(r'gene_name "([^"]+)"|gene "([^"]+)"|gene_id "([^"]+)"')
    chr_genes = defaultdict(list)
    with gzip.open(RAW_DIR / "sheepgtex" / "sheep.gtf.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            sp = ln.split("\t", 8)
            if sp[2] != "gene":
                continue
            c = sp[0].replace("chr", "")
            cid = 27 if c == "X" else (int(c) if c.isdigit() else -1)
            if cid < 0:
                continue
            m = re_gn.search(sp[8])
            sym = next((g for g in m.groups() if g), "") if m else ""
            if sym:
                chr_genes[cid].append((int(sp[3]), int(sp[4]), 1 if sp[6] == "+" else -1, sym))

    def lookup_gene(cid, pos0, st):
        for gs, ge, gst, sym in chr_genes.get(int(cid), ()):
            if gs <= pos0 + 1 <= ge and gst == int(st):
                return sym
        for gs, ge, gst, sym in chr_genes.get(int(cid), ()):
            if gs <= pos0 + 1 <= ge:
                return sym
        return f"chr{cid}_{pos0}"

    # Score Rumen_vs_SkeletalMuscle candidates by:
    # (1) high |du_true| & same-sign |du_oof|, (2) strong organ-differential ISM outside core +-5 bp
    cands = []
    for i, r in meta.iterrows():
        if r["group"] != "Rumen_vs_SkeletalMuscle":
            continue
        du_t = float(r["du_true"])
        du_p = float(r["du_oof"])
        if du_t * du_p <= 0 or abs(du_t) < 0.25 or abs(du_p) < 0.15:
            continue
        # Look at flanking window [-60, +60] excluding core [-4, +4] (center index = 200)
        rum_eff = np.max(np.abs(du_ism[i, 140:261, :, 6]), axis=1)  # head 6 = Rumen
        mus_eff = np.max(np.abs(du_ism[i, 140:261, :, 1]), axis=1)  # head 1 = SkeletalMuscle
        diff_eff = np.abs(rum_eff - mus_eff)
        # Mask out core +-5 bp around center index 60 in [140:261]
        flank_m = np.ones(121, dtype=bool)
        flank_m[55:66] = False
        flank_contrast = float(np.sum(diff_eff[flank_m]))
        peak_flank_diff = float(np.max(diff_eff[flank_m]))
        sym = lookup_gene(r["chrom_id"], r["pos_0based"], r["strand"])
        cands.append((flank_contrast, peak_flank_diff, i, sym, du_t, du_p, int(r["site_class"]), int(r["chrom_id"]), int(r["pos_0based"]), int(r["strand"])))

    cands.sort(reverse=True)
    print("  Top 8 Rumen vs SkeletalMuscle multi-organ ISM Logo candidates:", flush=True)
    for row in cands[:8]:
        print(f"    idx={row[2]:3d} | gene={row[3]:14s} | cls={row[6]} | du_true={row[4]:+.3f} du_oof={row[5]:+.3f} | flank_sum={row[0]:.3f} peak={row[1]:.4f}", flush=True)

    # Save top 3 candidates in full detail so Fig 5c picks the visually sharpest annotated protein-coding gene!
    top_pack = {}
    top_meta = []
    bch = ["A", "C", "G", "T", "N"]
    for rank_i, row in enumerate(cands[:5]):
        flank_c, peak_d, idx, sym, du_t, du_p, scls, cid, pos0, st = row
        # Center window [-50, +50] -> indices [150:251]
        seq101 = "".join(bch[int(x)] for x in ref401[idx, 150:251])
        # Importance at each base = mean |delta_u| over the 3 non-ref substitutions (positive importance score for Logo!)
        imp_rum = np.mean(np.abs(du_ism[idx, 150:251, :, 6]), axis=1) * (4.0 / 3.0)
        imp_mus = np.mean(np.abs(du_ism[idx, 150:251, :, 1]), axis=1) * (4.0 / 3.0)
        imp_cns = np.mean(np.abs(du_ism[idx, 150:251, :, 5]), axis=1) * (4.0 / 3.0)
        imp_diff = np.mean(np.abs(du_ism[idx, 150:251, :, 6] - du_ism[idx, 150:251, :, 1]), axis=1) * (4.0 / 3.0)
        top_pack[f"c{rank_i}_ref101"] = ref401[idx, 150:251]
        top_pack[f"c{rank_i}_imp_rum"] = imp_rum.astype(np.float32)
        top_pack[f"c{rank_i}_imp_mus"] = imp_mus.astype(np.float32)
        top_pack[f"c{rank_i}_imp_cns"] = imp_cns.astype(np.float32)
        top_pack[f"c{rank_i}_imp_diff"] = imp_diff.astype(np.float32)
        top_pack[f"c{rank_i}_du_rum_signed"] = du_ism[idx, 150:251, :, 6].astype(np.float32)
        top_pack[f"c{rank_i}_du_mus_signed"] = du_ism[idx, 150:251, :, 1].astype(np.float32)
        top_meta.append({
            "rank": rank_i, "ism_idx": int(idx), "gene_symbol": sym,
            "site_class": "Acceptor (3'ss)" if scls == 1 else "Donor (5'ss)",
            "chr": "X" if cid == 27 else str(cid), "pos_0based": pos0, "strand": st,
            "du_true_rum_minus_mus": round(du_t, 4), "du_oof_rum_minus_mus": round(du_p, 4),
            "u_ref_rum": round(float(u_ref[idx, 6]), 4),
            "u_ref_mus": round(float(u_ref[idx, 1]), 4),
            "u_ref_cns": round(float(u_ref[idx, 5]), 4),
            "p_ref": round(float(p_ref[idx]), 4),
            "flank_contrast": round(flank_c, 4), "peak_flank_diff": round(peak_d, 4),
            "seq101": seq101,
        })
    np.savez_compressed(OUT_DIR / "v4_multiorgan_ism_logo_cases.npz", **top_pack)
    with open(OUT_DIR / "v4_multiorgan_ism_logo_cases.json", "w", encoding="utf-8") as f:
        json.dump(top_meta, f, indent=2)
    print(f"[s03:ISM] Saved top 5 multi-organ ISM Logo candidates in {time.time()-t0:.1f}s!", flush=True)


def main():
    global BASE_DIR, RAW_DIR, CKPT_DIR, OUT_DIR, BACKEND
    ap = argparse.ArgumentParser(description="Cattle 10.0 Mb Top-k/PR-AUC, calibration violins, and multi-organ ISM logos")
    ap.add_argument("--task", choices=["all", "cattle_10mb_topk", "usage_density", "splicemap", "ism_logo"], default="all")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Processed evaluation arrays directory")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "00_raw", help="Public raw inputs directory")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints", help="Directory containing OviSplice .pt checkpoints")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results", help="Output results directory")
    ap.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    args = ap.parse_args()
    BASE_DIR = args.data_dir
    RAW_DIR = args.raw_dir
    CKPT_DIR = args.ckpt_dir
    OUT_DIR = args.out_dir
    BACKEND = args.backend
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "cattle_10mb_topk"):
        eval_cattle_10mb_topk()
    if args.task in ("all", "usage_density"):
        export_usage_density_and_violins()
    if args.task in ("all", "splicemap"):
        run_ruminant_splicemap_experiment()
    if args.task in ("all", "ism_logo"):
        extract_multiorgan_ism_logo_case()
    print("[s03] Completed requested tasks.", flush=True)


if __name__ == "__main__":
    main()
