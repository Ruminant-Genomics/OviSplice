#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s04_distance_free_ruminant_splicemap.py
=======================================
Evaluates the strictly distance-free `Ruminant SpliceMap` framework (AbSplice paradigm, zero
physical-distance proxies) under 5-fold leave-chromosome-out cross-validation across both
paralog-free SheepGTEx sQTLs (`N1` & `N2`) and zero-shot 28-tissue CattleGTEx sQTLs (`N1` & `N2`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice import (
    HEAD_NAMES,
    build_splicemap_index,
    fit_cv_splicemap,
    paired_delong_test,
    query_pure_splicemap,
    tie_invariant_auroc_auprc,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate distance-free Ruminant SpliceMap on sheep and cattle sQTL benchmarks")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Directory containing empirical splice-site atlases and benchmark tables")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results", help="Output directory for SpliceMap ablation tables")
    ap.add_argument("--species", choices=["both", "sheep", "cattle"], default="both")
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, pt_rows = [], []
    oof_save = {}

    if args.species in ("both", "sheep"):
        sz = np.load(args.data_dir / "01_labels" / "splice_sites_15heads.npz", allow_pickle=False)
        sh_sm = build_splicemap_index(sz["chrom_id"], sz["pos"], sz["strand"], sz["U_full"], sz["mask_full"])
        sz.close()

        df_sh = pd.read_csv(args.data_dir / "04_benchmark" / "oof_scores_master_complete.tsv.gz", sep="\t")
        df_sh_pf = df_sh[df_sh["paralog_in_fold"] == 0].copy().reset_index(drop=True)
        pan_p = args.data_dir / "04_benchmark" / "sheep_pf_pan_augmented.npz"
        if pan_p.is_file():
            pan_z = np.load(pan_p, allow_pickle=True)
            df_sh_pf["Ovi_pan_500"] = pan_z["ds_pan_500"].astype(np.float32)
            pan_z.close()

        sh_feats = query_pure_splicemap(
            sh_sm,
            df_sh_pf["chrom_id"].to_numpy(),
            df_sh_pf["POS"].to_numpy(),
            df_sh_pf["strand"].to_numpy(),
            df_sh_pf["head_idx"].to_numpy(),
            pan_col=0,
        )

        for neg_set in ("N1", "N2"):
            m = df_sh_pf["set_type"].isin(["POS", neg_set]).to_numpy()
            sub = df_sh_pf[m].reset_index(drop=True)
            y = sub["label"].to_numpy(dtype=np.int32)
            fl = sub["fold"].to_numpy(dtype=np.int32)
            sm = sh_feats[m]
            sm_only = fit_cv_splicemap(None, sm, y, fl)
            au_sm, ap_sm = tie_invariant_auroc_auprc(y, sm_only)
            all_rows.append({
                "species": "Sheep_PF", "neg_set": neg_set, "model": "SpliceMap-ONLY (no dist)",
                "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                "dna_auroc": round(au_sm, 4), "dna_auprc": round(ap_sm, 4),
                "splicemap_auroc": round(au_sm, 4), "splicemap_auprc": round(ap_sm, 4),
                "delta_auroc": 0.0, "delta_auprc": 0.0, "p_vs_dna": 1.0, "p_vs_splicemap_only": 1.0,
            })

            cand_cols = [
                ("OviSplice (matched)", "OviSplice_5fold_ens2_ds_usage_matched_500"),
                ("OviSplice (PanTissue)", "Ovi_pan_500"),
                ("Human Pangolin (max4)", "HP_ds_usage_max4_500"),
                ("SpliceAI (unmasked)", "SAI_unmasked_ds_500"),
            ]
            for m_lbl, col in cand_cols:
                if col not in sub.columns:
                    continue
                raw_s = sub[col].to_numpy(dtype=np.float64)
                fus_s = fit_cv_splicemap(raw_s, sm, y, fl)
                if m_lbl == "OviSplice (matched)":
                    oof_save[f"sheep_{neg_set}_y"] = y
                    oof_save[f"sheep_{neg_set}_ovi_dna"] = raw_s.astype(np.float32)
                    oof_save[f"sheep_{neg_set}_ovi_splicemap"] = fus_s.astype(np.float32)
                    oof_save[f"sheep_{neg_set}_sm_only"] = sm_only.astype(np.float32)
                au0, ap0 = tie_invariant_auroc_auprc(y, raw_s)
                au1, ap1 = tie_invariant_auroc_auprc(y, fus_s)
                p_dna = float(paired_delong_test(y, fus_s, raw_s)["p"])
                p_sm = float(paired_delong_test(y, fus_s, sm_only)["p"])
                all_rows.append({
                    "species": "Sheep_PF", "neg_set": neg_set, "model": m_lbl,
                    "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                    "dna_auroc": round(au0, 4), "dna_auprc": round(ap0, 4),
                    "splicemap_auroc": round(au1, 4), "splicemap_auprc": round(ap1, 4),
                    "delta_auroc": round(au1 - au0, 4), "delta_auprc": round(ap1 - ap0, 4),
                    "p_vs_dna": p_dna, "p_vs_splicemap_only": p_sm,
                })
                for hi in sorted(sub["head_idx"].unique()):
                    hm = sub["head_idx"].to_numpy() == hi
                    if y[hm].sum() >= 15 and (y[hm] == 0).sum() >= 15:
                        ta0, tp0 = tie_invariant_auroc_auprc(y[hm], raw_s[hm])
                        ta1, tp1 = tie_invariant_auroc_auprc(y[hm], fus_s[hm])
                        pt_rows.append({
                            "species": "Sheep_PF", "neg_set": neg_set, "model": m_lbl,
                            "organ": HEAD_NAMES[int(hi)], "n_pos": int(y[hm].sum()),
                            "dna_auroc": round(ta0, 4), "dna_auprc": round(tp0, 4),
                            "splicemap_auroc": round(ta1, 4), "splicemap_auprc": round(tp1, 4),
                        })

    if args.species in ("both", "cattle"):
        cz = np.load(args.data_dir / "05_crossspecies" / "cattle_12tissue_empirical_sites.npz", allow_pickle=False)
        cat_sm = build_splicemap_index(
            cz["chrom_id"], cz["pos"], cz["strand"],
            cz["U_full"].astype(np.float32), (cz["nge5_full"] >= 5).astype(np.uint8),
        )
        cz.close()

        df_cat = pd.read_csv(args.data_dir / "05_crossspecies" / "cattle_multitissue_master_scores.tsv.gz", sep="\t", low_memory=False)
        cat_head_to_emp = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 12: 9}

        def map_cat_emp_col(r):
            t = str(r.get("tissue", ""))
            if "Mammary" in t or "Milk" in t:
                return 11
            if "Embryo" in t or "Conceptus" in t or "Placenta" in t or "Cotyledon" in t:
                return 10
            return cat_head_to_emp.get(int(r["head_idx"]), 0)

        cids = np.array([30 if str(x).replace("chr", "") == "X" else int(str(x).replace("chr", "")) for x in df_cat["CHR"]], dtype=np.int32)
        tids = np.array([map_cat_emp_col(r) for r in df_cat[["tissue", "head_idx"]].to_dict("records")], dtype=np.int32)
        c_folds = (cids % 5).astype(np.int32)
        cat_feats = query_pure_splicemap(cat_sm, cids, df_cat["POS"].to_numpy(), df_cat["strand"].to_numpy(), tids, pan_col=None)

        for neg_set in ("N1_cattle", "N2_cattle"):
            m = df_cat["set_type"].isin(["POS_strict", neg_set]).to_numpy()
            sub = df_cat[m].copy()
            keep = ~sub.duplicated(subset=["set_type", "ukey"], keep="first").to_numpy()
            sub = sub.iloc[keep].reset_index(drop=True)
            y = (sub["set_type"] == "POS_strict").to_numpy(dtype=np.int32)
            fl = c_folds[m][keep]
            sm = cat_feats[m][keep]
            sm_only = fit_cv_splicemap(None, sm, y, fl)
            au_sm, ap_sm = tie_invariant_auroc_auprc(y, sm_only)
            all_rows.append({
                "species": "Cattle_28Tis", "neg_set": neg_set, "model": "SpliceMap-ONLY (no dist)",
                "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                "dna_auroc": round(au_sm, 4), "dna_auprc": round(ap_sm, 4),
                "splicemap_auroc": round(au_sm, 4), "splicemap_auprc": round(ap_sm, 4),
                "delta_auroc": 0.0, "delta_auprc": 0.0, "p_vs_dna": 1.0, "p_vs_splicemap_only": 1.0,
            })

            for m_lbl, col in (
                ("OviSplice (matched)", "Ovi_full_ds_usage_matched_500"),
                ("OviSplice (PanTissue)", "Ovi_full_ds_usage_pan_500"),
                ("Human Pangolin (max4)", "HP_ds_usage_max4_500"),
                ("SpliceAI (unmasked)", "SAI_unmasked_ds_500"),
            ):
                if col not in sub.columns:
                    continue
                raw_s = sub[col].to_numpy(dtype=np.float64)
                fus_s = fit_cv_splicemap(raw_s, sm, y, fl)
                if m_lbl == "OviSplice (matched)":
                    oof_save[f"cattle_{neg_set}_y"] = y
                    oof_save[f"cattle_{neg_set}_ovi_dna"] = raw_s.astype(np.float32)
                    oof_save[f"cattle_{neg_set}_ovi_splicemap"] = fus_s.astype(np.float32)
                    oof_save[f"cattle_{neg_set}_sm_only"] = sm_only.astype(np.float32)
                au0, ap0 = tie_invariant_auroc_auprc(y, raw_s)
                au1, ap1 = tie_invariant_auroc_auprc(y, fus_s)
                p_dna = float(paired_delong_test(y, fus_s, raw_s)["p"])
                p_sm = float(paired_delong_test(y, fus_s, sm_only)["p"])
                all_rows.append({
                    "species": "Cattle_28Tis", "neg_set": neg_set, "model": m_lbl,
                    "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                    "dna_auroc": round(au0, 4), "dna_auprc": round(ap0, 4),
                    "splicemap_auroc": round(au1, 4), "splicemap_auprc": round(ap1, 4),
                    "delta_auroc": round(au1 - au0, 4), "delta_auprc": round(ap1 - ap0, 4),
                    "p_vs_dna": p_dna, "p_vs_splicemap_only": p_sm,
                })
                org_col = "organ_system" if "organ_system" in sub.columns else "tissue"
                for org_v in sorted(sub[org_col].dropna().unique()):
                    om = sub[org_col].to_numpy() == org_v
                    if y[om].sum() >= 15 and (y[om] == 0).sum() >= 15:
                        ta0, tp0 = tie_invariant_auroc_auprc(y[om], raw_s[om])
                        ta1, tp1 = tie_invariant_auroc_auprc(y[om], fus_s[om])
                        pt_rows.append({
                            "species": "Cattle_28Tis", "neg_set": neg_set, "model": m_lbl,
                            "organ": str(org_v), "n_pos": int(y[om].sum()),
                            "dna_auroc": round(ta0, 4), "dna_auprc": round(tp0, 4),
                            "splicemap_auroc": round(ta1, 4), "splicemap_auprc": round(tp1, 4),
                        })

    df_all = pd.DataFrame(all_rows)
    df_pt = pd.DataFrame(pt_rows)
    df_all.to_csv(args.out_dir / "ruminant_splicemap_pure_ablation.tsv", sep="\t", index=False)
    df_pt.to_csv(args.out_dir / "ruminant_splicemap_pure_per_organ.tsv", sep="\t", index=False)
    np.savez_compressed(args.out_dir / "ruminant_splicemap_pure_oof.npz", **oof_save)
    print(f"[s04] Completed Ruminant SpliceMap evaluation in {time.time() - t0:.1f}s.", flush=True)
    print(df_all.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
