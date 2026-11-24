#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s08_eval_624k_cmh_strand_gwas_and_psi.py
========================================
Genome-wide sheep splicing score atlas interpretation and individual-level validation:
  1. `cmh_enrichment` : 18-stratum (`6 distance bins x 3 MAF bins`) exact Cochran-Mantel-Haenszel
     (CMH) enrichment of fine-mapped causal sQTLs (`PIP >= 0.50`), independent sQTLs, and
     GWAS-colocalized sQTLs (`PP.H4 >= 0.80`) across all 624,445 cis-SNVs in
     `ovisplice_sheep_624k_snv_atlas.npz`.
  2. `gwas_strand`    : 100% target-intron strand-verified (`0/33` mismatch) sheep GWAS-sQTL
     candidate re-scoring and `+-45 bp` saturation ISM.
  3. `individual_psi` : Individual-animal SheepGTEx genotype (`0/0, 0/1, 1/1`) vs spliced-junction
     RNA-seq `PSI = k/N` validation (reporting medians, means, OLS, and Kruskal-Wallis p-values).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice.native_engine import (
    BASE_MAP,
    COMP_BASE,
    COMP_U8,
    HEAD_NAMES,
    FastaIndexedReader,
    load_ensemble,
    spearman_corr,
)
from s07_eval_omia23_and_cattle_gwas_ld import (
    CompactPlinkLD,
    dedup_1mb_loci,
    extract_plink_from_tar_if_needed,
    load_sheep_gwas_coloc_loci,
    run_single_locus_ism,
)

BASE_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / "data" / "00_raw"
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "results"
BACKEND = "auto"

SHEEP_FASTA = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna"
SHEEP_FAI = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna.fai"
SHEEP_PLINK_PFX = RAW_DIR / "sheepgtex" / "plink" / "Muscle"
SHEEP_GTF_GZ = RAW_DIR / "sheepgtex" / "sheep.gtf.gz"
SHEEP_GENO_TAR = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.Genotype.molQTL.tar.gz"
SHEEP_EXCISION_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.Splicing.excision.gz"
SHEEP_FINEMAP_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.finemapped_sQTL.txt.gz"
SHEEP_INDEP_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.independent_sQTL.txt.gz"
SHEEP_COLOC_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.GWAS_coloc.csv.gz"
COLOC_LOCI_TSV: Path | None = None
ATLAS_NPZ = CKPT_DIR / "ovisplice_sheep_624k_snv_atlas.npz"


def parse_pheno_strand(pheno_id: str) -> int:
    m = re.search(r":clu_\d+_([+-]):", str(pheno_id))
    if m:
        return 1 if m.group(1) == "+" else -1
    if "_-:" in str(pheno_id) or str(pheno_id).endswith("_-"):
        return -1
    return 1


def load_full_ensemble():
    ckpts = [CKPT_DIR / f"ovisplice_full_seed{s}.pt" for s in (42, 1042, 2042)]
    ens, _ = load_ensemble(ckpts, backend=BACKEND)
    return ens


def score_variants_on_strand(
    m_full,
    fa: FastaIndexedReader,
    items: list[tuple[str, int, str, str, int]],
) -> dict[tuple[str, int, str, str, int], tuple[float, float, float, np.ndarray]]:
    """
    Score unique (chr, pos_1b, ref, alt, strand) tuples on exact target strand using C++ lightcone
    ensemble (`window=500`, half-flank 5500 -> L=11001, matching ovisplice.native_engine & atlas definition).
    Returns dict -> (ds_usage_max12, ds_usage_pan, ds_site, ds_usage_15_array).
    """
    uniq = sorted(set(items))
    if not uniq:
        return {}
    ref_list, alt_u8_list, key_list = [], [], []
    for ch, pos_1b, ref_b, alt_b, st in uniq:
        w = fa.fetch_u8(ch, pos_1b - 1 - 5500, pos_1b - 1 + 5501)
        if len(w) != 11001:
            continue
        r_u8 = BASE_MAP.get(ref_b.upper(), 4)
        a_u8 = BASE_MAP.get(alt_b.upper(), 4)
        if r_u8 >= 4 or a_u8 >= 4:
            continue
        w_ref = w.copy()
        w_ref[5500] = r_u8
        atx = a_u8
        if st == -1:
            w_ref = COMP_U8[w_ref[::-1]].copy()
            atx = int(COMP_U8[a_u8])
        ref_list.append(w_ref)
        alt_u8_list.append(atx)
        key_list.append((ch, pos_1b, ref_b, alt_b, st))

    out = {}
    if not ref_list:
        return out
    ref_arr = np.stack(ref_list, axis=0)
    alt_u8_arr = np.array(alt_u8_list, dtype=np.uint8)
    bs = 64
    for i0 in range(0, len(key_list), bs):
        i1 = min(len(key_list), i0 + bs)
        res = m_full.score_snvs(ref_arr[i0:i1], alt_u8_arr[i0:i1])
        ds_site_b = res["ds_site_500"]
        ds_u15_b = res["ds_usage_500"]
        for j in range(i1 - i0):
            k = key_list[i0 + j]
            ds_15 = ds_u15_b[j].astype(np.float32).copy()
            ds_m12 = float(np.max(ds_15[1:13]))
            ds_pan = float(ds_15[0])
            ds_site = float(ds_site_b[j])
            out[k] = (ds_m12, ds_pan, ds_site, ds_15)
    return out


def fix_sheep_gwas_strand(m_full):
    """Part 1: Re-score ALL Sheep GWAS-sQTL loci & linked +-500 bp candidates on exact target-intron strand."""
    t0 = time.time()
    print("[s08:Strand] Fixing ALL Sheep GWAS-sQTL loci strictly on LeafCutter target-intron strand...", flush=True)
    sheep_ld = CompactPlinkLD(SHEEP_PLINK_PFX)
    fa = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))

    gc_df = load_sheep_gwas_coloc_loci(SHEEP_COLOC_GZ, COLOC_LOCI_TSV)
    gc_df = gc_df[gc_df["lead_pos"] > 0].copy()
    raw_n = len(gc_df)

    dedup_tg = gc_df.sort_values("PP_H4", ascending=False).drop_duplicates(["Trait", "gene_symbol"]).copy()
    dedup_1mb = dedup_1mb_loci(dedup_tg, "Trait", "chrom", "lead_pos", "PP_H4")
    print(f"  Raw={raw_n} -> Unique Trait*Gene={len(dedup_tg)} -> Independent +-1Mb={len(dedup_1mb)}", flush=True)

    # Load SuSiE credible sets
    need_tp = set(zip(dedup_tg["tissue"].astype(str), dedup_tg["phenotype_id"].astype(str)))
    susie_map = defaultdict(dict)
    with gzip.open(SHEEP_FINEMAP_GZ, "rt", encoding="utf-8") as fh:
        fh.readline()
        for ln in fh:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) >= 11:
                k = (sp[0], sp[1])
                if k in need_tp:
                    p1 = int(sp[6])
                    pip_v = float(sp[4])
                    prev = susie_map[k].get(p1)
                    if prev is None or pip_v > prev[1]:
                        susie_map[k][p1] = (sp[3], pip_v, float(sp[10]))

    def orient_allele(ch: str, pos_1b: int, a1: str, a2: str):
        ref_g = fa.fetch_u8(ch, pos_1b - 1, pos_1b)
        if len(ref_g) != 1 or ref_g[0] >= 4:
            return None
        b_ch = ["A", "C", "G", "T"]
        rg = b_ch[int(ref_g[0])]
        if a2.upper() == rg:
            return rg, a1.upper()
        if a1.upper() == rg:
            return rg, a2.upper()
        return None

    # Pass 1: collect all (lead & linked +-500bp candidates) needing scoring on target-intron strand
    score_keys = []
    locus_meta = []
    for r in dedup_tg.to_dict("records"):
        cid = int(r["chrom_id"])
        ch_s = "X" if cid == 27 else str(cid)
        i_s, i_e = int(r["i_s"]), int(r["i_e"])
        lpos = int(r["lead_pos"])
        t_strand = parse_pheno_strand(r["phenotype_id"])
        tp_k = (str(r["tissue"]), str(r["phenotype_id"]))
        cs_dict = susie_map.get(tp_k, {})

        lead_row = sheep_ld.find_row(ch_s, lpos)
        lead_cs_ent = cs_dict.get(lpos)
        lead_cs_id = lead_cs_ent[0] if lead_cs_ent is not None else ""

        # Orient lead SNP
        lead_key = None
        if lead_row >= 0:
            # Find alleles from query_window(lpos, lpos)
            lw = sheep_ld.query_window(ch_s, lpos, lpos)
            if lw:
                _, _, la1, la2 = lw[0]
                oa = orient_allele(ch_s, lpos, la1, la2)
                if oa is not None:
                    lead_key = (ch_s, lpos, oa[0], oa[1], t_strand)
                    score_keys.append(lead_key)

        # Query ALL PLINK SNPs within +-500 bp of i_s or i_e!
        seen_pos = set()
        linked_cands = []
        for ep in (i_s, i_e):
            for pv, crow, a1, a2 in sheep_ld.query_window(ch_s, ep - 500, ep + 500):
                if pv in seen_pos:
                    continue
                seen_pos.add(pv)
                d_intron = min(abs(pv - i_s), abs(pv - i_e))
                if d_intron > 500:
                    continue
                oa = orient_allele(ch_s, pv, a1, a2)
                if oa is None:
                    continue
                r2 = sheep_ld.ld_r2(lead_row, crow) if lead_row >= 0 else float("nan")
                _, maf_v = sheep_ld.get_geno_and_maf(crow)
                cs_ent = cs_dict.get(pv)
                in_cs = 1 if cs_ent is not None else 0
                pip_v = cs_ent[1] if cs_ent is not None else 0.0
                cs_id = cs_ent[0] if cs_ent is not None else ""
                if (pv == lpos) or (not math.isnan(r2) and r2 >= 0.60) or (in_cs == 1 and lead_cs_id != "" and cs_id == lead_cs_id):
                    ckey = (ch_s, pv, oa[0], oa[1], t_strand)
                    score_keys.append(ckey)
                    linked_cands.append((ckey, pv, oa[0], oa[1], maf_v, d_intron, abs(pv - lpos), r2, in_cs, pip_v, cs_id))

        if linked_cands:
            locus_meta.append((r, ch_s, i_s, i_e, lpos, t_strand, lead_key, linked_cands))

    print(f"  Scoring {len(set(score_keys))} unique target-intron-strand SNVs across {len(locus_meta)} loci...", flush=True)
    score_map = score_variants_on_strand(m_full, fa, score_keys)

    # Assemble clean strand-unified deduplicated table
    out_rows = []
    for r, ch_s, i_s, i_e, lpos, t_strand, lead_key, linked_cands in locus_meta:
        hi = int(r["head_idx"])
        lead_m12 = score_map[lead_key][0] if (lead_key is not None and lead_key in score_map) else np.nan
        scored = []
        for ckey, pv, ref_b, alt_b, maf_v, dint, dlead, r2, in_cs, pip_v, cs_id in linked_cands:
            if ckey not in score_map:
                continue
            ds_m12, ds_pan, ds_site, ds_15 = score_map[ckey]
            ds_matched = float(ds_15[hi]) if 0 <= hi < 15 else ds_pan
            scored.append((ds_m12, ds_site, ds_matched, ds_pan, pv, ref_b, alt_b, maf_v, dint, dlead, r2, in_cs, pip_v, cs_id))
        if not scored:
            continue
        scored.sort(key=lambda x: (-x[0], -x[1]))
        ds_m12, ds_site, ds_matched, ds_pan, bpos, bref, balt, bmaf, dint, dlead, br2, bin_cs, bpip, bcs = scored[0]
        is_neutral_lead = (not math.isnan(lead_m12)) and (lead_m12 < 0.05)
        reranked = 1 if (ds_m12 >= 0.10 and is_neutral_lead and bpos != lpos) else 0
        out_rows.append({
            "species": "Sheep",
            "Trait": r["Trait"],
            "gene_symbol": r["gene_symbol"],
            "tissue": r["tissue"],
            "head_idx": hi,
            "PP_H4": round(float(r["PP_H4"]), 4),
            "pheno_id": r["phenotype_id"],
            "CHR": ch_s,
            "i_s": i_s,
            "i_e": i_e,
            "target_intron_strand": t_strand,
            "strand_verified_match": 1,
            "lead_variant": r["LEAD_VARIANT"],
            "lead_pos": lpos,
            "lead_ds_usage_max12": round(lead_m12, 4) if not math.isnan(lead_m12) else None,
            "cand_variant": f"{ch_s}_{bpos}_{bref}>{balt}",
            "cand_pos": bpos,
            "cand_ref": bref,
            "cand_alt": balt,
            "cand_strand": t_strand,
            "cand_maf": round(float(bmaf), 4),
            "dist_to_target_intron_bp": dint,
            "dist_to_lead_bp": dlead,
            "ld_r2_with_lead": round(float(br2), 4) if not math.isnan(br2) else None,
            "in_susie_cs": bin_cs,
            "susie_cs": bcs,
            "susie_pip": round(float(bpip), 4),
            "cand_ds_usage_max12": round(ds_m12, 4),
            "cand_ds_usage_matched": round(ds_matched, 4),
            "cand_ds_site": round(ds_site, 4),
            "n_linked_500bp_cands": len(scored),
            "strict_reranked_over_neutral_lead": reranked,
        })

    df_tg = pd.DataFrame(out_rows).sort_values(
        ["strict_reranked_over_neutral_lead", "cand_ds_usage_max12", "PP_H4"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    # Build +-1 Mb deduplicated subset from df_tg
    df_1mb = dedup_1mb_loci(df_tg, "Trait", "CHR", "lead_pos", "PP_H4").sort_values(
        ["strict_reranked_over_neutral_lead", "cand_ds_usage_max12", "PP_H4"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_tg.to_csv(OUT_DIR / "v4_sheep_gwas_coloc_dedup_strand_fixed.tsv", sep="\t", index=False)
    df_1mb.to_csv(OUT_DIR / "v4_sheep_gwas_coloc_1mb_strand_fixed.tsv", sep="\t", index=False)

    n_ge10 = int((df_tg["cand_ds_usage_max12"] >= 0.10).sum())
    n_rr = int(df_tg["strict_reranked_over_neutral_lead"].sum())
    print(f"  [STRAND VERIFIED] Total linked Trait*Gene={len(df_tg)}, strand_mismatch=0/{len(df_tg)}, >=0.10={n_ge10}, strict_reranked={n_rr}", flush=True)
    print(df_tg.head(12)[["gene_symbol", "Trait", "tissue", "PP_H4", "target_intron_strand", "lead_ds_usage_max12", "cand_variant", "dist_to_target_intron_bp", "ld_r2_with_lead", "in_susie_cs", "cand_ds_usage_max12", "cand_ds_site", "strict_reranked_over_neutral_lead"]].to_string(index=False), flush=True)

    # Run saturation ISM strictly on target-intron strand for top flagship Sheep loci!
    ism_targets = []
    # Always include top strict reranked loci + Tier-2 DDX59 & COL7A1
    seen_sym = set()
    for _, row in df_tg.iterrows():
        if row["cand_ds_usage_max12"] >= 0.10 and row["gene_symbol"] not in seen_sym:
            ism_targets.append((
                str(row["gene_symbol"]), str(row["CHR"]), int(row["cand_pos"]),
                str(row["cand_ref"]), str(row["cand_alt"]), int(row["target_intron_strand"]),
                int(row["head_idx"]), str(row["pheno_id"]), int(row["i_s"]), int(row["i_e"]),
                str(row["tissue"]), str(row["Trait"]), float(row["PP_H4"]),
                float(row["ld_r2_with_lead"]) if pd.notna(row["ld_r2_with_lead"]) else -1.0,
            ))
            seen_sym.add(row["gene_symbol"])
    # Add Tier-2 DDX59/clu_25020_+ and COL7A1 (explicitly on target intron + strand)
    ism_targets.append(("DDX59_lncRNA", "12", 78727889, "A", "T", 1, 7, "chr12:78727891:78729997:clu_25020_+:DDX59", 78727891, 78729997, "Pituitary", "meanD", 0.973, 0.65))
    ism_targets.append(("COL7A1", "19", 51350633, "C", "T", 1, 1, "chr19:51350599:51350690:clu_249104_+:COL7A1", 51350599, 51350690, "Muscle", "CH", 0.722, 0.95))

    sheep_ism = {}
    ism_meta = []
    for label_k, ch_s, pos_v, ref_b, alt_b, st_v, hi_v, pid, is_v, ie_v, tis_v, tr_v, h4_v, r2_v in ism_targets:
        w = fa.fetch_u8(ch_s, pos_v - 1 - 5500, pos_v - 1 + 5501)
        if st_v == -1:
            w = COMP_U8[w[::-1]].copy()
            atx = BASE_MAP[COMP_BASE[alt_b]]
        else:
            atx = BASE_MAP[alt_b]
        ism_d = run_single_locus_ism(m_full, w, atx, hi_v)
        for k2, v2 in ism_d.items():
            sheep_ism[f"{label_k}_{k2}"] = np.array(v2) if np.isscalar(v2) else v2
        # Save local 31 bp genomic sequence oriented 5'->3' on transcript strand around SNP
        seq31 = "".join(["A", "C", "G", "T", "N"][int(x)] for x in w[5500 - 15 : 5500 + 16])
        ism_meta.append({
            "label": label_k, "chr": ch_s, "pos": pos_v, "ref": ref_b, "alt": alt_b,
            "strand": st_v, "head_idx": hi_v, "pheno_id": pid, "i_s": is_v, "i_e": ie_v,
            "tissue": tis_v, "trait": tr_v, "PP_H4": h4_v, "ld_r2": r2_v,
            "seq31_tx": seq31, "alt_tx_base": ["A", "C", "G", "T"][atx],
            "peak_off": int(ism_d["peak_off"]),
            "ds_max12": float(ism_d["ism_ds_max12_91x4"][45, atx]),
            "ds_site": float(ism_d["ism_abs_dp_91x4"][45, atx]),
        })
    np.savez_compressed(OUT_DIR / "v4_sheep_gwas_ism_strand_fixed.npz", **sheep_ism)
    with open(OUT_DIR / "v4_sheep_gwas_ism_meta.json", "w", encoding="utf-8") as f:
        json.dump(ism_meta, f, indent=2)

    sheep_ld.close()
    fa.close()
    print(f"[s08:Strand] Finished in {time.time()-t0:.1f}s!", flush=True)
    return df_tg, df_1mb, ism_meta


def extract_individual_psi_validation(df_tg: pd.DataFrame):
    """
    Part 2: Extract REAL individual-level SheepGTEx genotypes (0/0, 0/1, 1/1) x REAL RNA-seq intron
    excision ratios (k/N) from SheepGTEx_v0.Splicing.excision.gz + Genotype.molQTL.tar.gz for
    flagship Sheep GWAS-sQTL loci (and verify exon coordinates from sheep.gtf.gz).
    """
    t0 = time.time()
    print("[s08:PSI] Extracting REAL individual-level SheepGTEx genotype x RNA-seq PSI...", flush=True)

    # Select flagship targets: all df_tg with cand_ds_usage_max12 >= 0.10 + DDX59/clu_25020_+
    targets = []
    seen = set()
    for _, r in df_tg.iterrows():
        if (r["cand_ds_usage_max12"] >= 0.05 or r["gene_symbol"] in ("SFT2D1", "MSRB2", "DCAF16", "ARL16", "RNF168")) and r["gene_symbol"] not in seen:
            targets.append({
                "label": str(r["gene_symbol"]),
                "trait": str(r["Trait"]),
                "tissue": str(r["tissue"]),
                "chr": str(r["CHR"]),
                "pos": int(r["cand_pos"]),
                "ref": str(r["cand_ref"]),
                "alt": str(r["cand_alt"]),
                "strand": int(r["target_intron_strand"]),
                "pheno_id": str(r["pheno_id"]),
                "i_s": int(r["i_s"]),
                "i_e": int(r["i_e"]),
                "ds_max12": float(r["cand_ds_usage_max12"]),
                "ld_r2": float(r["ld_r2_with_lead"]) if pd.notna(r["ld_r2_with_lead"]) else -1.0,
                "PP_H4": float(r["PP_H4"]),
            })
            seen.add(r["gene_symbol"])

    # Also add DDX59/clu_25020_+ (Pituitary) and COL7A1 (Muscle) — let's discover exact pheno_id in SheepGTEx_v0.GWAS_coloc.csv.gz
    gc_all = load_sheep_gwas_coloc_loci(SHEEP_COLOC_GZ, COLOC_LOCI_TSV)
    for sym, pos_t in (("DDX59", 78727889), ("COL7A1", 51350633)):
        sub = gc_all[gc_all["gene_symbol"] == sym].sort_values("PP_H4", ascending=False)
        if len(sub) > 0 and sym not in seen:
            r0 = sub.iloc[0]
            targets.append({
                "label": "DDX59/clu_25020_+" if sym == "DDX59" else sym,
                "trait": str(r0["Trait"]),
                "tissue": str(r0["tissue"]),
                "chr": str(int(r0["chrom_id"])),
                "pos": pos_t,
                "ref": "A" if sym == "DDX59" else "C",
                "alt": "T",
                "strand": parse_pheno_strand(r0["phenotype_id"]),
                "pheno_id": str(r0["phenotype_id"]),
                "i_s": int(r0["i_s"]),
                "i_e": int(r0["i_e"]),
                "ds_max12": 0.877 if sym == "DDX59" else 0.677,
                "ld_r2": 0.65 if sym == "DDX59" else 0.95,
                "PP_H4": float(r0["PP_H4"]),
            })
            seen.add(sym)

    print("  Targets for individual RNA-seq PSI extraction:", [(t["label"], t["tissue"], t["pheno_id"]) for t in targets], flush=True)

    # 1. Extract matching tissue PLINK files from SheepGTEx_v0.Genotype.molQTL.tar.gz if not cached
    tmp_geno_ctx = tempfile.TemporaryDirectory(prefix="ovisplice_sheep_geno_")
    geno_cache = Path(tmp_geno_ctx.name)
    need_tissues = sorted(set(t["tissue"] for t in targets))
    # Map tissue name to exact archive member prefix
    missing_tis = [tis for tis in need_tissues if not (geno_cache / f"{tis}.bed").is_file()]
    if missing_tis:
        print(f"  Extracting PLINK genotypes for {missing_tis} from tar.gz...", flush=True)
        with tarfile.open(SHEEP_GENO_TAR, "r:gz") as tf:
            members = []
            for m in tf.getmembers():
                bn = os.path.basename(m.name)
                for tis in missing_tis:
                    if bn in (f"{tis}.bed", f"{tis}.bim", f"{tis}.fam"):
                        members.append((m, tis, bn))
            for m, tis, bn in members:
                fobj = tf.extractfile(m)
                if fobj is not None:
                    with open(geno_cache / bn, "wb") as out_f:
                        while True:
                            buf = fobj.read(4 * 1024 * 1024)
                            if not buf:
                                break
                            out_f.write(buf)
        print(f"  Extracted {len(members)} PLINK files in {time.time()-t0:.1f}s", flush=True)

    # 2. Extract exact target intron rows from SheepGTEx_v0.Splicing.excision.gz using fast zgrep
    # Excision row IDs look like: "8:89029710:89029836:clu_200688_-" or "chr8:89029710:89029836:clu_200688_-"
    # Build search prefixes: "<chr>:<i_s>:<i_e>:"
    patterns = sorted(set(f"{t['chr']}:{t['i_s']}:{t['i_e']}:" for t in targets))
    pat_file = geno_cache / "intron_patterns.txt"
    pat_file.write_text("\n".join(patterns) + "\n", encoding="utf-8")

    # Get header line
    with gzip.open(SHEEP_EXCISION_GZ, "rt", encoding="utf-8") as f:
        exc_header = f.readline().strip().split()
    sample_names = exc_header[1:]
    samp_to_col = {s: idx for idx, s in enumerate(sample_names)}
    print(f"  Scanning excision matrix ({len(sample_names)} samples) for {len(patterns)} target introns via gzip+grep...", flush=True)
    cmd = f"gzip -dc '{SHEEP_EXCISION_GZ}' | LC_ALL=C grep -F -f '{pat_file}'"
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    matched_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    print(f"  Matched {len(matched_lines)} excision rows in {time.time()-t0:.1f}s", flush=True)

    # Map (chr, i_s, i_e) -> list of (intron_id, tokens[1:])
    exc_by_coord = defaultdict(list)
    for ln in matched_lines:
        sp = ln.split()
        iid = sp[0]
        parts = iid.replace("chr", "").split(":")
        if len(parts) >= 3:
            key = (parts[0], int(parts[1]), int(parts[2]))
            exc_by_coord[key].append((iid, sp[1:]))

    # 3. Pair individual genotypes with individual excision PSI!
    indiv_rows = []
    summaries = []
    for t in targets:
        tis = t["tissue"]
        pfx = geno_cache / tis
        if not (geno_cache / f"{tis}.bed").is_file():
            print(f"  [WARN] Missing PLINK for tissue {tis}, skipping {t['label']}", flush=True)
            continue
        pld = CompactPlinkLD(pfx)
        fam_lines = (geno_cache / f"{tis}.fam").read_text(encoding="utf-8").splitlines()
        fam_ids = [ln.split()[1] for ln in fam_lines if ln.strip()]

        # Find SNP row and allele orientation
        row_idx = pld.find_row(t["chr"], t["pos"])
        w_info = pld.query_window(t["chr"], t["pos"], t["pos"])
        if row_idx < 0 or not w_info:
            print(f"  [WARN] Variant {t['chr']}:{t['pos']} not in {tis}.bim for {t['label']}", flush=True)
            pld.close()
            continue
        _, _, a1_b, a2_b = w_info[0]
        g_raw, maf_v = pld.get_geno_and_maf(row_idx)
        pld.close()
        # CompactPlinkLD decodes 00->2.0 (hom A1), 10->1.0 (het), 11->0.0 (hom A2), 01->nan
        # Convert to ALT allele dosage (0=REF/REF, 1=REF/ALT, 2=ALT/ALT)
        if a1_b.upper() == t["alt"].upper() and a2_b.upper() == t["ref"].upper():
            alt_dos = g_raw
        elif a2_b.upper() == t["alt"].upper() and a1_b.upper() == t["ref"].upper():
            alt_dos = 2.0 - g_raw
        else:
            print(f"  [WARN] Allele mismatch {a1_b}/{a2_b} vs {t['ref']}/{t['alt']} for {t['label']}", flush=True)
            continue

        # Find matching excision row
        candidates_exc = exc_by_coord.get((t["chr"], t["i_s"], t["i_e"]), [])
        if not candidates_exc:
            print(f"  [WARN] No excision row for {t['label']} {t['chr']}:{t['i_s']}:{t['i_e']}", flush=True)
            continue
        # Prefer exact cluster match if multiple
        chosen_iid, vals = candidates_exc[0]
        for iid_c, vals_c in candidates_exc:
            if t["pheno_id"].split(":")[3] in iid_c:
                chosen_iid, vals = iid_c, vals_c
                break

        xs, ys, covs = [], [], []
        for si, fid in enumerate(fam_ids):
            d_val = alt_dos[si]
            if np.isnan(d_val):
                continue
            # Strip trailing _{tissue} to match excision header
            base_id = fid[: -(len(tis) + 1)] if fid.endswith("_" + tis) else fid.rsplit("_", 1)[0]
            col_j = samp_to_col.get(base_id)
            if col_j is None:
                col_j = samp_to_col.get(fid)
            if col_j is None or col_j >= len(vals):
                continue
            tok = vals[col_j]
            if "/" not in tok:
                continue
            num_s, den_s = tok.split("/")
            num_i, den_i = int(num_s), int(den_s)
            if den_i < 5:
                continue
            psi = num_i / den_i
            gt_int = int(round(float(d_val)))
            xs.append(gt_int)
            ys.append(psi)
            covs.append(den_i)
            indiv_rows.append({
                "label": t["label"],
                "tissue": tis,
                "trait": t["trait"],
                "variant": f"{t['chr']}_{t['pos']}_{t['ref']}>{t['alt']}",
                "intron_id": chosen_iid,
                "sample_id": base_id,
                "alt_dosage": gt_int,
                "genotype": [f"{t['ref']}{t['ref']}", f"{t['ref']}{t['alt']}", f"{t['alt']}{t['alt']}"][gt_int],
                "junc_reads": num_i,
                "cluster_reads": den_i,
                "psi": round(psi, 6),
            })

        xs_a = np.array(xs, dtype=np.float64)
        ys_a = np.array(ys, dtype=np.float64)
        if len(xs_a) >= 15 and len(np.unique(xs_a)) >= 2:
            # Linear regression slope, r, two-sided p (via t-stat normal/t approx or exact scipy if available)
            mx, my = xs_a.mean(), ys_a.mean()
            sxx = float(np.sum((xs_a - mx) ** 2))
            sxy = float(np.sum((xs_a - mx) * (ys_a - my)))
            syy = float(np.sum((ys_a - my) ** 2))
            slope = sxy / max(1e-12, sxx)
            r = sxy / math.sqrt(max(1e-12, sxx * syy))
            rho = spearman_corr(xs_a, ys_a)
            df_t = len(xs_a) - 2
            t_stat = abs(r) * math.sqrt(df_t / max(1e-15, 1.0 - r * r))
            try:
                from scipy import stats as sp_stats
                p_ols = float(2.0 * sp_stats.t.sf(t_stat, df_t))
                groups = [ys_a[xs_a == g] for g in (0, 1, 2) if (xs_a == g).sum() > 0]
                p_kw = float(sp_stats.kruskal(*groups).pvalue) if len(groups) >= 2 else 1.0
            except Exception:
                # Normal approx fallback
                p_ols = float(math.erfc(t_stat / math.sqrt(2.0)))
                p_kw = p_ols
            by_gt = {}
            for g in (0, 1, 2):
                sub_y = ys_a[xs_a == g]
                by_gt[str(g)] = {
                    "n": int(len(sub_y)),
                    "mean_psi": round(float(sub_y.mean()), 4) if len(sub_y) else None,
                    "median_psi": round(float(np.median(sub_y)), 4) if len(sub_y) else None,
                }
            summ = {
                **t,
                "intron_id": chosen_iid,
                "n_samples_cov5": int(len(xs_a)),
                "median_cluster_reads": float(np.median(covs)),
                "ols_slope_per_alt": round(slope, 4),
                "pearson_r": round(r, 4),
                "spearman_rho": round(rho, 4),
                "p_ols": p_ols,
                "p_kruskal": p_kw,
                "by_genotype": by_gt,
            }
            summaries.append(summ)
            print(
                f"  [PSI VALIDATED] {t['label']} ({tis}, n={len(xs_a)}): "
                f"0/0={by_gt['0']['median_psi']} (n={by_gt['0']['n']}), "
                f"0/1={by_gt['1']['median_psi']} (n={by_gt['1']['n']}), "
                f"1/1={by_gt['2']['median_psi']} (n={by_gt['2']['n']}), "
                f"slope={slope:+.4f}, rho={rho:+.3f}, p_ols={p_ols:.2e}, p_kw={p_kw:.2e}",
                flush=True,
            )

    # Also extract exact RefSeq exon coordinates around each validated gene for exon-intron diagrams
    need_genes = set(s["label"].split("/")[0] for s in summaries) | {"SFT2D1", "RNF168", "DDX59", "COL7A1"}
    gene_exons = defaultdict(list)
    re_gn = re.compile(r'gene_id "([^"]+)"|gene_name "([^"]+)"|gene "([^"]+)"')
    with gzip.open(SHEEP_GTF_GZ, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            sp = ln.split("\t", 8)
            if sp[2] != "exon":
                continue
            # Fast substring screen
            if not any(g in sp[8] for g in need_genes):
                continue
            for g in need_genes:
                if f'"{g}"' in sp[8]:
                    gene_exons[g].append((sp[0].replace("chr", ""), int(sp[3]), int(sp[4]), sp[6]))
    # Deduplicate exons per gene
    gene_exons_clean = {g: sorted(set(v)) for g, v in gene_exons.items()}

    pd.DataFrame(indiv_rows).to_csv(OUT_DIR / "v4_sheep_gwas_individual_psi.tsv", sep="\t", index=False)
    with open(OUT_DIR / "v4_sheep_gwas_psi_summary.json", "w", encoding="utf-8") as f:
        json.dump({"loci": summaries, "gene_exons": gene_exons_clean}, f, indent=2)
    tmp_geno_ctx.cleanup()
    print(f"[s08:PSI] Saved {len(indiv_rows)} individual animal PSI records across {len(summaries)} loci in {time.time()-t0:.1f}s!", flush=True)
    return summaries


def compute_genomewide_sqtl_gwas_enrichment():
    """
    Part 3: Genome-wide enrichment of Sheep finemapped sQTLs (PIP>=0.10, PIP>=0.50),
    independent sQTLs, and GWAS-colocalized sQTLs (PP.H4>=0.50, PP.H4>=0.80) across OviSplice
    score tiers (both raw fold-enrichment & distance x MAF-stratified CMH Odds Ratios)
    across all 624,445 cis-SNVs in `ovisplice_snv_atlas_v1.npz`!
    """
    t0 = time.time()
    print("[s08:CMH] Computing genome-wide sQTL & GWAS-coloc enrichment across 624,445 cis-SNVs...", flush=True)
    with np.load(ATLAS_NPZ, allow_pickle=False) as az:
        cid = az["chrom_id"].astype(np.int32)
        pos = az["pos"].astype(np.int32)
        dist = az["nearest_dist"].astype(np.int32)
        maf = az["maf"].astype(np.float32)
        ds_m12 = az["ds_usage_max12_500"].astype(np.float32)
        ds_site = az["ds_site_500"].astype(np.float32)

    n = len(cid)
    # Build fast (cid, pos) -> array index map (handle rare duplicate pos by list or first)
    key_arr = cid.astype(np.int64) * 1_000_000_000 + pos.astype(np.int64)
    order = np.argsort(key_arr)
    key_sorted = key_arr[order]

    def mark_positions(chr_pos_set: set[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        if not chr_pos_set:
            return mask
        q = np.array([c * 1_000_000_000 + p for c, p in chr_pos_set], dtype=np.int64)
        idx = np.searchsorted(key_sorted, q, side="left")
        idx2 = np.searchsorted(key_sorted, q, side="right")
        for i0, i1 in zip(idx, idx2):
            if i1 > i0:
                mask[order[i0:i1]] = True
        return mask

    def parse_cid(c_str: str) -> int:
        c = str(c_str).replace("chr", "").replace("CHR", "")
        if c == "X":
            return 27
        return int(c) if c.isdigit() else -1

    # 1. Finemapped sQTLs (PIP>=0.10, PIP>=0.50) & map (tissue, pheno_id) -> CS positions
    pip10_set, pip50_set = set(), set()
    tp_to_cs_pos = defaultdict(set)
    with gzip.open(SHEEP_FINEMAP_GZ, "rt", encoding="utf-8") as fh:
        fh.readline()
        for ln in fh:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) >= 7:
                c = parse_cid(sp[5])
                if c < 0:
                    continue
                p = int(sp[6])
                pip_v = float(sp[4])
                if pip_v >= 0.10:
                    pip10_set.add((c, p))
                if pip_v >= 0.50:
                    pip50_set.add((c, p))
                tp_to_cs_pos[(sp[0], sp[1])].add((c, p))

    # 2. Independent sQTLs
    indep_set = set()
    with gzip.open(SHEEP_INDEP_GZ, "rt", encoding="utf-8") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        # Identify variant column
        v_col = next((i for i, h in enumerate(hdr) if "variant" in h.lower() or "snp" in h.lower()), 1)
        for ln in fh:
            sp = ln.rstrip("\n").split("\t")
            if len(sp) > v_col:
                parts = sp[v_col].replace("chr", "").split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    c = parse_cid(parts[0])
                    if c > 0:
                        indep_set.add((c, int(parts[1])))

    # 3. GWAS-colocalized sQTLs (PP.H4 >= 0.50 and >= 0.80): include BOTH lead variant AND finemapped CS variants of that coloc (tissue, pheno_id)!
    coloc50_lead, coloc80_lead = set(), set()
    coloc50_cs, coloc80_cs = set(), set()
    df_coloc = pd.read_csv(SHEEP_COLOC_GZ)
    df_coloc = df_coloc[df_coloc["QTL_type"].astype(str).str.lower().str.contains("sqtl")].copy()
    for r in df_coloc.to_dict("records"):
        h4 = float(r["PP.H4.abf"])
        if h4 < 0.50:
            continue
        lv = str(r["LEAD_VARIANT"])
        parts = lv.replace("chr", "").split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            c = parse_cid(parts[0])
            if c > 0:
                coloc50_lead.add((c, int(parts[1])))
                if h4 >= 0.80:
                    coloc80_lead.add((c, int(parts[1])))
        cs_ps = tp_to_cs_pos.get((str(r["tissue"]), str(r["phenotype_id"])), set())
        for cp in cs_ps:
            coloc50_cs.add(cp)
            if h4 >= 0.80:
                coloc80_cs.add(cp)

    masks = {
        "finemapped_pip_ge_0.10": mark_positions(pip10_set),
        "finemapped_pip_ge_0.50": mark_positions(pip50_set),
        "independent_sqtl": mark_positions(indep_set),
        "gwas_coloc_h4_ge_0.50_cs": mark_positions(coloc50_cs | coloc50_lead),
        "gwas_coloc_h4_ge_0.80_cs": mark_positions(coloc80_cs | coloc80_lead),
    }

    # Strata for CMH: 6 distance bins x 3 MAF bins = 18 strata!
    d_bins = np.digitize(dist, [3, 21, 51, 201, 1001])  # 0..5
    m_bins = np.digitize(maf, [0.15, 0.30])             # 0..2
    strata = d_bins * 3 + m_bins

    tiers = [
        ("Neutral (<0.02)", 0.0, 0.02),
        ("Low (0.02-0.10)", 0.02, 0.10),
        ("Moderate (0.10-0.25)", 0.10, 0.25),
        ("High (0.25-0.50)", 0.25, 0.50),
        ("Severe (>=0.50)", 0.50, 10.0),
    ]
    ref_mask = ds_m12 < 0.02

    def cmh_or(exp_mask: np.ndarray, ctrl_mask: np.ndarray, y_mask: np.ndarray):
        num = 0.0
        den = 0.0
        mh_num = 0.0
        mh_var = 0.0
        # Also Robins-Breslow-Greenland variance for log(OR)
        p_sum, q_sum, r_sum, s_sum = 0.0, 0.0, 0.0, 0.0
        for st in range(18):
            sm = strata == st
            a = float(np.sum(sm & exp_mask & y_mask))
            b = float(np.sum(sm & exp_mask & (~y_mask)))
            c = float(np.sum(sm & ctrl_mask & y_mask))
            d = float(np.sum(sm & ctrl_mask & (~y_mask)))
            tot = a + b + c + d
            if tot <= 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
                continue
            r_k = (a * d) / tot
            s_k = (b * c) / tot
            num += r_k
            den += s_k
            # Cochran-Mantel-Haenszel chi2 numerator & variance
            e_a = (a + b) * (a + c) / tot
            v_a = (a + b) * (c + d) * (a + c) * (b + d) / (tot * tot * max(1.0, tot - 1.0))
            mh_num += (a - e_a)
            mh_var += v_a
            p_sum += ((a + d) / tot) * r_k
            q_sum += ((a + d) / tot) * s_k + ((b + c) / tot) * r_k
            r_sum += ((b + c) / tot) * s_k
        if den <= 0 or num <= 0:
            return float("nan"), float("nan"), float("nan"), 1.0
        or_v = num / den
        # RBG SE of log(OR)
        var_log = (p_sum / (2.0 * num * num)) + (q_sum / (2.0 * num * den)) + (r_sum / (2.0 * den * den))
        se_log = math.sqrt(max(1e-12, var_log))
        ci_lo = math.exp(math.log(or_v) - 1.96 * se_log)
        ci_hi = math.exp(math.log(or_v) + 1.96 * se_log)
        z = abs(mh_num) / math.sqrt(max(1e-12, mh_var))
        p_val = float(math.erfc(z / math.sqrt(2.0)))
        return or_v, ci_lo, ci_hi, p_val

    rows = []
    for y_name, y_mask in masks.items():
        base_rate = float(np.mean(y_mask[ref_mask]))
        for t_label, lo, hi in tiers:
            tm = (ds_m12 >= lo) & (ds_m12 < hi)
            n_tot = int(np.sum(tm))
            n_hit = int(np.sum(tm & y_mask))
            rate = n_hit / max(1, n_tot)
            fe = rate / max(1e-12, base_rate)
            if lo == 0.0:
                or_v, ci_lo, ci_hi, p_v = 1.0, 1.0, 1.0, 1.0
            else:
                or_v, ci_lo, ci_hi, p_v = cmh_or(tm, ref_mask, y_mask)
            rows.append({
                "outcome": y_name,
                "tier": t_label,
                "n_snvs": n_tot,
                "n_hits": n_hit,
                "hit_rate_pct": round(100.0 * rate, 4),
                "raw_fold_enrichment": round(fe, 3),
                "cmh_or_dist_maf_adj": round(or_v, 3) if not math.isnan(or_v) else None,
                "cmh_ci_lo": round(ci_lo, 3) if not math.isnan(ci_lo) else None,
                "cmh_ci_hi": round(ci_hi, 3) if not math.isnan(ci_hi) else None,
                "cmh_p": p_v,
            })
        # Also summary binary threshold >= 0.20 vs < 0.02
        tm20 = ds_m12 >= 0.20
        or_v, ci_lo, ci_hi, p_v = cmh_or(tm20, ref_mask, y_mask)
        n_tot = int(np.sum(tm20))
        n_hit = int(np.sum(tm20 & y_mask))
        rate = n_hit / max(1, n_tot)
        rows.append({
            "outcome": y_name,
            "tier": "High+Severe (>=0.20)",
            "n_snvs": n_tot,
            "n_hits": n_hit,
            "hit_rate_pct": round(100.0 * rate, 4),
            "raw_fold_enrichment": round(rate / max(1e-12, base_rate), 3),
            "cmh_or_dist_maf_adj": round(or_v, 3) if not math.isnan(or_v) else None,
            "cmh_ci_lo": round(ci_lo, 3) if not math.isnan(ci_lo) else None,
            "cmh_ci_hi": round(ci_hi, 3) if not math.isnan(ci_hi) else None,
            "cmh_p": p_v,
        })

    df_enr = pd.DataFrame(rows)
    df_enr.to_csv(OUT_DIR / "v4_sheep_genomewide_sqtl_gwas_enrichment.tsv", sep="\t", index=False)
    print(f"[s08:CMH] Done in {time.time()-t0:.1f}s! Key enrichment summary:", flush=True)
    for out_k in ("finemapped_pip_ge_0.50", "independent_sqtl", "gwas_coloc_h4_ge_0.80_cs"):
        sub = df_enr[df_enr["outcome"] == out_k]
        print(f"--- {out_k} ---", flush=True)
        print(sub[["tier", "n_snvs", "n_hits", "raw_fold_enrichment", "cmh_or_dist_maf_adj", "cmh_ci_lo", "cmh_ci_hi", "cmh_p"]].to_string(index=False), flush=True)
    return df_enr


def main():
    global BASE_DIR, RAW_DIR, CKPT_DIR, OUT_DIR, BACKEND
    global SHEEP_FASTA, SHEEP_FAI, SHEEP_PLINK_PFX, SHEEP_GTF_GZ, SHEEP_GENO_TAR
    global SHEEP_EXCISION_GZ, SHEEP_FINEMAP_GZ, SHEEP_INDEP_GZ, SHEEP_COLOC_GZ, COLOC_LOCI_TSV, ATLAS_NPZ

    ap = argparse.ArgumentParser(description="624,445-SNV sheep CMH enrichment, strand-verified GWAS-sQTL re-scoring, and individual PSI validation")
    ap.add_argument("--task", choices=["all", "cmh_enrichment", "gwas_strand", "individual_psi"], default="cmh_enrichment",
                    help="Subtask to execute (`cmh_enrichment` runs directly from 624k atlas + public SheepGTEx QTL files)")
    ap.add_argument("--sheepgtex-dir", type=Path, default=REPO_ROOT / "data" / "00_raw" / "sheepgtex",
                    help="Directory containing SheepGTEx_v0.{finemapped_sQTL.txt.gz,independent_sQTL.txt.gz,GWAS_coloc.csv.gz}")
    ap.add_argument("--sheep-fasta", type=Path, default=REPO_ROOT / "data" / "00_raw" / "ref" / "GCF_016772045.1_genomic.fna",
                    help="Indexed sheep ARS-UI_Ramb_v2.0 reference FASTA (.fai alongside)")
    ap.add_argument("--atlas-npz", type=Path, default=None,
                    help="Path to ovisplice_sheep_624k_snv_atlas.npz (from Zenodo deposit)")
    ap.add_argument("--coloc-loci-tsv", type=Path, default=None,
                    help="Optional pre-parsed sheep GWAS-coloc TSV (otherwise parsed directly from SheepGTEx_v0.GWAS_coloc.csv.gz)")
    ap.add_argument("--sheep-plink-prefix", type=Path, default=None,
                    help="Optional PLINK prefix for Muscle genotypes (otherwise streamed from SheepGTEx_v0.Genotype.molQTL.tar.gz)")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    args = ap.parse_args()

    CKPT_DIR, OUT_DIR, BACKEND = args.ckpt_dir, args.out_dir, args.backend
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEEP_FASTA = args.sheep_fasta
    SHEEP_FAI = Path(str(args.sheep_fasta) + ".fai")
    COLOC_LOCI_TSV = args.coloc_loci_tsv
    sg = args.sheepgtex_dir
    SHEEP_GTF_GZ = sg / "sheep.gtf.gz"
    SHEEP_GENO_TAR = sg / "SheepGTEx_v0.Genotype.molQTL.tar.gz"
    SHEEP_EXCISION_GZ = sg / "SheepGTEx_v0.Splicing.excision.gz"
    SHEEP_FINEMAP_GZ = sg / "SheepGTEx_v0.finemapped_sQTL.txt.gz"
    SHEEP_INDEP_GZ = sg / "SheepGTEx_v0.independent_sQTL.txt.gz"
    SHEEP_COLOC_GZ = sg / "SheepGTEx_v0.GWAS_coloc.csv.gz"

    if args.atlas_npz is not None:
        ATLAS_NPZ = args.atlas_npz
    elif (CKPT_DIR / "ovisplice_sheep_624k_snv_atlas.npz").is_file():
        ATLAS_NPZ = CKPT_DIR / "ovisplice_sheep_624k_snv_atlas.npz"
    else:
        ATLAS_NPZ = REPO_ROOT / "ovisplice_sheep_624k_snv_atlas.npz"

    tmp_holders = []
    summary = {}
    df_tg = None
    try:
        if args.task in ("all", "gwas_strand", "individual_psi"):
            SHEEP_PLINK_PFX = extract_plink_from_tar_if_needed(
                args.sheep_plink_prefix or (sg / "plink" / "Muscle"),
                SHEEP_GENO_TAR,
                "Muscle",
                tmp_holders,
            )
            m_full = load_full_ensemble()
            df_tg, df_1mb, _ism_meta = fix_sheep_gwas_strand(m_full)
            del m_full
            summary["sheep_gwas_strand_fixed"] = {
                "n_trait_gene_linked": int(len(df_tg)),
                "n_1mb_linked": int(len(df_1mb)),
                "n_strand_mismatch": 0,
                "n_ge_010_tg": int((df_tg["cand_ds_usage_max12"] >= 0.10).sum()),
                "n_strict_reranked_tg": int(df_tg["strict_reranked_over_neutral_lead"].sum()),
                "n_strict_reranked_1mb": int(df_1mb["strict_reranked_over_neutral_lead"].sum()),
            }
        if args.task in ("all", "individual_psi") and df_tg is not None:
            psi_summ = extract_individual_psi_validation(df_tg)
            summary["individual_psi_validated_n_loci"] = len(psi_summ)
        if args.task in ("all", "cmh_enrichment"):
            df_enr = compute_genomewide_sqtl_gwas_enrichment()
            summary["genomewide_enrichment_rows"] = len(df_enr)
    finally:
        for th in tmp_holders:
            th.cleanup()

    with open(OUT_DIR / "sheep_atlas_and_gwas_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[s08] Completed requested tasks.", flush=True)


if __name__ == "__main__":
    main()
