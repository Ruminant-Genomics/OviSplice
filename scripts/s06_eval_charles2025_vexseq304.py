#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s06_eval_charles2025_vexseq304.py
=================================
External wet-lab bovine MPRA benchmark on Charles et al. (*Nature Communications* 2025,
doi:10.1038/s41467-025-58970-5):
  - Parses all 304 QC-passed bovine Vex-seq variants (`Var.GWAS n=183` + `Var.P n=121`)
    assayed in bovine mammary epithelial (`MAC-T`) and `HEK293T` cells.
  - Verifies genomic REF allele concordance against `ARS-UCD1.2` (`GCF_002263795.1`).
  - Scores `OviSplice` (`ovisplice_full_seed42/1042/2042.pt`), `OviSplice-F1`, and
    `OviSplice-SiteOnly`, computing AUROC, AUPRC, signed Pearson r / Spearman rho, paired
    DeLong p-values, bootstrap 95% CIs, and score-bucket TP/TN calibration.
"""
from __future__ import annotations

import argparse
import gzip
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice.native_engine import (
    BASE_MAP,
    COMP_U8,
    load_ensemble,
    paired_delong_test,
    spearman_corr,
    tie_invariant_auroc_auprc,
)

RAW_C25 = REPO_ROOT / "data" / "00_raw" / "charles2025_vexseq"
CATTLE_FNA_GZ = REPO_ROOT / "data" / "00_raw" / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz"
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "results"
BACKEND = "auto"


def parse_float(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s in ("", ".", "NA", "nan", "None"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def load_charles_vexseq_304() -> pd.DataFrame:
    records = []
    for ds_name, fn in (("Var.GWAS", "41467_2025_58970_MOESM5_ESM.xlsx"), ("Var.P", "41467_2025_58970_MOESM6_ESM.xlsx")):
        wb = openpyxl.load_workbook(RAW_C25 / fn, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = list(ws.iter_rows(values_only=True))
        hdr = [str(c).strip() if c is not None else "" for c in rows[3]]
        cidx = {h: i for i, h in enumerate(hdr)}
        for r in rows[4:]:
            if r[0] is None:
                continue
            ch = str(r[cidx["Chr"]]).strip().replace("chr", "")
            if not (ch.isdigit() or ch in ("X", "Y")):
                continue
            pos = int(r[cidx["Position"]])
            ref = str(r[cidx["REF allele"]]).strip().upper()
            alt = str(r[cidx["ALT allele"]]).strip().upper()
            reg = str(r[cidx["Vex-seq region"]]).strip()
            m_st = re.search(r"\(([+-])\)", reg)
            strand = 1 if (m_st and m_st.group(1) == "+") else -1
            m_reg = re.match(r"(\d+)-(\d+)", reg)
            reg_s = int(m_reg.group(1)) if m_reg else pos - 75
            reg_e = int(m_reg.group(2)) if m_reg else pos + 75

            sp_score = parse_float(r[cidx["SpliceAI - score"]])
            sp_cons = str(r[cidx["SpliceAI - consequence"]]).strip()
            # Charles et al. Fig 4c sign convention: SpliceAI loss (AL/DL) -> negative, gain (AG/DG) -> positive
            if sp_cons in ("AL", "DL"):
                sp_signed = -abs(sp_score)
            elif sp_cons in ("AG", "DG"):
                sp_signed = abs(sp_score)
            else:
                sp_signed = 0.0 if math.isnan(sp_score) else sp_score

            pan_signed = parse_float(r[cidx["Pangolin - score"]])
            if math.isnan(pan_signed):
                pan_signed = 0.0

            hek_dpsi = parse_float(r[cidx["HEK293T / ΔPSI(M)"]])
            hek_fdr = parse_float(r[cidx["HEK293T / ΔPSI(FDR)"]])
            mact_dpsi = parse_float(r[cidx["MAC-T / ΔPSI(M)"]])
            mact_fdr = parse_float(r[cidx["MAC-T / ΔPSI(FDR)"]])
            avg_dpsi = np.nanmean([hek_dpsi, mact_dpsi])

            records.append({
                "dataset": ds_name,
                "chr": ch,
                "pos": pos,
                "rsid": str(r[cidx["rsID"]]).strip(),
                "ref": ref,
                "alt": alt,
                "gene_symbol": str(r[cidx["Gene symbol"]]).strip(),
                "ensembl_gene_id": str(r[cidx["Ensembl Gene ID"]]).strip(),
                "consequence": str(r[cidx["Variant consequence"]]).strip(),
                "strand": strand,
                "vex_start": reg_s,
                "vex_end": reg_e,
                "gerp": parse_float(r[cidx["GERP score"]]),
                "spliceai_cons": sp_cons,
                "spliceai_abs": abs(sp_score) if not math.isnan(sp_score) else 0.0,
                "spliceai_signed": sp_signed,
                "pangolin_signed": pan_signed,
                "pangolin_abs": abs(pan_signed),
                "hek_dpsi": hek_dpsi,
                "hek_fdr": hek_fdr,
                "mact_dpsi": mact_dpsi,
                "mact_fdr": mact_fdr,
                "avg_dpsi": avg_dpsi,
                "mact_sdv_fdr01": int((not math.isnan(mact_fdr)) and (mact_fdr < 0.01) and (abs(mact_dpsi) >= 5.0)),
                "mact_sdv_fdr05": int((not math.isnan(mact_fdr)) and (mact_fdr < 0.05) and (abs(mact_dpsi) >= 5.0)),
                "hek_sdv_fdr01": int((not math.isnan(hek_fdr)) and (hek_fdr < 0.01) and (abs(hek_dpsi) >= 5.0)),
                "hek_sdv_fdr05": int((not math.isnan(hek_fdr)) and (hek_fdr < 0.05) and (abs(hek_dpsi) >= 5.0)),
                "avg_sdv_fdr01": int(
                    ((not math.isnan(mact_fdr)) and (mact_fdr < 0.01) and (abs(mact_dpsi) >= 5.0))
                    or ((not math.isnan(hek_fdr)) and (hek_fdr < 0.01) and (abs(hek_dpsi) >= 5.0))
                ),
            })
        wb.close()
    df = pd.DataFrame(records)
    print(f"[s06] Loaded Charles et al. 2025 Vex-seq: total={len(df)} (Var.GWAS={(df['dataset']=='Var.GWAS').sum()}, Var.P={(df['dataset']=='Var.P').sum()})", flush=True)
    print(f"  MAC-T SDV (FDR<0.01,|ΔPSI|>=5%): {df['mact_sdv_fdr01'].sum()} | (FDR<0.05): {df['mact_sdv_fdr05'].sum()} | HEK FDR<0.01: {df['hek_sdv_fdr01'].sum()}", flush=True)
    return df


def load_cattle_chr_seqs(need_chrs: set[str]) -> dict[str, np.ndarray]:
    nc_to_ch = {f"NC_{37327 + i:06d}.1": str(i) for i in range(1, 30)}
    nc_to_ch["NC_037357.1"] = "X"
    lut = np.full(256, 4, dtype=np.uint8)
    for b, v in ((ord("A"), 0), (ord("a"), 0), (ord("C"), 1), (ord("c"), 1),
                 (ord("G"), 2), (ord("g"), 2), (ord("T"), 3), (ord("t"), 3)):
        lut[b] = v
    seqs = {}
    cur_ch = None
    chunks = []
    with gzip.open(CATTLE_FNA_GZ, "rb") as fh:
        for ln in fh:
            if ln.startswith(b">"):
                if cur_ch is not None:
                    seqs[cur_ch] = lut[np.frombuffer(b"".join(chunks), dtype=np.uint8)]
                acc = ln[1:].split()[0].decode("ascii", "ignore")
                ch = nc_to_ch.get(acc)
                cur_ch = ch if ch in need_chrs else None
                chunks = []
            elif cur_ch is not None:
                chunks.append(ln.strip())
    if cur_ch is not None:
        seqs[cur_ch] = lut[np.frombuffer(b"".join(chunks), dtype=np.uint8)]
    return seqs


def score_vexseq_with_ovi(df: pd.DataFrame, chr_seqs: dict[str, np.ndarray]):
    """
    Build exact 11,001-bp (flank=5,500 -> center 1,001 bp) REF & ALT sequences oriented on gene strand
    for all 304 variants, verify REF allele 100%, score with OviSplice full ens3, F1 ens2, and M1,
    returning signed and unsigned scores at both +-50 bp (Pangolin/SpliceAI standard) and +-500 bp.
    """
    ref_seqs, alt_seqs = [], []
    ref_match_n = 0
    valid_mask = []
    for r in df.to_dict("records"):
        arr = chr_seqs[r["chr"]]
        p0 = int(r["pos"]) - 1
        lo = p0 - 5500
        hi = p0 + 5501
        w = np.full(11001, 4, dtype=np.uint8)
        s0 = max(0, lo)
        e0 = min(len(arr), hi)
        w[s0 - lo : e0 - lo] = arr[s0:e0]
        r_u8 = BASE_MAP.get(r["ref"][0], 4)
        a_u8 = BASE_MAP.get(r["alt"][0], 4)
        is_snv = (len(r["ref"]) == 1) and (len(r["alt"]) == 1) and (r_u8 < 4) and (a_u8 < 4)
        if is_snv and int(w[5500]) == r_u8:
            ref_match_n += 1
        w_ref = w.copy()
        w_alt = w.copy()
        if is_snv:
            w_ref[5500] = r_u8
            w_alt[5500] = a_u8
        if int(r["strand"]) == -1:
            w_ref = COMP_U8[w_ref[::-1]].copy()
            w_alt = COMP_U8[w_alt[::-1]].copy()
        ref_seqs.append(w_ref)
        alt_seqs.append(w_alt)
        valid_mask.append(is_snv)

    ref_arr = np.stack(ref_seqs, axis=0)
    alt_arr = np.stack(alt_seqs, axis=0)
    print(f"[s06] ARS-UCD1.2 REF match: {ref_match_n}/{len(df)} (valid SNVs={sum(valid_mask)})", flush=True)

    # Save sequence bundle for Cloud GPU (SpliceTransformer + OpenSpliceAI)!
    np.savez_compressed(
        OUT_DIR / "v4_charles2025_vexseq_304_seqs.npz",
        ref_u8=ref_arr,
        alt_u8=alt_arr,
        valid_snv=np.array(valid_mask, dtype=np.uint8),
        rsid=df["rsid"].to_numpy(dtype=object),
        chr=df["chr"].to_numpy(dtype=object),
        pos=df["pos"].to_numpy(dtype=np.int32),
        strand=df["strand"].to_numpy(dtype=np.int8),
    )

    # Score full 1,001-bp center output windows (offset -500..+500) so we get EXACT signed & unsigned
    # deltas at BOTH +-50 bp (matching Pangolin/SpliceAI default D=50 in Charles et al.) and +-500 bp!
    cand_specs = {
        "ovi_full": [CKPT_DIR / f"ovisplice_full_seed{s}.pt" for s in (42, 1042, 2042)],
        "ovi_f1": [CKPT_DIR / f"ovisplice_fold1_seed{s}.pt" for s in (42, 1042)],
        "ovi_siteonly": [CKPT_DIR / f"ovisplice_siteonly_fold{f}_seed42.pt" for f in (1, 2, 3, 4, 5)],
    }
    model_specs = {k: [str(p) for p in ps] for k, ps in cand_specs.items() if all(p.is_file() for p in ps)}
    if "ovi_full" not in model_specs:
        raise FileNotFoundError(f"Default 3-seed OviSplice checkpoints not found in {CKPT_DIR}")

    n = len(df)
    for mname, ckpts in model_specs.items():
        t0 = time.time()
        ens, _ = load_ensemble(ckpts, backend=BACKEND)
        has_usage = (mname != "ovi_siteonly")
        p_ref, u_ref = ens.score_windows(ref_arr, compute_usage=has_usage)
        p_alt, u_alt = ens.score_windows(alt_arr, compute_usage=has_usage)
        del ens

        # p_splice = p_acc + p_don; gate = max(p_ref, p_alt)
        ps_ref = (p_ref[:, :, 0] + p_ref[:, :, 1]).astype(np.float32)  # (N, 1001)
        ps_alt = (p_alt[:, :, 0] + p_alt[:, :, 1]).astype(np.float32)
        gate = np.maximum(ps_ref, ps_alt)  # (N, 1001)

        if not has_usage:
            u_ref = np.zeros((n, ps_ref.shape[1], 15), dtype=np.float32)
            u_alt = np.zeros((n, ps_alt.shape[1], 15), dtype=np.float32)

        # Signed raw usage delta per head: du = (u_alt - u_ref) * gate
        du_all = (u_alt - u_ref) * gate[:, :, None]   # (N, 1001, 15)
        du_raw = (u_alt - u_ref)                       # (N, 1001, 15) ungated
        dp_site = ps_alt - ps_ref                      # (N, 1001)

        for win_bp in (50, 500):
            s0 = 500 - win_bp
            s1 = 500 + win_bp + 1
            # 1. Signed & unsigned gated usage max across 12 main organs (heads 1..12)
            du12 = du_all[:, s0:s1, 1:13]  # (N, W, 12)
            # For each variant: find position & head with max |du|
            flat12 = du12.reshape(n, -1)
            idx12 = np.argmax(np.abs(flat12), axis=1)
            signed_m12 = flat12[np.arange(n), idx12]
            abs_m12 = np.abs(signed_m12)

            # 2. PanTissue (head 0) — note Mammary Gland maps to PanTissue head 0 in OviSplice!
            du0 = du_all[:, s0:s1, 0]
            idx0 = np.argmax(np.abs(du0), axis=1)
            signed_pan = du0[np.arange(n), idx0]
            abs_pan = np.abs(signed_pan)

            # 3. Ungated raw usage delta (matching Pangolin's raw usage difference definition!)
            dur12 = du_raw[:, s0:s1, 1:13].reshape(n, -1)
            idxr12 = np.argmax(np.abs(dur12), axis=1)
            signed_raw12 = dur12[np.arange(n), idxr12]
            abs_raw12 = np.abs(signed_raw12)

            # 4. Site probability delta (analogous to SpliceAI)
            dps = dp_site[:, s0:s1]
            idxp = np.argmax(np.abs(dps), axis=1)
            signed_site = dps[np.arange(n), idxp]
            abs_site = np.abs(signed_site)

            df[f"{mname}_m12_abs_{win_bp}"] = abs_m12
            df[f"{mname}_m12_signed_{win_bp}"] = signed_m12
            df[f"{mname}_pan_abs_{win_bp}"] = abs_pan
            df[f"{mname}_pan_signed_{win_bp}"] = signed_pan
            df[f"{mname}_raw12_abs_{win_bp}"] = abs_raw12
            df[f"{mname}_raw12_signed_{win_bp}"] = signed_raw12
            df[f"{mname}_site_abs_{win_bp}"] = abs_site
            df[f"{mname}_site_signed_{win_bp}"] = signed_site

        print(f"  Scored {mname} ({len(ckpts)} ckpts) on {n} variants in {time.time()-t0:.1f}s", flush=True)
    return df


def eval_vexseq_metrics(df: pd.DataFrame):
    models = [
        ("OviSplice (max12, +-50bp)", "ovi_full_m12_abs_50", "ovi_full_m12_signed_50"),
        ("OviSplice (max12, +-500bp)", "ovi_full_m12_abs_500", "ovi_full_m12_signed_500"),
        ("OviSplice (PanTissue, +-50bp)", "ovi_full_pan_abs_50", "ovi_full_pan_signed_50"),
        ("OviSplice (raw usage, +-50bp)", "ovi_full_raw12_abs_50", "ovi_full_raw12_signed_50"),
        ("OviSplice (site prob, +-50bp)", "ovi_full_site_abs_50", "ovi_full_site_signed_50"),
        ("OviSplice-F1 (0-ortholog, +-50bp)", "ovi_f1_m12_abs_50", "ovi_f1_m12_signed_50"),
        ("OviSplice-SiteOnly (+-50bp)", "ovi_siteonly_site_abs_50", "ovi_siteonly_site_signed_50"),
        ("Human Pangolin (Charles2025)", "pangolin_abs", "pangolin_signed"),
        ("SpliceAI (Charles2025)", "spliceai_abs", "spliceai_signed"),
    ]

    # Evaluate across MAC-T (FDR<0.01, FDR<0.05), HEK293T (FDR<0.01), Average (FDR<0.01)
    # Note: Charles et al. Fig 4e uses FDR<0.01 (regardless of 5% cutoff or with |ΔPSI|>=5% — check both!)
    df["mact_sdv_fdr01_any"] = ((df["mact_fdr"] < 0.01) & df["mact_fdr"].notna()).astype(int)
    df["hek_sdv_fdr01_any"] = ((df["hek_fdr"] < 0.01) & df["hek_fdr"].notna()).astype(int)
    df["avg_sdv_fdr01_any"] = ((df["mact_sdv_fdr01_any"] == 1) | (df["hek_sdv_fdr01_any"] == 1)).astype(int)

    tasks = [
        ("MAC-T (FDR<0.01, |dPSI|>=5%)", "mact_sdv_fdr01", "mact_dpsi"),
        ("MAC-T (FDR<0.01, all sig)", "mact_sdv_fdr01_any", "mact_dpsi"),
        ("MAC-T (FDR<0.05, |dPSI|>=5%)", "mact_sdv_fdr05", "mact_dpsi"),
        ("HEK293T (FDR<0.01, |dPSI|>=5%)", "hek_sdv_fdr01", "hek_dpsi"),
        ("HEK293T (FDR<0.01, all sig)", "hek_sdv_fdr01_any", "hek_dpsi"),
        ("Combined MAC-T/HEK (FDR<0.01, |dPSI|>=5%)", "avg_sdv_fdr01", "avg_dpsi"),
    ]

    summary_rows = []
    for t_name, y_col, dpsi_col in tasks:
        sub = df[df[dpsi_col].notna()].copy()
        y = sub[y_col].to_numpy(dtype=np.int32)
        dpsi = sub[dpsi_col].to_numpy(dtype=np.float64)
        hp_s = sub["pangolin_abs"].to_numpy(dtype=np.float64)
        sa_s = sub["spliceai_abs"].to_numpy(dtype=np.float64)

        print(f"\n=== Vex-seq Task: {t_name} (n={len(sub)}, pos={int(y.sum())}) ===", flush=True)
        for m_label, abs_c, sig_c in models:
            if abs_c not in sub.columns:
                continue
            sc_abs = sub[abs_c].to_numpy(dtype=np.float64)
            sc_sig = sub[sig_c].to_numpy(dtype=np.float64)
            auroc, auprc = tie_invariant_auroc_auprc(y, sc_abs)
            # Signed Pearson r & Spearman rho vs ΔPSI
            r_signed = float(np.corrcoef(sc_sig, dpsi)[0, 1]) if np.std(sc_sig) > 0 else 0.0
            rho_signed = spearman_corr(sc_sig, dpsi)
            rho_abs = spearman_corr(sc_abs, np.abs(dpsi))
            p_vs_hp = float(paired_delong_test(y, sc_abs, hp_s)["p"]) if abs_c != "pangolin_abs" else 1.0
            p_vs_sa = float(paired_delong_test(y, sc_abs, sa_s)["p"]) if abs_c != "spliceai_abs" else 1.0

            # Bootstrap 95% CI on AUROC & AUPRC (2000 reps)
            rng = np.random.default_rng(42)
            idx_b = rng.integers(0, len(y), size=(1000, len(y)))
            au_b, ap_b = [], []
            for row_i in idx_b:
                yb = y[row_i]
                if yb.sum() == 0 or yb.sum() == len(yb):
                    continue
                a1, a2 = tie_invariant_auroc_auprc(yb, sc_abs[row_i])
                au_b.append(a1)
                ap_b.append(a2)
            au_lo, au_hi = float(np.percentile(au_b, 2.5)), float(np.percentile(au_b, 97.5))
            ap_lo, ap_hi = float(np.percentile(ap_b, 2.5)), float(np.percentile(ap_b, 97.5))

            summary_rows.append({
                "task": t_name,
                "model": m_label,
                "abs_col": abs_c,
                "n_total": int(len(sub)),
                "n_pos": int(y.sum()),
                "auroc": round(auroc, 4),
                "auroc_ci_lo": round(au_lo, 4),
                "auroc_ci_hi": round(au_hi, 4),
                "auprc": round(auprc, 4),
                "auprc_ci_lo": round(ap_lo, 4),
                "auprc_ci_hi": round(ap_hi, 4),
                "pearson_r_signed": round(r_signed, 4),
                "spearman_rho_signed": round(rho_signed, 4),
                "spearman_rho_abs": round(rho_abs, 4),
                "delong_p_vs_hp": p_vs_hp,
                "delong_p_vs_spliceai": p_vs_sa,
            })
            print(
                f"  {m_label:36s} | AUROC={auroc:.4f} [{au_lo:.3f},{au_hi:.3f}] | "
                f"AUPRC={auprc:.4f} | r_sig={r_signed:.4f} | rho_abs={rho_abs:.4f} | "
                f"p_HP={p_vs_hp:.4g} | p_SAI={p_vs_sa:.4g}",
                flush=True,
            )

    # Also compute stratified TP/TN rates across Charles et al. Fig 4d buckets ([0,0.2), [0.2,0.5), [0.5,0.8), [0.8,1.0])
    bucket_rows = []
    for cell_lbl, y_col, dpsi_c in (("MAC-T_FDR01", "mact_sdv_fdr01", "mact_dpsi"), ("HEK293T_FDR01", "hek_sdv_fdr01", "hek_dpsi")):
        df_q = df[df[dpsi_c].notna()]
        for m_lbl, col in (
            ("OviSplice_m12_50", "ovi_full_m12_abs_50"),
            ("OviSplice_raw12_50", "ovi_full_raw12_abs_50"),
            ("Human_Pangolin", "pangolin_abs"),
            ("SpliceAI", "spliceai_abs"),
        ):
            sc = df_q[col].to_numpy(dtype=np.float64)
            y = df_q[y_col].to_numpy(dtype=np.int32)
            for blbl, lo, hi in (("[0,0.2)", 0.0, 0.2), ("[0.2,0.5)", 0.2, 0.5), ("[0.5,0.8)", 0.5, 0.8), ("[0.8,1.0]", 0.8, 1.01)):
                m = (sc >= lo) & (sc < hi)
                nb = int(m.sum())
                npos = int(y[m].sum())
                rate = (1.0 - npos / nb) if lo == 0.0 else (npos / nb if nb > 0 else float("nan"))
                bucket_rows.append({
                    "cell": cell_lbl, "model": m_lbl, "bucket": blbl,
                    "n_variants": nb, "n_sdv": npos,
                    "metric": "TN_rate" if lo == 0.0 else "TP_rate",
                    "rate": round(rate, 4) if not math.isnan(rate) else None,
                })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "v4_charles2025_vexseq_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(bucket_rows).to_csv(OUT_DIR / "v4_charles2025_vexseq_buckets.tsv", sep="\t", index=False)
    df.to_csv(OUT_DIR / "v4_charles2025_vexseq_304_scored.tsv", sep="\t", index=False)


def main():
    global RAW_C25, CATTLE_FNA_GZ, CKPT_DIR, OUT_DIR, BACKEND
    ap = argparse.ArgumentParser(description="Evaluate OviSplice on Charles et al. (2025) 304 bovine Vex-seq MPRA variants")
    ap.add_argument("--vexseq-dir", type=Path, default=REPO_ROOT / "data" / "00_raw" / "charles2025_vexseq",
                    help="Directory containing 41467_2025_58970_MOESM5_ESM.xlsx and MOESM6_ESM.xlsx")
    ap.add_argument("--cattle-fna-gz", type=Path, default=REPO_ROOT / "data" / "00_raw" / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz",
                    help="Path to gzipped ARS-UCD1.2 genomic FASTA")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints", help="Directory containing OviSplice .pt checkpoints")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results", help="Output directory for scored TSV and metrics")
    ap.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    args = ap.parse_args()

    RAW_C25 = args.vexseq_dir
    CATTLE_FNA_GZ = args.cattle_fna_gz
    CKPT_DIR = args.ckpt_dir
    OUT_DIR = args.out_dir
    BACKEND = args.backend
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_charles_vexseq_304()
    need_chrs = set(df["chr"].unique())
    print(f"[s06] Loading {len(need_chrs)} ARS-UCD1.2 chromosomes...", flush=True)
    chr_seqs = load_cattle_chr_seqs(need_chrs)
    df = score_vexseq_with_ovi(df, chr_seqs)
    eval_vexseq_metrics(df)
    print(f"[s06] ALL CHARLES 2025 VEX-SEQ EVALUATIONS COMPLETE in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
