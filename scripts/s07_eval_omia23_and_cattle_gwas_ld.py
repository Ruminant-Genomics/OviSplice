#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s07_eval_omia23_and_cattle_gwas_ld.py
=====================================
Evaluates:
  1. Purified 23 OMIA Mendelian splice-disrupting mutations (21 cattle + 2 sheep strict OOF;
     14 canonical + 9 non-canonical) against 368 same-gene common benign controls (`MAF >= 0.05`).
  2. Dual-species GWAS-sQTL colocalization candidate prioritization under strict `Trait x Gene`
     and `+-1 Mb` locus deduplication, exact population PLINK LD `r^2 >= 0.60` / SuSiE 95% CS
     verification, and `+-45 bp` saturation ISM profiling.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import re
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice.native_engine import (
    BASE_MAP,
    CATTLE_TISSUE_MAP,
    COMP_BASE,
    COMP_U8,
    HEAD_NAMES,
    FastaIndexedReader,
    load_ensemble,
    paired_delong_test,
    paired_mcnemar_exact,
    tie_invariant_auroc_auprc,
)

SHEEP_QTL_TISSUE_TO_HEAD = {
    "Muscle": 1, "Longissimus_muscle": 1, "Infraspinatus_muscle": 1, "Skeletal_muscle": 1,
    "Liver": 2,
    "Adipose": 3, "Caul_adipose": 3, "Intermuscular_adipose": 3,
    "Perirenal_adipose": 3, "Subcutaneous_adipose": 3, "Tail_adipose": 3,
    "Small_intestine": 4, "Duodenum": 4, "Ileum": 4, "Jejunum": 4,
    "Brain": 5, "Brainstem": 5, "Cerebellum": 5, "Cerebrum": 5, "Cerebral_cortex": 5,
    "Hypothalamus": 5, "Hippocampus": 5, "Optic_chiasm": 5, "Pineal": 5,
    "Rumen": 6,
    "Pituitary": 7, "Adenohypophysis": 7,
    "Lung": 8,
    "Abomasum": 9,
    "Ovary": 10, "Ovarian_follicle": 10, "Granulosa_cell": 10, "Testis": 10,
    "Heart": 11,
    "Skin": 12, "Hair_follicle": 12,
}


def load_sheep_gwas_coloc_loci(coloc_csv_gz: Path, coloc_loci_tsv: Path | None = None) -> pd.DataFrame:
    """Parse SheepGTEx_v0.GWAS_coloc.csv.gz (`sQTL` & `PP.H4.abf >= 0.50`) directly on the fly."""
    if coloc_loci_tsv is not None and coloc_loci_tsv.is_file():
        return pd.read_csv(coloc_loci_tsv, sep="\t")
    gc_raw = pd.read_csv(coloc_csv_gz)
    sq = gc_raw[(gc_raw["QTL_type"] == "sQTL") & (gc_raw["PP.H4.abf"] >= 0.5)].copy()
    keep_rows = []
    for r in sq.to_dict("records"):
        parts = str(r["phenotype_id"]).split(":")
        if len(parts) < 4:
            continue
        c_str = parts[0].replace("chr", "")
        cid = 27 if c_str == "X" else (int(c_str) if c_str.isdigit() and 1 <= int(c_str) <= 26 else 0)
        if cid == 0:
            continue
        try:
            i_s, i_e = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if i_s <= 0 or i_e <= 0:
            continue
        st = -1 if parts[3].endswith("_-") else (1 if parts[3].endswith("_+") else 0)
        sym_p = parts[4] if len(parts) >= 5 else ""
        lead_parts = str(r["LEAD_VARIANT"]).split("_")
        lead_pos = int(lead_parts[1]) if len(lead_parts) >= 2 and lead_parts[1].isdigit() else 0
        gene_sym = str(r["gene_symbol"]) if pd.notna(r["gene_symbol"]) else sym_p
        keep_rows.append({
            "Trait": r["Trait"],
            "PP_H4": round(float(r["PP.H4.abf"]), 6),
            "LEAD_VARIANT": r["LEAD_VARIANT"],
            "lead_pos": lead_pos,
            "chrom_id": cid,
            "chrom": r["chrom"],
            "i_s": i_s,
            "i_e": i_e,
            "pheno_strand_hint": st,
            "phenotype_id": r["phenotype_id"],
            "tissue": r["tissue"],
            "head_idx": SHEEP_QTL_TISSUE_TO_HEAD.get(str(r["tissue"]), 0),
            "Loci": r["Loci"],
            "gene_symbol": gene_sym,
        })
    return pd.DataFrame(keep_rows).sort_values("PP_H4", ascending=False).reset_index(drop=True)


def extract_plink_from_tar_if_needed(
    plink_pfx: Path,
    tar_gz: Path | None,
    basename: str,
    tmp_holders: list,
) -> Path:
    """Return `plink_pfx` if `.bed` exists, else stream `{basename}.{bed,bim,fam}` from `tar_gz`."""
    if Path(str(plink_pfx) + ".bed").is_file():
        return plink_pfx
    if tar_gz is None or not tar_gz.is_file():
        raise FileNotFoundError(f"PLINK prefix {plink_pfx}.bed not found (and archive {tar_gz} missing).")
    tctx = tempfile.TemporaryDirectory(prefix=f"ovisplice_{basename}_plink_")
    tmp_holders.append(tctx)
    td = Path(tctx.name)
    targets = {f"{basename}.bed", f"{basename}.bim", f"{basename}.fam"}
    with tarfile.open(tar_gz, "r:gz") as tf:
        for m in tf.getmembers():
            bn = os.path.basename(m.name)
            if bn in targets:
                fobj = tf.extractfile(m)
                if fobj is not None:
                    with open(td / bn, "wb") as out_f:
                        while True:
                            buf = fobj.read(4 * 1024 * 1024)
                            if not buf:
                                break
                            out_f.write(buf)
    return td / basename


BASE_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / "data" / "00_raw"
OMIA_DIR = RAW_DIR / "omia"
CATTLEGTEX_DIR = RAW_DIR / "cattlegtex"
OMIA_CURATED_TSV = REPO_ROOT / "examples" / "omia_mendelian_variants_curated.tsv"
SHEEP_COLOC_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.GWAS_coloc.csv.gz"
COLOC_LOCI_TSV: Path | None = None
CKPT_DIR = REPO_ROOT / "checkpoints"
ATLAS_NPZ = REPO_ROOT / "checkpoints" / "ovisplice_sheep_624k_snv_atlas.npz"
OUT_DIR = REPO_ROOT / "results"
BACKEND = "auto"

SHEEP_FASTA = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna"
SHEEP_FAI = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna.fai"
CATTLE_FNA_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz"
CATTLE_GTF_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.gtf.gz"
SHEEP_GTF_GZ = RAW_DIR / "sheepgtex" / "sheep.gtf.gz"
SHEEP_PLINK_PFX = RAW_DIR / "sheepgtex" / "plink" / "Muscle"
CATTLE_PLINK_PFX = RAW_DIR / "cattle" / "plink" / "Rumen"


class CompactPlinkLD:
    """Memory-lean (<60 MB RAM) exact pairwise genotype LD r^2 calculator on PLINK .bed/.bim/.fam."""

    _DECODE = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float32)

    def __init__(self, prefix: Path):
        t0 = time.time()
        self.bed_path = str(prefix) + ".bed"
        bim_path = str(prefix) + ".bim"
        fam_path = str(prefix) + ".fam"
        with open(fam_path, "r", encoding="utf-8", errors="replace") as f:
            self.n_samples = sum(1 for ln in f if ln.strip())
        self.bytes_per_snp = (self.n_samples + 3) // 4
        bvals = np.arange(256, dtype=np.uint8)[:, None]
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, :]
        self.lut = self._DECODE[(bvals >> shifts) & 3]

        # Store per-chr sorted int32 pos, int32 bim_row, uint8 a1, uint8 a2
        bmap = {"A": 0, "C": 1, "G": 2, "T": 3}
        tmp = defaultdict(lambda: ([], [], [], []))
        n_rows = 0
        with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, ln in enumerate(f):
                sp = ln.split()
                if len(sp) < 6:
                    continue
                ch = sp[0].replace("chr", "").replace("CHR", "")
                a1 = bmap.get(sp[4], 4)
                a2 = bmap.get(sp[5], 4)
                if a1 < 4 and a2 < 4:
                    p_l, r_l, a1_l, a2_l = tmp[ch]
                    p_l.append(int(sp[3]))
                    r_l.append(idx)
                    a1_l.append(a1)
                    a2_l.append(a2)
                n_rows += 1
        self.by_chr = {}
        for ch, (p_l, r_l, a1_l, a2_l) in tmp.items():
            pa = np.array(p_l, dtype=np.int32)
            ra = np.array(r_l, dtype=np.int32)
            a1a = np.array(a1_l, dtype=np.uint8)
            a2a = np.array(a2_l, dtype=np.uint8)
            o = np.argsort(pa, kind="mergesort")
            self.by_chr[ch] = (pa[o], ra[o], a1a[o], a2a[o])
        self._fh = open(self.bed_path, "rb")
        magic = self._fh.read(3)
        assert magic == b"\x6c\x1b\x01"
        self._cache = {}
        print(f"[LD] Indexed {prefix.name}: {n_rows:,} SNPs x {self.n_samples} samples in {time.time()-t0:.1f}s", flush=True)

    def close(self):
        if getattr(self, "_fh", None):
            self._fh.close()
            self._fh = None

    def find_row(self, ch: str, pos_1b: int) -> int:
        ent = self.by_chr.get(str(ch).replace("chr", ""))
        if ent is None:
            return -1
        pa, ra, _a1, _a2 = ent
        i = int(np.searchsorted(pa, pos_1b, side="left"))
        if i < len(pa) and int(pa[i]) == pos_1b:
            return int(ra[i])
        return -1

    def query_window(self, ch: str, lo_1b: int, hi_1b: int):
        ent = self.by_chr.get(str(ch).replace("chr", ""))
        if ent is None:
            return []
        pa, ra, a1a, a2a = ent
        i0 = int(np.searchsorted(pa, lo_1b, side="left"))
        i1 = int(np.searchsorted(pa, hi_1b, side="right"))
        b_ch = ["A", "C", "G", "T"]
        out = []
        for k in range(i0, i1):
            out.append((int(pa[k]), int(ra[k]), b_ch[int(a1a[k])], b_ch[int(a2a[k])]))
        return out

    def get_geno_and_maf(self, row_idx: int):
        if row_idx in self._cache:
            return self._cache[row_idx]
        self._fh.seek(3 + row_idx * self.bytes_per_snp)
        raw = np.frombuffer(self._fh.read(self.bytes_per_snp), dtype=np.uint8)
        g = self.lut[raw].ravel()[: self.n_samples].copy()
        valid = g[~np.isnan(g)]
        p = float(valid.mean() / 2.0) if len(valid) > 0 else 0.0
        maf = min(p, 1.0 - p)
        if len(self._cache) > 40000:
            self._cache.clear()
        self._cache[row_idx] = (g, maf)
        return g, maf

    def ld_r2(self, row_a: int, row_b: int) -> float:
        if row_a < 0 or row_b < 0:
            return float("nan")
        if row_a == row_b:
            return 1.0
        ga, _ = self.get_geno_and_maf(row_a)
        gb, _ = self.get_geno_and_maf(row_b)
        m = (~np.isnan(ga)) & (~np.isnan(gb))
        if int(m.sum()) < 20:
            return float("nan")
        xa = ga[m] - ga[m].mean()
        xb = gb[m] - gb[m].mean()
        va = float(np.dot(xa, xa))
        vb = float(np.dot(xb, xb))
        if va <= 1e-12 or vb <= 1e-12:
            return 0.0
        r = float(np.dot(xa, xb)) / math.sqrt(va * vb)
        return float(min(1.0, r * r))


def load_cattle_chr_u8_subset(needed_chrs: set) -> dict:
    """Load ONLY needed cattle chromosomes as uint8 arrays (keeps peak RAM <400 MB)."""
    nc_to_str = {f"NC_{37327 + i:06d}.1": str(i) for i in range(1, 30)}
    nc_to_str["NC_037357.1"] = "X"
    lut = np.full(256, 4, dtype=np.uint8)
    for ch_b, val in ((b"A"[0], 0), (b"a"[0], 0), (b"C"[0], 1), (b"c"[0], 1), (b"G"[0], 2), (b"g"[0], 2), (b"T"[0], 3), (b"t"[0], 3)):
        lut[ch_b] = val
    out = {}
    cur = None
    chunks = []
    with gzip.open(CATTLE_FNA_GZ, "rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = lut[np.frombuffer("".join(chunks).encode("ascii"), dtype=np.uint8)]
                    if len(out) == len(needed_chrs):
                        break
                acc = line[1:].split()[0]
                cs = nc_to_str.get(acc)
                cur = cs if cs in needed_chrs else None
                chunks = []
            elif cur is not None:
                chunks.append(line.strip().upper())
        if cur is not None and cur not in out:
            out[cur] = lut[np.frombuffer("".join(chunks).encode("ascii"), dtype=np.uint8)]
    return out


def fetch_snv_window_u8(seq_u8: np.ndarray, pos_1b: int, ref_b: str, alt_b: str, strand: int):
    g0 = pos_1b - 1
    win = seq_u8[g0 - 5500 : g0 + 5501].copy()
    r_tx = BASE_MAP[ref_b if strand == 1 else COMP_BASE[ref_b]]
    a_tx = BASE_MAP[alt_b if strand == 1 else COMP_BASE[alt_b]]
    if strand == -1:
        win = COMP_U8[win[::-1]].copy()
    if int(win[5500]) != r_tx:
        return None, None
    return win, a_tx


def run_single_locus_ism(ens, win_11001: np.ndarray, alt_tx_u8: int, head_idx: int):
    """
    Exact +-45 bp saturation ISM (91 x 3 = 273 alt mutations) + +-150 bp REF/ALT profile.
    All coordinates are strictly on the + transcript strand (5' -> 3', offset 0 = variant base).
    """
    # 1. REF & ALT base profiles on +-150 bp around center (output length 1,001 -> center 500 -> [350:651])
    w_ref = win_11001[None, :].copy()
    w_alt = win_11001[None, :].copy()
    w_alt[0, 5500] = alt_tx_u8
    p_ra, u_ra = ens.score_windows(np.vstack([w_ref, w_alt]), compute_usage=True)
    # p_ra: [2, 1001, 2], u_ra: [2, 1001, 15]
    p_ref_301 = p_ra[0, 350:651, :].T.copy()  # [2, 301]
    p_alt_301 = p_ra[1, 350:651, :].T.copy()  # [2, 301]
    u_ref_301 = u_ra[0, 350:651, head_idx].copy()
    u_alt_301 = u_ra[1, 350:651, head_idx].copy()

    # Locate the exact perturbed splice site peak inside +-500 bp (where |p_alt_any - p_ref_any| or gated |dU| peaks!)
    p_r_any = p_ra[0, :, 0] + p_ra[0, :, 1]
    p_a_any = p_ra[1, :, 0] + p_ra[1, :, 1]
    dp_abs = np.max(np.abs(p_ra[1] - p_ra[0]), axis=1)
    peak_idx = int(np.argmax(dp_abs))  # in [0..1000], relative to center 500
    peak_off = peak_idx - 500

    # 2. Saturation ISM over [-45, +45] (273 single-nt mutations scored via ultra-fast lightcone!)
    ref_seq_91 = win_11001[5500 - 45 : 5500 + 46].copy()
    ism_abs_dp = np.zeros((91, 4), dtype=np.float32)
    ism_ds_max12 = np.zeros((91, 4), dtype=np.float32)
    ism_signed_peak_mod = np.zeros((91, 4), dtype=np.float32)

    mut_wins = []
    mut_alts = []
    mut_meta = []
    mut_rs = []
    mut_re = []
    for k in range(91):
        off = k - 45
        rb = int(ref_seq_91[k])
        for ab in range(4):
            if ab == rb:
                continue
            # Shift window so mutated base is at 5500 -> peak_idx shifts by -off!
            g_off = 5500 + off
            # Need 11,001 window centered at g_off; pad edges with 4 if needed
            w_m = np.full(11001, 4, dtype=np.uint8)
            src_lo = max(0, g_off - 5500)
            src_hi = min(11001, g_off + 5501)
            dst_lo = src_lo - (g_off - 5500)
            w_m[dst_lo : dst_lo + (src_hi - src_lo)] = win_11001[src_lo:src_hi]
            mut_wins.append(w_m)
            mut_alts.append(ab)
            mut_meta.append((k, ab))
            pk_rel = 500 + (peak_off - off)
            mut_rs.append(pk_rel)
            mut_re.append(pk_rel)

    res_ism = ens.score_snvs(
        np.vstack(mut_wins),
        np.array(mut_alts, dtype=np.uint8),
        np.array(mut_rs, dtype=np.int32),
        np.array(mut_re, dtype=np.int32),
    )
    for j, (k, ab) in enumerate(mut_meta):
        ism_abs_dp[k, ab] = res_ism["ds_site_500"][j]
        ism_ds_max12[k, ab] = float(res_ism["ds_usage_500"][j, 1:13].max())
        ism_signed_peak_mod[k, ab] = res_ism["delta_intron_mod"][j, head_idx]

    return {
        "ref_seq_91": ref_seq_91,
        "p_ref_301": p_ref_301,
        "p_alt_301": p_alt_301,
        "u_ref_301": u_ref_301,
        "u_alt_301": u_alt_301,
        "peak_off": int(peak_off),
        "ism_abs_dp_91x4": ism_abs_dp,
        "ism_ds_max12_91x4": ism_ds_max12,
        "ism_signed_peak_mod_91x4": ism_signed_peak_mod,
    }


def run_omia_purified_audit(cattle_ld: CompactPlinkLD, m_full, m_m1):
    """Purify OMIA to 23 true splicing variants, compute exact OviSplice metrics + Bootstrap CI + DeLong/McNemar."""
    t0 = time.time()
    cur_raw = pd.read_csv(OMIA_CURATED_TSV, sep="\t")
    om_c = pd.read_csv(OMIA_DIR / "omia_cattle_9913_variants.csv")
    om_s = pd.read_csv(OMIA_DIR / "omia_sheep_9940_variants.csv")

    eff_map = {}
    pmid_map = {}
    for sp, df_o in (("cattle", om_c), ("sheep", om_s)):
        id_c = [c for c in df_o.columns if "Variant ID" in c][0]
        ef_c = [c for c in df_o.columns if "Variant Effect" in c][0]
        pm_c = [c for c in df_o.columns if "PubMed" in c or "Reference" in c or "Year" in c]
        for r in df_o.to_dict("records"):
            eff_map[(sp, int(r[id_c]))] = str(r.get(ef_c, ""))

    cur_raw["omia_variant_effect"] = [eff_map.get((r.species, int(r.omia_variant_id)), "") for r in cur_raw.itertuples()]
    excluded_df = cur_raw[~cur_raw["omia_variant_effect"].str.lower().str.contains("splic")].copy()
    purified_pos = cur_raw[cur_raw["omia_variant_effect"].str.lower().str.contains("splic")].copy().reset_index(drop=True)
    print(f"[s07:OMIA] Purified OMIA: kept {len(purified_pos)} splicing variants, excluded {len(excluded_df)} regulatory: {excluded_df['locus_id'].tolist()}", flush=True)

    # Parse gene spans from Cattle GTF & Sheep GTF for same-gene control sampling (same-gene +/-5 kb window)
    c_needed = set(purified_pos.loc[purified_pos["species"] == "cattle", "CHR"].astype(str))
    c_u8 = load_cattle_chr_u8_subset(c_needed)
    fa_sh = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))

    nc_to_cstr = {f"NC_{37327 + i:06d}.1": str(i) for i in range(1, 30)}
    nc_to_cstr["NC_037357.1"] = "X"
    re_gn = re.compile(r'(?:gene_name|gene) "([^"]+)"')
    c_gene_span = {}
    with gzip.open(CATTLE_GTF_GZ, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            f = ln.split("\t", 8)
            if f[2] != "gene":
                continue
            cs = nc_to_cstr.get(f[0])
            if cs not in c_needed:
                continue
            m = re_gn.search(f[8])
            if m:
                c_gene_span[m.group(1).upper()] = (cs, int(f[3]), int(f[4]))

    s_gene_span = {}
    with gzip.open(SHEEP_GTF_GZ, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            f = ln.split("\t", 8)
            if f[2] != "gene":
                continue
            ch = f[0].replace("chr", "")
            m = re_gn.search(f[8])
            if m:
                s_gene_span[m.group(1).upper()] = (ch, int(f[3]), int(f[4]))

    # Load Sheep common SNPs (MAF>=0.05) from atlas
    with np.load(ATLAS_NPZ, allow_pickle=False) as sz:
        b_arr = np.array(["A", "C", "G", "T"])
        s_chr_a = sz["chrom_id"].astype(np.int32)
        s_pos_a = sz["pos"].astype(np.int32)
        s_a1_a = b_arr[sz["ref_u8"]]
        s_a2_a = b_arr[sz["alt_u8"]]
        s_maf_a = sz["maf"].astype(np.float32)

    rng = np.random.default_rng(20261119)
    rows = []
    wins = []
    alts = []
    b_ch = ["A", "C", "G", "T"]

    for r in purified_pos.itertuples():
        sp = str(r.species)
        ch = str(r.CHR)
        pos_p = int(r.POS)
        st = int(r.strand)
        gsym = str(r.gene_symbol)

        if sp == "cattle":
            w, a_tx = fetch_snv_window_u8(c_u8[ch], pos_p, str(r.REF), str(r.ALT), st)
        else:
            seq_s = fa_sh.fetch_u8(ch, pos_p - 1 - 5500, pos_p - 1 + 5501)
            if st == -1:
                seq_s = COMP_U8[seq_s[::-1]].copy()
            w = seq_s
            a_tx = BASE_MAP[str(r.ALT) if st == 1 else COMP_BASE[str(r.ALT)]]
        assert w is not None
        wins.append(w); alts.append(a_tx)
        rows.append({
            "locus_id": r.locus_id, "species": sp, "gene_symbol": gsym, "label": 1,
            "mechanism_category": r.mechanism_category, "CHR": ch, "POS": pos_p,
            "sheep_fold": int(r.sheep_fold),
        })

        # Same-gene common benign controls (MAF >= 0.05, up to 20)
        g_info = (c_gene_span if sp == "cattle" else s_gene_span).get(gsym.upper())
        gs = g_info[1] - 5000 if g_info else pos_p - 25000
        ge = g_info[2] + 5000 if g_info else pos_p + 25000

        added = 0
        if sp == "cattle":
            cands = cattle_ld.query_window(ch, gs, ge)
            if len(cands) < 10:
                cands = cattle_ld.query_window(ch, gs - 100000, ge + 100000)
            rng.shuffle(cands)
            seq_c = c_u8[ch]
            for pv, ridx, a1, a2 in cands:
                if pv == pos_p or pv < 6000 or pv > len(seq_c) - 6000:
                    continue
                _, maf = cattle_ld.get_geno_and_maf(ridx)
                if maf < 0.05:
                    continue
                fb = b_ch[int(seq_c[pv - 1])] if int(seq_c[pv - 1]) < 4 else "N"
                if fb == a1:
                    rb, ab = a1, a2
                elif fb == a2:
                    rb, ab = a2, a1
                else:
                    continue
                wc, atx = fetch_snv_window_u8(seq_c, pv, rb, ab, st)
                if wc is None:
                    continue
                wins.append(wc); alts.append(atx)
                rows.append({
                    "locus_id": r.locus_id, "species": sp, "gene_symbol": gsym, "label": 0,
                    "mechanism_category": r.mechanism_category, "CHR": ch, "POS": pv,
                    "sheep_fold": 0,
                })
                added += 1
                if added >= 20:
                    break
        else:
            cid_int = 27 if ch == "X" else int(ch)
            m_bg = np.flatnonzero((s_chr_a == cid_int) & (s_pos_a >= gs) & (s_pos_a <= ge) & (s_pos_a != pos_p) & (s_maf_a >= 0.05))
            if len(m_bg) < 10:
                m_bg = np.flatnonzero((s_chr_a == cid_int) & (s_pos_a >= gs - 100000) & (s_pos_a <= ge + 100000) & (s_pos_a != pos_p) & (s_maf_a >= 0.05))
            rng.shuffle(m_bg)
            for idx_s in m_bg.tolist():
                pv = int(s_pos_a[idx_s])
                rb, ab = str(s_a1_a[idx_s]), str(s_a2_a[idx_s])
                wc = fa_sh.fetch_u8(ch, pv - 1 - 5500, pv - 1 + 5501)
                if int(wc[5500]) != BASE_MAP[rb]:
                    if int(wc[5500]) == BASE_MAP[ab]:
                        rb, ab = ab, rb
                    else:
                        continue
                if st == -1:
                    wc = COMP_U8[wc[::-1]].copy()
                    atx = BASE_MAP[COMP_BASE[ab]]
                else:
                    atx = BASE_MAP[ab]
                wins.append(wc); alts.append(atx)
                rows.append({
                    "locus_id": r.locus_id, "species": sp, "gene_symbol": gsym, "label": 0,
                    "mechanism_category": r.mechanism_category, "CHR": ch, "POS": pv,
                    "sheep_fold": int(r.sheep_fold),
                })
                added += 1
                if added >= 20:
                    break

    fa_sh.close()
    del c_u8
    gc.collect()

    df_all = pd.DataFrame(rows)
    w_mat = np.vstack(wins).astype(np.uint8)
    a_arr = np.array(alts, dtype=np.uint8)
    print(f"[s07:OMIA] Scoring {len(df_all)} purified OMIA records (23 POS + {(df_all['label']==0).sum()} same-gene CTRL)...", flush=True)

    res_full = m_full.score_snvs(w_mat, a_arr)
    res_m1 = m_m1.score_snvs(w_mat, a_arr)

    s_max12 = res_full["ds_usage_500"][:, 1:13].max(axis=1).copy()
    s_pan = res_full["ds_usage_500"][:, 0].copy()
    s_site = res_full["ds_site_500"].copy()

    # Strict OOF scoring for Sheep folds 4 & 5!
    for fk in sorted(set(df_all["sheep_fold"]) - {0}):
        m_fk = np.flatnonzero(df_all["sheep_fold"].to_numpy() == fk)
        ckpts_fk = [CKPT_DIR / f"ovisplice_fold{fk}_seed{s}.pt" for s in (42, 1042)]
        ens_fk, _ = load_ensemble(ckpts_fk, backend=BACKEND)
        rfk = ens_fk.score_snvs(w_mat[m_fk], a_arr[m_fk])
        s_max12[m_fk] = rfk["ds_usage_500"][:, 1:13].max(axis=1)
        s_pan[m_fk] = rfk["ds_usage_500"][:, 0]
        s_site[m_fk] = rfk["ds_site_500"]
        del ens_fk

    df_all["Ovi_max12"] = np.round(s_max12, 6)
    df_all["Ovi_pan"] = np.round(s_pan, 6)
    df_all["Ovi_site"] = np.round(s_site, 6)
    df_all["Ovi_composite"] = np.maximum(df_all["Ovi_max12"], df_all["Ovi_site"])
    df_all["Ovi_SiteOnly"] = np.round(res_m1["ds_site_500"], 6)

    # Record per-locus same-gene control count & Ovi_max12 rank for all 23 purified variants
    ctrl_cnt = df_all[df_all["label"] == 0].groupby("locus_id").size().to_dict()
    rank_info = {}
    for lid, g in df_all.groupby("locus_id", sort=False):
        gv = g["Ovi_max12"].to_numpy(dtype=np.float64)
        pi = int(np.flatnonzero(g["label"].to_numpy() == 1)[0])
        pv = gv[pi]
        ng = int((gv > pv).sum()); ne = int((gv == pv).sum())
        rank_info[lid] = (ctrl_cnt.get(lid, 0), ng + 0.5 * (ne + 1.0), 1 if (ng == 0 and ne == 1 and ctrl_cnt.get(lid, 0) >= 5) else 0)

    purified_pos["n_same_gene_controls"] = [rank_info[lid][0] for lid in purified_pos["locus_id"]]
    purified_pos["ovi_max12_within_gene_rank"] = [rank_info[lid][1] for lid in purified_pos["locus_id"]]
    purified_pos["ovi_max12_top1_hit"] = [rank_info[lid][2] for lid in purified_pos["locus_id"]]
    purified_pos.to_csv(OUT_DIR / "v3_omia_purified_23_variants.tsv", sep="\t", index=False)
    excluded_df.to_csv(OUT_DIR / "v3_omia_excluded_3_regulatory.tsv", sep="\t", index=False)

    # Compute purified metrics across ALL 4 transparent strata + Canonical Core, with 2,000-draw Bootstrap 95% CI & DeLong/McNemar!
    strata = [
        ("All_Purified_Splicing_23",     df_all),
        ("Cattle_ZeroShot_Splicing_21",  df_all[df_all["species"] == "cattle"]),
        ("Sheep_Strict_OOF_Splicing_2",  df_all[df_all["species"] == "sheep"]),
        ("NonCanonical_Splicing_9",      df_all[df_all["mechanism_category"] != "Canonical_Core_1_2bp"]),
        ("Canonical_Core_1_2bp_14",      df_all[df_all["mechanism_category"] == "Canonical_Core_1_2bp"]),
    ]
    ovi_models = [
        ("OviSplice_max12",     "Ovi_max12"),
        ("OviSplice_pan",       "Ovi_pan"),
        ("OviSplice_site",      "Ovi_site"),
        ("OviSplice_composite", "Ovi_composite"),
        ("OviSplice_SiteOnly",   "Ovi_SiteOnly"),
    ]
    rng_b = np.random.default_rng(20261119)
    m_rows = []
    for sname, sdf in strata:
        y = sdf["label"].to_numpy(dtype=np.int64)
        loci_g = [g for _, g in sdf.groupby("locus_id", sort=False) if (g["label"] == 1).sum() == 1 and (g["label"] == 0).sum() >= 5]
        s_ref = sdf["Ovi_max12"].to_numpy(dtype=np.float64)
        ref_hits_b = np.array([
            int((g["Ovi_max12"].to_numpy() > g.loc[g["label"] == 1, "Ovi_max12"].iloc[0]).sum()) == 0
            for g in loci_g
        ], dtype=bool)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)

        for mn, col in ovi_models:
            sm = sdf[col].to_numpy(dtype=np.float64)
            auc, ap = tie_invariant_auroc_auprc(y, sm)
            # Stratified 1,000-draw bootstrap 95% CI
            b_aucs, b_aps = [], []
            for _ in range(1000):
                pi_b = rng_b.choice(pos_idx, size=len(pos_idx), replace=True)
                ni_b = rng_b.choice(neg_idx, size=len(neg_idx), replace=True)
                idx_b = np.concatenate([pi_b, ni_b])
                a_b, p_b = tie_invariant_auroc_auprc(y[idx_b], sm[idx_b])
                b_aucs.append(a_b); b_aps.append(p_b)
            dl = paired_delong_test(y, s_ref, sm) if col != "Ovi_max12" else {"p": 1.0}

            t1s, rrs, hits_b = [], [], []
            for g in loci_g:
                gv = g[col].to_numpy(dtype=np.float64)
                pv = gv[int(np.flatnonzero(g["label"].to_numpy() == 1)[0])]
                ng = int((gv > pv).sum()); ne = int((gv == pv).sum())
                t1s.append((1.0 / ne) if ng == 0 else 0.0)
                rrs.append(1.0 / (ng + 0.5 * (ne + 1.0)))
                hits_b.append(ng == 0 and ne == 1)
            mc = paired_mcnemar_exact(ref_hits_b, np.array(hits_b, dtype=bool))["p_mcnemar"] if (col != "Ovi_max12" and len(loci_g) > 0) else 1.0
            pv_all = sm[pos_idx]
            m_rows.append({
                "stratum": sname, "model": mn,
                "n_pos": int(len(pos_idx)), "n_ctrl": int(len(neg_idx)), "n_rankable_loci": len(loci_g),
                "auroc": round(auc, 4), "auroc_ci_lo": round(float(np.percentile(b_aucs, 2.5)), 4), "auroc_ci_hi": round(float(np.percentile(b_aucs, 97.5)), 4),
                "auprc": round(ap, 4), "auprc_ci_lo": round(float(np.percentile(b_aps, 2.5)), 4), "auprc_ci_hi": round(float(np.percentile(b_aps, 97.5)), 4),
                "top1_rate": round(float(np.mean(t1s)), 4) if t1s else None,
                "mrr": round(float(np.mean(rrs)), 4) if rrs else None,
                "pos_median": round(float(np.median(pv_all)), 4),
                "recall_ge02": round(float((pv_all >= 0.20).mean()), 4),
                "delong_p_vs_max12": dl["p"], "mcnemar_p_vs_max12": mc,
            })

    # Also compute exact 23-variant positive-only summary for ALL external baselines (HP, SpliceAI, MaxEntScan)
    ext_pos_rows = []
    ext_specs = [
        ("OviSplice_max12", "Ovi_strict_max12", 0.20),
        ("OviSplice_site", "Ovi_strict_site", 0.20),
        ("OviSplice_SiteOnly", "Ovi_SiteOnly", 0.20),
        ("Human_Pangolin_usage", "HP_ds_usage_max4_500", 0.20),
        ("Human_Pangolin_psplice", "HP_ds_psplice_max4_500", 0.20),
        ("SpliceAI_masked", "SAI_masked_ds_500", 0.20),
        ("SpliceAI_unmasked", "SAI_unmasked_ds_500", 0.20),
        ("MaxEntScan_core", "maxent_core_abs", 2.0),
    ]
    for sname, sub_p in (
        ("All_Purified_Splicing_23", purified_pos),
        ("Cattle_ZeroShot_Splicing_21", purified_pos[purified_pos["species"] == "cattle"]),
        ("Sheep_Strict_OOF_Splicing_2", purified_pos[purified_pos["species"] == "sheep"]),
        ("NonCanonical_Splicing_9", purified_pos[purified_pos["mechanism_category"] != "Canonical_Core_1_2bp"]),
    ):
        for mn, col, thr in ext_specs:
            vals = sub_p[col].to_numpy(dtype=np.float64)
            ext_pos_rows.append({
                "stratum": sname, "model": mn, "n_pos": len(vals),
                "pos_median": round(float(np.median(vals)), 4),
                "recall_ge_thr": round(float((vals >= thr).mean()), 4),
                "n_recalled": int((vals >= thr).sum()),
            })

    df_m = pd.DataFrame(m_rows)
    df_m.to_csv(OUT_DIR / "v3_omia_purified_23_metrics.tsv", sep="\t", index=False)
    df_ext = pd.DataFrame(ext_pos_rows)
    df_ext.to_csv(OUT_DIR / "v3_omia_purified_23_allmodels_pos_recall.tsv", sep="\t", index=False)
    print(f"[s07:OMIA] Completed purified OMIA audit in {time.time()-t0:.1f}s!", flush=True)
    print(df_m[df_m["model"] == "OviSplice_max12"][["stratum", "n_pos", "n_ctrl", "n_rankable_loci", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "top1_rate", "mrr", "recall_ge02"]].to_string(index=False))
    return {"purified_metrics": m_rows, "ext_pos_recall": ext_pos_rows, "excluded_regulatory": excluded_df[["locus_id", "gene_symbol", "dist", "omia_variant_effect", "Ovi_strict_max12", "HP_ds_usage_max4_500"]].to_dict(orient="records")}


def dedup_1mb_loci(df: pd.DataFrame, trait_col: str, chr_col: str, pos_col: str, score_col: str) -> pd.DataFrame:
    """Greedy clumping within +-1 Mb per Trait (retaining highest PP.H4 representative per physical locus)."""
    df_s = df.sort_values(score_col, ascending=False)
    kept = []
    seen = defaultdict(list)  # (trait, chr) -> list of kept lead positions
    for r in df_s.to_dict("records"):
        key = (str(r[trait_col]), str(r[chr_col]))
        p = int(r[pos_col])
        if any(abs(p - kp) <= 1_000_000 for kp in seen[key]):
            continue
        seen[key].append(p)
        kept.append(r)
    return pd.DataFrame(kept)


def run_sheep_gwas_ld_audit(sheep_ld: CompactPlinkLD, m_full):
    """Sheep GWAS-sQTL: strict deduplication + exact PLINK LD r^2 + SuSiE CS + +-500bp target-intron verification + COL7A1 ISM fix."""
    t0 = time.time()
    gc_df = load_sheep_gwas_coloc_loci(SHEEP_COLOC_GZ, COLOC_LOCI_TSV)
    gc_df = gc_df[gc_df["lead_pos"] > 0].copy()
    raw_n = len(gc_df)

    # Deduplicate 1: unique (Trait, gene_symbol) keeping highest PP_H4
    dedup_tg = gc_df.sort_values("PP_H4", ascending=False).drop_duplicates(["Trait", "gene_symbol"]).copy()
    # Deduplicate 2: +-1 Mb independent physical loci per Trait
    dedup_1mb = dedup_1mb_loci(dedup_tg, "Trait", "chrom", "lead_pos", "PP_H4")
    print(f"[s07:SheepGWAS] Raw={raw_n} -> Unique Trait*Gene={len(dedup_tg)} -> Independent +-1Mb loci={len(dedup_1mb)}", flush=True)

    # Load SuSiE credible sets for matching (tissue, pheno_id)
    need_tp = set(zip(dedup_tg["tissue"].astype(str), dedup_tg["phenotype_id"].astype(str)))
    susie_map = defaultdict(dict)  # (tissue, pheno_id) -> {pos_1b: (cs, pip, beta)}
    with gzip.open(RAW_DIR / "sheepgtex" / "SheepGTEx_v0.finemapped_sQTL.txt.gz", "rt", encoding="utf-8") as fh:
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

    # Index Sheep 624k SNV atlas by chrom_id -> sorted pos_1b for fast +-500bp lookup
    with np.load(ATLAS_NPZ, allow_pickle=False) as az:
        a_cid = az["chrom_id"].astype(np.int32)
        a_pos = az["pos"].astype(np.int32)
        a_ref = az["ref_u8"]
        a_alt = az["alt_u8"]
        a_str = az["strand"]
        a_maf = az["maf"].astype(np.float32)
        a_m12 = az["ds_usage_max12_500"].astype(np.float32)
        a_site = az["ds_site_500"].astype(np.float32)
        a_u15 = az["ds_usage_15_500"].astype(np.float32)

    b_ch = ["A", "C", "G", "T"]
    atlas_by_chr = {}
    for c in range(1, 28):
        m = np.flatnonzero(a_cid == c)
        if len(m) > 0:
            o = np.argsort(a_pos[m], kind="mergesort")
            ms = m[o]
            atlas_by_chr[c] = (a_pos[ms], ms)

    def eval_sheep_loci_df(loci_df: pd.DataFrame):
        out_rows = []
        for r in loci_df.to_dict("records"):
            cid = int(r["chrom_id"])
            ch_s = "X" if cid == 27 else str(cid)
            i_s, i_e = int(r["i_s"]), int(r["i_e"])
            lpos = int(r["lead_pos"])
            hi = int(r["head_idx"])
            tp_k = (str(r["tissue"]), str(r["phenotype_id"]))
            cs_dict = susie_map.get(tp_k, {})

            # Lookup lead SNV in atlas & PLINK
            lead_row = sheep_ld.find_row(ch_s, lpos)
            lead_m12 = np.nan
            lead_site = np.nan
            if cid in atlas_by_chr:
                pa, ms = atlas_by_chr[cid]
                i0 = int(np.searchsorted(pa, lpos, side="left"))
                if i0 < len(pa) and int(pa[i0]) == lpos:
                    ai = int(ms[i0])
                    lead_m12 = float(a_m12[ai])
                    lead_site = float(a_site[ai])

            lead_cs_ent = cs_dict.get(lpos)
            lead_cs_id = lead_cs_ent[0] if lead_cs_ent is not None else ""

            # Gather all atlas SNVs within +-500 bp of target intron endpoints i_s or i_e!
            cands_linked = []
            cands_all_500 = []
            if cid in atlas_by_chr:
                pa, ms = atlas_by_chr[cid]
                for ep in (i_s, i_e):
                    j0 = int(np.searchsorted(pa, ep - 500, side="left"))
                    j1 = int(np.searchsorted(pa, ep + 500, side="right"))
                    for jj in range(j0, j1):
                        ai = int(ms[jj])
                        pv = int(a_pos[ai])
                        d_intron = min(abs(pv - i_s), abs(pv - i_e))
                        if d_intron > 500:
                            continue
                        sc_m12 = float(a_m12[ai])
                        sc_site = float(a_site[ai])
                        crow = sheep_ld.find_row(ch_s, pv)
                        r2 = sheep_ld.ld_r2(lead_row, crow)
                        cs_ent = cs_dict.get(pv)
                        in_cs = 1 if cs_ent is not None else 0
                        pip_v = cs_ent[1] if cs_ent is not None else 0.0
                        cs_id = cs_ent[0] if cs_ent is not None else ""
                        item = (sc_m12, sc_site, pv, b_ch[int(a_ref[ai])], b_ch[int(a_alt[ai])], int(a_str[ai]), float(a_maf[ai]), d_intron, abs(pv - lpos), r2, in_cs, pip_v, cs_id)
                        cands_all_500.append(item)
                        if (pv == lpos) or (not math.isnan(r2) and r2 >= 0.60) or (in_cs == 1 and lead_cs_id != "" and cs_id == lead_cs_id):
                            cands_linked.append(item)

            if not cands_linked:
                continue
            cands_linked.sort(key=lambda x: (-x[0], -x[1]))
            best = cands_linked[0]
            sc_m12, sc_site, bpos, bref, balt, bstr, bmaf, dint, dlead, br2, bin_cs, bpip, bcs = best
            is_neutral_lead = (not math.isnan(lead_m12)) and (lead_m12 < 0.05)
            reranked = 1 if (sc_m12 >= 0.10 and is_neutral_lead and bpos != lpos) else 0
            out_rows.append({
                "species": "Sheep", "Trait": r["Trait"], "gene_symbol": r["gene_symbol"],
                "tissue": r["tissue"], "head_idx": hi, "PP_H4": round(float(r["PP_H4"]), 4),
                "pheno_id": r["phenotype_id"], "CHR": ch_s, "i_s": i_s, "i_e": i_e,
                "lead_variant": r["LEAD_VARIANT"], "lead_pos": lpos,
                "lead_ds_usage_max12": round(lead_m12, 4) if not math.isnan(lead_m12) else None,
                "cand_variant": f"{ch_s}_{bpos}_{bref}>{balt}", "cand_pos": bpos, "cand_ref": bref, "cand_alt": balt,
                "cand_strand": bstr, "cand_maf": round(bmaf, 4),
                "dist_to_target_intron_bp": dint, "dist_to_lead_bp": dlead,
                "ld_r2_with_lead": round(br2, 4) if not math.isnan(br2) else None,
                "in_susie_cs": bin_cs, "susie_cs": bcs, "susie_pip": round(bpip, 4),
                "cand_ds_usage_max12": round(sc_m12, 4), "cand_ds_site": round(sc_site, 4),
                "n_linked_500bp_cands": len(cands_linked), "strict_reranked_over_neutral_lead": reranked,
            })
        return pd.DataFrame(out_rows).sort_values(["strict_reranked_over_neutral_lead", "cand_ds_usage_max12", "PP_H4"], ascending=[False, False, False])

    df_tg = eval_sheep_loci_df(dedup_tg)
    df_1mb = eval_sheep_loci_df(dedup_1mb)
    df_tg.to_csv(OUT_DIR / "v3_sheep_gwas_coloc_dedup_ld_verified.tsv", sep="\t", index=False)
    df_1mb.to_csv(OUT_DIR / "v3_sheep_gwas_coloc_1mb_ld_verified.tsv", sep="\t", index=False)

    # Recompute fixed saturation ISM for Sheep DDX59 & COL7A1 (fixing Fig S6c white-at-offset-0 paradox!)
    fa_sh = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))
    sheep_ism = {}
    for label_k, ch_s, pos_v, ref_b, alt_b, st_v, hi_v in (
        ("DDX59", "12", 78727889, "A", "T", 1, 7),
        ("COL7A1", "19", 51350633, "C", "T", -1, 6),
    ):
        w = fa_sh.fetch_u8(ch_s, pos_v - 1 - 5500, pos_v - 1 + 5501)
        if st_v == -1:
            w = COMP_U8[w[::-1]].copy()
            atx = BASE_MAP[COMP_BASE[alt_b]]
        else:
            atx = BASE_MAP[alt_b]
        ism_d = run_single_locus_ism(m_full, w, atx, hi_v)
        for k2, v2 in ism_d.items():
            sheep_ism[f"{label_k}_{k2}"] = np.array(v2) if np.isscalar(v2) else v2
    fa_sh.close()
    np.savez_compressed(OUT_DIR / "v3_sheep_gwas_ism_fixed.npz", **sheep_ism)

    n_tg_ge10 = int((df_tg["cand_ds_usage_max12"] >= 0.10).sum()) if len(df_tg) else 0
    n_tg_rr = int(df_tg["strict_reranked_over_neutral_lead"].sum()) if len(df_tg) else 0
    n_1mb_ge10 = int((df_1mb["cand_ds_usage_max12"] >= 0.10).sum()) if len(df_1mb) else 0
    n_1mb_rr = int(df_1mb["strict_reranked_over_neutral_lead"].sum()) if len(df_1mb) else 0
    print(f"[s07:SheepGWAS] Done in {time.time()-t0:.1f}s! Trait*Gene linked={len(df_tg)} (>=0.10: {n_tg_ge10}, reranked={n_tg_rr}); 1Mb linked={len(df_1mb)} (>=0.10: {n_1mb_ge10}, reranked={n_1mb_rr})", flush=True)
    if len(df_tg):
        print(df_tg.head(10)[["gene_symbol", "Trait", "tissue", "PP_H4", "lead_ds_usage_max12", "cand_variant", "dist_to_target_intron_bp", "dist_to_lead_bp", "ld_r2_with_lead", "in_susie_cs", "cand_ds_usage_max12", "cand_ds_site"]].to_string(index=False))
    return {
        "raw_coloc_rows_h50": raw_n,
        "unique_trait_gene_loci": len(dedup_tg),
        "independent_1mb_loci": len(dedup_1mb),
        "trait_gene_with_linked_500bp_cand": len(df_tg),
        "trait_gene_linked_ds_ge010": n_tg_ge10,
        "trait_gene_strict_reranked": n_tg_rr,
        "indep_1mb_with_linked_500bp_cand": len(df_1mb),
        "indep_1mb_linked_ds_ge010": n_1mb_ge10,
        "indep_1mb_strict_reranked": n_1mb_rr,
        "top10": df_tg.head(10).to_dict(orient="records"),
    }


def run_cattle_gwas_ld_audit(cattle_ld: CompactPlinkLD, m_full):
    """Cattle GWAS-sQTL: drop lead_pos=0, strict dedup, exact PLINK LD r^2 + SuSiE CS + +-500bp target-intron + unified ds_usage_max12 + ISM."""
    t0 = time.time()
    # Map cattle tissues to head_idx

    # Load Cattle gene coordinates from RefSeq GTF
    nc_to_cstr = {f"NC_{37327 + i:06d}.1": str(i) for i in range(1, 30)}
    nc_to_cstr["NC_037357.1"] = "X"
    re_gn = re.compile(r'(?:gene_name|gene) "([^"]+)"')
    chr_genes = defaultdict(list)
    with gzip.open(CATTLE_GTF_GZ, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if not ln or ln[0] == "#":
                continue
            f = ln.split("\t", 8)
            if f[2] != "gene" or f[6] not in ("+", "-"):
                continue
            cs = nc_to_cstr.get(f[0])
            if not cs:
                continue
            m = re_gn.search(f[8])
            if m:
                chr_genes[cs].append((int(f[3]), int(f[4]), 1 if f[6] == "+" else -1, m.group(1)))
    for cs in chr_genes:
        chr_genes[cs].sort()

    def resolve_gene(ch: str, mid: int):
        for gs, ge, st, sym in chr_genes.get(ch, ()):
            if gs <= mid <= ge:
                return sym, st
        return f"chr{ch}_{mid//100000}00k", 1

    # Load lead sQTL SNPs from molGenes/sGene.txt.gz
    lead_map = {}
    with tarfile.open(CATTLEGTEX_DIR / "molGenes_CattleGTEx_phase1.tar.gz", "r:gz") as tf:
        with gzip.GzipFile(fileobj=tf.extractfile("molGenes/sGene.txt.gz")) as gz:
            gz.readline()
            for ln in gz:
                sp = ln.decode("utf-8", "replace").rstrip("\n").split("\t")
                if len(sp) >= 16:
                    lead_map[(sp[15], sp[2])] = sp[7]  # chr_pos_ALT_REF

    coloc = pd.read_csv(CATTLEGTEX_DIR / "total_sQTL_coloc.csv")
    if "X" in coloc.columns:
        coloc = coloc[coloc["X"] == "PP.H4"].copy()
    coloc = coloc[coloc["tissue_name"].isin(CATTLE_TISSUE_MAP.keys())].copy()
    n_h50_raw = int((coloc["res.summary"] >= 0.50).sum())
    coloc_h70 = coloc[coloc["res.summary"] >= 0.70].copy()
    n_h70_raw = len(coloc_h70)

    # Parse coordinates, resolve gene, require valid lead_pos > 0 (eliminating all UBE2O lead_pos=0 dirty rows!)
    parsed = []
    n_dropped_no_lead = 0
    for r in coloc_h70.to_dict("records"):
        tis = str(r["tissue_name"])
        pheno = str(r["gene_id"])
        parts = pheno.split(":")
        if len(parts) < 4 or parts[0] not in chr_genes:
            continue
        ch = parts[0]
        i_s, i_e = int(parts[1]), int(parts[2])
        gsym, st = resolve_gene(ch, (i_s + i_e) // 2)
        lvid = lead_map.get((tis, pheno))
        if not lvid:
            n_dropped_no_lead += 1
            continue
        vp = lvid.split("_")
        if len(vp) < 4 or vp[0] != ch or (not vp[1].isdigit()) or len(vp[2]) != 1 or len(vp[3]) != 1:
            n_dropped_no_lead += 1
            continue
        lpos = int(vp[1])
        parsed.append({
            "tissue": tis, "head_idx": CATTLE_TISSUE_MAP[tis][0], "organ_system": CATTLE_TISSUE_MAP[tis][1],
            "Trait": str(r["traits"]), "PP_H4": float(r["res.summary"]), "pheno_id": pheno,
            "CHR": ch, "i_s": i_s, "i_e": i_e, "gene_symbol": gsym, "strand_hint": st,
            "lead_vid": lvid, "lead_pos": lpos, "lead_a_raw": vp[2], "lead_r_raw": vp[3],
        })
    df_p = pd.DataFrame(parsed)
    # Strict deduplication: (1) unique Trait x Gene, (2) +-1 Mb independent loci per Trait
    dedup_tg = df_p.sort_values("PP_H4", ascending=False).drop_duplicates(["Trait", "gene_symbol"]).copy()
    dedup_1mb = dedup_1mb_loci(dedup_tg, "Trait", "CHR", "lead_pos", "PP_H4")
    print(f"[s07:CattleGWAS] Raw H50={n_h50_raw}, H70={n_h70_raw} (dropped_no_lead={n_dropped_no_lead}) -> Unique Trait*Gene={len(dedup_tg)} -> Independent +-1Mb={len(dedup_1mb)}", flush=True)

    # Load Cattle SuSiE credible sets matching our deduplicated (tissue, pheno_id)
    need_tp = set(zip(dedup_tg["tissue"].astype(str), dedup_tg["pheno_id"].astype(str)))
    susie_map = defaultdict(dict)
    with tarfile.open(CATTLEGTEX_DIR / "finemapping_CattleGTEx_phase1.tar.gz", "r:gz") as tf:
        with gzip.GzipFile(fileobj=tf.extractfile("finemapping/sQTL_finemapped_credible.gz")) as gz:
            gz.readline()
            for ln in gz:
                sp = ln.decode("utf-8", "replace").rstrip("\n").split("\t")
                if len(sp) >= 11:
                    k = (sp[0], sp[1])
                    if k in need_tp:
                        p1 = int(sp[6])
                        pip_v = float(sp[4])
                        prev = susie_map[k].get(p1)
                        if prev is None or pip_v > prev[1]:
                            susie_map[k][p1] = (sp[3], pip_v, float(sp[10]))

    # Step 1: For all deduplicated Trait x Gene loci, find all +-500bp target-intron SNVs passing LD r^2 >= 0.60 OR in_susie_cs == 1
    locus_jobs = []
    snvs_to_score = {}  # (ch, pos_1b, a1, a2, st) -> idx
    needed_chrs = set()
    for r in dedup_tg.to_dict("records"):
        ch = r["CHR"]; i_s = r["i_s"]; i_e = r["i_e"]; lpos = r["lead_pos"]
        cs_dict = susie_map.get((r["tissue"], r["pheno_id"]), {})
        lead_cs_ent = cs_dict.get(lpos)
        lead_cs_id = lead_cs_ent[0] if lead_cs_ent is not None else ""
        lead_row = cattle_ld.find_row(ch, lpos)
        linked = []
        seen_p = set()
        for ep in (i_s, i_e):
            for pv, ridx, a1, a2 in cattle_ld.query_window(ch, ep - 500, ep + 500):
                if pv in seen_p:
                    continue
                seen_p.add(pv)
                d_int = min(abs(pv - i_s), abs(pv - i_e))
                if d_int > 500:
                    continue
                _, maf = cattle_ld.get_geno_and_maf(ridx)
                if maf < 0.01:
                    continue
                r2 = cattle_ld.ld_r2(lead_row, ridx)
                cs_ent = cs_dict.get(pv)
                in_cs = 1 if cs_ent is not None else 0
                cs_id = cs_ent[0] if cs_ent is not None else ""
                if (pv == lpos) or (not math.isnan(r2) and r2 >= 0.60) or (in_cs == 1 and lead_cs_id != "" and cs_id == lead_cs_id):
                    linked.append((pv, a1, a2, maf, d_int, r2, in_cs, cs_ent[1] if cs_ent else 0.0, cs_id))
        if not linked:
            continue
        needed_chrs.add(ch)
        r["linked_cands"] = linked
        locus_jobs.append(r)

    print(f"[s07:CattleGWAS] Found {len(locus_jobs)} unique Trait*Gene loci with LD>=0.6/SuSiE-CS candidates inside +-500bp target intron; loading {len(needed_chrs)} cattle chrs...", flush=True)
    c_u8 = load_cattle_chr_u8_subset(needed_chrs)
    b_ch = ["A", "C", "G", "T"]

    # Resolve intron canonical strand & pack unique SNVs to score with OviSplice OviSplice_full_ens3
    win_list = []
    alt_list = []
    snv_key_to_i = {}
    snv_ref_alt = {}

    def register_snv(ch: str, pv: int, a1: str, a2: str, st: int):
        seq = c_u8[ch]
        if pv < 6000 or pv > len(seq) - 6000:
            return -1, None, None
        fb = b_ch[int(seq[pv - 1])] if int(seq[pv - 1]) < 4 else "N"
        if fb == a1:
            rb, ab = a1, a2
        elif fb == a2:
            rb, ab = a2, a1
        else:
            return -1, None, None
        key = (ch, pv, rb, ab, st)
        if key not in snv_key_to_i:
            w, atx = fetch_snv_window_u8(seq, pv, rb, ab, st)
            if w is None:
                return -1, None, None
            snv_key_to_i[key] = len(win_list)
            win_list.append(w)
            alt_list.append(atx)
            snv_ref_alt[key] = (rb, ab)
        return snv_key_to_i[key], rb, ab

    resolved_jobs = []
    for r in locus_jobs:
        ch = r["CHR"]; i_s = r["i_s"]; i_e = r["i_e"]; seq = c_u8[ch]
        st = r["strand_hint"]
        if 3 <= i_s and i_e <= len(seq) - 3:
            # Check canonical intron dinucleotides
            d1 = (int(seq[i_s]), int(seq[i_s + 1]))
            d2 = (int(seq[i_e - 3]), int(seq[i_e - 2]))
            if (d1, d2) in (((2, 3), (0, 2)), ((2, 1), (0, 2)), ((0, 3), (0, 1))):
                st = 1
            elif (d1, d2) in (((1, 3), (0, 1)), ((1, 3), (2, 1)), ((2, 3), (0, 3))):
                st = -1
        lead_ci, lref, lalt = register_snv(ch, r["lead_pos"], r["lead_r_raw"], r["lead_a_raw"], st)
        c_entries = []
        for pv, a1, a2, maf, d_int, r2, in_cs, pip_v, cs_id in r["linked_cands"]:
            ci, crb, cab = register_snv(ch, pv, a1, a2, st)
            if ci >= 0:
                c_entries.append((ci, pv, crb, cab, maf, d_int, r2, in_cs, pip_v, cs_id))
        if c_entries:
            r["strand"] = st
            r["lead_ci"] = lead_ci
            r["cand_entries"] = c_entries
            resolved_jobs.append(r)

    print(f"[s07:CattleGWAS] Scoring {len(win_list)} unique linked cattle SNVs across {len(resolved_jobs)} loci with OviSplice_full_ens3...", flush=True)
    if len(win_list) > 0:
        w_mat = np.vstack(win_list).astype(np.uint8)
        a_arr = np.array(alt_list, dtype=np.uint8)
        sc = m_full.score_snvs(w_mat, a_arr)
        s_m12 = sc["ds_usage_500"][:, 1:13].max(axis=1)
        s_site = sc["ds_site_500"]
    else:
        s_m12 = np.zeros(0); s_site = np.zeros(0)

    out_rows = []
    for r in resolved_jobs:
        lci = r["lead_ci"]
        l_m12 = float(s_m12[lci]) if lci >= 0 else np.nan
        best = None
        for ci, pv, crb, cab, maf, d_int, r2, in_cs, pip_v, cs_id in r["cand_entries"]:
            v_m12 = float(s_m12[ci]); v_site = float(s_site[ci])
            if best is None or v_m12 > best[0]:
                best = (v_m12, v_site, ci, pv, crb, cab, maf, d_int, r2, in_cs, pip_v, cs_id)
        v_m12, v_site, bci, bpos, bref, balt, bmaf, dint, br2, bin_cs, bpip, bcs = best
        is_neutral_lead = (not math.isnan(l_m12)) and (l_m12 < 0.05)
        reranked = 1 if (v_m12 >= 0.10 and is_neutral_lead and bpos != r["lead_pos"]) else 0
        out_rows.append({
            "species": "Cattle", "Trait": r["Trait"], "gene_symbol": r["gene_symbol"],
            "tissue": r["tissue"], "organ_system": r["organ_system"], "head_idx": r["head_idx"],
            "PP_H4": round(r["PP_H4"], 4), "pheno_id": r["pheno_id"], "CHR": r["CHR"],
            "i_s": r["i_s"], "i_e": r["i_e"], "strand": r["strand"],
            "lead_variant": r["lead_vid"], "lead_pos": r["lead_pos"],
            "lead_ds_usage_max12": round(l_m12, 4) if not math.isnan(l_m12) else None,
            "cand_variant": f"chr{r['CHR']}:{bpos}_{bref}>{balt}", "cand_pos": bpos, "cand_ref": bref, "cand_alt": balt,
            "cand_maf": round(bmaf, 4), "dist_to_target_intron_bp": dint,
            "dist_to_lead_bp": abs(bpos - r["lead_pos"]),
            "ld_r2_with_lead": round(br2, 4) if not math.isnan(br2) else None,
            "in_susie_cs": bin_cs, "susie_cs": bcs, "susie_pip": round(bpip, 4),
            "cand_ds_usage_max12": round(v_m12, 4), "cand_ds_site": round(v_site, 4),
            "n_linked_500bp_cands": len(r["cand_entries"]), "strict_reranked_over_neutral_lead": reranked,
            "_bci": bci,
        })

    df_ct = pd.DataFrame(out_rows).sort_values(["strict_reranked_over_neutral_lead", "cand_ds_usage_max12", "PP_H4"], ascending=[False, False, False])
    # Also compute +-1Mb clumped subset
    df_ct_1mb = dedup_1mb_loci(df_ct, "Trait", "CHR", "lead_pos", "PP_H4").sort_values(
        ["strict_reranked_over_neutral_lead", "cand_ds_usage_max12", "PP_H4"], ascending=[False, False, False]
    )

    # Run saturation ISM on top 2 distinct-gene strictly validated reranked Cattle loci for Fig 7d & Fig S6d!
    top_ism_loci = []
    seen_g = set()
    for r in df_ct[df_ct["strict_reranked_over_neutral_lead"] == 1].to_dict("records"):
        if r["gene_symbol"] not in seen_g:
            seen_g.add(r["gene_symbol"])
            top_ism_loci.append(r)
        if len(top_ism_loci) >= 2:
            break
    cattle_ism = {}
    ism_meta = []
    for idx_i, r in enumerate(top_ism_loci, start=1):
        bci = int(r["_bci"])
        ism_d = run_single_locus_ism(m_full, w_mat[bci], int(a_arr[bci]), int(r["head_idx"]))
        pfx = f"cattle_coloc_{idx_i}_{r['gene_symbol']}_"
        for k2, v2 in ism_d.items():
            cattle_ism[pfx + k2] = np.array(v2) if np.isscalar(v2) else v2
        ism_meta.append({k: v for k, v in r.items() if k != "_bci"})
        ism_meta[-1]["ism_prefix"] = pfx
        ism_meta[-1]["peak_off"] = ism_d["peak_off"]
    cattle_ism["meta_json"] = np.array(json.dumps(ism_meta))
    np.savez_compressed(OUT_DIR / "v3_cattle_gwas_ism_validated.npz", **cattle_ism)

    df_ct.drop(columns=["_bci"]).to_csv(OUT_DIR / "v3_cattle_gwas_coloc_dedup_ld_verified.tsv", sep="\t", index=False)
    df_ct_1mb.drop(columns=["_bci"]).to_csv(OUT_DIR / "v3_cattle_gwas_coloc_1mb_ld_verified.tsv", sep="\t", index=False)

    del c_u8
    gc.collect()

    n_tg_ge10 = int((df_ct["cand_ds_usage_max12"] >= 0.10).sum()) if len(df_ct) else 0
    n_tg_rr = int(df_ct["strict_reranked_over_neutral_lead"].sum()) if len(df_ct) else 0
    n_1mb_ge10 = int((df_ct_1mb["cand_ds_usage_max12"] >= 0.10).sum()) if len(df_ct_1mb) else 0
    n_1mb_rr = int(df_ct_1mb["strict_reranked_over_neutral_lead"].sum()) if len(df_ct_1mb) else 0
    print(f"[s07:CattleGWAS] Done in {time.time()-t0:.1f}s! Trait*Gene linked={len(df_ct)} (>=0.10: {n_tg_ge10}, reranked={n_tg_rr}); 1Mb linked={len(df_ct_1mb)} (>=0.10: {n_1mb_ge10}, reranked={n_1mb_rr})", flush=True)
    if len(df_ct):
        print(df_ct.drop(columns=["_bci"]).head(10)[["gene_symbol", "Trait", "tissue", "PP_H4", "lead_ds_usage_max12", "cand_variant", "dist_to_target_intron_bp", "dist_to_lead_bp", "ld_r2_with_lead", "in_susie_cs", "cand_ds_usage_max12", "cand_ds_site"]].to_string(index=False))
    return {
        "raw_coloc_rows_h50": n_h50_raw,
        "raw_coloc_rows_h70": n_h70_raw,
        "dropped_invalid_lead_rows": n_dropped_no_lead,
        "unique_trait_gene_loci_h70": len(dedup_tg),
        "independent_1mb_loci_h70": len(dedup_1mb),
        "trait_gene_with_linked_500bp_cand": len(df_ct),
        "trait_gene_linked_ds_ge010": n_tg_ge10,
        "trait_gene_strict_reranked": n_tg_rr,
        "indep_1mb_with_linked_500bp_cand": len(df_ct_1mb),
        "indep_1mb_linked_ds_ge010": n_1mb_ge10,
        "indep_1mb_strict_reranked": n_1mb_rr,
        "ism_featured_loci": ism_meta,
        "top10": df_ct.drop(columns=["_bci"]).head(10).to_dict(orient="records"),
    }


def main():
    global BASE_DIR, RAW_DIR, OMIA_DIR, CATTLEGTEX_DIR, OMIA_CURATED_TSV, SHEEP_COLOC_GZ, COLOC_LOCI_TSV
    global CKPT_DIR, ATLAS_NPZ, OUT_DIR, BACKEND
    global SHEEP_FASTA, SHEEP_FAI, CATTLE_FNA_GZ, CATTLE_GTF_GZ, SHEEP_GTF_GZ, SHEEP_PLINK_PFX, CATTLE_PLINK_PFX
    ap = argparse.ArgumentParser(description="Purified 23 OMIA Mendelian benchmark and dual-species GWAS-sQTL LD/SuSiE audit")
    ap.add_argument("--task", choices=["all", "omia23", "cattle_gwas", "sheep_gwas"], default="all")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "00_raw")
    ap.add_argument("--omia-dir", type=Path, default=None, help="Directory containing omia_cattle_9913_variants.csv & omia_sheep_9940_variants.csv")
    ap.add_argument("--cattlegtex-dir", type=Path, default=None, help="Directory containing CattleGTEx tarballs, total_sQTL_coloc.csv, and Rumen_genotype.tar.gz")
    ap.add_argument("--omia-curated-tsv", type=Path, default=REPO_ROOT / "examples" / "omia_mendelian_variants_curated.tsv")
    ap.add_argument("--coloc-loci-tsv", type=Path, default=None, help="Optional pre-parsed sheep GWAS-coloc TSV (otherwise parsed directly from SheepGTEx_v0.GWAS_coloc.csv.gz)")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints")
    ap.add_argument("--atlas-npz", type=Path, default=None)
    ap.add_argument("--cattle-plink-prefix", type=Path, default=None, help="PLINK prefix (.bed/.bim/.fam) for CattleGTEx Rumen genotypes")
    ap.add_argument("--sheep-plink-prefix", type=Path, default=None, help="PLINK prefix (.bed/.bim/.fam) for SheepGTEx Muscle genotypes")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    args = ap.parse_args()

    BASE_DIR, RAW_DIR, CKPT_DIR, OUT_DIR, BACKEND = (
        args.data_dir, args.raw_dir, args.ckpt_dir, args.out_dir, args.backend
    )
    OMIA_DIR = args.omia_dir or (RAW_DIR / "omia")
    CATTLEGTEX_DIR = args.cattlegtex_dir or (RAW_DIR / "cattlegtex")
    OMIA_CURATED_TSV = args.omia_curated_tsv
    COLOC_LOCI_TSV = args.coloc_loci_tsv
    SHEEP_COLOC_GZ = RAW_DIR / "sheepgtex" / "SheepGTEx_v0.GWAS_coloc.csv.gz"
    if args.atlas_npz is not None:
        ATLAS_NPZ = args.atlas_npz
    elif (CKPT_DIR / "ovisplice_sheep_624k_snv_atlas.npz").is_file():
        ATLAS_NPZ = CKPT_DIR / "ovisplice_sheep_624k_snv_atlas.npz"
    else:
        ATLAS_NPZ = REPO_ROOT / "ovisplice_sheep_624k_snv_atlas.npz"

    SHEEP_FASTA = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna"
    SHEEP_FAI = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna.fai"
    CATTLE_FNA_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz"
    CATTLE_GTF_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.gtf.gz"
    SHEEP_GTF_GZ = RAW_DIR / "sheepgtex" / "sheep.gtf.gz"

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_full, _ = load_ensemble([CKPT_DIR / f"ovisplice_full_seed{s}.pt" for s in (42, 1042, 2042)], backend=BACKEND)
    so_p = CKPT_DIR / "ovisplice_siteonly_fold1_seed42.pt"
    m_m1, _ = load_ensemble([so_p], backend=BACKEND) if so_p.is_file() else (m_full, "")

    tmp_holders = []
    summary = {}
    try:
        if args.task in ("all", "omia23", "cattle_gwas"):
            c_pfx = extract_plink_from_tar_if_needed(
                args.cattle_plink_prefix or (CATTLEGTEX_DIR / "Rumen"),
                CATTLEGTEX_DIR / "Rumen_genotype.tar.gz",
                "Rumen",
                tmp_holders,
            )
            cattle_ld = CompactPlinkLD(c_pfx)
            if args.task in ("all", "omia23"):
                summary["omia_purified_23"] = run_omia_purified_audit(cattle_ld, m_full, m_m1)
            if args.task in ("all", "cattle_gwas"):
                summary["cattle_gwas_ld_verified"] = run_cattle_gwas_ld_audit(cattle_ld, m_full)
            cattle_ld.close()
            del cattle_ld
            gc.collect()

        if args.task in ("all", "sheep_gwas"):
            s_pfx = extract_plink_from_tar_if_needed(
                args.sheep_plink_prefix or (RAW_DIR / "sheepgtex" / "plink" / "Muscle"),
                RAW_DIR / "sheepgtex" / "SheepGTEx_v0.Genotype.molQTL.tar.gz",
                "Muscle",
                tmp_holders,
            )
            sheep_ld = CompactPlinkLD(s_pfx)
            summary["sheep_gwas_ld_verified"] = run_sheep_gwas_ld_audit(sheep_ld, m_full)
            sheep_ld.close()
    finally:
        for th in tmp_holders:
            th.cleanup()

    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "omia23_and_gwas_ld_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[s07] ALL DONE in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
