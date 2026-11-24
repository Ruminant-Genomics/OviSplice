#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s05_syntenic_22mya_te_dual_controls.py
======================================
Synteny-stratified (`E1` >90 Mya, `E2` 22-60 Mya Bovidae-shared, `E3` <22 Mya lineage-specific)
ruminant transposable-element (`SINE/BovA`, `LINE/RTE-BovB`, ancestral `SINE/MIR`) exonization
analysis across sheep and cattle with dual controls:
  1. Unadjusted comparison vs ancient mammalian `E1` splice sites (Haldane OR).
  2. Evolutionary-age- and local-GC-matched non-splice intronic controls & canonical `GT/AG` decoys
     (GC-stratified Cochran-Mantel-Haenszel OR) plus genome-wide `FPR = 1e-3` recall calibration.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice.native_engine import (
    COMP_U8,
    FastaIndexedReader,
    cmh_2x2xk_test,
    fisher_2x2_exact_and_haldane,
    load_ensemble,
    norm_sf_2sided,
)

BASE_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / "data" / "00_raw"
CKPT_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "results"
BACKEND = "auto"

SHEEP_PANEL_NPZ = BASE_DIR / "06_interpret" / "pangolin_site_panel_6k_win.npz"
SHEEP_ALL_SITES = BASE_DIR / "01_labels" / "splice_sites_15heads.npz"
CATTLE_ALL_SITES = BASE_DIR / "05_crossspecies" / "cattle_12tissue_empirical_sites.npz"

SHEEP_FASTA = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna"
SHEEP_FAI = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna.fai"
CATTLE_FNA_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz"

SHEEP_RM = RAW_DIR / "ref" / "GCF_016772045.1.repeatMasker.out.gz"
CHAIN_DIR = RAW_DIR / "chains_rmsk"
CATTLE_RM = CHAIN_DIR / "GCF_002263795.1.repeatMasker.out.gz"
CHAIN_BT_TO_SH = CHAIN_DIR / "bosTau9ToGCF_016772045.2.over.chain.gz"
CHAIN_BT_TO_HG = CHAIN_DIR / "bosTau9ToHg38.over.chain.gz"

TE_CLASSES_KEEP = [
    "Ruminant_TE_combined",
    "Ruminant_SINE_BovA_BovtA",
    "Ruminant_LINE_BovB",
    "Ancestral_SINE_MIR",
    "LINE_L1",
    "Any_TE",
]


def bh_fdr(pvals):
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranked = arr[order] * n / np.arange(1, n + 1, dtype=np.float64)
    mono = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(mono, 0.0, 1.0)
    return out.tolist()


def classify_rm_row(rname: str, rclass: str):
    if rclass in ("Simple_repeat", "Low_complexity", "Satellite", "rRNA", "tRNA", "snRNA", "scRNA", "srpRNA", "Unknown"):
        return []
    hits = []
    is_rum_sine = (rclass in ("SINE/tRNA-Core-RTE", "SINE/Core-RTE")) or rname.startswith(("Bov-tA", "BOV-A", "ART2A"))
    is_rum_line = (rclass == "LINE/RTE-BovB") or rname.startswith("BovB")
    if is_rum_sine:
        hits.append("Ruminant_SINE_BovA_BovtA")
    if is_rum_line:
        hits.append("Ruminant_LINE_BovB")
    if is_rum_sine or is_rum_line:
        hits.append("Ruminant_TE_combined")
    if rclass == "SINE/MIR":
        hits.append("Ancestral_SINE_MIR")
    if rclass == "LINE/L1":
        hits.append("LINE_L1")
    if rclass.startswith(("SINE", "LINE", "LTR", "DNA", "Retroposon", "RC")):
        hits.append("Any_TE")
    return hits


def parse_rmsk_lean(rm_path: Path, acc_to_cid: dict):
    """Compact RepeatMasker parser storing only merged intervals for 6 key classes + ruminant TE exact names."""
    t0 = time.time()
    raw_ivs = {tc: defaultdict(list) for tc in TE_CLASSES_KEEP}
    rum_exact = defaultdict(list)
    with gzip.open(rm_path, "rt", encoding="ascii", errors="ignore") as fh:
        for _ in range(3):
            fh.readline()
        for line in fh:
            p = line.split()
            if len(p) < 11:
                continue
            cid = acc_to_cid.get(p[4], 0)
            if cid == 0:
                continue
            rname, rclass = p[9], p[10]
            cats = classify_rm_row(rname, rclass)
            if not cats:
                continue
            b0 = int(p[5]) - 1
            e0 = int(p[6])
            for tc in cats:
                raw_ivs[tc][cid].append((b0, e0))
            if "Ruminant_TE_combined" in cats:
                rum_exact[cid].append((b0, e0, rname, rclass))
    merged = {tc: {} for tc in TE_CLASSES_KEEP}
    for tc in TE_CLASSES_KEEP:
        for cid, ivs in raw_ivs[tc].items():
            ivs.sort()
            ms, me = [ivs[0][0]], [ivs[0][1]]
            for a, b in ivs[1:]:
                if a <= me[-1]:
                    if b > me[-1]:
                        me[-1] = b
                else:
                    ms.append(a); me.append(b)
            merged[tc][cid] = (np.array(ms, dtype=np.int32), np.array(me, dtype=np.int32))
        raw_ivs[tc].clear()
    lookup = {}
    for cid, ivs in rum_exact.items():
        ivs.sort()
        lookup[cid] = (
            np.array([x[0] for x in ivs], dtype=np.int32),
            np.array([x[1] for x in ivs], dtype=np.int32),
            [x[2] for x in ivs],
            [x[3] for x in ivs],
        )
    print(f"[s05] Parsed {rm_path.name} (lean) in {time.time()-t0:.1f}s", flush=True)
    return merged, lookup


def query_te_mask(merged_tc: dict, cid_arr: np.ndarray, pos0_arr: np.ndarray, half_win: int, max_cid: int) -> np.ndarray:
    out = np.zeros(len(pos0_arr), dtype=bool)
    lo = pos0_arr - half_win
    hi = pos0_arr + half_win + 1
    for c in range(1, max_cid + 1):
        if c not in merged_tc:
            continue
        m = cid_arr == c
        if not m.any():
            continue
        ms, me = merged_tc[c]
        idx = np.searchsorted(ms, hi[m], side="left") - 1
        ok = (idx >= 0) & (me[np.maximum(idx, 0)] > lo[m])
        out[m] = ok
    return out


def find_ruminant_te(rum_lookup: dict, cid: int, pos0: int, half_win: int = 25):
    ent = rum_lookup.get(cid)
    if ent is None:
        return None
    ms, me, names, classes = ent
    lo, hi = pos0 - half_win, pos0 + half_win + 1
    idx = int(np.searchsorted(ms, hi, side="left")) - 1
    while idx >= 0 and ms[idx] > lo - 15000:
        if ms[idx] < hi and me[idx] > lo:
            return (names[idx], classes[idx])
        idx -= 1
    return None


def stream_chain_liftover(
    chain_path: Path,
    t_map: dict,
    q_map,
    fwd_queries=None,
    rev_queries=None,
):
    """C++17 zero-RAM streaming UCSC over.chain liftover (<3s per chain!)."""
    t0 = time.time()
    cache_dir = Path.home() / ".cache" / "ovisplice"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_p = cache_dir / "fast_chain_lift"
    cpp_p = SCRIPT_DIR / "cpp" / "fast_chain_lift.cpp"
    if (not bin_p.exists()) or (cpp_p.stat().st_mtime > bin_p.stat().st_mtime):
        subprocess.run(["g++", "-O3", str(cpp_p), "-o", str(bin_p)], check=True)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="ovisplice_chain_")
    tmp_d = Path(tmp_ctx.name)
    t_f = tmp_d / "tmap.tsv"
    q_f = tmp_d / "qmap.tsv"
    in_f = tmp_d / "in.tsv"
    out_f = tmp_d / "out.tsv"

    t_f.write_text("\n".join(f"{k}\t{v}" for k, v in t_map.items()) + "\n", encoding="utf-8")
    if q_map is not None:
        q_f.write_text("\n".join(f"{k}\t{v}" for k, v in q_map.items()) + "\n", encoding="utf-8")
        q_arg = str(q_f)
    else:
        q_arg = "ANYCHR"

    fwd_lens = []
    rev_lens = []
    with open(in_f, "w", encoding="utf-8") as fh:
        for item in (fwd_queries or []):
            c_arr, p_arr, s_arr = item[0], item[1], item[2]
            r3 = 1 if (len(item) > 3 and item[3]) else 0
            fwd_lens.append(len(c_arr))
            for c, p, s in zip(c_arr.tolist(), p_arr.tolist(), s_arr.tolist()):
                fh.write(f"0\t{r3}\t{c}\t{p}\t{s}\n")
        for item in (rev_queries or []):
            c_arr, p_arr, s_arr = item[0], item[1], item[2]
            rev_lens.append(len(c_arr))
            for c, p, s in zip(c_arr.tolist(), p_arr.tolist(), s_arr.tolist()):
                fh.write(f"1\t0\t{c}\t{p}\t{s}\n")

    cmd = f"gzip -dc '{chain_path}' | '{bin_p}' '{t_f}' '{q_arg}' '{in_f}' '{out_f}'"
    subprocess.run(["bash", "-c", cmd], check=True)

    raw = np.loadtxt(out_f, dtype=np.int32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    fwd_res = []
    pos = 0
    for L in fwd_lens:
        sl = raw[pos : pos + L]
        fwd_res.append((sl[:, 0].astype(np.int16), sl[:, 1].astype(np.int32), sl[:, 2].astype(np.int8)))
        pos += L
    rev_res = []
    for L in rev_lens:
        sl = raw[pos : pos + L]
        rev_res.append((sl[:, 0].astype(np.int16), sl[:, 1].astype(np.int32), sl[:, 2].astype(np.int8)))
        pos += L

    tmp_ctx.cleanup()
    print(f"[s05] C++ streamed {chain_path.name} ({pos:,} queries) in {time.time()-t0:.2f}s", flush=True)
    return fwd_res, rev_res


def has_canonical_dinuc_u8(seq_u8: np.ndarray, pos0: int, strand: int, sc: int) -> bool:
    # 0=A, 1=C, 2=G, 3=T, 4=N
    if pos0 < 3 or pos0 + 3 >= len(seq_u8):
        return False
    if strand == 1:
        if sc == 2:
            a, b = int(seq_u8[pos0 + 1]), int(seq_u8[pos0 + 2])
            return (a == 2 and (b == 3 or b == 1)) or (a == 0 and b == 3)
        a, b = int(seq_u8[pos0 - 2]), int(seq_u8[pos0 - 1])
        return a == 0 and (b == 2 or b == 1)
    else:
        if sc == 2:
            a, b = int(seq_u8[pos0 - 2]), int(seq_u8[pos0 - 1])
            return (a == 0 and (b == 1 or b == 3)) or (a == 2 and b == 1)
        a, b = int(seq_u8[pos0 + 1]), int(seq_u8[pos0 + 2])
        return (b == 3) and (a == 1 or a == 2)


def local_gc_bin_u8(seq_u8: np.ndarray, pos0: int) -> int:
    lo = max(0, pos0 - 50)
    hi = min(len(seq_u8), pos0 + 51)
    sub = seq_u8[lo:hi]
    if len(sub) == 0:
        return 2
    gc_v = float(((sub == 1) | (sub == 2)).sum()) / float(len(sub))
    if gc_v < 0.35:
        return 0
    if gc_v < 0.42:
        return 1
    if gc_v < 0.50:
        return 2
    if gc_v < 0.58:
        return 3
    return 4


def mwu_two_sided_z_p(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 1.0
    all_v = np.concatenate([x, y])
    order = np.argsort(all_v, kind="mergesort")
    vs = all_v[order]
    N = nx + ny
    ranks = np.empty(N, dtype=np.float64)
    tie_corr = 0.0
    i = 0
    while i < N:
        j = i + 1
        while j < N and vs[j] == vs[i]:
            j += 1
        t = j - i
        if t > 1:
            tie_corr += t * (t * t - 1)
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    u_x = float(np.sum(ranks[:nx])) - nx * (nx + 1) / 2.0
    mu = nx * ny / 2.0
    var = (nx * ny / 12.0) * ((N + 1.0) - tie_corr / (N * (N - 1.0)))
    if var <= 0:
        return 1.0
    return norm_sf_2sided((u_x - mu) / math.sqrt(var))


def compute_ovi_bg_fpr_thresholds():
    cache_p = OUT_DIR / "sheep_f1_300win_bg_fpr.json"
    if cache_p.is_file():
        return json.loads(cache_p.read_text(encoding="utf-8"))

    t0 = time.time()
    lbl = np.load(BASE_DIR / "02_colab_pack" / "ovisplice_labels_v1.npz", allow_pickle=False)
    pf_f1 = np.flatnonzero((lbl["win_fold"] == 1) & (lbl["win_paralog"] == 0))
    rng = np.random.default_rng(20261119)
    pick = np.sort(rng.choice(pf_f1, size=min(300, len(pf_f1)), replace=False))

    fa = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))
    ckpts = [CKPT_DIR / f"ovisplice_fold1_seed{s}.pt" for s in (42, 1042)]
    ens, _ = load_ensemble(ckpts, backend=BACKEND)

    ptr = lbl["site_win_ptr"]
    rpos = lbl["site_rel_pos"]
    neg_chunks = []
    for b0 in range(0, len(pick), 50):
        sub_p = pick[b0 : b0 + 50]
        w_u8 = np.empty((len(sub_p), 15000), dtype=np.uint8)
        nmask = np.ones((len(sub_p), 5000), dtype=bool)
        for i, wi in enumerate(sub_p.tolist()):
            ch = str(int(lbl["win_chrom"][wi]))
            ws = int(lbl["win_start"][wi])
            st = int(lbl["win_strand"][wi])
            seq = fa.fetch_u8(ch, ws - 5000, ws + 10000)
            if st < 0:
                seq = COMP_U8[seq[::-1]]
            w_u8[i] = seq
            p0, p1 = int(ptr[wi]), int(ptr[wi + 1])
            if p1 > p0:
                nmask[i, rpos[p0:p1]] = False
        p_ad, _ = ens.score_windows(w_u8, compute_usage=False)
        p_any = np.clip(p_ad[:, :, 0] + p_ad[:, :, 1], 0.0, 1.0)
        neg_chunks.append(p_any[nmask])
    fa.close()
    del ens
    gc.collect()

    neg_scores = np.concatenate(neg_chunks)
    res = {"n_windows": len(pick), "n_neg_bases": int(len(neg_scores)), "fpr_thresholds": {}, "fpr_at_fixed_p": {}}
    for fpr in (1e-4, 5e-4, 1e-3, 5e-3, 1e-2):
        res["fpr_thresholds"][f"{fpr:g}"] = round(float(np.quantile(neg_scores, 1.0 - fpr)), 6)
    for p_fix in (0.05, 0.10, 0.20, 0.50):
        res["fpr_at_fixed_p"][f"{p_fix:.2f}"] = round(float((neg_scores >= p_fix).mean()), 7)

    cache_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[s05:FPR] Computed OviSplice background FPR on {len(neg_scores):,} bases in {time.time()-t0:.1f}s: {res['fpr_thresholds']}", flush=True)
    return res


def sample_chr_controls(anc_pos, anc_str, anc_sc, site_arr, seq_u8, k_per_site, canonical_decoy, rng):
    slen = len(seq_u8)
    p_out, s_out, sc_out, gc_out = [], [], [], []
    for pos0, st, sc in zip(anc_pos.tolist(), anc_str.tolist(), anc_sc.tolist()):
        found = 0
        offsets = rng.integers(-4000, 4001, size=k_per_site * (28 if canonical_decoy else 4))
        for off in offsets.tolist():
            if abs(off) < 250:
                continue
            cand = pos0 + off
            if cand < 100 or cand >= slen - 100 or seq_u8[cand] == 4:
                continue
            if site_arr is not None and len(site_arr) > 0:
                ii = int(np.searchsorted(site_arr, cand))
                if (ii < len(site_arr) and abs(int(site_arr[ii]) - cand) < 150) or (ii > 0 and abs(int(site_arr[ii - 1]) - cand) < 150):
                    continue
            if canonical_decoy and not has_canonical_dinuc_u8(seq_u8, cand, st, sc):
                continue
            p_out.append(cand); s_out.append(st); sc_out.append(sc)
            gc_out.append(local_gc_bin_u8(seq_u8, cand))
            found += 1
            if found >= k_per_site:
                break
    return p_out, s_out, sc_out, gc_out


def main():
    global BASE_DIR, RAW_DIR, CKPT_DIR, OUT_DIR, BACKEND
    global SHEEP_PANEL_NPZ, SHEEP_ALL_SITES, CATTLE_ALL_SITES, SHEEP_FASTA, SHEEP_FAI, CATTLE_FNA_GZ
    global SHEEP_RM, CHAIN_DIR, CATTLE_RM, CHAIN_BT_TO_SH, CHAIN_BT_TO_HG
    ap = argparse.ArgumentParser(description="22-Mya syntenic ruminant TE exonization dual-control analysis")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Directory containing processed splice-site panels")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "00_raw", help="Directory containing reference FASTAs and RepeatMasker")
    ap.add_argument("--chain-dir", type=Path, default=None, help="Directory containing UCSC over.chain.gz and cattle RepeatMasker files")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO_ROOT / "checkpoints", help="Directory containing OviSplice checkpoints")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results", help="Output directory")
    ap.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    args = ap.parse_args()

    BASE_DIR, RAW_DIR, CKPT_DIR, OUT_DIR, BACKEND = args.data_dir, args.raw_dir, args.ckpt_dir, args.out_dir, args.backend
    SHEEP_PANEL_NPZ = BASE_DIR / "06_interpret" / "pangolin_site_panel_6k_win.npz"
    SHEEP_ALL_SITES = BASE_DIR / "01_labels" / "splice_sites_15heads.npz"
    CATTLE_ALL_SITES = BASE_DIR / "05_crossspecies" / "cattle_12tissue_empirical_sites.npz"
    SHEEP_FASTA = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna"
    SHEEP_FAI = RAW_DIR / "ref" / "GCF_016772045.1_genomic.fna.fai"
    CATTLE_FNA_GZ = RAW_DIR / "cattle" / "GCF_002263795.1_ARS-UCD1.2_genomic.fna.gz"
    SHEEP_RM = RAW_DIR / "ref" / "GCF_016772045.1.repeatMasker.out.gz"
    CHAIN_DIR = args.chain_dir if args.chain_dir is not None else (RAW_DIR / "chains_rmsk")
    CATTLE_RM = CHAIN_DIR / "GCF_002263795.1.repeatMasker.out.gz"
    CHAIN_BT_TO_SH = CHAIN_DIR / "bosTau9ToGCF_016772045.2.over.chain.gz"
    CHAIN_BT_TO_HG = CHAIN_DIR / "bosTau9ToHg38.over.chain.gz"

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # First compute OviSplice background FPR while RAM is completely clean
    fpr_info = compute_ovi_bg_fpr_thresholds()

    # Load chromosome maps
    sheep_acc_to_cid = {}
    with open(SHEEP_FAI, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= 27:
                break
            sheep_acc_to_cid[line.split()[0]] = idx + 1
    cattle_nc_to_cid = {f"NC_{37327 + i:06d}.1": i for i in range(1, 30)}
    cattle_nc_to_cid["NC_037357.1"] = 30
    cattle_ucsc_to_cid = {f"chr{i}": i for i in range(1, 30)}
    cattle_ucsc_to_cid["chrX"] = 30

    # Load empirical site sets + sorted positions per chr
    sh_all = np.load(SHEEP_ALL_SITES, allow_pickle=False)
    sh_emp_m = sh_all["source"] < 2
    sheep_emp_set = set(zip(
        sh_all["chrom_id"][sh_emp_m].tolist(), (sh_all["pos"][sh_emp_m] - 1).tolist(),
        sh_all["strand"][sh_emp_m].tolist(), sh_all["site_class"][sh_emp_m].tolist(),
    ))
    sh_all_pos_by_chr = {c: np.sort((sh_all["pos"][sh_all["chrom_id"] == c] - 1).astype(np.int32)) for c in range(1, 28)}
    del sh_all

    cat_all = np.load(CATTLE_ALL_SITES, allow_pickle=False)
    cattle_emp_set = set(zip(
        cat_all["chrom_id"].tolist(), (cat_all["pos"] - 1).tolist(),
        cat_all["strand"].tolist(), cat_all["site_class"].tolist(),
    ))
    ct_all_pos_by_chr = {c: np.sort((cat_all["pos"][cat_all["chrom_id"] == c] - 1).astype(np.int32)) for c in range(1, 31)}

    # Load 6,000 sheep sites & 10,000 cattle sites
    pan_sh = np.load(SHEEP_PANEL_NPZ, allow_pickle=False)
    sh_cid = pan_sh["chrom_id"].astype(np.int32)
    sh_pos0 = pan_sh["pos_0based"].astype(np.int32)
    sh_str = pan_sh["strand"].astype(np.int32)
    sh_sc = pan_sh["site_class"].astype(np.int32)
    sh_src = pan_sh["site_source"].astype(np.int32)
    sh_p_ovi_m = pan_sh["oof_p_matched"].astype(np.float64)
    sh_p_ovi_tot = pan_sh["oof_p_total"].astype(np.float64)
    sh_p_hp_max4 = pan_sh["hp_psplice_max4"].astype(np.float64)
    sh_p_hp_mean4 = pan_sh["hp_psplice_mean4"].astype(np.float64)
    pan_sh.close()



    rng = np.random.default_rng(20261119)
    n_cat_tot = len(cat_all["pos"])
    cat_idx_sample = np.sort(rng.choice(n_cat_tot, size=min(10000, n_cat_tot), replace=False))
    ct_cid = cat_all["chrom_id"][cat_idx_sample].astype(np.int32)
    ct_pos0 = cat_all["pos"][cat_idx_sample].astype(np.int32) - 1
    ct_str = cat_all["strand"][cat_idx_sample].astype(np.int32)
    ct_sc = cat_all["site_class"][cat_idx_sample].astype(np.int32)
    del cat_all
    gc.collect()

    # Step A: Stream Sheep indexed FASTA -> GC bins & sheep intronic controls
    fa_sh = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))
    sh_gc = np.zeros(len(sh_cid), dtype=np.int8)
    sh_bg_rand = {"c": [], "p": [], "s": [], "sc": [], "gc": []}
    sh_bg_dec = {"c": [], "p": [], "s": [], "sc": [], "gc": []}
    rng_sh_r = np.random.default_rng(20261119)
    rng_sh_d = np.random.default_rng(20261120)
    for c in range(1, 28):
        clen = fa_sh.idx[str(c)][0]
        seq_u8 = fa_sh.fetch_u8(str(c), 0, clen)
        m = np.flatnonzero(sh_cid == c)
        for idx_i in m.tolist():
            sh_gc[idx_i] = local_gc_bin_u8(seq_u8, int(sh_pos0[idx_i]))
        if len(m) > 0:
            p_r, s_r, sc_r, gc_r = sample_chr_controls(sh_pos0[m], sh_str[m], sh_sc[m], sh_all_pos_by_chr.get(c), seq_u8, 5, False, rng_sh_r)
            sh_bg_rand["c"].extend([c] * len(p_r)); sh_bg_rand["p"].extend(p_r); sh_bg_rand["s"].extend(s_r); sh_bg_rand["sc"].extend(sc_r); sh_bg_rand["gc"].extend(gc_r)
            p_d, s_d, sc_d, gc_d = sample_chr_controls(sh_pos0[m], sh_str[m], sh_sc[m], sh_all_pos_by_chr.get(c), seq_u8, 5, True, rng_sh_d)
            sh_bg_dec["c"].extend([c] * len(p_d)); sh_bg_dec["p"].extend(p_d); sh_bg_dec["s"].extend(s_d); sh_bg_dec["sc"].extend(sc_d); sh_bg_dec["gc"].extend(gc_d)
        del seq_u8
    fa_sh.close()
    sh_bg_rand = {k: np.array(v, dtype=np.int32 if k != "gc" else np.int8) for k, v in sh_bg_rand.items()}
    sh_bg_dec = {k: np.array(v, dtype=np.int32 if k != "gc" else np.int8) for k, v in sh_bg_dec.items()}

    # Step B: Stream Cattle FASTA 1 chr at a time -> GC bins & cattle intronic controls
    ct_gc = np.zeros(len(ct_cid), dtype=np.int8)
    ct_bg_rand = {"c": [], "p": [], "s": [], "sc": [], "gc": []}
    ct_bg_dec = {"c": [], "p": [], "s": [], "sc": [], "gc": []}
    rng_ct_r = np.random.default_rng(20261119)
    rng_ct_d = np.random.default_rng(20261120)
    lut = np.full(256, 4, dtype=np.uint8)
    for ch_b, val in ((b"A"[0], 0), (b"a"[0], 0), (b"C"[0], 1), (b"c"[0], 1), (b"G"[0], 2), (b"g"[0], 2), (b"T"[0], 3), (b"t"[0], 3)):
        lut[ch_b] = val

    def sample_cattle_chr(cid_cur: int, raw_str: str):
        seq_u8 = lut[np.frombuffer(raw_str.encode("ascii"), dtype=np.uint8)]
        m = np.flatnonzero(ct_cid == cid_cur)
        for idx_i in m.tolist():
            ct_gc[idx_i] = local_gc_bin_u8(seq_u8, int(ct_pos0[idx_i]))
        if len(m) > 0:
            p_r, s_r, sc_r, gc_r = sample_chr_controls(ct_pos0[m], ct_str[m], ct_sc[m], ct_all_pos_by_chr.get(cid_cur), seq_u8, 5, False, rng_ct_r)
            ct_bg_rand["c"].extend([cid_cur] * len(p_r)); ct_bg_rand["p"].extend(p_r); ct_bg_rand["s"].extend(s_r); ct_bg_rand["sc"].extend(sc_r); ct_bg_rand["gc"].extend(gc_r)
            p_d, s_d, sc_d, gc_d = sample_chr_controls(ct_pos0[m], ct_str[m], ct_sc[m], ct_all_pos_by_chr.get(cid_cur), seq_u8, 5, True, rng_ct_d)
            ct_bg_dec["c"].extend([cid_cur] * len(p_d)); ct_bg_dec["p"].extend(p_d); ct_bg_dec["s"].extend(s_d); ct_bg_dec["sc"].extend(sc_d); ct_bg_dec["gc"].extend(gc_d)
        del seq_u8

    cur_c, chunks, seen_ct = 0, [], 0
    with gzip.open(CATTLE_FNA_GZ, "rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_c > 0:
                    sample_cattle_chr(cur_c, "".join(chunks))
                    seen_ct += 1
                    if seen_ct == 30:
                        break
                cur_c = cattle_nc_to_cid.get(line[1:].split()[0], 0)
                chunks = []
            elif cur_c > 0:
                chunks.append(line.strip().upper())
        if cur_c > 0 and seen_ct < 30:
            sample_cattle_chr(cur_c, "".join(chunks))
    ct_bg_rand = {k: np.array(v, dtype=np.int32 if k != "gc" else np.int8) for k, v in ct_bg_rand.items()}
    ct_bg_dec = {k: np.array(v, dtype=np.int32 if k != "gc" else np.int8) for k, v in ct_bg_dec.items()}

    # Step C: ZERO-RAM Streaming Liftover Pass 1: Cattle <-> Sheep (all 6 query sets at once!)
    fwd_ct_sh, rev_sh_ct = stream_chain_liftover(
        CHAIN_BT_TO_SH, cattle_ucsc_to_cid, sheep_acc_to_cid,
        fwd_queries=[
            (ct_cid, ct_pos0, ct_str, False),
            (ct_bg_rand["c"], ct_bg_rand["p"], ct_bg_rand["s"], False),
            (ct_bg_dec["c"], ct_bg_dec["p"], ct_bg_dec["s"], False),
        ],
        rev_queries=[
            (sh_cid, sh_pos0, sh_str),
            (sh_bg_rand["c"], sh_bg_rand["p"], sh_bg_rand["s"]),
            (sh_bg_dec["c"], sh_bg_dec["p"], sh_bg_dec["s"]),
        ],
    )
    (ct_to_sh_c, ct_to_sh_p, ct_to_sh_s) = fwd_ct_sh[0]
    (ct_r_to_sh_c, ct_r_to_sh_p, ct_r_to_sh_s) = fwd_ct_sh[1]
    (ct_d_to_sh_c, ct_d_to_sh_p, ct_d_to_sh_s) = fwd_ct_sh[2]
    (sh_to_ct_c, sh_to_ct_p, sh_to_ct_s) = rev_sh_ct[0]
    (sh_r_to_ct_c, sh_r_to_ct_p, sh_r_to_ct_s) = rev_sh_ct[1]
    (sh_d_to_ct_c, sh_d_to_ct_p, sh_d_to_ct_s) = rev_sh_ct[2]

    # Step D: ZERO-RAM Streaming Liftover Pass 2: Cattle -> Human hg38 (all 6 sets in Cattle coordinates!)
    fwd_ct_hg, _ = stream_chain_liftover(
        CHAIN_BT_TO_HG, cattle_ucsc_to_cid, None,
        fwd_queries=[
            (ct_cid, ct_pos0, ct_str, True),
            (ct_bg_rand["c"], ct_bg_rand["p"], ct_bg_rand["s"], True),
            (ct_bg_dec["c"], ct_bg_dec["p"], ct_bg_dec["s"], True),
            (sh_to_ct_c.astype(np.int32), sh_to_ct_p, sh_to_ct_s.astype(np.int32), True),
            (sh_r_to_ct_c.astype(np.int32), sh_r_to_ct_p, sh_r_to_ct_s.astype(np.int32), True),
            (sh_d_to_ct_c.astype(np.int32), sh_d_to_ct_p, sh_d_to_ct_s.astype(np.int32), True),
        ],
        rev_queries=None,
    )
    ct_hg_ok = fwd_ct_hg[0][0] > 0
    ct_r_hg_ok = fwd_ct_hg[1][0] > 0
    ct_d_hg_ok = fwd_ct_hg[2][0] > 0
    sh_hg_ok = fwd_ct_hg[3][0] > 0
    sh_r_hg_ok = fwd_ct_hg[4][0] > 0
    sh_d_hg_ok = fwd_ct_hg[5][0] > 0

    # Step E: Check lifted canonical dinucleotides in sibling genomes (streaming 1 chr at a time)
    ct_sh_dinuc_ok = np.zeros(len(ct_cid), dtype=bool)
    ct_d_sh_dinuc_ok = np.zeros(len(ct_bg_dec["c"]), dtype=bool)
    by_sh = defaultdict(list)
    for i in np.flatnonzero(ct_to_sh_c > 0).tolist():
        by_sh[int(ct_to_sh_c[i])].append((0, i, int(ct_to_sh_p[i]), int(ct_to_sh_s[i]), int(ct_sc[i])))
    for i in np.flatnonzero(ct_d_to_sh_c > 0).tolist():
        by_sh[int(ct_d_to_sh_c[i])].append((1, i, int(ct_d_to_sh_p[i]), int(ct_d_to_sh_s[i]), int(ct_bg_dec["sc"][i])))
    fa_sh = FastaIndexedReader(str(SHEEP_FASTA), str(SHEEP_FAI))
    for c, lst in by_sh.items():
        seq_u8 = fa_sh.fetch_u8(str(c), 0, fa_sh.idx[str(c)][0])
        for kind, i, p0, st, sc in lst:
            ok = has_canonical_dinuc_u8(seq_u8, p0, st, sc)
            if kind == 0:
                ct_sh_dinuc_ok[i] = ok
            else:
                ct_d_sh_dinuc_ok[i] = ok
        del seq_u8
    fa_sh.close()

    sh_ct_dinuc_ok = np.zeros(len(sh_cid), dtype=bool)
    sh_d_ct_dinuc_ok = np.zeros(len(sh_bg_dec["c"]), dtype=bool)
    by_ct = defaultdict(list)
    for i in np.flatnonzero(sh_to_ct_c > 0).tolist():
        by_ct[int(sh_to_ct_c[i])].append((0, i, int(sh_to_ct_p[i]), int(sh_to_ct_s[i]), int(sh_sc[i])))
    for i in np.flatnonzero(sh_d_to_ct_c > 0).tolist():
        by_ct[int(sh_d_to_ct_c[i])].append((1, i, int(sh_d_to_ct_p[i]), int(sh_d_to_ct_s[i]), int(sh_bg_dec["sc"][i])))
    cur_c, chunks, seen_ct = 0, [], 0
    with gzip.open(CATTLE_FNA_GZ, "rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_c > 0:
                    if cur_c in by_ct:
                        seq_u8 = lut[np.frombuffer("".join(chunks).encode("ascii"), dtype=np.uint8)]
                        for kind, i, p0, st, sc in by_ct[cur_c]:
                            ok = has_canonical_dinuc_u8(seq_u8, p0, st, sc)
                            if kind == 0:
                                sh_ct_dinuc_ok[i] = ok
                            else:
                                sh_d_ct_dinuc_ok[i] = ok
                        del seq_u8
                    seen_ct += 1
                    if seen_ct == 30:
                        break
                cur_c = cattle_nc_to_cid.get(line[1:].split()[0], 0)
                chunks = []
            elif cur_c > 0 and cur_c in by_ct:
                chunks.append(line.strip().upper())
        if cur_c > 0 and seen_ct < 30 and cur_c in by_ct:
            seq_u8 = lut[np.frombuffer("".join(chunks).encode("ascii"), dtype=np.uint8)]
            for kind, i, p0, st, sc in by_ct[cur_c]:
                ok = has_canonical_dinuc_u8(seq_u8, p0, st, sc)
                if kind == 0:
                    sh_ct_dinuc_ok[i] = ok
                else:
                    sh_d_ct_dinuc_ok[i] = ok
            del seq_u8

    # Step F: Assign E1 / E2 / E3 tiers to all splice sites and controls
    sh_tier = []
    sh_dual_rnaseq = np.zeros(len(sh_cid), dtype=bool)
    sh_cat_coord = []
    for i in range(len(sh_cid)):
        if sh_to_ct_c[i] <= 0:
            sh_tier.append("E3_Lineage_Specific"); sh_cat_coord.append(None); continue
        c_cid, c_pos0, c_str = int(sh_to_ct_c[i]), int(sh_to_ct_p[i]), int(sh_to_ct_s[i])
        c_rnaseq = any((c_cid, c_pos0 + d, c_str, int(sh_sc[i])) in cattle_emp_set for d in (0, -1, 1))
        sh_cat_coord.append((c_cid, c_pos0, c_str))
        if not (sh_ct_dinuc_ok[i] or c_rnaseq):
            sh_tier.append("E3_Lineage_Specific"); continue
        sh_dual_rnaseq[i] = c_rnaseq
        sh_tier.append("E1_Mammalian_Conserved" if sh_hg_ok[i] else "E2_Bovidae_Shared")
    sh_tier = np.array(sh_tier)

    sh_rand_tier = np.where(sh_r_to_ct_c <= 0, "E3_Lineage_Specific", np.where(sh_r_hg_ok, "E1_Mammalian_Conserved", "E2_Bovidae_Shared"))
    sh_dec_tier = np.where((sh_d_to_ct_c <= 0) | (~sh_d_ct_dinuc_ok), "E3_Lineage_Specific", np.where(sh_d_hg_ok, "E1_Mammalian_Conserved", "E2_Bovidae_Shared"))

    ct_tier = []
    for i in range(len(ct_cid)):
        s_rna = (ct_to_sh_c[i] > 0) and any((int(ct_to_sh_c[i]), int(ct_to_sh_p[i]) + d, int(ct_to_sh_s[i]), int(ct_sc[i])) in sheep_emp_set for d in (0, -1, 1))
        s_ok = bool(ct_sh_dinuc_ok[i]) or s_rna
        if ct_hg_ok[i] and s_ok:
            ct_tier.append("E1_Mammalian_Conserved")
        elif s_ok:
            ct_tier.append("E2_Bovidae_Shared")
        else:
            ct_tier.append("E3_Lineage_Specific")
    ct_tier = np.array(ct_tier)

    ct_rand_tier = np.where(ct_r_to_sh_c <= 0, "E3_Lineage_Specific", np.where(ct_r_hg_ok, "E1_Mammalian_Conserved", "E2_Bovidae_Shared"))
    ct_dec_tier = np.where((ct_d_to_sh_c <= 0) | (~ct_d_sh_dinuc_ok), "E3_Lineage_Specific", np.where(ct_d_hg_ok, "E1_Mammalian_Conserved", "E2_Bovidae_Shared"))

    del sheep_emp_set, cattle_emp_set
    gc.collect()

    print("[s05] Verified splice tier counts:",
          "Sheep=", {t: int((sh_tier == t).sum()) for t in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific")},
          "Cattle=", {t: int((ct_tier == t).sum()) for t in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific")}, flush=True)

    # Step D: Sequential RepeatMasker evaluation (Sheep first, then free, then Cattle!)
    age_bg_rows = []
    excl_rows = []
    bg_dist_summary = {}

    # 1) SHEEP RepeatMasker
    sh_rm_m, sh_rum_lu = parse_rmsk_lean(SHEEP_RM, sheep_acc_to_cid)
    sh_rum25 = query_te_mask(sh_rm_m["Ruminant_TE_combined"], sh_cid, sh_pos0, 25, 27)

    # Exclusive Fig 5c breakdown (Sheep)
    bova_s = query_te_mask(sh_rm_m["Ruminant_SINE_BovA_BovtA"], sh_cid, sh_pos0, 25, 27)
    bovb_s = query_te_mask(sh_rm_m["Ruminant_LINE_BovB"], sh_cid, sh_pos0, 25, 27)
    mir_s = query_te_mask(sh_rm_m["Ancestral_SINE_MIR"], sh_cid, sh_pos0, 25, 27)
    for tr in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific"):
        mt = sh_tier == tr; nt = int(mt.sum())
        excl_rows.append({
            "species": "Sheep", "tier": tr, "n_sites": nt,
            "bova_only_n": int((mt & bova_s & (~bovb_s)).sum()), "bova_only_pct": round(100 * float((mt & bova_s & (~bovb_s)).sum()) / nt, 3),
            "bovb_only_n": int((mt & (~bova_s) & bovb_s).sum()), "bovb_only_pct": round(100 * float((mt & (~bova_s) & bovb_s).sum()) / nt, 3),
            "both_n": int((mt & bova_s & bovb_s).sum()), "both_pct": round(100 * float((mt & bova_s & bovb_s).sum()) / nt, 3),
            "union_ruminant_te_n": int((mt & (bova_s | bovb_s)).sum()), "union_ruminant_te_pct": round(100 * float((mt & (bova_s | bovb_s)).sum()) / nt, 3),
            "mir_n": int((mt & mir_s).sum()), "mir_pct": round(100 * float((mt & mir_s).sum()) / nt, 3),
        })

    # Collect Sheep E2 ruminant TE candidates for fossil check (exact 176!)
    e2_sh_rum_list = []
    for i in np.flatnonzero((sh_tier == "E2_Bovidae_Shared") & sh_rum25).tolist():
        hit = find_ruminant_te(sh_rum_lu, int(sh_cid[i]), int(sh_pos0[i]), 25)
        e2_sh_rum_list.append((i, hit[0] if hit else "Ruminant_TE", hit[1] if hit else "Ruminant_TE"))

    # Sheep same-age GC-matched intronic background
    for bg_type, bg_d, btier in (
        ("Random_Intronic_SameAge_GC", sh_bg_rand, sh_rand_tier),
        ("Canonical_GTAG_Decoy_SameAge_GC", sh_bg_dec, sh_dec_tier),
    ):
        bg_dist_summary[f"Sheep_{bg_type}"] = {t: int((btier == t).sum()) for t in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific")}
        for win_lbl, hw in (("0bp_exact", 0), ("+-25bp", 25)):
            for tc in TE_CLASSES_KEEP:
                sp_hit = query_te_mask(sh_rm_m[tc], sh_cid, sh_pos0, hw, 27)
                bg_hit = query_te_mask(sh_rm_m[tc], bg_d["c"], bg_d["p"], hw, 27)
                for tr in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific"):
                    ms = sh_tier == tr; mb = btier == tr
                    a = int((ms & sp_hit).sum()); b = int((ms & (~sp_hit)).sum())
                    c_ = int((mb & bg_hit).sum()); d_ = int((mb & (~bg_hit)).sum())
                    fr = fisher_2x2_exact_and_haldane(a, b, c_, d_)
                    tabs = []
                    for gb in range(5):
                        mg_s = ms & (sh_gc == gb); mg_b = mb & (bg_d["gc"] == gb)
                        if mg_s.any() and mg_b.any():
                            tabs.append((int((mg_s & sp_hit).sum()), int((mg_s & (~sp_hit)).sum()),
                                         int((mg_b & bg_hit).sum()), int((mg_b & (~bg_hit)).sum())))
                    cr = cmh_2x2xk_test(tabs)
                    age_bg_rows.append({
                        "species": "Sheep", "bg_control_type": bg_type, "window_bp": win_lbl,
                        "evolutionary_tier": tr, "te_class": tc,
                        "splice_hit": a, "splice_n": a + b, "splice_rate": round(a / max(1, a + b), 5),
                        "bg_hit": c_, "bg_n": c_ + d_, "bg_rate": round(c_ / max(1, c_ + d_), 5),
                        "haldane_or": round(fr["haldane_or"], 4), "ci95_lo": round(fr["ci_lo"], 4), "ci95_hi": round(fr["ci_hi"], 4),
                        "p_fisher": fr["p_fisher"],
                        "gc_cmh_or": round(cr["cmh_or"], 4) if math.isfinite(cr["cmh_or"]) else None,
                        "gc_cmh_ci_lo": round(cr["ci_lo"], 4) if math.isfinite(cr["ci_lo"]) else None,
                        "gc_cmh_ci_hi": round(cr["ci_hi"], 4) if math.isfinite(cr["ci_hi"]) else None,
                        "p_gc_cmh": cr["p_cmh"],
                    })
    del sh_rm_m, sh_rum_lu
    gc.collect()

    # 2) CATTLE RepeatMasker
    ct_rm_m, ct_rum_lu = parse_rmsk_lean(CATTLE_RM, cattle_nc_to_cid)
    bova_c = query_te_mask(ct_rm_m["Ruminant_SINE_BovA_BovtA"], ct_cid, ct_pos0, 25, 30)
    bovb_c = query_te_mask(ct_rm_m["Ruminant_LINE_BovB"], ct_cid, ct_pos0, 25, 30)
    mir_c = query_te_mask(ct_rm_m["Ancestral_SINE_MIR"], ct_cid, ct_pos0, 25, 30)
    for tr in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific"):
        mt = ct_tier == tr; nt = int(mt.sum())
        excl_rows.append({
            "species": "Cattle", "tier": tr, "n_sites": nt,
            "bova_only_n": int((mt & bova_c & (~bovb_c)).sum()), "bova_only_pct": round(100 * float((mt & bova_c & (~bovb_c)).sum()) / nt, 3),
            "bovb_only_n": int((mt & (~bova_c) & bovb_c).sum()), "bovb_only_pct": round(100 * float((mt & (~bova_c) & bovb_c).sum()) / nt, 3),
            "both_n": int((mt & bova_c & bovb_c).sum()), "both_pct": round(100 * float((mt & bova_c & bovb_c).sum()) / nt, 3),
            "union_ruminant_te_n": int((mt & (bova_c | bovb_c)).sum()), "union_ruminant_te_pct": round(100 * float((mt & (bova_c | bovb_c)).sum()) / nt, 3),
            "mir_n": int((mt & mir_c).sum()), "mir_pct": round(100 * float((mt & mir_c).sum()) / nt, 3),
        })

    # Complete unified Sheep E2 co-orthologous fossil check against Cattle RepeatMasker (denominator = exact 176!)
    fossil_rows = []
    n_e2_sh_rum = len(e2_sh_rum_list)
    n_e2_coorth_same = 0
    n_e2_coorth_dual_rna = 0
    for i, sh_rn, sh_rc in e2_sh_rum_list:
        cc_t = sh_cat_coord[i]
        if cc_t is None:
            continue
        c_cid, c_pos0, c_str = cc_t
        ct_hit = find_ruminant_te(ct_rum_lu, c_cid, c_pos0, 25)
        if ct_hit is not None:
            n_e2_coorth_same += 1
            if sh_dual_rnaseq[i]:
                n_e2_coorth_dual_rna += 1
            fossil_rows.append({
                "sheep_chr": f"chr{sh_cid[i]}" if sh_cid[i] <= 26 else "chrX",
                "sheep_pos_1b": int(sh_pos0[i]) + 1, "sheep_strand": "+" if sh_str[i] == 1 else "-",
                "site_type": "Donor" if sh_sc[i] == 2 else "Acceptor",
                "sheep_te_name": sh_rn, "sheep_te_class": sh_rc,
                "cattle_chr": f"chr{c_cid}" if c_cid <= 29 else "chrX",
                "cattle_pos_1b": c_pos0 + 1, "cattle_strand": "+" if c_str == 1 else "-",
                "cattle_te_name": ct_hit[0], "cattle_te_class": ct_hit[1],
                "dual_species_rnaseq_spliced": int(sh_dual_rnaseq[i]),
                "p_ovisplice": round(float(sh_p_ovi_m[i]), 5),
                "p_hp_mean4": round(float(sh_p_hp_mean4[i]), 5),
                "p_hp_max4": round(float(sh_p_hp_max4[i]), 5),
            })

    # Cattle same-age GC-matched intronic background
    for bg_type, bg_d, btier in (
        ("Random_Intronic_SameAge_GC", ct_bg_rand, ct_rand_tier),
        ("Canonical_GTAG_Decoy_SameAge_GC", ct_bg_dec, ct_dec_tier),
    ):
        bg_dist_summary[f"Cattle_{bg_type}"] = {t: int((btier == t).sum()) for t in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific")}
        for win_lbl, hw in (("0bp_exact", 0), ("+-25bp", 25)):
            for tc in TE_CLASSES_KEEP:
                sp_hit = query_te_mask(ct_rm_m[tc], ct_cid, ct_pos0, hw, 30)
                bg_hit = query_te_mask(ct_rm_m[tc], bg_d["c"], bg_d["p"], hw, 30)
                for tr in ("E1_Mammalian_Conserved", "E2_Bovidae_Shared", "E3_Lineage_Specific"):
                    ms = ct_tier == tr; mb = btier == tr
                    a = int((ms & sp_hit).sum()); b = int((ms & (~sp_hit)).sum())
                    c_ = int((mb & bg_hit).sum()); d_ = int((mb & (~bg_hit)).sum())
                    fr = fisher_2x2_exact_and_haldane(a, b, c_, d_)
                    tabs = []
                    for gb in range(5):
                        mg_s = ms & (ct_gc == gb); mg_b = mb & (bg_d["gc"] == gb)
                        if mg_s.any() and mg_b.any():
                            tabs.append((int((mg_s & sp_hit).sum()), int((mg_s & (~sp_hit)).sum()),
                                         int((mg_b & bg_hit).sum()), int((mg_b & (~bg_hit)).sum())))
                    cr = cmh_2x2xk_test(tabs)
                    age_bg_rows.append({
                        "species": "Cattle", "bg_control_type": bg_type, "window_bp": win_lbl,
                        "evolutionary_tier": tr, "te_class": tc,
                        "splice_hit": a, "splice_n": a + b, "splice_rate": round(a / max(1, a + b), 5),
                        "bg_hit": c_, "bg_n": c_ + d_, "bg_rate": round(c_ / max(1, c_ + d_), 5),
                        "haldane_or": round(fr["haldane_or"], 4), "ci95_lo": round(fr["ci_lo"], 4), "ci95_hi": round(fr["ci_hi"], 4),
                        "p_fisher": fr["p_fisher"],
                        "gc_cmh_or": round(cr["cmh_or"], 4) if math.isfinite(cr["cmh_or"]) else None,
                        "gc_cmh_ci_lo": round(cr["ci_lo"], 4) if math.isfinite(cr["ci_lo"]) else None,
                        "gc_cmh_ci_hi": round(cr["ci_hi"], 4) if math.isfinite(cr["ci_hi"]) else None,
                        "p_gc_cmh": cr["p_cmh"],
                    })
    del ct_rm_m, ct_rum_lu
    gc.collect()

    pd.DataFrame(excl_rows).to_csv(OUT_DIR / "v3_te_exclusive_fig5c_breakdown.tsv", sep="\t", index=False)
    pd.DataFrame(fossil_rows).sort_values(["dual_species_rnaseq_spliced", "p_ovisplice"], ascending=[False, False]).to_csv(
        OUT_DIR / "v3_coorthologous_ruminant_te_fossils.tsv", sep="\t", index=False
    )
    df_age_bg = pd.DataFrame(age_bg_rows)
    df_age_bg["fdr_gc_cmh"] = bh_fdr(df_age_bg["p_gc_cmh"].tolist())
    df_age_bg.to_csv(OUT_DIR / "v3_te_same_age_gc_intronic_background.tsv", sep="\t", index=False)

    # Step E: Fair calibration & equal-E1-recall operating point comparison (M3 #2)
    m_e1 = sh_tier == "E1_Mammalian_Conserved"
    thr_e1_70_ovi = float(np.quantile(sh_p_ovi_tot[m_e1], 0.30))
    thr_e1_70_hpm = float(np.quantile(sh_p_hp_mean4[m_e1], 0.30))
    thr_e1_70_hpx = float(np.quantile(sh_p_hp_max4[m_e1], 0.30))
    thr_fpr_1e3 = float(fpr_info["fpr_thresholds"]["0.001"])
    thr_fpr_5e3 = float(fpr_info["fpr_thresholds"]["0.005"])

    recall_strata = [
        ("E1_Mammalian_Conserved_gt90Mya",   m_e1),
        ("E2_Bovidae_Shared_all",            sh_tier == "E2_Bovidae_Shared"),
        ("E2a_Bovidae_Shared_DualRNAseq",    (sh_tier == "E2_Bovidae_Shared") & sh_dual_rnaseq),
        ("E2_Bovidae_Shared_NonTE",          (sh_tier == "E2_Bovidae_Shared") & (~sh_rum25)),
        ("E2_Bovidae_Shared_RuminantTE",     (sh_tier == "E2_Bovidae_Shared") & sh_rum25),
        ("E3_Sheep_Specific_all",            sh_tier == "E3_Lineage_Specific"),
        ("E3_Sheep_Specific_NonTE",          (sh_tier == "E3_Lineage_Specific") & (~sh_rum25)),
        ("E3_Sheep_Specific_RuminantTE",     (sh_tier == "E3_Lineage_Specific") & sh_rum25),
    ]
    rec_rows = []
    for sname, ms in recall_strata:
        po = sh_p_ovi_tot[ms]; pm = sh_p_hp_mean4[ms]; px = sh_p_hp_max4[ms]
        rec_rows.append({
            "stratum": sname, "n_sites": int(ms.sum()), "novel_frac": round(float((sh_src[ms] == 1).mean()), 4),
            "ovi_median_p": round(float(np.median(po)), 4),
            "hp_mean4_median_p": round(float(np.median(pm)), 4),
            "hp_max4_median_p": round(float(np.median(px)), 4),
            "ovi_recall_p02": round(float((po >= 0.20).mean()), 4),
            "hp_mean4_recall_p02": round(float((pm >= 0.20).mean()), 4),
            "hp_max4_recall_p02": round(float((px >= 0.20).mean()), 4),
            "ovi_recall_eqE1_70": round(float((po >= thr_e1_70_ovi).mean()), 4),
            "hp_mean4_recall_eqE1_70": round(float((pm >= thr_e1_70_hpm).mean()), 4),
            "hp_max4_recall_eqE1_70": round(float((px >= thr_e1_70_hpx).mean()), 4),
            "ovi_recall_fpr1e3": round(float((po >= thr_fpr_1e3).mean()), 4),
            "ovi_recall_fpr5e3": round(float((po >= thr_fpr_5e3).mean()), 4),
            "mwu_p_ovi_vs_hp_mean4": mwu_two_sided_z_p(po, pm),
            "mwu_p_ovi_vs_hp_max4": mwu_two_sided_z_p(po, px),
        })
    df_rec = pd.DataFrame(rec_rows)
    df_rec.to_csv(OUT_DIR / "v3_te_strata_model_calibration_recall.tsv", sep="\t", index=False)

    summary = {
        "unified_e2_coorthologous_fossils": {
            "e2_sheep_ruminant_te_sites_unified": n_e2_sh_rum,
            "e2_cattle_coorthologous_same_te_sites": n_e2_coorth_same,
            "coorthologous_retention_rate": round(n_e2_coorth_same / max(1, n_e2_sh_rum), 4),
            "coorthologous_dual_rnaseq_spliced": n_e2_coorth_dual_rna,
        },
        "bg_tier_counts": bg_dist_summary,
        "ovi_background_fpr": fpr_info,
        "equal_e1_70_thresholds": {"ovi_total": round(thr_e1_70_ovi, 5), "hp_mean4": round(thr_e1_70_hpm, 5), "hp_max4": round(thr_e1_70_hpx, 5)},
        "exclusive_fig5c_breakdown": excl_rows,
        "key_same_age_intron_bg_25bp": df_age_bg[
            (df_age_bg["window_bp"] == "+-25bp")
            & (df_age_bg["te_class"].isin(["Ruminant_TE_combined", "Ruminant_SINE_BovA_BovtA", "Ruminant_LINE_BovB", "Ancestral_SINE_MIR"]))
        ].to_dict(orient="records"),
        "strata_calibration_recall": rec_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "v3_part3_te_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[s05] Finished in {time.time()-t0:.1f}s! Unified E2 Fossils: {n_e2_coorth_same}/{n_e2_sh_rum} ({100*n_e2_coorth_same/max(1,n_e2_sh_rum):.2f}%), dual-RNA={n_e2_coorth_dual_rna}", flush=True)
    print("\n=== SAME-AGE GC-MATCHED INTRONIC BACKGROUND (+-25bp, Ruminant_TE_combined & MIR) ===")
    sub_p = df_age_bg[(df_age_bg["window_bp"] == "+-25bp") & (df_age_bg["te_class"].isin(["Ruminant_TE_combined", "Ancestral_SINE_MIR"]))]
    print(sub_p[["species", "bg_control_type", "evolutionary_tier", "te_class", "splice_rate", "bg_rate", "haldane_or", "gc_cmh_or", "p_gc_cmh"]].to_string(index=False))
    print("\n=== FAIR CALIBRATION & EQUAL-E1-RECALL COMPARISON ===")
    print(df_rec[["stratum", "n_sites", "ovi_median_p", "hp_mean4_median_p", "hp_max4_median_p", "ovi_recall_eqE1_70", "hp_mean4_recall_eqE1_70", "hp_max4_recall_eqE1_70", "ovi_recall_fpr1e3"]].to_string(index=False))


if __name__ == "__main__":
    main()
