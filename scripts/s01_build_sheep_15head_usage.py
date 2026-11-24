#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s01_build_sheep_15head_usage.py
===============================
Constructs the 15-head cluster-relative splice-site usage atlas (`splice_sites_15heads.npz`)
on sheep `ARS-UI_Ramb_v2.0` (`chr1..chr26, chrX`) from aggregated LeafCutter junction counts,
dual annotations (`NCBI RefSeq 104` + `Ensembl 113`), and per-head median gene expression.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_INTRON_NPZ = REPO_ROOT / "data" / "01_labels" / "intron_agg.npz"
IN_TPM_NPZ = REPO_ROOT / "data" / "01_labels" / "gene_tpm_medians_15heads.npz"
FASTA = REPO_ROOT / "data" / "00_raw" / "ref" / "GCF_016772045.1_genomic.fna"
FAI = REPO_ROOT / "data" / "00_raw" / "ref" / "GCF_016772045.1_genomic.fna.fai"
REFSEQ_GTF = REPO_ROOT / "data" / "00_raw" / "sheepgtex" / "sheep.gtf.gz"
ENS_GTF = REPO_ROOT / "data" / "00_raw" / "ref" / "Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.113.gtf.gz"
ENS_SYM = REPO_ROOT / "data" / "00_raw" / "ref" / "Ovis_aries_rambouillet.113_gene_symbols.tsv"
CLU2GENE = REPO_ROOT / "data" / "00_raw" / "sheepgtex" / "SheepGTEx_v0.Splicing.clusters_to_genes.txt"

OUT_NPZ = REPO_ROOT / "data" / "01_labels" / "splice_sites_15heads.npz"
OUT_META = REPO_ROOT / "data" / "01_labels" / "splice_sites_meta.tsv.gz"
OUT_JSON = REPO_ROOT / "data" / "01_labels" / "site_usage_summary.json"

HEAD_NAMES = [
    "PanTissue",
    "SkeletalMuscle", "Liver", "Adipose", "SmallIntestine", "CNS", "Rumen",
    "Pituitary", "Lung", "Abomasum", "Gonad", "Heart", "Skin",
    "SkeletalMuscle_Prenatal", "SkeletalMuscle_Postnatal",
]

PLUS_DINUCS = {(b"GT", b"AG"), (b"GC", b"AG"), (b"AT", b"AC")}
MINUS_DINUCS = {(b"CT", b"AC"), (b"CT", b"GC"), (b"GT", b"AT")}
CANON_DONOR_PLUS = {b"GT", b"GC", b"AT"}
CANON_ACC_PLUS = {b"AG", b"AC"}
CANON_DONOR_MINUS_GENOMIC = {b"AC", b"GC", b"AT"}  # rc -> GT, GC, AT
CANON_ACC_MINUS_GENOMIC = {b"CT", b"GT"}          # rc -> AG, AC

COMP_BYTES = bytes.maketrans(b"ACGTN", b"TGCAN")


def rc_bytes(b: bytes) -> bytes:
    return b.translate(COMP_BYTES)[::-1]


def parse_chr_to_cid(chrom_str: str) -> int:
    s = chrom_str[3:] if chrom_str.startswith("chr") else chrom_str
    if s == "X":
        return 27
    if s.isdigit():
        v = int(s)
        if 1 <= v <= 26:
            return v
    return 0


def pack_keys(cid: np.ndarray, pos: np.ndarray, strand: np.ndarray, sc: np.ndarray) -> np.ndarray:
    sbit = (strand > 0).astype(np.uint64)
    return (
        (cid.astype(np.uint64) << 40)
        | (pos.astype(np.uint64) << 8)
        | (sbit << 4)
        | sc.astype(np.uint64)
    )


def load_main_chrom_seqs():
    """Load NC_056054.1..NC_056080.1 (chrom_id 1..27) uppercase bytes via .fai offsets."""
    seqs = {}
    lens = {}
    with open(FAI, "r", encoding="utf-8") as fai_fh, open(FASTA, "rb") as fa_fh:
        for idx, line in enumerate(fai_fh):
            if idx >= 27:
                break
            cid = idx + 1
            parts = line.split()
            length = int(parts[1])
            offset = int(parts[2])
            bl = int(parts[3])
            bw = int(parts[4])
            n_full = length // bl
            rem = length % bl
            n_bytes = n_full * bw + rem
            fa_fh.seek(offset)
            raw = fa_fh.read(n_bytes)
            seq = raw.translate(None, b"\r\n").upper()
            assert len(seq) == length, f"FAI read mismatch chr{cid}: {len(seq)} vs {length}"
            seqs[cid] = seq
            lens[cid] = length
    assert len(seqs) == 27
    return seqs, lens


def load_gtf_union_and_gene_map(tpm_gene_ids: np.ndarray, tpm_pan: np.ndarray):
    """
    Parse RefSeq 104 + Ensembl 113 GTFs on chrom_id 1..27.
    Returns:
      - refseq_gene_strand: dict RefSeq gene_id -> strand_int (+1 / -1)
      - gtf_intron_strand: dict (cid, i_s, i_e) -> strand_int (+1 / -1 if unambiguous)
      - ann_keys: sorted uint64 array of annotated (cid, pos, strand_int, site_class)
      - ann_tpm_idx: corresponding int32 TPM row index (-1 if unmapped to TPM)
      - stats: dict of counts
    """
    tpm_map = {str(g): i for i, g in enumerate(tpm_gene_ids)}

    refseq_gene_strand = {}
    # For coordinate overlap mapping of Ensembl genes -> RefSeq TPM gene index:
    # refseq_iv[(cid, strand_int)] -> list of (start, end, tpm_idx)
    refseq_iv = defaultdict(list)
    refseq_tx_exons = defaultdict(list)

    re_gid = re.compile(r'gene_id "([^"]+)"')
    re_tid = re.compile(r'transcript_id "([^"]+)"')
    re_gname = re.compile(r'gene_name "([^"]+)"')

    with gzip.open(REFSEQ_GTF, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split("\t", 8)
            ftype = f[2]
            if ftype not in ("gene", "exon"):
                continue
            cid = parse_chr_to_cid(f[0])
            if cid == 0:
                continue
            strand_int = 1 if f[6] == "+" else (-1 if f[6] == "-" else 0)
            if strand_int == 0:
                continue
            s = int(f[3])
            e = int(f[4])
            attr = f[8]
            mg = re_gid.search(attr)
            if not mg:
                continue
            gid = mg.group(1)
            if ftype == "gene":
                refseq_gene_strand[gid] = strand_int
                tidx = tpm_map.get(gid, -1)
                if tidx >= 0:
                    refseq_iv[(cid, strand_int)].append((s, e, tidx))
            else:
                mt = re_tid.search(attr)
                if not mt:
                    continue
                refseq_tx_exons[(cid, strand_int, gid, mt.group(1))].append((s, e))

    # Build interval start arrays for fast overlap lookup
    refseq_iv_sorted = {}
    refseq_max_end = {}
    for k, ivs in refseq_iv.items():
        ivs.sort(key=lambda x: x[0])
        starts = [x[0] for x in ivs]
        refseq_iv_sorted[k] = (starts, ivs)
        refseq_max_end[k] = max(e for _, e, _ in ivs)

    def find_refseq_tpm_overlap(cid: int, strand_int: int, gs: int, ge: int) -> int:
        entry = refseq_iv_sorted.get((cid, strand_int))
        if entry is None:
            return -1
        starts, ivs = entry
        # Backtrack conservatively using binary search
        hi = bisect.bisect_right(starts, ge)
        best_idx = -1
        best_ov = 0
        best_pan = -1.0
        # Scan backward from hi-1 while end could overlap; track window ~2Mb max gene length
        for j in range(hi - 1, -1, -1):
            rs, re_, tidx = ivs[j]
            if gs - rs > 2_500_000:
                break
            ov = min(ge, re_) - max(gs, rs) + 1
            if ov > 0:
                pan_val = float(tpm_pan[tidx])
                if ov > best_ov or (ov == best_ov and pan_val > best_pan):
                    best_ov = ov
                    best_pan = pan_val
                    best_idx = tidx
        return best_idx

    # Load Ensembl 113 symbols
    ens_sym_df = pd.read_csv(ENS_SYM, sep="\t", dtype=str)
    ens_to_sym = {}
    for _, r in ens_sym_df.iterrows():
        sym = str(r["gene_symbol"]) if pd.notna(r["gene_symbol"]) else "-"
        if sym != "-" and sym:
            ens_to_sym[str(r["gene_id"])] = sym

    ens_gene_to_tpm = {}
    ens_tx_exons = defaultdict(list)

    with gzip.open(ENS_GTF, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split("\t", 8)
            ftype = f[2]
            if ftype not in ("gene", "exon"):
                continue
            cid = parse_chr_to_cid(f[0])
            if cid == 0:
                continue
            strand_int = 1 if f[6] == "+" else (-1 if f[6] == "-" else 0)
            if strand_int == 0:
                continue
            s = int(f[3])
            e = int(f[4])
            attr = f[8]
            mg = re_gid.search(attr)
            if not mg:
                continue
            egid = mg.group(1)
            if ftype == "gene":
                tidx = tpm_map.get(egid, -1)
                if tidx < 0:
                    mn = re_gname.search(attr)
                    if mn and mn.group(1) in tpm_map:
                        tidx = tpm_map[mn.group(1)]
                    elif egid in ens_to_sym and ens_to_sym[egid] in tpm_map:
                        tidx = tpm_map[ens_to_sym[egid]]
                    else:
                        tidx = find_refseq_tpm_overlap(cid, strand_int, s, e)
                ens_gene_to_tpm[egid] = tidx
            else:
                mt = re_tid.search(attr)
                if not mt:
                    continue
                ens_tx_exons[(cid, strand_int, egid, mt.group(1))].append((s, e))

    gtf_intron_strand = {}
    ambig_introns = set()
    # key_uint64 -> (best_pan, tpm_idx)
    ann_site_best = {}

    def process_tx_dict(tx_dict, gid_to_tidx_fn):
        n_introns = 0
        for (cid, strand_int, gid, _tid), ex_list in tx_dict.items():
            if len(ex_list) < 2:
                continue
            ex_list.sort()
            # Merge/dedup exact duplicate exon tuples
            dedup = [ex_list[0]]
            for ex in ex_list[1:]:
                if ex != dedup[-1]:
                    dedup.append(ex)
            tidx = gid_to_tidx_fn(gid)
            pan = float(tpm_pan[tidx]) if tidx >= 0 else -1.0
            for k in range(len(dedup) - 1):
                i_s = dedup[k][1]
                i_e = dedup[k + 1][0]
                if i_e - i_s < 4:
                    continue
                n_introns += 1
                ikey = (cid, i_s, i_e)
                if ikey not in ambig_introns:
                    prev = gtf_intron_strand.get(ikey)
                    if prev is None:
                        gtf_intron_strand[ikey] = strand_int
                    elif prev != strand_int:
                        del gtf_intron_strand[ikey]
                        ambig_introns.add(ikey)

                # Annotate donor (2) and acceptor (1)
                if strand_int == 1:
                    pairs = ((i_s, 2), (i_e, 1))
                else:
                    pairs = ((i_e, 2), (i_s, 1))
                sbit = 1 if strand_int == 1 else 0
                for pos, sc in pairs:
                    ukey = (cid << 40) | (pos << 8) | (sbit << 4) | sc
                    cur = ann_site_best.get(ukey)
                    if cur is None or pan > cur[0] or (cur[1] < 0 and tidx >= 0):
                        ann_site_best[ukey] = (pan, tidx)
        return n_introns

    n_ref_in = process_tx_dict(refseq_tx_exons, lambda g: tpm_map.get(g, -1))
    n_ens_in = process_tx_dict(ens_tx_exons, lambda g: ens_gene_to_tpm.get(g, -1))

    ann_keys = np.array(sorted(ann_site_best.keys()), dtype=np.uint64)
    ann_tpm_idx = np.array([ann_site_best[int(k)][1] for k in ann_keys], dtype=np.int32)
    ens_mapped = sum(1 for v in ens_gene_to_tpm.values() if v >= 0)
    stats = {
        "refseq_genes_with_strand": len(refseq_gene_strand),
        "refseq_tx_count": len(refseq_tx_exons),
        "ens113_genes_seen": len(ens_gene_to_tpm),
        "ens113_genes_mapped_to_tpm": ens_mapped,
        "ens113_tx_count": len(ens_tx_exons),
        "gtf_union_unique_introns": len(gtf_intron_strand),
        "gtf_union_unique_splice_sites": len(ann_keys),
        "gtf_sites_with_tpm_mapping": int((ann_tpm_idx >= 0).sum()),
    }
    return refseq_gene_strand, gtf_intron_strand, ann_keys, ann_tpm_idx, stats


def load_clusters_to_genes_strand(refseq_gene_strand: dict):
    """Map (cid, clu_id) and clu_id -> strand_int from clusters_to_genes.txt + RefSeq strand."""
    c2s_chr = {}
    c2s_clu = {}
    with open(CLU2GENE, "r", encoding="utf-8") as fh:
        hdr = fh.readline()
        for line in fh:
            sp = line.strip().split()
            if len(sp) < 2:
                continue
            clu_full, genes_str = sp[0], sp[1]
            parts = clu_full.split(":")
            clu_id = parts[-1]
            cid = parse_chr_to_cid(parts[0]) if len(parts) >= 2 else 0
            for g in genes_str.split(","):
                st = refseq_gene_strand.get(g)
                if st is not None:
                    if cid > 0:
                        c2s_chr[(cid, clu_id)] = st
                    if clu_id not in c2s_clu:
                        c2s_clu[clu_id] = st
                    break
    return c2s_chr, c2s_clu


def parse_args():
    ap = argparse.ArgumentParser(description="Build 15-head cluster-relative sheep splice-site usage atlas")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "00_raw", help="Directory containing ref/ and sheepgtex/ inputs")
    ap.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data" / "01_labels", help="Directory containing intron_agg.npz and gene_tpm_medians_15heads.npz")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "01_labels", help="Output directory for splice_sites_15heads.npz")
    return ap.parse_args()


def main():
    global IN_INTRON_NPZ, IN_TPM_NPZ, FASTA, FAI, REFSEQ_GTF, ENS_GTF, ENS_SYM, CLU2GENE, OUT_NPZ, OUT_META, OUT_JSON
    args = parse_args()
    IN_INTRON_NPZ = args.labels_dir / "intron_agg.npz"
    IN_TPM_NPZ = args.labels_dir / "gene_tpm_medians_15heads.npz"
    FASTA = args.raw_dir / "ref" / "GCF_016772045.1_genomic.fna"
    FAI = args.raw_dir / "ref" / "GCF_016772045.1_genomic.fna.fai"
    REFSEQ_GTF = args.raw_dir / "sheepgtex" / "sheep.gtf.gz"
    ENS_GTF = args.raw_dir / "ref" / "Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.113.gtf.gz"
    ENS_SYM = args.raw_dir / "ref" / "Ovis_aries_rambouillet.113_gene_symbols.tsv"
    CLU2GENE = args.raw_dir / "sheepgtex" / "SheepGTEx_v0.Splicing.clusters_to_genes.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUT_NPZ = args.out_dir / "splice_sites_15heads.npz"
    OUT_META = args.out_dir / "splice_sites_meta.tsv.gz"
    OUT_JSON = args.out_dir / "site_usage_summary.json"
    t0 = time.time()
    print("[s01] Loading gene_tpm_medians_15heads.npz ...", flush=True)
    tpm_npz = np.load(IN_TPM_NPZ, allow_pickle=False)
    tpm_gene_ids = tpm_npz["gene_ids"]
    tpm_medians = tpm_npz["medians"].astype(np.float32)  # [G, 15]
    assert int(tpm_npz["loco_excluded_count"][0]) == 23

    print("[s01] Loading ARS-UI_Ramb_v2.0 chromosomes 1..27 ...", flush=True)
    seqs, chrom_lens = load_main_chrom_seqs()

    print("[s01] Parsing RefSeq 104 U Ensembl 113 GTFs ...", flush=True)
    refseq_g2s, gtf_in_strand, ann_keys, ann_tpm_idx, gtf_stats = load_gtf_union_and_gene_map(
        tpm_gene_ids, tpm_medians[:, 0]
    )
    print(f"  GTF stats: {gtf_stats}", flush=True)
    c2s_chr, c2s_clu = load_clusters_to_genes_strand(refseq_g2s)

    print("[s01] Loading intron_agg.npz ...", flush=True)
    in_npz = np.load(IN_INTRON_NPZ, allow_pickle=True)
    cid = in_npz["chrom_id"].astype(np.int16)
    i_s = in_npz["i_s"].astype(np.int32)
    i_e = in_npz["i_e"].astype(np.int32)
    clu_id = np.array([str(x) for x in in_npz["clu_id"]])  # clean unicode strings
    num_ab = in_npz["num_ab"].astype(np.int64)             # [N, 2, 15]
    nge5_ab = in_npz["nge5_ab"].astype(np.int32)           # [N, 2, 15]
    n_introns = len(cid)
    print(f"  Loaded {n_introns} introns ({time.time() - t0:.1f}s)", flush=True)

    # Step 1: Strand resolution (NEVER use clu_id suffix)
    strand = np.zeros(n_introns, dtype=np.int8)
    is_canon_intron = np.zeros(n_introns, dtype=bool)
    for k in range(n_introns):
        c = int(cid[k])
        s = int(i_s[k])
        e = int(i_e[k])
        if s < 3 or e > chrom_lens[c] - 3 or e - s < 4:
            continue
        seq = seqs[c]
        dl = seq[s : s + 2]
        dr = seq[e - 3 : e - 1]
        pair = (dl, dr)
        if pair in PLUS_DINUCS:
            strand[k] = 1
            is_canon_intron[k] = True
        elif pair in MINUS_DINUCS:
            strand[k] = -1
            is_canon_intron[k] = True

    n_canon = int(is_canon_intron.sum())
    # Priority 1: GTF intron strand
    p1_res = 0
    for k in np.flatnonzero(strand == 0):
        st = gtf_in_strand.get((int(cid[k]), int(i_s[k]), int(i_e[k])))
        if st is not None:
            strand[k] = st
            p1_res += 1

    # Priority 2: PanTissue read-weighted strand majority within (chrom_id, clu_id)
    clu_votes = defaultdict(lambda: [0, 0])  # [plus_reads, minus_reads]
    num_pan_full = num_ab[:, 0, 0] + num_ab[:, 1, 0]
    for k in np.flatnonzero(strand != 0):
        v = clu_votes[(int(cid[k]), clu_id[k])]
        if strand[k] == 1:
            v[0] += int(num_pan_full[k])
        else:
            v[1] += int(num_pan_full[k])

    p2_res = 0
    p3_res = 0
    p4_drop = 0
    for k in np.flatnonzero(strand == 0):
        key_c = (int(cid[k]), clu_id[k])
        v = clu_votes.get(key_c)
        if v is not None and v[0] != v[1]:
            strand[k] = 1 if v[0] > v[1] else -1
            p2_res += 1
            continue
        st = c2s_chr.get(key_c)
        if st is None:
            st = c2s_clu.get(clu_id[k])
        if st is not None:
            strand[k] = st
            p3_res += 1
        else:
            p4_drop += 1

    print(
        f"  Strand resolution: canon={n_canon} ({n_canon/n_introns:.4f}), "
        f"GTF={p1_res}, CluMaj={p2_res}, Clu2Gene={p3_res}, UnresolvedDropped={p4_drop}",
        flush=True,
    )

    keep_idx = np.flatnonzero(strand != 0)
    cid_k = cid[keep_idx]
    s_k = i_s[keep_idx]
    e_k = i_e[keep_idx]
    clu_k = clu_id[keep_idx]
    str_k = strand[keep_idx]

    # Build [N_kept, 45] matrices (0..14 Full, 15..29 Half A, 30..44 Half B)
    CNT = np.empty((len(keep_idx), 45), dtype=np.float64)
    CNT[:, 0:15] = num_ab[keep_idx, 0, :] + num_ab[keep_idx, 1, :]
    CNT[:, 15:30] = num_ab[keep_idx, 0, :]
    CNT[:, 30:45] = num_ab[keep_idx, 1, :]

    NGE5 = np.empty((len(keep_idx), 45), dtype=np.int32)
    NGE5[:, 0:15] = nge5_ab[keep_idx, 0, :] + nge5_ab[keep_idx, 1, :]
    NGE5[:, 15:30] = nge5_ab[keep_idx, 0, :]
    NGE5[:, 30:45] = nge5_ab[keep_idx, 1, :]
    del num_ab, nge5_ab

    # Step 2: Sort kept introns by integer group ID for (chrom_id, clu_id, strand_int)
    print("[s01] Grouping by (chrom_id, clu_id, strand_int) & computing SpliSER U ...", flush=True)
    grp_map = {}
    grp_ids = np.empty(len(keep_idx), dtype=np.int32)
    nxt = 0
    for i in range(len(keep_idx)):
        gk = (int(cid_k[i]), clu_k[i], int(str_k[i]))
        gid = grp_map.get(gk)
        if gid is None:
            gid = nxt
            grp_map[gk] = gid
            nxt += 1
        grp_ids[i] = gid
    del grp_map

    order = np.argsort(grp_ids, kind="mergesort")
    grp_s = grp_ids[order]
    cid_s = cid_k[order]
    s_s = s_k[order]
    e_s = e_k[order]
    str_s = str_k[order]
    CNT = CNT[order]
    NGE5 = NGE5[order]
    del order, grp_ids, cid_k, s_k, e_k, clu_k, str_k

    # Find contiguous group spans
    diffs = np.flatnonzero(grp_s[1:] != grp_s[:-1]) + 1
    starts = np.concatenate(([0], diffs))
    ends = np.concatenate((diffs, [len(grp_s)]))
    sizes = ends - starts

    # Fast vectorized path for K == 1 clusters
    k1_pos_idx = starts[sizes == 1]
    n_k1 = len(k1_pos_idx)
    out_cid = []
    out_pos = []
    out_str = []
    out_sc = []
    out_U = []
    out_denom = []
    out_nge5 = []
    out_beta_pos = []

    if n_k1 > 0:
        c_k1 = CNT[k1_pos_idx]
        u_k1 = np.where(c_k1 > 0.0, 1.0, 0.0).astype(np.float32)
        n5_k1 = NGE5[k1_pos_idx].astype(np.int32)
        st_k1 = str_s[k1_pos_idx]
        # Left site (i_s): + strand -> Donor(2), - strand -> Acceptor(1)
        sc_left = np.where(st_k1 == 1, 2, 1).astype(np.uint8)
        sc_right = np.where(st_k1 == 1, 1, 2).astype(np.uint8)
        bp_zero = np.zeros((n_k1, 15), dtype=np.uint8)

        out_cid.extend([cid_s[k1_pos_idx], cid_s[k1_pos_idx]])
        out_pos.extend([s_s[k1_pos_idx], e_s[k1_pos_idx]])
        out_str.extend([st_k1, st_k1])
        out_sc.extend([sc_left, sc_right])
        out_U.extend([u_k1, u_k1])
        out_denom.extend([c_k1, c_k1])
        out_nge5.extend([n5_k1, n5_k1])
        out_beta_pos.extend([bp_zero, bp_zero])

    # General vectorized SpliSER path for K >= 2 clusters
    multi_mask = sizes >= 2
    m_starts = starts[multi_mask]
    m_ends = ends[multi_mask]
    m_cid, m_pos, m_str, m_sc = [], [], [], []
    m_U, m_den, m_n5, m_bp = [], [], [], []

    for lo, hi in zip(m_starts, m_ends):
        c_val = int(cid_s[lo])
        st_val = int(str_s[lo])
        sa = s_s[lo:hi]
        ea = e_s[lo:hi]
        ca = CNT[lo:hi]         # [K, 45]
        n5a = NGE5[lo:hi].max(axis=0)  # identical cluster den>=5 counts across cluster introns

        for is_left, (side, partner) in ((True, (sa, ea)), (False, (ea, sa))):
            sc_val = (2 if st_val == 1 else 1) if is_left else (1 if st_val == 1 else 2)
            seen = set()
            for idx_k in range(hi - lo):
                pos = int(side[idx_k])
                if pos in seen:
                    continue
                seen.add(pos)
                use_m = side == pos
                span_m = (sa < pos) & (pos < ea)
                comp_m = (~use_m) & (~span_m)

                alpha = ca[use_m].sum(axis=0)
                beta1 = ca[span_m].sum(axis=0) if span_m.any() else np.zeros(45, dtype=np.float64)
                beta2 = np.zeros(45, dtype=np.float64)
                if comp_m.any():
                    partners_pos = np.unique(partner[use_m])
                    for p in partners_pos:
                        pm_comp = comp_m & (partner == p)
                        if not pm_comp.any():
                            continue
                        pm_use = use_m & (partner == p)
                        c_ap = ca[pm_use].sum(axis=0)
                        c_bp = ca[pm_comp].sum(axis=0)
                        beta2 += np.where(alpha > 0.0, (c_ap / np.maximum(alpha, 1.0)) * c_bp, 0.0)

                beta_tot = beta1 + beta2
                denom = alpha + beta_tot
                u_val = np.where(denom > 0.0, alpha / np.maximum(denom, 1.0), 0.0).astype(np.float32)

                m_cid.append(c_val)
                m_pos.append(pos)
                m_str.append(st_val)
                m_sc.append(sc_val)
                m_U.append(u_val)
                m_den.append(denom)
                m_n5.append(n5a)
                m_bp.append((beta_tot[:15] > 0.0).astype(np.uint8))

    if m_cid:
        out_cid.append(np.array(m_cid, dtype=np.int16))
        out_pos.append(np.array(m_pos, dtype=np.int32))
        out_str.append(np.array(m_str, dtype=np.int8))
        out_sc.append(np.array(m_sc, dtype=np.uint8))
        out_U.append(np.vstack(m_U))
        out_denom.append(np.vstack(m_den))
        out_nge5.append(np.vstack(m_n5))
        out_beta_pos.append(np.vstack(m_bp))

    del CNT, NGE5, s_s, e_s, cid_s, str_s, grp_s

    all_cid = np.concatenate(out_cid)
    all_pos = np.concatenate(out_pos)
    all_str = np.concatenate(out_str)
    all_sc = np.concatenate(out_sc)
    all_U = np.vstack(out_U)
    all_den = np.vstack(out_denom)
    all_n5 = np.vstack(out_nge5)
    all_bp = np.vstack(out_beta_pos)
    del out_cid, out_pos, out_str, out_sc, out_U, out_denom, out_nge5, out_beta_pos

    print(f"  Raw extracted splice sites: {len(all_cid)} ({time.time() - t0:.1f}s)", flush=True)

    # Step 3: Deduplicate identical (chrom_id, pos, strand_int, site_class) keeping max Full PanTissue denom
    all_keys = pack_keys(all_cid, all_pos, all_str, all_sc)
    sort_idx = np.lexsort((-all_den[:, 0], all_keys))
    keys_sorted = all_keys[sort_idx]
    keep_first = np.empty(len(keys_sorted), dtype=bool)
    keep_first[0] = True
    keep_first[1:] = keys_sorted[1:] != keys_sorted[:-1]
    u_idx = sort_idx[keep_first]

    u_keys = all_keys[u_idx]
    e_cid = all_cid[u_idx]
    e_pos = all_pos[u_idx]
    e_str = all_str[u_idx]
    e_sc = all_sc[u_idx]
    e_U = all_U[u_idx]
    e_den = all_den[u_idx]
    e_n5 = all_n5[u_idx]
    e_bp = all_bp[u_idx]
    del all_keys, sort_idx, keys_sorted, keep_first, u_idx
    del all_cid, all_pos, all_str, all_sc, all_U, all_den, all_n5, all_bp

    # Coverage masks
    tau_s_full = np.full(15, 20, dtype=np.int32)
    tau_s_full[13] = 8
    tau_r_full = np.full(15, 20.0, dtype=np.float64)
    tau_r_full[13] = 10.0

    tau_s_half = np.full(15, 10, dtype=np.int32)
    tau_s_half[13] = 4
    tau_r_half = np.full(15, 10.0, dtype=np.float64)
    tau_r_half[13] = 5.0

    m_full = ((e_n5[:, 0:15] >= tau_s_full) & (e_den[:, 0:15] >= tau_r_full)).astype(np.uint8)
    m_A = ((e_n5[:, 15:30] >= tau_s_half) & (e_den[:, 15:30] >= tau_r_half)).astype(np.uint8)
    m_B = ((e_n5[:, 30:45] >= tau_s_half) & (e_den[:, 30:45] >= tau_r_half)).astype(np.uint8)

    # Filter empirical sites:
    #   source=0 if annotated in RefSeq 104 U Ensembl 113 AND covered in >=1 head
    #   source=1 if unannotated canonical dinucleotide AND covered in >=3 of Heads 1..12
    is_ann = np.isin(u_keys, ann_keys)
    is_canon_site = np.zeros(len(u_keys), dtype=bool)
    for idx in range(len(u_keys)):
        c = int(e_cid[idx])
        p = int(e_pos[idx])
        st = int(e_str[idx])
        sc = int(e_sc[idx])
        if p < 3 or p > chrom_lens[c] - 3:
            continue
        seq = seqs[c]
        if st == 1:
            if sc == 2:
                is_canon_site[idx] = seq[p : p + 2] in CANON_DONOR_PLUS
            else:
                is_canon_site[idx] = seq[p - 3 : p - 1] in CANON_ACC_PLUS
        else:
            if sc == 2:
                is_canon_site[idx] = seq[p - 3 : p - 1] in CANON_DONOR_MINUS_GENOMIC
            else:
                is_canon_site[idx] = seq[p : p + 2] in CANON_ACC_MINUS_GENOMIC

    n_main_cov = m_full[:, 1:13].sum(axis=1)
    has_any_cov = m_full.sum(axis=1) >= 1
    keep_ann = is_ann & has_any_cov
    keep_nov = (~is_ann) & is_canon_site & (n_main_cov >= 3)
    emp_keep = keep_ann | keep_nov

    e_src = np.where(keep_ann[emp_keep], 0, 1).astype(np.uint8)
    emp_keys_kept = u_keys[emp_keep]
    emp_cid = e_cid[emp_keep]
    emp_pos = e_pos[emp_keep]
    emp_str = e_str[emp_keep]
    emp_sc = e_sc[emp_keep]
    emp_U_full = e_U[emp_keep, 0:15].astype(np.float32)
    emp_m_full = m_full[emp_keep]
    emp_bp = e_bp[emp_keep]
    emp_U_A = e_U[emp_keep, 15:30].astype(np.float32)
    emp_m_A = m_A[emp_keep]
    emp_U_B = e_U[emp_keep, 30:45].astype(np.float32)
    emp_m_B = m_B[emp_keep]

    print(
        f"  Empirical sites kept: total={len(e_src)} (ann={int((e_src==0).sum())}, novel={int((e_src==1).sum())}; "
        f"dropped_unann_or_uncov={int((~emp_keep).sum())})",
        flush=True,
    )

    # Step 4: Add Constitutive Annotated Sites (source=2) where host gene median TPM >= 1.0
    print("[s01] Adding constitutive GTF sites (source=2, TPM >= 1.0) ...", flush=True)
    missing_mask = (~np.isin(ann_keys, emp_keys_kept)) & (ann_tpm_idx >= 0)
    cand_keys = ann_keys[missing_mask]
    cand_tidx = ann_tpm_idx[missing_mask]

    cand_cid = (cand_keys >> 40).astype(np.int16)
    cand_pos = ((cand_keys >> 8) & 0xFFFFFFFF).astype(np.int32)
    cand_str = np.where(((cand_keys >> 4) & 1) == 1, 1, -1).astype(np.int8)
    cand_sc = (cand_keys & 0xF).astype(np.uint8)

    cand_tpm = tpm_medians[cand_tidx]  # [C, 15]
    cand_mask_tpm = cand_tpm >= 1.0
    has_expr = cand_mask_tpm.any(axis=1)

    is_canon_const = np.zeros(len(cand_keys), dtype=bool)
    for idx in np.flatnonzero(has_expr):
        c = int(cand_cid[idx])
        p = int(cand_pos[idx])
        st = int(cand_str[idx])
        sc = int(cand_sc[idx])
        if p < 3 or p > chrom_lens[c] - 3:
            continue
        seq = seqs[c]
        if st == 1:
            if sc == 2:
                is_canon_const[idx] = seq[p : p + 2] in CANON_DONOR_PLUS
            else:
                is_canon_const[idx] = seq[p - 3 : p - 1] in CANON_ACC_PLUS
        else:
            if sc == 2:
                is_canon_const[idx] = seq[p - 3 : p - 1] in CANON_DONOR_MINUS_GENOMIC
            else:
                is_canon_const[idx] = seq[p : p + 2] in CANON_ACC_MINUS_GENOMIC

    c_keep = has_expr & is_canon_const
    n_c2 = int(c_keep.sum())
    c2_cid = cand_cid[c_keep]
    c2_pos = cand_pos[c_keep]
    c2_str = cand_str[c_keep]
    c2_sc = cand_sc[c_keep]
    c2_src = np.full(n_c2, 2, dtype=np.uint8)
    c2_m_full = cand_mask_tpm[c_keep].astype(np.uint8)
    c2_U_full = np.where(c2_m_full == 1, 1.0, 0.0).astype(np.float32)
    c2_zeros_f = np.zeros((n_c2, 15), dtype=np.float32)
    c2_zeros_u = np.zeros((n_c2, 15), dtype=np.uint8)

    print(f"  Constitutive GTF sites added (source=2): {n_c2}", flush=True)

    # Concatenate and sort by (chrom_id, pos, strand_int, site_class)
    f_cid = np.concatenate([emp_cid, c2_cid])
    f_pos = np.concatenate([emp_pos, c2_pos])
    f_str = np.concatenate([emp_str, c2_str])
    f_sc = np.concatenate([emp_sc, c2_sc])
    f_src = np.concatenate([e_src, c2_src])
    f_Uf = np.vstack([emp_U_full, c2_U_full])
    f_mf = np.vstack([emp_m_full, c2_m_full])
    f_bp = np.vstack([emp_bp, c2_zeros_u])
    f_UA = np.vstack([emp_U_A, c2_zeros_f])
    f_mA = np.vstack([emp_m_A, c2_zeros_u])
    f_UB = np.vstack([emp_U_B, c2_zeros_f])
    f_mB = np.vstack([emp_m_B, c2_zeros_u])

    final_order = np.lexsort((f_sc, f_str, f_pos, f_cid))
    f_cid = f_cid[final_order]
    f_pos = f_pos[final_order]
    f_str = f_str[final_order]
    f_sc = f_sc[final_order]
    f_src = f_src[final_order]
    f_Uf = f_Uf[final_order]
    f_mf = f_mf[final_order]
    f_bp = f_bp[final_order]
    f_UA = f_UA[final_order]
    f_mA = f_mA[final_order]
    f_UB = f_UB[final_order]
    f_mB = f_mB[final_order]

    # Mask un-covered empirical U values to 0.0 for tidy storage
    f_Uf = np.where(f_mf == 1, f_Uf, 0.0).astype(np.float32)
    f_UA = np.where(f_mA == 1, f_UA, 0.0).astype(np.float32)
    f_UB = np.where(f_mB == 1, f_UB, 0.0).astype(np.float32)

    # Step 5: Mandatory SheepCompactDataset Symmetry Check (2,000 '+' and 2,000 '-' sites)
    print("[s01] Running SheepCompactDataset symmetry check (2000 '+' and 2000 '-') ...", flush=True)
    rng = np.random.default_rng(42)
    sym_stats = {}
    for s_val, s_label in ((1, "plus"), (-1, "minus")):
        eligible = []
        for idx in np.flatnonzero(f_str == s_val):
            p = int(f_pos[idx])
            clen = chrom_lens[int(f_cid[idx])]
            if 6000 <= p <= clen - 6000:
                eligible.append(idx)
        eligible = np.array(eligible, dtype=np.int64)
        sample_idx = rng.choice(eligible, size=min(2000, len(eligible)), replace=False)

        n_don, n_don_ok = 0, 0
        n_acc, n_acc_ok = 0, 0
        for idx in sample_idx:
            c = int(f_cid[idx])
            pos = int(f_pos[idx])
            sc = int(f_sc[idx])
            seq = seqs[c]
            g0 = pos - 1
            w_start = (g0 // 5000) * 5000
            win_seq = seq[w_start - 5000 : w_start + 10000]
            if s_val == -1:
                win_seq = rc_bytes(win_seq)
                rel_pos = (w_start + 5000 - 1) - g0
            else:
                rel_pos = g0 - w_start
            p_win = 5000 + rel_pos
            if sc == 2:
                n_don += 1
                di = win_seq[p_win + 1 : p_win + 3]
                if di in CANON_DONOR_PLUS:
                    n_don_ok += 1
            else:
                n_acc += 1
                di = win_seq[p_win - 2 : p_win]
                if di in CANON_ACC_PLUS:
                    n_acc_ok += 1

        don_rate = n_don_ok / max(1, n_don)
        acc_rate = n_acc_ok / max(1, n_acc)
        tot_rate = (n_don_ok + n_acc_ok) / max(1, n_don + n_acc)
        sym_stats[s_label] = {
            "checked": int(n_don + n_acc),
            "donor_checked": int(n_don),
            "donor_canon_rate": round(don_rate, 6),
            "acceptor_checked": int(n_acc),
            "acceptor_canon_rate": round(acc_rate, 6),
            "overall_canon_rate": round(tot_rate, 6),
        }
        assert tot_rate >= 0.95 and don_rate >= 0.95 and acc_rate >= 0.95, (
            f"SheepCompactDataset symmetry check FAILED on {s_label}: {sym_stats[s_label]}"
        )
    print(f"  Symmetry PASSED: {sym_stats}", flush=True)

    # Step 6: Save outputs (strictly non-object dtypes in NPZ!)
    head_arr = np.array(HEAD_NAMES, dtype="U32")
    np.savez_compressed(
        OUT_NPZ,
        chrom_id=f_cid,
        pos=f_pos,
        strand=f_str,
        site_class=f_sc,
        source=f_src,
        U_full=f_Uf,
        mask_full=f_mf,
        beta_pos=f_bp,
        U_A=f_UA,
        mask_A=f_mA,
        U_B=f_UB,
        mask_B=f_mB,
        head_names=head_arr,
    )
    # Assert allow_pickle=False works cleanly
    chk = np.load(OUT_NPZ, allow_pickle=False)
    assert len(chk["pos"]) == len(f_pos)

    df_meta = pd.DataFrame({
        "chrom_id": f_cid,
        "pos": f_pos,
        "strand": np.where(f_str == 1, "+", "-"),
        "site_class": np.where(f_sc == 2, "donor", "acceptor"),
        "source": f_src,
        "pan_U": np.round(f_Uf[:, 0], 4),
        "pan_mask": f_mf[:, 0],
        "n_main_tissues_covered": f_mf[:, 1:13].sum(axis=1),
    })
    df_meta.to_csv(OUT_META, sep="\t", index=False, compression="gzip")

    summary = {
        "input_introns_main_chr": int(n_introns),
        "intron_strand_canonical_dinuc": n_canon,
        "intron_strand_gtf_resolved": p1_res,
        "intron_strand_cluster_majority_resolved": p2_res,
        "intron_strand_clusters_to_genes_resolved": p3_res,
        "intron_strand_unresolved_dropped": p4_drop,
        "gtf_stats": gtf_stats,
        "total_final_splice_sites": int(len(f_pos)),
        "source_counts": {
            "0_annotated_empirical": int((f_src == 0).sum()),
            "1_novel_canonical_empirical": int((f_src == 1).sum()),
            "2_constitutive_gtf_tpm_ge1": int((f_src == 2).sum()),
        },
        "site_class_counts": {
            "1_acceptor": int((f_sc == 1).sum()),
            "2_donor": int((f_sc == 2).sum()),
        },
        "strand_counts": {
            "plus": int((f_str == 1).sum()),
            "minus": int((f_str == -1).sum()),
        },
        "sheep_compact_dataset_symmetry": sym_stats,
        "npz_bytes": int(OUT_NPZ.stat().st_size),
        "meta_tsv_gz_bytes": int(OUT_META.stat().st_size),
        "allow_pickle_false_verified": True,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as w:
        json.dump(summary, w, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
