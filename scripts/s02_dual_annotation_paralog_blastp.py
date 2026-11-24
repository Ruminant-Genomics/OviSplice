#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s02_dual_annotation_paralog_blastp.py
=====================================
Extracts canonical/longest-CDS protein sequences from dual sheep annotations (`Ensembl 113` +
`NCBI RefSeq 104`) on `chr1..chr26, chrX` (`--step extract`) and computes the 5-fold LOCO
cross-fold paralog bitmask from all-vs-all BLASTP hits (`--step parse`; E <= 1e-5, >=50% identity
over >=50% shorter-protein coverage).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FASTA = REPO_ROOT / "data" / "00_raw" / "ref" / "GCF_016772045.1_genomic.fna"
FAI = REPO_ROOT / "data" / "00_raw" / "ref" / "GCF_016772045.1_genomic.fna.fai"
REFSEQ_GTF = REPO_ROOT / "data" / "00_raw" / "sheepgtex" / "sheep.gtf.gz"
ENS_GTF = REPO_ROOT / "data" / "00_raw" / "ref" / "Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.113.gtf.gz"
FOLDS_JSON = REPO_ROOT / "data" / "pack" / "folds.json"
TMP_DIR = REPO_ROOT / "data" / "pack" / "_tmp_prot"
OUT_TSV = REPO_ROOT / "data" / "pack" / "paralog_genes.tsv.gz"
OUT_JSON = REPO_ROOT / "results" / "paralog_summary.json"

COMP_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def rc_str(s: str) -> str:
    return s.translate(COMP_TRANS)[::-1]


def parse_chr_to_cid(chrom_str: str) -> int:
    s = chrom_str[3:] if chrom_str.startswith("chr") else chrom_str
    if s == "X":
        return 27
    if s.isdigit():
        v = int(s)
        if 1 <= v <= 26:
            return v
    return 0


def cid_to_chr_name(cid: int) -> str:
    return "chrX" if cid == 27 else f"chr{cid}"


def load_main_chrom_strs():
    seqs = {}
    with open(FAI, "r", encoding="utf-8") as fai_fh, open(FASTA, "rb") as fa_fh:
        for idx, line in enumerate(fai_fh):
            if idx >= 27:
                break
            cid = idx + 1
            parts = line.split()
            length, offset, bl, bw = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            n_bytes = (length // bl) * bw + (length % bl)
            fa_fh.seek(offset)
            raw = fa_fh.read(n_bytes)
            seqs[cid] = raw.translate(None, b"\r\n").decode("ascii").upper()
    return seqs


def translate_cds(dna: str) -> str:
    aa = []
    lim = len(dna) - (len(dna) % 3)
    for i in range(0, lim, 3):
        c = CODON_TABLE.get(dna[i : i + 3], "X")
        aa.append(c)
    if aa and aa[-1] == "*":
        aa.pop()
    return "".join(c if c != "*" else "X" for c in aa)


def collect_gtf_cds(gtf_path: str, prefix: str):
    pc_genes = {}
    tx_cds = defaultdict(list)
    re_gid = re.compile(r'gene_id "([^"]+)"')
    re_tid = re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(gtf_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split("\t", 8)
            ftype = f[2]
            if ftype not in ("gene", "CDS"):
                continue
            cid = parse_chr_to_cid(f[0])
            if cid == 0 or f[6] not in ("+", "-"):
                continue
            attr = f[8]
            mg = re_gid.search(attr)
            if not mg:
                continue
            gid = f"{prefix}:{mg.group(1)}"
            if ftype == "gene":
                if 'gene_biotype "protein_coding"' in attr:
                    pc_genes[gid] = (cid, f[6], int(f[3]), int(f[4]))
            else:
                if gid not in pc_genes:
                    continue
                mt = re_tid.search(attr)
                if not mt:
                    continue
                frame = int(f[7]) if f[7] in ("0", "1", "2") else 0
                tx_cds[(gid, mt.group(1))].append((int(f[3]), int(f[4]), frame))
    return pc_genes, tx_cds


def step_extract():
    t0 = time.time()
    os.makedirs(TMP_DIR, exist_ok=True)
    print("[s02] Loading sheep genome chromosomes 1..27 ...", flush=True)
    seqs = load_main_chrom_strs()

    meta_rows = []
    aa_to_uid = {}
    unique_pairs = []

    for src_label, gtf_path, prefix in (
        ("Ensembl113", ENS_GTF, "ENS"),
        ("RefSeq104", REFSEQ_GTF, "RS"),
    ):
        pc_genes, tx_cds = collect_gtf_cds(gtf_path, prefix)
        best_tx = {}  # gid -> (total_len, cds_list)
        for (gid, tid), segs in tx_cds.items():
            tlen = sum(e - s + 1 for s, e, _ in segs)
            cur = best_tx.get(gid)
            if cur is None or tlen > cur[0]:
                best_tx[gid] = (tlen, segs)

        n_written = 0
        for gid, (cid, strand, gs, ge) in pc_genes.items():
            ent = best_tx.get(gid)
            if ent is None:
                continue
            segs = sorted(ent[1], key=lambda x: (x[0], x[1]), reverse=(strand == "-"))
            seq = seqs[cid]
            frags = []
            first_frame = segs[0][2]
            for s, e, _fr in segs:
                sub = seq[s - 1 : e]
                frags.append(rc_str(sub) if strand == "-" else sub)
            dna = "".join(frags)
            if first_frame > 0 and len(dna) > first_frame:
                dna = dna[first_frame:]
            aa = translate_cds(dna)
            if len(aa) < 20:
                continue
            uid = aa_to_uid.get(aa)
            if uid is None:
                uid = f"U{len(unique_pairs)}"
                aa_to_uid[aa] = uid
                unique_pairs.append((uid, aa))
            meta_rows.append((gid, uid, src_label, cid, cid_to_chr_name(cid), strand, gs, ge, len(aa)))
            n_written += 1
        print(f"  {src_label}: pc_genes={len(pc_genes)}, written_proteins={n_written}", flush=True)

    uniq_faa_path = os.path.join(TMP_DIR, "unique_proteins.faa")
    with open(uniq_faa_path, "w", encoding="utf-8", newline="\n") as fw:
        for uid, aa in unique_pairs:
            fw.write(f">{uid}\n{aa}\n")

    # Sort descending by length and distribute round-robin across 24 shards
    sorted_pairs = sorted(unique_pairs, key=lambda x: len(x[1]), reverse=True)
    shard_fhs = [
        open(os.path.join(TMP_DIR, f"q_{s:02d}.faa"), "w", encoding="utf-8", newline="\n")
        for s in range(24)
    ]
    try:
        for idx, (uid, aa) in enumerate(sorted_pairs):
            shard_fhs[idx % 24].write(f">{uid}\n{aa}\n")
    finally:
        for fh in shard_fhs:
            fh.close()

    meta_path = os.path.join(TMP_DIR, "gene_meta.tsv")
    df_meta = pd.DataFrame(
        meta_rows,
        columns=["gid", "uid", "source", "chrom_id", "chr_name", "strand", "g_start", "g_end", "aa_len"],
    )
    df_meta.to_csv(meta_path, sep="\t", index=False)
    print(f"[s02] Wrote {len(df_meta)} genes ({len(unique_pairs)} unique proteins) across 24 shards ({time.time() - t0:.1f}s)", flush=True)


def step_parse():
    t0 = time.time()
    meta_path = os.path.join(TMP_DIR, "gene_meta.tsv")
    hits_path = os.path.join(TMP_DIR, "hits.tsv")
    df_meta = pd.read_csv(meta_path, sep="\t")
    with open(FOLDS_JSON, "r", encoding="utf-8") as fh:
        folds = json.load(fh)["folds"]

    uid_to_genes = defaultdict(list)
    for gid, uid, cname in zip(df_meta["gid"], df_meta["uid"], df_meta["chr_name"]):
        uid_to_genes[uid].append((gid, cname))
    n_unique = int(df_meta["uid"].nunique())

    uid_hit_chrs = defaultdict(set)
    seen_self = set()
    n_raw_hits = 0
    n_qual_hits = 0

    with open(hits_path, "r", encoding="utf-8") as fh:
        for line in fh:
            n_raw_hits += 1
            parts = line.rstrip("\r\n").split("\t")
            q, s = parts[0], parts[1]
            if q == s:
                seen_self.add(q)
                continue
            pident = float(parts[2])
            alen = int(parts[3])
            qlen = int(parts[4])
            slen = int(parts[5])
            eval_ = float(parts[6])
            cov = alen / min(qlen, slen)
            if pident >= 50.0 and eval_ <= 1e-5 and cov >= 0.50:
                s_genes = uid_to_genes.get(s)
                if s_genes:
                    for _sgid, sc in s_genes:
                        uid_hit_chrs[q].add(sc)
                    n_qual_hits += 1

    assert len(seen_self) >= int(0.995 * n_unique), f"Incomplete BLASTP self-hits: {len(seen_self)}/{n_unique}"

    train_sets = {k: set(folds[f"fold_{k}"]["train_chroms"]) for k in range(1, 6)}
    test_sets = {k: set(folds[f"fold_{k}"]["test_chroms"]) for k in range(1, 6)}

    masks = []
    for gid, uid, qchr in zip(df_meta["gid"], df_meta["uid"], df_meta["chr_name"]):
        targets = set(uid_hit_chrs.get(uid, ())) | {ochr for ogid, ochr in uid_to_genes[uid] if ogid != gid}
        bm = 0
        if targets:
            for k in range(1, 6):
                tset = train_sets[k]
                if (qchr not in tset) and (targets & tset):
                    bm |= 1 << (k - 1)
        masks.append(bm)
    df_meta["paralog_bitmask"] = masks
    df_meta.to_csv(OUT_TSV, sep="\t", index=False, compression="gzip")

    assert (df_meta[df_meta["source"] == "Ensembl113"].groupby("chrom_id")["paralog_bitmask"].apply(lambda s: (s > 0).sum()) > 0).all(), "Ensembl113 missing paralogs on some chrom"
    assert (df_meta[df_meta["source"] == "RefSeq104"].groupby("chrom_id")["paralog_bitmask"].apply(lambda s: (s > 0).sum()) > 0).all(), "RefSeq104 missing paralogs on some chrom"

    per_fold = {}
    for k in range(1, 6):
        bit = 1 << (k - 1)
        sub_all = df_meta[df_meta["chr_name"].isin(test_sets[k])]
        flagged_all = int(((sub_all["paralog_bitmask"] & bit) != 0).sum())
        sub_ens = sub_all[sub_all["source"] == "Ensembl113"]
        flagged_ens = int(((sub_ens["paralog_bitmask"] & bit) != 0).sum())
        sub_rs = sub_all[sub_all["source"] == "RefSeq104"]
        flagged_rs = int(((sub_rs["paralog_bitmask"] & bit) != 0).sum())
        per_fold[f"fold_{k}"] = {
            "test_pc_genes_union": int(len(sub_all)),
            "paralog_flagged_union": flagged_all,
            "paralog_free_union": int(len(sub_all) - flagged_all),
            "test_pc_genes_ens113": int(len(sub_ens)),
            "paralog_flagged_ens113": flagged_ens,
            "paralog_free_ens113": int(len(sub_ens) - flagged_ens),
            "test_pc_genes_refseq104": int(len(sub_rs)),
            "paralog_flagged_refseq104": flagged_rs,
            "paralog_free_refseq104": int(len(sub_rs) - flagged_rs),
        }

    ens_flagged_tot = int(((df_meta["source"] == "Ensembl113") & (df_meta["paralog_bitmask"] > 0)).sum())
    rs_flagged_tot = int(((df_meta["source"] == "RefSeq104") & (df_meta["paralog_bitmask"] > 0)).sum())
    summary = {
        "total_proteins": int(len(df_meta)),
        "unique_proteins": n_unique,
        "self_hit_uids": int(len(seen_self)),
        "ens113_proteins": int((df_meta["source"] == "Ensembl113").sum()),
        "refseq104_proteins": int((df_meta["source"] == "RefSeq104").sum()),
        "ens113_flagged_total": ens_flagged_tot,
        "refseq104_flagged_total": rs_flagged_tot,
        "raw_blastp_hits": n_raw_hits,
        "qualifying_cross_gene_hits_id50_cov50": n_qual_hits,
        "genes_with_any_cross_fold_paralog": int((df_meta["paralog_bitmask"] > 0).sum()),
        "per_fold": per_fold,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as w:
        json.dump(summary, w, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    global FASTA, FAI, REFSEQ_GTF, ENS_GTF, FOLDS_JSON, TMP_DIR, OUT_TSV, OUT_JSON
    ap = argparse.ArgumentParser(description="Dual-annotation sheep proteome extraction and 5-fold BLASTP paralog masking")
    ap.add_argument("--step", choices=["extract", "parse"], required=True, help="Pipeline stage: extract proteins or parse BLASTP hits")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "00_raw", help="Directory containing ref/ and sheepgtex/ inputs")
    ap.add_argument("--folds-json", type=Path, default=REPO_ROOT / "data" / "pack" / "folds.json", help="Path to 5-fold chromosome split JSON")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "data" / "pack" / "_tmp_prot", help="Working directory for protein FASTA shards and BLASTP hits.tsv")
    ap.add_argument("--out-tsv", type=Path, default=REPO_ROOT / "data" / "pack" / "paralog_genes.tsv.gz", help="Output gzipped paralog bitmask TSV")
    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "results" / "paralog_summary.json", help="Output summary JSON")
    args = ap.parse_args()
    FASTA = args.raw_dir / "ref" / "GCF_016772045.1_genomic.fna"
    FAI = args.raw_dir / "ref" / "GCF_016772045.1_genomic.fna.fai"
    REFSEQ_GTF = args.raw_dir / "sheepgtex" / "sheep.gtf.gz"
    ENS_GTF = args.raw_dir / "ref" / "Ovis_aries_rambouillet.ARS-UI_Ramb_v2.0.113.gtf.gz"
    FOLDS_JSON = args.folds_json
    TMP_DIR = str(args.work_dir)
    OUT_TSV = str(args.out_tsv)
    OUT_JSON = str(args.out_json)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.step == "extract":
        step_extract()
    else:
        step_parse()


if __name__ == "__main__":
    main()
