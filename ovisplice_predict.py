#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ovisplice_predict.py — Command-Line Interface for OviSplice Multi-Tissue Ruminant Splicing Prediction.

Subcommands:
  1. `demo`     : Run quick verification on synthetic canonical splice-donor/neutral test variants (`examples/`).
  2. `fasta`    : Predict 3-class splice-site probabilities & 15-head usage across FASTA sequences.
  3. `variants` : Score SNVs from a TSV table (using an indexed reference FASTA or pre-extracted 11,001-bp windows).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovisplice import COMP_U8, HEAD_NAMES, load_ensemble, seq_to_u8
from ovisplice.native_engine import FastaIndexedReader


def resolve_checkpoints(ensemble: str, custom_ckpts: list[str] | None) -> list[Path]:
    if custom_ckpts:
        return [Path(p) for p in custom_ckpts]
    cdir = REPO_ROOT / "checkpoints"
    if ensemble == "full":
        paths = [cdir / f"ovisplice_full_seed{s}.pt" for s in (42, 1042, 2042)]
    elif ensemble.startswith("sheep_fold"):
        f = int(ensemble.replace("sheep_fold", ""))
        paths = [cdir / f"ovisplice_fold{f}_seed{s}.pt" for s in (42, 1042)]
    elif ensemble.startswith("siteonly_fold"):
        f = int(ensemble.replace("siteonly_fold", ""))
        paths = [cdir / f"ovisplice_siteonly_fold{f}_seed42.pt"]
    elif ensemble == "scratch_fold1":
        paths = [cdir / "ovisplice_scratch_fold1_seed42.pt"]
    else:
        raise ValueError(f"Unknown ensemble preset: {ensemble}")
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint(s) {missing} not found in {cdir}. "
            "Note: Default 3-seed cross-species weights are bundled in GitHub; "
            "download `ovisplice_all_19_checkpoints_v1.0.tar.gz` from Zenodo for LOCO fold/ablation weights."
        )
    return paths


def read_fasta_records(fa_path: Path) -> list[tuple[str, str]]:
    records, hdr, chunks = [], None, []
    with open(fa_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            if s.startswith(">"):
                if hdr is not None:
                    records.append((hdr, "".join(chunks)))
                hdr = s[1:].split()[0]
                chunks = []
            else:
                chunks.append(s)
    if hdr is not None:
        records.append((hdr, "".join(chunks)))
    return records


def cmd_demo(args: argparse.Namespace) -> int:
    t0 = time.time()
    demo_tsv = REPO_ROOT / "examples" / "example_variants.tsv"
    df = pd.read_csv(demo_tsv, sep="\t")

    ckpts_full = resolve_checkpoints("full", None)
    ens_full, backend_name = load_ensemble(ckpts_full, backend=args.backend)
    print(f"[OviSplice] Loaded 3-seed pre-trained ensemble via backend: {backend_name}")

    rows_out = []
    for r in df.itertuples():
        ref_u8 = seq_to_u8(r.seq_11001bp_5to3)[None, :]
        alt_u8 = seq_to_u8(str(r.alt_5to3).upper())
        res = ens_full.score_snvs(ref_u8, alt_u8)
        u500 = res["ds_usage_500"][0]
        max12 = float(np.max(u500[1:13]))
        top_h = int(1 + np.argmax(u500[1:13]))
        rows_out.append({
            "Variant_ID": r.variant_id,
            "Category": r.category,
            "Alleles": f"{r.ref}>{r.alt} ({r.strand})",
            "dS_site_500": round(float(res["ds_site_500"][0]), 4),
            "dS_PanTissue": round(float(u500[0]), 4),
            "dS_max12": round(max12, 4),
            "Top_Organ": f"{HEAD_NAMES[top_h]} ({u500[top_h]:.4f})",
        })

    out_df = pd.DataFrame(rows_out)
    print("\n" + out_df.to_string(index=False))
    assert out_df.loc[0, "dS_max12"] > 0.50 and out_df.loc[1, "dS_max12"] < 0.05, "Unexpected demo output!"
    print(f"\n[PASS] Self-test passed in {time.time() - t0:.2f}s (high-impact donor disruption vs neutral intronic control).")
    return 0


def cmd_fasta(args: argparse.Namespace) -> int:
    ckpts = resolve_checkpoints(args.ensemble, args.ckpt)
    ens, backend_name = load_ensemble(ckpts, backend=args.backend)
    is_so = bool(getattr(ens, "is_site_only", False))
    recs = read_fasta_records(Path(args.input))
    print(f"[OviSplice] Scoring {len(recs)} FASTA record(s) using {len(ckpts)} checkpoint(s) [{backend_name}] ...")
    if is_so:
        print("[OviSplice] Note: site-only checkpoint selected; saving 3-class site probabilities only.")
    out_dict = {"head_names": np.array([] if is_so else HEAD_NAMES)}
    for name, seq in recs:
        u8 = seq_to_u8(seq)
        if len(u8) < 10001:
            pad_l = (10001 - len(u8)) // 2
            pad_r = 10001 - len(u8) - pad_l
            u8 = np.concatenate([np.full(pad_l, 4, dtype=np.uint8), u8, np.full(pad_r, 4, dtype=np.uint8)])
        elif args.pad_flanks:
            pad = np.full(5000, 4, dtype=np.uint8)
            u8 = np.concatenate([pad, u8, pad])
        p_site, u_15 = ens.score_windows(u8[None, :], compute_usage=(not is_so))
        out_dict[f"{name}__site_prob_3ss_5ss"] = p_site[0]
        if not is_so and u_15 is not None:
            out_dict[f"{name}__usage_15heads"] = u_15[0]
    np.savez_compressed(args.output, **out_dict)
    print(f"[OviSplice] Saved nucleotide-resolution predictions to {args.output}")
    return 0


def cmd_variants(args: argparse.Namespace) -> int:
    ckpts = resolve_checkpoints(args.ensemble, args.ckpt)
    ens, backend_name = load_ensemble(ckpts, backend=args.backend)
    is_so = bool(getattr(ens, "is_site_only", False))
    df = pd.read_csv(args.tsv, sep="\t")
    N = len(df)
    ref_wins = np.full((N, 11001), 4, dtype=np.uint8)
    alt_u8 = np.full(N, 4, dtype=np.uint8)

    if args.seq_col and args.seq_col in df.columns:
        for i, r in enumerate(df.itertuples()):
            s_u8 = seq_to_u8(getattr(r, args.seq_col))
            assert len(s_u8) == 11001, f"Row {i}: expected 11,001-bp 5'->3' window, got {len(s_u8)}"
            ref_wins[i] = s_u8
            alt_u8[i] = seq_to_u8(str(getattr(r, args.alt_col))[0])[0]
    else:
        if not args.fasta:
            raise ValueError("Provide either --fasta <indexed.fa> or --seq-col <11001bp_column>.")
        reader = FastaIndexedReader(args.fasta, args.fasta + ".fai")
        for i, r in enumerate(df.itertuples()):
            ch = str(r.chrom)
            pos1 = int(r.pos)
            st = str(getattr(r, "strand", "+"))
            win = reader.fetch_u8(ch, pos1 - 1 - 5500, pos1 - 1 + 5501)
            ab = seq_to_u8(str(r.alt)[0])[0]
            if st in ("-", "-1"):
                win = COMP_U8[win[::-1]]
                ab = int(COMP_U8[ab])
            ref_wins[i] = win
            alt_u8[i] = ab
        reader.close()

    print(f"[OviSplice] Scoring {N} variant(s) (R={args.radius} bp) via {backend_name} ...")
    res = ens.score_snvs(ref_wins, alt_u8)
    use_key = "ds_usage_50" if args.radius == 50 else "ds_usage_500"
    site_key = "ds_site_50" if args.radius == 50 else "ds_site_500"
    umat = res[use_key]

    out = df.drop(columns=[args.seq_col], errors="ignore") if args.seq_col else df.copy()
    out[f"ds_site_{args.radius}"] = np.round(res[site_key], 5)
    if is_so:
        print(f"[OviSplice] Note: site-only checkpoint selected; usage delta columns are set to NaN.")
        out[f"ds_usage_pan_{args.radius}"] = np.nan
        out[f"ds_usage_max12_{args.radius}"] = np.nan
        for hname in HEAD_NAMES:
            out[f"ds_{hname}_{args.radius}"] = np.nan
    else:
        out[f"ds_usage_pan_{args.radius}"] = np.round(umat[:, 0], 5)
        out[f"ds_usage_max12_{args.radius}"] = np.round(np.max(umat[:, 1:13], axis=1), 5)
        for hi, hname in enumerate(HEAD_NAMES):
            out[f"ds_{hname}_{args.radius}"] = np.round(umat[:, hi], 5)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"[OviSplice] Wrote scored table ({len(out)} variants) to {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OviSplice Multi-Tissue Ruminant Splicing Predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Run instant self-test on synthetic example variants")
    p_demo.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")

    p_fa = sub.add_parser("fasta", help="Score FASTA sequences across 3 splice classes + 15 usage heads")
    p_fa.add_argument("--input", required=True, help="Input FASTA file")
    p_fa.add_argument("--output", required=True, help="Output compressed .npz file")
    p_fa.add_argument("--ensemble", default="full", help="Preset: full | sheep_fold1..5 | siteonly_fold1..5 | scratch_fold1")
    p_fa.add_argument("--ckpt", nargs="+", help="Optional custom .pt checkpoint path(s)")
    p_fa.add_argument("--pad-flanks", action="store_true", help="Pad 5,000 bp N flanks on each side")
    p_fa.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")

    p_var = sub.add_parser("variants", help="Score SNVs from a TSV file")
    p_var.add_argument("--tsv", required=True, help="Input TSV (chrom, pos, ref, alt, strand)")
    p_var.add_argument("--fasta", help="Indexed reference FASTA (.fai required) when --seq-col is not used")
    p_var.add_argument("--seq-col", help="Column containing pre-extracted 11,001-bp 5'->3' window")
    p_var.add_argument("--alt-col", default="alt", help="Column containing alternate allele")
    p_var.add_argument("--radius", type=int, choices=[50, 500], default=500, help="Scanning radius in bp (500 or 50)")
    p_var.add_argument("--ensemble", default="full", help="Preset: full | sheep_fold1..5 | siteonly_fold1..5 | scratch_fold1")
    p_var.add_argument("--ckpt", nargs="+", help="Optional custom .pt checkpoint path(s)")
    p_var.add_argument("--backend", choices=["auto", "cpp", "numpy"], default="auto")
    p_var.add_argument("--output", required=True, help="Output scored TSV")

    args = ap.parse_args()
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "fasta":
        return cmd_fasta(args)
    return cmd_variants(args)


if __name__ == "__main__":
    raise SystemExit(main())
