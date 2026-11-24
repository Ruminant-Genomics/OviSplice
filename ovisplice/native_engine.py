# -*- coding: utf-8 -*-
"""
Pure Python + C++17 (ctypes/OpenMP) exact inference & statistical engine for OviSplice.
Zero dependency on PyTorch or SciPy — bit-accurate (<2e-5) to PyTorch reference checkpoints.
"""
import ctypes
import math
import os
import pickle
import subprocess
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BLOCK_SPECS = [(11, 1)] * 4 + [(11, 4)] * 4 + [(21, 10)] * 4 + [(41, 25)] * 4
COMP_U8 = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMP_BASE = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}

HEAD_NAMES = [
    "PanTissue",
    "SkeletalMuscle",
    "Liver",
    "Adipose",
    "SmallIntestine",
    "CNS",
    "Rumen",
    "Pituitary",
    "Lung",
    "Abomasum",
    "Gonad",
    "Heart",
    "Skin",
    "SkeletalMuscle_Prenatal",
    "SkeletalMuscle_Postnatal",
]

_SO_PATH = Path.home() / ".cache" / "ovisplice" / "libovi_native_v1.so"
_CPP_PATH = Path(__file__).resolve().parent / "ovi_NativeEngine.cpp"
_LIB = None


def get_ovi_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    _SO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if (not _SO_PATH.exists()) or (_CPP_PATH.stat().st_mtime > _SO_PATH.stat().st_mtime):
        cmd = [
            "g++", "-O3", "-march=native", "-funroll-loops", "-fopenmp", "-shared", "-fPIC",
            str(_CPP_PATH), "-o", str(_SO_PATH),
        ]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(_SO_PATH))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    vp = ctypes.c_void_p

    lib.create_model.argtypes = [
        ctypes.c_int,
        fp, fp, fp, fp,
        ip, ip,
        fp, fp, fp, fp, fp, fp, fp, fp,
        fp, fp, fp, fp,
        fp,
    ]
    lib.create_model.restype = vp
    lib.destroy_model.argtypes = [vp]
    lib.destroy_model.restype = None

    lib.score_windows_ensemble.argtypes = [
        ctypes.POINTER(vp), ctypes.c_int,
        u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        fp, fp,
    ]
    lib.score_windows_ensemble.restype = None

    lib.score_snvs_lightcone_ensemble.argtypes = [
        ctypes.POINTER(vp), ctypes.c_int,
        u8p, u8p, ip, ip, ctypes.c_int,
        fp, fp, fp, fp, fp, fp, fp,
    ]
    lib.score_snvs_lightcone_ensemble.restype = None

    lib.score_chunk_snvs_ensemble.argtypes = [
        ctypes.POINTER(vp), ctypes.c_int,
        u8p, ctypes.c_int, ip, u8p, ip, ip, ctypes.c_int,
        fp, fp, fp, fp, fp, fp, fp,
    ]
    lib.score_chunk_snvs_ensemble.restype = None
    _LIB = lib
    return lib


def load_pytorch_ckpt_numpy(ckpt_path: str) -> dict:
    """Unpack PyTorch ZIP checkpoint (.pt) directly into float32/int64 NumPy arrays without torch."""
    zf = zipfile.ZipFile(ckpt_path, "r")
    prefix = zf.namelist()[0].split("/")[0]

    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        n = int(np.prod(size)) if len(size) > 0 else 1
        arr = storage[storage_offset : storage_offset + n]
        return arr.reshape(size)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if "Storage" in name:
                return type(name, (), {})
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, stype, key, _, _numel = pid
            raw = zf.read(f"{prefix}/data/{key}")
            st = getattr(stype, "__name__", str(stype))
            if "Half" in st:
                dt = np.float16
            elif "Long" in st:
                dt = np.int64
            elif "Double" in st:
                dt = np.float64
            else:
                dt = np.float32
            return np.frombuffer(raw, dtype=dt).copy()

    with zf.open(f"{prefix}/data.pkl") as f:
        res = Unpickler(f).load()
    zf.close()
    return res


class CppOviEnsemble:
    """Wrapper holding K native C++ OviSplice15Head models."""

    def __init__(self, ckpt_paths: Sequence[str]):
        self.lib = get_ovi_lib()
        self.ptrs = []
        so_flags = []
        for p in ckpt_paths:
            raw = load_pytorch_ckpt_numpy(str(p))
            sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw["model_state"].items()}
            is_so = ("pan_head.weight" not in sd) and (raw.get("head_type") != "linear_15head")
            so_flags.append(is_so)
            is_mlp = 1 if "pan_head.weight" in sd else (-1 if is_so else 0)
            c1w = np.ascontiguousarray(sd["conv1.weight"], dtype=np.float32)
            c1b = np.ascontiguousarray(sd["conv1.bias"], dtype=np.float32)
            skw = np.ascontiguousarray(sd["skip.weight"], dtype=np.float32)
            skb = np.ascontiguousarray(sd["skip.bias"], dtype=np.float32)

            blk_k = np.array([w for w, _ in BLOCK_SPECS], dtype=np.int32)
            blk_d = np.array([d for _, d in BLOCK_SPECS], dtype=np.int32)
            bn1_sc, bn1_sh, bw1, bb1 = [], [], [], []
            bn2_sc, bn2_sh, bw2, bb2 = [], [], [], []
            eps = 1e-5
            for bi in range(16):
                px = f"resblocks.{bi}."
                sc1 = sd[px + "bn1.weight"].astype(np.float32) / np.sqrt(sd[px + "bn1.running_var"].astype(np.float32) + eps)
                sh1 = sd[px + "bn1.bias"].astype(np.float32) - sd[px + "bn1.running_mean"].astype(np.float32) * sc1
                sc2 = sd[px + "bn2.weight"].astype(np.float32) / np.sqrt(sd[px + "bn2.running_var"].astype(np.float32) + eps)
                sh2 = sd[px + "bn2.bias"].astype(np.float32) - sd[px + "bn2.running_mean"].astype(np.float32) * sc2
                bn1_sc.append(sc1); bn1_sh.append(sh1)
                bn2_sc.append(sc2); bn2_sh.append(sh2)
                bw1.append(sd[px + "conv1.weight"].astype(np.float32).ravel())
                bb1.append(sd[px + "conv1.bias"].astype(np.float32).ravel())
                bw2.append(sd[px + "conv2.weight"].astype(np.float32).ravel())
                bb2.append(sd[px + "conv2.bias"].astype(np.float32).ravel())

            bn1_sc = np.ascontiguousarray(np.concatenate(bn1_sc), dtype=np.float32)
            bn1_sh = np.ascontiguousarray(np.concatenate(bn1_sh), dtype=np.float32)
            bw1 = np.ascontiguousarray(np.concatenate(bw1), dtype=np.float32)
            bb1 = np.ascontiguousarray(np.concatenate(bb1), dtype=np.float32)
            bn2_sc = np.ascontiguousarray(np.concatenate(bn2_sc), dtype=np.float32)
            bn2_sh = np.ascontiguousarray(np.concatenate(bn2_sh), dtype=np.float32)
            bw2 = np.ascontiguousarray(np.concatenate(bw2), dtype=np.float32)
            bb2 = np.ascontiguousarray(np.concatenate(bb2), dtype=np.float32)

            cvw = np.ascontiguousarray(np.concatenate([sd[f"convs.{g}.weight"].astype(np.float32).ravel() for g in range(4)]), dtype=np.float32)
            cvb = np.ascontiguousarray(np.concatenate([sd[f"convs.{g}.bias"].astype(np.float32).ravel() for g in range(4)]), dtype=np.float32)
            stw = np.ascontiguousarray(sd["site_head.weight"], dtype=np.float32)
            stb = np.ascontiguousarray(sd["site_head.bias"], dtype=np.float32)

            if is_mlp == -1:
                h_blob = np.zeros(15 * 32 + 15, dtype=np.float32)
            elif is_mlp == 0:
                h_blob = np.ascontiguousarray(np.concatenate([
                    sd["usage_head.weight"].astype(np.float32).ravel(),
                    sd["usage_head.bias"].astype(np.float32).ravel(),
                ]), dtype=np.float32)
            else:
                h_blob = np.ascontiguousarray(np.concatenate([
                    sd["usage_mlp.0.weight"].astype(np.float32).ravel(),
                    sd["usage_mlp.0.bias"].astype(np.float32).ravel(),
                    sd["usage_mlp.3.weight"].astype(np.float32).ravel(),
                    sd["usage_mlp.3.bias"].astype(np.float32).ravel(),
                    sd["pan_head.weight"].astype(np.float32).ravel(),
                    sd["pan_head.bias"].astype(np.float32).ravel(),
                    sd["delta_head.weight"].astype(np.float32).ravel(),
                    sd["delta_head.bias"].astype(np.float32).ravel(),
                ]), dtype=np.float32)

            fp = ctypes.POINTER(ctypes.c_float)
            ip = ctypes.POINTER(ctypes.c_int32)
            ptr = self.lib.create_model(
                is_mlp,
                c1w.ctypes.data_as(fp), c1b.ctypes.data_as(fp),
                skw.ctypes.data_as(fp), skb.ctypes.data_as(fp),
                blk_k.ctypes.data_as(ip), blk_d.ctypes.data_as(ip),
                bn1_sc.ctypes.data_as(fp), bn1_sh.ctypes.data_as(fp),
                bw1.ctypes.data_as(fp), bb1.ctypes.data_as(fp),
                bn2_sc.ctypes.data_as(fp), bn2_sh.ctypes.data_as(fp),
                bw2.ctypes.data_as(fp), bb2.ctypes.data_as(fp),
                cvw.ctypes.data_as(fp), cvb.ctypes.data_as(fp),
                stw.ctypes.data_as(fp), stb.ctypes.data_as(fp),
                h_blob.ctypes.data_as(fp),
            )
            self.ptrs.append(ptr)
        arr_t = ctypes.c_void_p * len(self.ptrs)
        self.c_ptrs = arr_t(*self.ptrs)
        self.is_site_only = bool(so_flags and all(so_flags))

    def __del__(self):
        if getattr(self, "lib", None) and getattr(self, "ptrs", None):
            for p in self.ptrs:
                self.lib.destroy_model(p)
            self.ptrs = []

    def score_windows(self, seqs_u8: np.ndarray, compute_usage: bool = True):
        seqs = np.ascontiguousarray(seqs_u8, dtype=np.uint8)
        N, L_in = seqs.shape
        assert L_in >= 10001
        L_out = L_in - 10000
        p_out = np.zeros((N, L_out, 2), dtype=np.float32)
        u_out = np.zeros((N, L_out, 15), dtype=np.float32) if compute_usage else None
        fp = ctypes.POINTER(ctypes.c_float)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self.lib.score_windows_ensemble(
            self.c_ptrs, len(self.ptrs),
            seqs.ctypes.data_as(u8p), N, L_in, 1 if compute_usage else 0,
            p_out.ctypes.data_as(fp),
            u_out.ctypes.data_as(fp) if compute_usage else ctypes.cast(0, fp),
        )
        return p_out, u_out

    def score_snvs(
        self,
        ref_wins_u8: np.ndarray,
        alt_u8: np.ndarray,
        rel_s: Optional[np.ndarray] = None,
        rel_e: Optional[np.ndarray] = None,
    ) -> dict:
        rw = np.ascontiguousarray(ref_wins_u8, dtype=np.uint8)
        au = np.ascontiguousarray(alt_u8, dtype=np.uint8)
        N, L_in = rw.shape
        assert L_in == 11001 and len(au) == N
        s50 = np.zeros(N, dtype=np.float32)
        s500 = np.zeros(N, dtype=np.float32)
        u50 = np.zeros((N, 15), dtype=np.float32)
        u500 = np.zeros((N, 15), dtype=np.float32)
        uraw500 = np.zeros((N, 15), dtype=np.float32)

        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int32)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        if rel_s is not None and rel_e is not None:
            rs = np.ascontiguousarray(rel_s, dtype=np.int32)
            re = np.ascontiguousarray(rel_e, dtype=np.int32)
            di_u = np.zeros((N, 15), dtype=np.float32)
            di_m = np.zeros((N, 15), dtype=np.float32)
            rs_p, re_p = rs.ctypes.data_as(ip), re.ctypes.data_as(ip)
            du_p, dm_p = di_u.ctypes.data_as(fp), di_m.ctypes.data_as(fp)
        else:
            di_u = di_m = None
            rs_p = re_p = ctypes.cast(0, ip)
            du_p = dm_p = ctypes.cast(0, fp)

        self.lib.score_snvs_lightcone_ensemble(
            self.c_ptrs, len(self.ptrs),
            rw.ctypes.data_as(u8p), au.ctypes.data_as(u8p),
            rs_p, re_p, N,
            s50.ctypes.data_as(fp), s500.ctypes.data_as(fp),
            u50.ctypes.data_as(fp), u500.ctypes.data_as(fp),
            uraw500.ctypes.data_as(fp), du_p, dm_p,
        )
        return {
            "ds_site_50": s50,
            "ds_site_500": s500,
            "ds_usage_50": u50,
            "ds_usage_500": u500,
            "du_raw_500": uraw500,
            "delta_intron_u": di_u,
            "delta_intron_mod": di_m,
        }

    def score_snvs_on_genome(
        self,
        genome_u8_dict: Dict[str, np.ndarray],
        chrom_keys: Sequence[str],
        pos_1b: np.ndarray,
        alt_genomic_u8: np.ndarray,
        strand: np.ndarray,
        i_s_1b: Optional[np.ndarray] = None,
        i_e_1b: Optional[np.ndarray] = None,
        max_merge_gap: int = 8000,
        max_chunk_span: int = 120000,
    ) -> dict:
        """
        Cluster nearby SNVs on the same (chrom, strand) into contiguous REF chunks so
        REF trunk + heads run ONCE per gene/cluster and each SNV runs only the 4,001-bp ALT cone!
        """
        N = len(pos_1b)
        s50 = np.zeros(N, dtype=np.float32)
        s500 = np.zeros(N, dtype=np.float32)
        u50 = np.zeros((N, 15), dtype=np.float32)
        u500 = np.zeros((N, 15), dtype=np.float32)
        uraw500 = np.zeros((N, 15), dtype=np.float32)
        has_intron = (i_s_1b is not None) and (i_e_1b is not None)
        di_u = np.zeros((N, 15), dtype=np.float32) if has_intron else None
        di_m = np.zeros((N, 15), dtype=np.float32) if has_intron else None

        # Group indices by (chrom_key, strand)
        groups: Dict[Tuple[str, int], List[int]] = {}
        for i in range(N):
            groups.setdefault((str(chrom_keys[i]), int(strand[i])), []).append(i)

        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int32)
        u8p = ctypes.POINTER(ctypes.c_uint8)

        for (ck, st), idxs in groups.items():
            idxs_arr = np.array(idxs, dtype=np.int32)
            p_sub = pos_1b[idxs_arr]
            order = np.argsort(p_sub, kind="mergesort")
            idxs_s = idxs_arr[order]
            p_s = p_sub[order]
            g_seq = genome_u8_dict[ck]
            g_len = len(g_seq)

            # Partition into merged intervals
            c_start = 0
            while c_start < len(idxs_s):
                c_end = c_start + 1
                p_first = int(p_s[c_start])
                while c_end < len(idxs_s):
                    if int(p_s[c_end]) - int(p_s[c_end - 1]) <= max_merge_gap and int(p_s[c_end]) - p_first <= max_chunk_span:
                        c_end += 1
                    else:
                        break
                sub_orig = idxs_s[c_start:c_end]
                M = len(sub_orig)
                p_min = int(p_s[c_start]) - 1
                p_max = int(p_s[c_end - 1]) - 1
                g0 = p_min - 5500
                g1 = p_max + 5501
                L_chunk = g1 - g0
                if g0 >= 0 and g1 <= g_len:
                    chunk = g_seq[g0:g1].copy()
                else:
                    chunk = np.full(L_chunk, 4, dtype=np.uint8)
                    a0, a1 = max(0, g0), min(g_len, g1)
                    if a1 > a0:
                        chunk[a0 - g0 : a1 - g0] = g_seq[a0:a1]

                offs = np.empty(M, dtype=np.int32)
                alts = np.empty(M, dtype=np.uint8)
                rs_loc = np.empty(M, dtype=np.int32) if has_intron else None
                re_loc = np.empty(M, dtype=np.int32) if has_intron else None

                if st > 0:
                    for k_i, oi in enumerate(sub_orig):
                        pv = int(pos_1b[oi])
                        offs[k_i] = (pv - 1 - 5500) - g0
                        alts[k_i] = int(alt_genomic_u8[oi])
                        if has_intron:
                            rs_loc[k_i] = 500 + (int(i_s_1b[oi]) - pv)
                            re_loc[k_i] = 500 + (int(i_e_1b[oi]) - pv)
                else:
                    chunk = COMP_U8[chunk[::-1]].copy()
                    for k_i, oi in enumerate(sub_orig):
                        pv = int(pos_1b[oi])
                        # In reversed chunk of [g0, g1), genomic index g_idx = pv - 1 maps to rev_idx = (g1 - 1) - (pv - 1) = g1 - pv
                        rev_center = g1 - pv
                        offs[k_i] = rev_center - 5500
                        alts[k_i] = int(COMP_U8[int(alt_genomic_u8[oi])])
                        if has_intron:
                            rs_loc[k_i] = 500 + (pv - int(i_s_1b[oi]))
                            re_loc[k_i] = 500 + (pv - int(i_e_1b[oi]))

                ts50 = np.zeros(M, dtype=np.float32)
                ts500 = np.zeros(M, dtype=np.float32)
                tu50 = np.zeros((M, 15), dtype=np.float32)
                tu500 = np.zeros((M, 15), dtype=np.float32)
                traw500 = np.zeros((M, 15), dtype=np.float32)
                tdiu = np.zeros((M, 15), dtype=np.float32) if has_intron else None
                tdim = np.zeros((M, 15), dtype=np.float32) if has_intron else None

                self.lib.score_chunk_snvs_ensemble(
                    self.c_ptrs, len(self.ptrs),
                    chunk.ctypes.data_as(u8p), L_chunk,
                    offs.ctypes.data_as(ip), alts.ctypes.data_as(u8p),
                    rs_loc.ctypes.data_as(ip) if has_intron else ctypes.cast(0, ip),
                    re_loc.ctypes.data_as(ip) if has_intron else ctypes.cast(0, ip),
                    M,
                    ts50.ctypes.data_as(fp), ts500.ctypes.data_as(fp),
                    tu50.ctypes.data_as(fp), tu500.ctypes.data_as(fp),
                    traw500.ctypes.data_as(fp),
                    tdiu.ctypes.data_as(fp) if has_intron else ctypes.cast(0, fp),
                    tdim.ctypes.data_as(fp) if has_intron else ctypes.cast(0, fp),
                )
                s50[sub_orig] = ts50
                s500[sub_orig] = ts500
                u50[sub_orig] = tu50
                u500[sub_orig] = tu500
                uraw500[sub_orig] = traw500
                if has_intron:
                    di_u[sub_orig] = tdiu
                    di_m[sub_orig] = tdim
                c_start = c_end

        return {
            "ds_site_50": s50,
            "ds_site_500": s500,
            "ds_usage_50": u50,
            "ds_usage_500": u500,
            "du_raw_500": uraw500,
            "delta_intron_u": di_u,
            "delta_intron_mod": di_m,
        }


# ======================================================================
# Exact Pure-NumPy / Math Statistical Functions (No SciPy required)
# ======================================================================

def norm_sf_2sided(z: float) -> float:
    if np.isnan(z):
        return 1.0
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def tie_invariant_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Exact tie-aware AUROC (trapezoidal / Mann-Whitney) and tie-invariant Average Precision."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # Group unique score levels in descending order
    diffs = np.flatnonzero(np.diff(s_sorted) != 0)
    starts = np.concatenate(([0], diffs + 1))
    ends = np.concatenate((diffs + 1, [len(s_sorted)]))

    cum_tp = 0.0
    cum_fp = 0.0
    auc_num = 0.0
    ap_sum = 0.0
    for a, b in zip(starts, ends):
        tp_g = float(y_sorted[a:b].sum())
        fp_g = float((b - a) - tp_g)
        # Mann-Whitney / trapezoidal AUC
        auc_num += tp_g * (n_neg - cum_fp - 0.5 * fp_g)
        # Tie-invariant exact expected AP over random permutations within tie group
        if tp_g > 0:
            k_tot = int(b - a)
            if k_tot == 1:
                ap_sum += tp_g * ((cum_tp + 1.0) / (cum_tp + cum_fp + 1.0))
            else:
                # Exact analytical expectation across j = 1..k_tot in a tied block:
                # For rank r = cum_n + j, expected TP up to j (given j is positive) is cum_tp + 1 + (j-1)*(tp_g-1)/(k_tot-1)
                j = np.arange(1, k_tot + 1, dtype=np.float64)
                e_tp_at_j = cum_tp + 1.0 + (j - 1.0) * ((tp_g - 1.0) / (k_tot - 1.0))
                ap_sum += (tp_g / k_tot) * float(np.sum(e_tp_at_j / (cum_tp + cum_fp + j)))
        cum_tp += tp_g
        cum_fp += fp_g

    auroc = auc_num / (float(n_pos) * float(n_neg))
    auprc = ap_sum / float(n_pos)
    return float(auroc), float(auprc)


def _delong_structural_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Exact midrank Sun & Xu (2014) fast DeLong structural components V10 and V01."""
    m = len(pos_scores)
    n = len(neg_scores)
    all_s = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_s, kind="mergesort")
    s_sorted = all_s[order]
    N = m + n
    midranks = np.empty(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i + 1
        while j < N and s_sorted[j] == s_sorted[i]:
            j += 1
        mid = 0.5 * (i + 1 + j)
        midranks[order[i:j]] = mid
        i = j

    # Also midranks within positives and within negatives
    def _within_midranks(x):
        L = len(x)
        o = np.argsort(x, kind="mergesort")
        xs = x[o]
        mr = np.empty(L, dtype=np.float64)
        ii = 0
        while ii < L:
            jj = ii + 1
            while jj < L and xs[jj] == xs[ii]:
                jj += 1
            mr[o[ii:jj]] = 0.5 * (ii + 1 + jj)
            ii = jj
        return mr

    tz_pos = midranks[:m]
    tz_neg = midranks[m:]
    tx = _within_midranks(pos_scores)
    ty = _within_midranks(neg_scores)

    v10 = (tz_pos - tx) / float(n)
    v01 = 1.0 - (tz_neg - ty) / float(m)
    auc = float(np.mean(v10))
    return auc, v10, v01


def paired_delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    m_ok = (~np.isnan(sa)) & (~np.isnan(sb))
    y, sa, sb = y[m_ok], sa[m_ok], sb[m_ok]
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m < 2 or n < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "delta": float("nan"), "se": float("nan"), "z": 0.0, "p": 1.0}
    auc_a, v10_a, v01_a = _delong_structural_components(sa[pos], sa[neg])
    auc_b, v10_b, v01_b = _delong_structural_components(sb[pos], sb[neg])
    d10 = v10_a - v10_b
    d01 = v01_a - v01_b
    var_d = float(np.var(d10, ddof=1)) / m + float(np.var(d01, ddof=1)) / n
    delta = auc_a - auc_b
    if var_d <= 1e-18:
        return {"auc_a": auc_a, "auc_b": auc_b, "delta": delta, "se": 0.0, "z": 0.0, "p": 1.0 if abs(delta) < 1e-12 else 0.0}
    se = math.sqrt(var_d)
    z = delta / se
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": delta, "se": se, "z": z, "p": norm_sf_2sided(z)}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    def _rank(v):
        o = np.argsort(v, kind="mergesort")
        vs = v[o]
        r = np.empty(len(v), dtype=np.float64)
        i = 0
        N = len(v)
        while i < N:
            j = i + 1
            while j < N and vs[j] == vs[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1)
            i = j
        return r
    rx = _rank(x)
    ry = _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / den) if den > 0 else 0.0


def fisher_2x2_exact_and_haldane(a: int, b: int, c: int, d: int) -> dict:
    """
    2x2 contingency table:
      group1: a (exposed), b (unexposed) -> n1 = a + b
      group2: c (exposed), d (unexposed) -> n2 = c + d
    Returns raw OR, Haldane (+0.5) OR, 95% Woolf CI on Haldane OR, and exact two-sided Fisher p-value.
    """
    raw_or = (a * d) / (b * c) if (b * c) > 0 else float("inf")
    ah, bh, ch, dh = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    haldane_or = (ah * dh) / (bh * ch)
    se_ln = math.sqrt(1.0 / ah + 1.0 / bh + 1.0 / ch + 1.0 / dh)
    ci_lo = math.exp(math.log(haldane_or) - 1.959963984540054 * se_ln)
    ci_hi = math.exp(math.log(haldane_or) + 1.959963984540054 * se_ln)

    # Exact two-sided Fisher test via log-gamma hypergeometric probabilities
    r1 = a + b
    r2 = c + d
    c1 = a + c
    N = r1 + r2
    k_min = max(0, c1 - r2)
    k_max = min(r1, c1)

    def log_hyper(k):
        return (
            math.lgamma(r1 + 1) - math.lgamma(k + 1) - math.lgamma(r1 - k + 1)
            + math.lgamma(r2 + 1) - math.lgamma(c1 - k + 1) - math.lgamma(r2 - (c1 - k) + 1)
            - (math.lgamma(N + 1) - math.lgamma(c1 + 1) - math.lgamma(N - c1 + 1))
        )

    ks = np.arange(k_min, k_max + 1, dtype=np.int64)
    lps = np.array([log_hyper(int(k)) for k in ks], dtype=np.float64)
    lp_obs = log_hyper(a)
    mask = lps <= lp_obs + 1e-12
    if not mask.any():
        p_val = 1.0
    else:
        mx = float(np.max(lps[mask]))
        p_val = min(1.0, float(math.exp(mx) * np.sum(np.exp(lps[mask] - mx))))
    return {
        "raw_or": float(raw_or),
        "haldane_or": float(haldane_or),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_fisher": float(p_val),
    }


def cmh_2x2xk_test(tables: List[Tuple[int, int, int, int]]) -> dict:
    """Cochran-Mantel-Haenszel pooled OR, 95% CI, and two-sided chi2(df=1) p-value."""
    num_or = 0.0
    den_or = 0.0
    sum_diff = 0.0
    sum_var = 0.0
    # Robins-Breslow-Greenland variance components
    P_sum = Q_sum = R_sum = S_sum = PR_sum = PS_QR_sum = QS_sum = 0.0
    for a, b, c, d in tables:
        n = float(a + b + c + d)
        if n <= 1:
            continue
        r1 = float(a + b)
        r2 = float(c + d)
        c1 = float(a + c)
        c2 = float(b + d)
        R_i = (a * d) / n
        S_i = (b * c) / n
        P_i = (a + d) / n
        Q_i = (b + c) / n
        num_or += R_i
        den_or += S_i
        PR_sum += P_i * R_i
        PS_QR_sum += P_i * S_i + Q_i * R_i
        QS_sum += Q_i * S_i
        sum_diff += a - (r1 * c1) / n
        sum_var += (r1 * r2 * c1 * c2) / (n * n * (n - 1.0))

    if den_or <= 0 or num_or <= 0 or sum_var <= 0:
        return {"cmh_or": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_cmh": 1.0}
    cmh_or = num_or / den_or
    var_ln = (PR_sum / (2.0 * num_or * num_or)) + (PS_QR_sum / (2.0 * num_or * den_or)) + (QS_sum / (2.0 * den_or * den_or))
    se_ln = math.sqrt(max(0.0, var_ln))
    ci_lo = math.exp(math.log(cmh_or) - 1.959963984540054 * se_ln)
    ci_hi = math.exp(math.log(cmh_or) + 1.959963984540054 * se_ln)
    z = abs(sum_diff) / math.sqrt(sum_var)
    return {"cmh_or": float(cmh_or), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_cmh": norm_sf_2sided(z)}


def exact_binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    ks = np.arange(0, n + 1, dtype=np.int64)
    lp = math.log(p)
    lq = math.log(1.0 - p)
    lg_n = math.lgamma(n + 1)
    lprobs = np.array([lg_n - math.lgamma(int(i) + 1) - math.lgamma(n - int(i) + 1) + i * lp + (n - i) * lq for i in ks])
    lp_obs = lprobs[k]
    m = lprobs <= lp_obs + 1e-12
    mx = float(np.max(lprobs[m]))
    return min(1.0, float(math.exp(mx) * np.sum(np.exp(lprobs[m] - mx))))


def paired_mcnemar_exact(hit_a: np.ndarray, hit_b: np.ndarray) -> dict:
    a = np.asarray(hit_a, dtype=bool)
    b = np.asarray(hit_b, dtype=bool)
    n10 = int((a & (~b)).sum())
    n01 = int(((~a) & b).sum())
    return {"n_a_only": n10, "n_b_only": n01, "p_mcnemar": exact_binom_two_sided(n10, n10 + n01, 0.5)}


class FastaIndexedReader:
    """Instant random-access FASTA reader via .fai index (avoids reading multi-GB genome archives)."""

    _ASCII_U8 = np.full(256, 4, dtype=np.uint8)
    for _ch, _v in ((ord("A"), 0), (ord("a"), 0), (ord("C"), 1), (ord("c"), 1),
                    (ord("G"), 2), (ord("g"), 2), (ord("T"), 3), (ord("t"), 3)):
        _ASCII_U8[_ch] = _v

    def __init__(self, fna_path: str, fai_path: str):
        self.fh = open(fna_path, "rb")
        self.idx = {}
        # Also build RefSeq NC_ accession -> 1..27 (sheep) or 1..30 (cattle) lookup
        self.acc_list = []
        with open(fai_path, "r", encoding="utf-8") as f:
            for ln in f:
                sp = ln.rstrip("\n").split("\t")
                if len(sp) >= 5:
                    name = sp[0]
                    entry = (int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4]))
                    self.idx[name] = entry
                    self.acc_list.append(name)
        # Map "1".."27" for sheep NC_056054.1..NC_056080.1
        nc_main = [a for a in self.acc_list if a.startswith("NC_")]
        for k_i, acc in enumerate(nc_main, start=1):
            self.idx[str(k_i)] = self.idx[acc]
            self.idx[f"chr{k_i}"] = self.idx[acc]
        if nc_main:
            # Last NC_ before MT (check length > 1,000,000 for X)
            nc_nuclear = [a for a in nc_main if self.idx[a][0] > 1_000_000]
            if nc_nuclear:
                self.idx["X"] = self.idx[nc_nuclear[-1]]
                self.idx["chrX"] = self.idx[nc_nuclear[-1]]

    def close(self):
        if getattr(self, "fh", None):
            self.fh.close()
            self.fh = None

    def fetch_u8(self, chrom_key: str, start_0b: int, end_0b: int) -> np.ndarray:
        L = end_0b - start_0b
        seq_len, byte_off, line_bases, line_bytes = self.idx[str(chrom_key)]
        out = np.full(L, 4, dtype=np.uint8)
        a0 = max(0, start_0b)
        a1 = min(seq_len, end_0b)
        if a1 <= a0:
            return out
        b_start = byte_off + (a0 // line_bases) * line_bytes + (a0 % line_bases)
        b_end = byte_off + ((a1 - 1) // line_bases) * line_bytes + ((a1 - 1) % line_bases) + 1
        self.fh.seek(b_start)
        raw = self.fh.read(b_end - b_start)
        # Strip CR/LF bytes
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = arr[(arr != 10) & (arr != 13)]
        out[a0 - start_0b : a0 - start_0b + len(arr)] = self._ASCII_U8[arr]
        return out


class PlinkBedReader:
    """Fast random-access PLINK .bed/.bim/.fam genotype reader for exact pairwise LD r^2."""

    # PLINK SNP-major 2-bit lookup: 00->hom1(2), 01->missing(-1), 10->het(1), 11->hom2(0)
    _DECODE = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float32)

    def __init__(self, bed_path: str, bim_path: str, fam_path: str):
        self.bed_path = str(bed_path)
        with open(fam_path, "r", encoding="utf-8", errors="replace") as f:
            self.n_samples = sum(1 for ln in f if ln.strip())
        self.bytes_per_snp = (self.n_samples + 3) // 4
        # Build byte -> 4 float32 table [256, 4]
        bvals = np.arange(256, dtype=np.uint8)[:, None]
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, :]
        codes = (bvals >> shifts) & 3
        self.lut = self._DECODE[codes]

        # Index (chrom_clean, pos_1b) -> list of (snp_row_idx, snp_id, a1, a2)
        self.pos_index: Dict[Tuple[str, int], List[Tuple[int, str, str, str]]] = {}
        self.id_index: Dict[str, int] = {}
        with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, ln in enumerate(f):
                sp = ln.rstrip("\n").split()
                if len(sp) < 6:
                    continue
                ch = sp[0].replace("chr", "").replace("CHR", "")
                sid = sp[1]
                pos = int(sp[3])
                a1, a2 = sp[4], sp[5]
                self.pos_index.setdefault((ch, pos), []).append((idx, sid, a1, a2))
                self.id_index[sid] = idx
        self._fh = open(self.bed_path, "rb")
        magic = self._fh.read(3)
        assert magic == b"\x6c\x1b\x01", f"Invalid PLINK SNP-major bed magic: {magic}"
        self._cache: Dict[int, np.ndarray] = {}

    def close(self):
        if getattr(self, "_fh", None):
            self._fh.close()
            self._fh = None

    def get_geno_by_idx(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]
        self._fh.seek(3 + idx * self.bytes_per_snp)
        raw = np.frombuffer(self._fh.read(self.bytes_per_snp), dtype=np.uint8)
        g = self.lut[raw].ravel()[: self.n_samples].copy()
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[idx] = g
        return g

    def lookup_snp_idx(self, ch: str, pos: int, snp_id: Optional[str] = None) -> int:
        if snp_id and snp_id in self.id_index:
            return self.id_index[snp_id]
        ch_c = str(ch).replace("chr", "").replace("CHR", "")
        hits = self.pos_index.get((ch_c, int(pos)), [])
        return hits[0][0] if hits else -1

    def ld_r2(self, idx_a: int, idx_b: int) -> float:
        if idx_a < 0 or idx_b < 0:
            return float("nan")
        if idx_a == idx_b:
            return 1.0
        ga = self.get_geno_by_idx(idx_a)
        gb = self.get_geno_by_idx(idx_b)
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
        return float(r * r)


CATTLE_TISSUE_MAP = {
    "Muscle":               (1,  "Skeletal_Muscle",    0),
    "Liver":                (2,  "Liver",              1),
    "Adipose":              (3,  "Adipose",           -1),
    "Subcutaneous_Adipose": (3,  "Adipose",           -1),
    "Visceral_Fat":         (3,  "Adipose",           -1),
    "Small_Intestine":      (4,  "Intestine",         -1),
    "Jejunum":              (4,  "Intestine",         -1),
    "Ileum":                (4,  "Intestine",         -1),
    "Large_Intestine":      (4,  "Intestine",         -1),
    "Brain":                (5,  "CNS_Brain",          2),
    "Hypothalamus":         (5,  "CNS_Brain",          2),
    "Rumen":                (6,  "Rumen",             -1),
    "Pituitary":            (7,  "Pituitary",          2),
    "Lung":                 (8,  "Lung",              -1),
    "Testis":               (10, "Gonad_Reproductive", 3),
    "Epididymis":           (10, "Gonad_Reproductive", 3),
    "Vas_Deferens":         (10, "Gonad_Reproductive", 3),
    "Uterus":               (10, "Gonad_Reproductive", 3),
    "Corpus_Luteum":        (10, "Gonad_Reproductive", 3),
    "Skin":                 (12, "Skin",              -1),
    "Mammary_Gland":        (0,  "Mammary_Lactation", -1),
    "Milk":                 (0,  "Mammary_Lactation", -1),
    "Embryo":               (0,  "Embryo_Placenta",   -1),
    "Placenta":             (0,  "Embryo_Placenta",   -1),
    "Whole_Blood":          (0,  "Blood_Immune",      -1),
    "Leukocyte":            (0,  "Blood_Immune",      -1),
    "Lymph_Node":           (0,  "Blood_Immune",      -1),
    "Monocyte":             (0,  "Blood_Immune",      -1),
    "Tonsil":               (0,  "Blood_Immune",      -1),
}

try:
    from scipy.special import erf as _erf_fn
except ImportError:
    _erf_fn = np.vectorize(math.erf, otypes=[np.float32])


def seq_to_u8(seq: str) -> np.ndarray:
    lut = np.full(256, 4, dtype=np.uint8)
    for b, v in ((ord("A"), 0), (ord("a"), 0), (ord("C"), 1), (ord("c"), 1),
                 (ord("G"), 2), (ord("g"), 2), (ord("T"), 3), (ord("t"), 3)):
        lut[b] = v
    return lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)].copy()


def u8_to_seq(arr: np.ndarray) -> str:
    alph = np.frombuffer(b"ACGTN", dtype=np.uint8)
    return alph[np.clip(arr, 0, 4)].tobytes().decode("ascii")


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + _erf_fn(x / np.float32(1.4142135623730951)))).astype(np.float32)


def _conv1d_same_dilated_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """Exact float32 1D dilated same-padded convolution for [B, C_in, L] -> [B, C_out, L]."""
    B, C_in, L = x.shape
    C_out, _, K = w.shape
    pad = (K - 1) * dilation // 2
    out = np.broadcast_to(b[None, :, None], (B, C_out, L)).copy()
    for ki in range(K):
        shift = ki * dilation - pad
        if shift < 0:
            xs = x[:, :, : L + shift]
            dst_s, dst_e = -shift, L
        elif shift > 0:
            xs = x[:, :, shift:]
            dst_s, dst_e = 0, L - shift
        else:
            xs = x
            dst_s, dst_e = 0, L
        out[:, :, dst_s:dst_e] += np.einsum("bcl,oc->bol", xs, w[:, :, ki], optimize=True)
    return out


class NumPyOviEnsemble:
    """
    Zero-compiler, zero-PyTorch, zero-SciPy pure-NumPy exact float32 fallback for OviSplice15Head.
    Guarantees out-of-the-box execution on Windows/macOS/Linux even without g++ or torch.
    """

    def __init__(self, ckpt_paths: Sequence[str]):
        self.models = []
        so_flags = []
        for p in ckpt_paths:
            raw = load_pytorch_ckpt_numpy(str(p))
            sd = {k.replace("_orig_mod.", "").replace("module.", ""): v.astype(np.float32)
                  for k, v in raw["model_state"].items()}
            is_so = ("pan_head.weight" not in sd) and (raw.get("head_type") != "linear_15head")
            so_flags.append(is_so)
            self.models.append((sd, is_so))
        self.is_site_only = bool(so_flags and all(so_flags))

    def _forward_single(self, sd: dict, is_so: bool, x_oh: np.ndarray, compute_usage: bool = True):
        eps = np.float32(1e-5)
        conv = np.einsum("bcl,oc->bol", x_oh, sd["conv1.weight"][:, :, 0]) + sd["conv1.bias"][None, :, None]
        skip = np.einsum("bcl,oc->bol", conv, sd["skip.weight"][:, :, 0]) + sd["skip.bias"][None, :, None]
        j = 0
        for i, (_k, dil) in enumerate(BLOCK_SPECS):
            px = f"resblocks.{i}."
            sc1 = sd[px + "bn1.weight"] / np.sqrt(sd[px + "bn1.running_var"] + eps)
            sh1 = sd[px + "bn1.bias"] - sd[px + "bn1.running_mean"] * sc1
            h = np.maximum(0.0, conv * sc1[None, :, None] + sh1[None, :, None])
            h = _conv1d_same_dilated_np(h, sd[px + "conv1.weight"], sd[px + "conv1.bias"], dil)

            sc2 = sd[px + "bn2.weight"] / np.sqrt(sd[px + "bn2.running_var"] + eps)
            sh2 = sd[px + "bn2.bias"] - sd[px + "bn2.running_mean"] * sc2
            h = np.maximum(0.0, h * sc2[None, :, None] + sh2[None, :, None])
            h = _conv1d_same_dilated_np(h, sd[px + "conv2.weight"], sd[px + "conv2.bias"], dil)
            conv = conv + h
            if (i + 1) % 4 == 0 or (i + 1) == len(BLOCK_SPECS):
                skip = skip + np.einsum("bcl,oc->bol", conv, sd[f"convs.{j}.weight"][:, :, 0]) + sd[f"convs.{j}.bias"][None, :, None]
                j += 1

        skip = skip[:, :, 5000:-5000]
        s_log = np.einsum("bcl,oc->bol", skip, sd["site_head.weight"][:, :, 0]) + sd["site_head.bias"][None, :, None]
        s_max = np.max(s_log, axis=1, keepdims=True)
        s_exp = np.exp(s_log - s_max)
        s_prob = s_exp / np.sum(s_exp, axis=1, keepdims=True)

        if not compute_usage:
            return np.transpose(s_prob[:, 1:3, :], (0, 2, 1)), None

        if is_so:
            u_prob = np.zeros((x_oh.shape[0], 15, skip.shape[2]), dtype=np.float32)
        elif "pan_head.weight" in sd:
            u = _gelu_np(np.einsum("bcl,oc->bol", skip, sd["usage_mlp.0.weight"][:, :, 0]) + sd["usage_mlp.0.bias"][None, :, None])
            u = _gelu_np(np.einsum("bcl,oc->bol", u, sd["usage_mlp.3.weight"][:, :, 0]) + sd["usage_mlp.3.bias"][None, :, None])
            pan = np.einsum("bcl,oc->bol", u, sd["pan_head.weight"][:, :, 0]) + sd["pan_head.bias"][None, :, None]
            dlt = np.einsum("bcl,oc->bol", u, sd["delta_head.weight"][:, :, 0]) + sd["delta_head.bias"][None, :, None]
            u_log = np.concatenate([pan, pan + dlt], axis=1)
            u_prob = 1.0 / (1.0 + np.exp(-np.clip(u_log, -30.0, 30.0)))
        else:
            u_log = np.einsum("bcl,oc->bol", skip, sd["usage_head.weight"][:, :, 0]) + sd["usage_head.bias"][None, :, None]
            u_prob = 1.0 / (1.0 + np.exp(-np.clip(u_log, -30.0, 30.0)))
        return np.transpose(s_prob[:, 1:3, :], (0, 2, 1)), np.transpose(u_prob.astype(np.float32), (0, 2, 1))

    def score_windows(self, seqs_u8: np.ndarray, compute_usage: bool = True):
        seqs = np.asarray(seqs_u8, dtype=np.uint8)
        B, L_in = seqs.shape
        assert L_in >= 10001
        x_oh = np.zeros((B, 4, L_in), dtype=np.float32)
        for c in range(4):
            x_oh[:, c, :] = (seqs == c)
        p_acc, u_acc = None, None
        for sd, is_so in self.models:
            p_i, u_i = self._forward_single(sd, is_so, x_oh, compute_usage=compute_usage)
            p_acc = p_i if p_acc is None else (p_acc + p_i)
            if compute_usage:
                u_acc = u_i if u_acc is None else (u_acc + u_i)
        K = np.float32(len(self.models))
        return p_acc / K, (u_acc / K if compute_usage else None)

    def score_snvs(self, ref_wins_u8: np.ndarray, alt_u8: np.ndarray, rel_s=None, rel_e=None) -> dict:
        rw = np.asarray(ref_wins_u8, dtype=np.uint8)
        au = np.asarray(alt_u8, dtype=np.uint8)
        N, L_in = rw.shape
        assert L_in == 11001 and len(au) == N
        aw = rw.copy()
        aw[:, 5500] = au
        p_ref, u_ref = self.score_windows(rw, compute_usage=True)
        p_alt, u_alt = self.score_windows(aw, compute_usage=True)

        p_any_ref = p_ref.sum(axis=-1)
        p_any_alt = p_alt.sum(axis=-1)
        p_gate = np.maximum(p_any_ref, p_any_alt)
        dp_abs = np.max(np.abs(p_alt - p_ref), axis=-1)
        du_raw_diff = u_alt - u_ref
        du_abs = np.abs(du_raw_diff) * p_gate[:, :, None]

        s500 = np.max(dp_abs, axis=1).astype(np.float32)
        s50 = np.max(dp_abs[:, 450:551], axis=1).astype(np.float32)
        u500 = np.max(du_abs, axis=1).astype(np.float32)
        u50 = np.max(du_abs[:, 450:551, :], axis=1).astype(np.float32)
        uraw500 = np.max(np.abs(du_raw_diff), axis=1).astype(np.float32)

        di_u = di_m = None
        if rel_s is not None and rel_e is not None:
            rs = np.asarray(rel_s, dtype=np.int32)
            re_a = np.asarray(rel_e, dtype=np.int32)
            di_u = np.full((N, 15), np.nan, dtype=np.float32)
            di_m = np.full((N, 15), np.nan, dtype=np.float32)
            L_out = p_ref.shape[1]
            for n_i in range(N):
                pts = [int(rs[n_i])] if int(rs[n_i]) == int(re_a[n_i]) else [int(rs[n_i]), int(re_a[n_i])]
                valid_pts = [rp for rp in pts if 0 <= rp < L_out]
                if valid_pts:
                    acc_u = np.zeros(15, dtype=np.float32)
                    acc_m = np.zeros(15, dtype=np.float32)
                    for rp in valid_pts:
                        acc_u += u_alt[n_i, rp] - u_ref[n_i, rp]
                        acc_m += u_alt[n_i, rp] * p_any_alt[n_i, rp] - u_ref[n_i, rp] * p_any_ref[n_i, rp]
                    inv_c = np.float32(1.0 / len(valid_pts))
                    di_u[n_i] = acc_u * inv_c
                    di_m[n_i] = acc_m * inv_c

        return {
            "ds_site_50": s50,
            "ds_site_500": s500,
            "ds_usage_50": u50,
            "ds_usage_500": u500,
            "du_raw_500": uraw500,
            "delta_intron_u": di_u,
            "delta_intron_mod": di_m,
        }


def load_ensemble(ckpt_paths: Sequence[str | Path], backend: str = "auto"):
    """Load OviSplice ensemble using C++/AVX2 engine when available, falling back to pure NumPy."""
    paths = [str(p) for p in ckpt_paths]
    if backend in ("auto", "cpp"):
        try:
            return CppOviEnsemble(paths), "cpp_avx2"
        except Exception as exc:
            if backend == "cpp":
                raise
            return NumPyOviEnsemble(paths), f"numpy_fallback ({type(exc).__name__})"
    return NumPyOviEnsemble(paths), "numpy"
