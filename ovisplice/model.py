# -*- coding: utf-8 -*-
"""
OviSplice Core PyTorch Architecture, Compact Dataset Loader, Trainer & Evaluator.

Architecture & Loss Summary:
  - Input: 15,000 bp uint8 sequence during training (5,000 bp left + 5,000 bp target + 5,000 bp right)
    or 10,001+ bp sequence during point-wise inference.
  - Backbone: 16 one-dimensional dilated residual blocks (`[(11,1)]x4 + [(11,4)]x4 + [(21,10)]x4 + [(41,25)]x4`,
    exact receptive field = 10,001 bp).
  - Heads:
      * `site_head`: 3-class splice-site classifier (neither=0, acceptor=1, donor=2)
      * `residual_mlp` usage module (`719,410` total trainable parameters): `PanTissue` + 12 ruminant
        organ heads + Prenatal/Postnatal Skeletal Muscle (`delta_head` zero-initialized)
      * `site_only` ablation mode (`696,739` total trainable parameters): omits usage head
  - Four-term composite objective: `cls_scale=500.0`, `lambda_usage=1.0`, `lambda_emp=1.0`, `lambda_diff=4.0`.
"""
import argparse
import csv
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HEAD_NAMES = [
    "PanTissue",
    "SkeletalMuscle", "Liver", "Adipose", "SmallIntestine", "CNS", "Rumen",
    "Pituitary", "Lung", "Abomasum", "Gonad", "Heart", "Skin",
    "SkeletalMuscle_Prenatal", "SkeletalMuscle_Postnatal",
]

CONTRAST_PAIRS = [
    (5, 1, "CNS_vs_SkeletalMuscle"),
    (2, 1, "Liver_vs_SkeletalMuscle"),
    (10, 1, "Gonad_vs_SkeletalMuscle"),
    (6, 1, "Rumen_vs_SkeletalMuscle"),
    (13, 14, "Prenatal_vs_Postnatal_Muscle"),
]

# SpliceAI-10k / Pangolin exact receptive field specification:
#   4*(2*5*1) + 4*(2*5*4) + 4*(2*10*10) + 4*(2*20*25) = 5,000 bp each side (10,000 bp total)
BLOCK_SPECS = (
    [(11, 1)] * 4
    + [(11, 4)] * 4
    + [(21, 10)] * 4
    + [(41, 25)] * 4
)

COMP_U8 = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


class ResBlock1D(nn.Module):
    """Exact match to tkzeng/Pangolin ResBlock so pretrained human weights load 1:1."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class OviSplice15Head(nn.Module):
    """
    Trunk attribute names (`conv1`, `skip`, `resblocks`, `convs`) and skip-crop (no post-skip ReLU)
    strictly match tkzeng/Pangolin so pretrained Pangolin checkpoints transfer 236/236 tensors.
    """

    def __init__(self, channels: int = 32, n_usage_heads: int = 15, flank: int = 5000, head_type: str = "residual_mlp"):
        super().__init__()
        self.flank = flank
        self.n_usage_heads = n_usage_heads
        self.head_type = head_type
        self.conv1 = nn.Conv1d(4, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i, (w, ar) in enumerate(BLOCK_SPECS):
            self.resblocks.append(ResBlock1D(channels, w, ar))
            if (i + 1) % 4 == 0 or (i + 1) == len(BLOCK_SPECS):
                self.convs.append(nn.Conv1d(channels, channels, kernel_size=1))

        self.site_head = nn.Conv1d(channels, 3, kernel_size=1)
        if head_type == "site_only":
            pass
        elif head_type == "linear":
            self.usage_head = nn.Conv1d(channels, n_usage_heads, kernel_size=1)
        elif head_type == "residual_mlp":
            self.usage_mlp = nn.Sequential(
                nn.Conv1d(channels, 128, kernel_size=1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(128, 128, kernel_size=1),
                nn.GELU(),
            )
            self.pan_head = nn.Conv1d(128, 1, kernel_size=1)
            self.delta_head = nn.Conv1d(128, n_usage_heads - 1, kernel_size=1)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def load_pangolin_trunk(self, ckpt_path: str) -> int:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        clean = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
        own = self.state_dict()
        loaded = 0
        for k, v in clean.items():
            if k in own and own[k].shape == v.shape and not k.startswith("conv_last"):
                own[k].copy_(v)
                loaded += 1
        self.load_state_dict(own)
        return loaded

    def forward(self, seq_onehot: torch.Tensor):
        """
        seq_onehot: [B, 4, L_in] (e.g. L_in = 15,000 or 11,001)
        returns:
          site_logits:  [B, 3, L_in - 2*flank]
          usage_logits: [B, 15, L_in - 2*flank]
        """
        conv = self.conv1(seq_onehot)
        skip = self.skip(conv)
        j = 0
        for i, blk in enumerate(self.resblocks):
            conv = blk(conv)
            if (i + 1) % 4 == 0 or (i + 1) == len(self.resblocks):
                skip = skip + self.convs[j](conv)
                j += 1
        if self.flank > 0:
            skip = skip[:, :, self.flank:-self.flank]
        s_log = self.site_head(skip)
        if self.head_type == "site_only":
            u_neg = torch.full(
                (s_log.shape[0], self.n_usage_heads, s_log.shape[2]),
                -100.0,
                device=s_log.device,
                dtype=s_log.dtype,
            )
            return s_log, u_neg
        if self.head_type == "linear":
            return s_log, self.usage_head(skip)
        h = self.usage_mlp(skip)
        pan = self.pan_head(h)
        return s_log, torch.cat([pan, pan + self.delta_head(h)], dim=1)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _is_siteonly_arch(arch_val: str) -> bool:
    return str(arch_val).strip().lower() in ("siteonly", "site_only", "m1")


def _normalize_arch_name(arch_val: str) -> str:
    low = str(arch_val).strip().lower()
    if low in ("siteonly", "site_only", "m1"):
        return "siteonly"
    if low in ("scratch", "m2"):
        return "scratch"
    if low in ("ovisplice_full", "full", "m3_full"):
        return "ovisplice_full"
    return "ovisplice"


def load_ovisplice_ckpt(ckpt_path: str, device: torch.device):
    st = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in st["model_state"].items()}
    if "pan_head.weight" in sd:
        head_type = "residual_mlp"
    elif st.get("head_type") == "linear_15head":
        head_type = "linear"
    else:
        head_type = "site_only"
        sd = {k: v for k, v in sd.items() if not k.startswith("usage_head.")}
    m = OviSplice15Head(head_type=head_type).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, st


def uint8_to_onehot_gpu(seq_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert [B, L] uint8 (0=A,1=C,2=G,3=T,4=N) on GPU to [B, 4, L] floating one-hot."""
    oh = F.one_hot(seq_u8.long().clamp(0, 4), num_classes=5)[..., :4]
    return oh.permute(0, 2, 1).contiguous().to(dtype=dtype)


class SheepCompactDataset(Dataset):
    """
    Memory-mapped / RAM-resident compact dataset (<35 MB label pack + genome_npz):
      genome_npz: {chrom_id_str: uint8 array of 0..4}
      labels_npz:
        win_chrom, win_start, win_strand, win_fold, win_paralog,
        site_win_ptr, site_rel_pos, site_class, site_source, site_usage, site_mask
    When `annotated_only=True` (`siteonly` ablation), unannotated `site_source == 1` sites
    are excluded from `cls_y` labels.
    """

    def __init__(
        self,
        genome_dict,
        labels_npz,
        keep_mask: np.ndarray,
        target_len: int = 5000,
        flank: int = 5000,
        annotated_only: bool = False,
        return_source: bool = False,
    ):
        self.genome = genome_dict
        self.L = target_len
        self.flank = flank
        self.annotated_only = annotated_only
        self.return_source = return_source
        self.indices = np.flatnonzero(keep_mask).astype(np.int32)
        self.win_chrom = labels_npz["win_chrom"]
        self.win_start = labels_npz["win_start"]
        self.win_strand = labels_npz["win_strand"]
        self.ptr = labels_npz["site_win_ptr"]
        self.rel_pos = labels_npz["site_rel_pos"]
        self.s_cls = labels_npz["site_class"]
        self.s_src = labels_npz["site_source"]
        self.s_use = labels_npz["site_usage"]
        self.s_msk = labels_npz["site_mask"]
        self.rc_lut = COMP_U8
        self.bit_shifts = np.arange(15, dtype=np.uint16)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        w = int(self.indices[idx])
        chrom = str(int(self.win_chrom[w]))
        s0 = int(self.win_start[w]) - self.flank
        s1 = int(self.win_start[w]) + self.L + self.flank
        g = self.genome[chrom]
        total_len = self.L + 2 * self.flank

        if s0 >= 0 and s1 <= len(g):
            seq = g[s0:s1].copy()
        else:
            seq = np.full(total_len, 4, dtype=np.uint8)
            a0, a1 = max(0, s0), min(len(g), s1)
            if a1 > a0:
                seq[a0 - s0:a1 - s0] = g[a0:a1]

        if self.win_strand[w] < 0:
            seq = self.rc_lut[seq[::-1]]

        cls_y = np.zeros(self.L, dtype=np.int64)
        use_y = np.zeros((15, self.L), dtype=np.float32)
        msk_y = np.zeros((15, self.L), dtype=np.float32)
        src_y = np.full(self.L, 255, dtype=np.uint8) if self.return_source else None

        p0, p1 = int(self.ptr[w]), int(self.ptr[w + 1])
        if p1 > p0:
            pos = self.rel_pos[p0:p1].astype(np.int64)
            s_c = self.s_cls[p0:p1]
            s_sr = self.s_src[p0:p1]
            if self.annotated_only:
                ann_m = s_sr != 1
                pos = pos[ann_m]
                s_c = s_c[ann_m]
            if len(pos) > 0:
                cls_y[pos] = s_c
                if not self.annotated_only:
                    use_y[:, pos] = self.s_use[p0:p1].astype(np.float32).T
                    bits = ((self.s_msk[p0:p1, None] >> self.bit_shifts[None, :]) & 1).astype(np.float32)
                    msk_y[:, pos] = bits.T
                if self.return_source:
                    src_y[pos] = s_sr if not self.annotated_only else s_sr[ann_m]

        if self.return_source:
            return (
                torch.from_numpy(seq),
                torch.from_numpy(cls_y),
                torch.from_numpy(use_y),
                torch.from_numpy(msk_y),
                torch.from_numpy(src_y),
            )
        return (
            torch.from_numpy(seq),
            torch.from_numpy(cls_y),
            torch.from_numpy(use_y),
            torch.from_numpy(msk_y),
        )


def compute_loss(
    site_logits,
    usage_logits,
    cls_y,
    use_y,
    msk_y,
    src_y=None,
    annotation_only: bool = False,
    cls_scale: float = 500.0,
    lambda_usage: float = 1.0,
    lambda_emp: float = 1.0,
    lambda_diff: float = 4.0,
):
    loss_cls = F.cross_entropy(site_logits.float(), cls_y, reduction="mean") * cls_scale
    if annotation_only:
        return loss_cls, loss_cls.detach(), torch.tensor(0.0, device=site_logits.device)

    u_log = usage_logits.float()
    head_win_active = (msk_y.sum(dim=2, keepdim=True) > 0).float()  # [B, 15, 1]
    non_site = (cls_y == 0).unsqueeze(1).float()                    # [B, 1, L]
    dense_msk = torch.clamp(msk_y + non_site * head_win_active, 0.0, 1.0)
    dense_target = use_y * msk_y
    bce_all = F.binary_cross_entropy_with_logits(u_log, dense_target, reduction="none")
    d_cnt = dense_msk.sum(dim=(0, 2))
    d_act = d_cnt > 0
    if d_act.any():
        loss_dense = ((bce_all * dense_msk).sum(dim=(0, 2))[d_act] / d_cnt[d_act]).mean() * cls_scale
    else:
        loss_dense = torch.tensor(0.0, device=site_logits.device)

    if src_y is not None:
        emp_msk = msk_y * (src_y < 2).unsqueeze(1).float()
    else:
        emp_msk = msk_y
    e_cnt = emp_msk.sum(dim=(0, 2))
    e_act = e_cnt > 0
    if e_act.any():
        loss_emp = ((bce_all * emp_msk).sum(dim=(0, 2))[e_act] / e_cnt[e_act]).mean()
    else:
        loss_emp = torch.tensor(0.0, device=site_logits.device)

    u_prob = torch.sigmoid(u_log)
    if lambda_diff > 0.0 and src_y is not None:
        both_pan = emp_msk[:, 1:, :] * emp_msk[:, 0:1, :]
        du_true = use_y[:, 1:, :] - use_y[:, 0:1, :]
        du_pred = u_prob[:, 1:, :] - u_prob[:, 0:1, :]
        w_pan = both_pan * (1.0 + 4.0 * (du_true.abs() >= 0.10).float())
        pan_denom = w_pan.sum()
        if pan_denom > 0:
            loss_diff_pan = (F.smooth_l1_loss(du_pred, du_true, beta=0.05, reduction="none") * w_pan).sum() / pan_denom
        else:
            loss_diff_pan = torch.tensor(0.0, device=site_logits.device)

        both_dev = emp_msk[:, 13, :] * emp_msk[:, 14, :]
        du_dev_t = use_y[:, 13, :] - use_y[:, 14, :]
        du_dev_p = u_prob[:, 13, :] - u_prob[:, 14, :]
        w_dev = both_dev * (1.0 + 4.0 * (du_dev_t.abs() >= 0.10).float())
        dev_denom = w_dev.sum()
        if dev_denom > 0:
            loss_diff_dev = (F.smooth_l1_loss(du_dev_p, du_dev_t, beta=0.05, reduction="none") * w_dev).sum() / dev_denom
        else:
            loss_diff_dev = torch.tensor(0.0, device=site_logits.device)
    else:
        loss_diff_pan = torch.tensor(0.0, device=site_logits.device)
        loss_diff_dev = torch.tensor(0.0, device=site_logits.device)

    loss_use = lambda_usage * loss_dense + lambda_emp * loss_emp + lambda_diff * (loss_diff_pan + 0.5 * loss_diff_dev)
    return loss_cls + loss_use, loss_cls.detach(), loss_use.detach()


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_pack_arrays(pack_dir: Path):
    g_npz = np.load(pack_dir / "sheep_genome_uint8.npz", allow_pickle=False)
    genome_dict = {k: g_npz[k] for k in g_npz.files}
    labels_npz = np.load(pack_dir / "ovisplice_labels_v1.npz", allow_pickle=False)
    with open(pack_dir / "folds.json", "r", encoding="utf-8") as fh:
        folds = json.load(fh)["folds"]
    return genome_dict, labels_npz, folds


def parse_chr_to_cid(s: str) -> int:
    c = s[3:] if s.startswith("chr") else s
    if c == "X":
        return 27
    if c.isdigit():
        return int(c)
    return 0


def build_window_masks(labels_npz, folds: dict, fold: int, arch: str):
    """Return train_mask, val_mask, test_mask (paralog_bit applied externally in eval)."""
    w_chr = labels_npz["win_chrom"]
    w_para = labels_npz["win_paralog"]
    ptr = labels_npz["site_win_ptr"]
    has_sites = (ptr[1:] - ptr[:-1]) > 0

    if _is_siteonly_arch(arch):
        # Restrict training/val windows for site-only ablation to windows containing >=1 annotated site (source != 1)
        s_src = labels_npz["site_source"]
        n_w = len(w_chr)
        ann_site_wins = np.zeros(n_w, dtype=bool)
        ann_flags = s_src != 1
        # Fast cumulative check
        cs = np.concatenate(([0], np.cumsum(ann_flags, dtype=np.int32)))
        ann_site_wins = (cs[ptr[1:]] - cs[ptr[:-1]]) > 0
        pos_win_mask = ann_site_wins
    else:
        pos_win_mask = has_sites

    if fold == 0:
        # ovisplice_full: train on all non-val chromosomes (use chr25 as small early-stop sentinel)
        val_cids = {parse_chr_to_cid("chr25")}
        is_val = np.isin(w_chr, list(val_cids))
        train_mask = (~is_val) & pos_win_mask
        val_mask = is_val & pos_win_mask
        test_mask = np.zeros(len(w_chr), dtype=bool)
        return train_mask, val_mask, test_mask

    fspec = folds[f"fold_{fold}"]
    tr_cids = [parse_chr_to_cid(c) for c in fspec["train_chroms"]]
    va_cids = [parse_chr_to_cid(c) for c in fspec["val_chroms"]]
    te_cids = [parse_chr_to_cid(c) for c in fspec["test_chroms"]]

    train_mask = np.isin(w_chr, tr_cids) & pos_win_mask
    val_mask = np.isin(w_chr, va_cids) & pos_win_mask
    test_mask = np.isin(w_chr, te_cids) & has_sites  # Evaluate on all test site-containing windows
    return train_mask, val_mask, test_mask


def ap_from_hist_and_pos(pos_scores: np.ndarray, neg_hist: np.ndarray, n_bins: int = 100000) -> float:
    """Exact/1e-5-resolution step Average Precision using sorted positives + 100k negative histogram."""
    if len(pos_scores) == 0:
        return float("nan")
    pos_hist = np.bincount(
        np.clip((pos_scores * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins
    )
    tp_cum = np.cumsum(pos_hist[::-1], dtype=np.float64)
    fp_cum = np.cumsum(neg_hist[::-1], dtype=np.float64)
    tot_pos = float(len(pos_scores))
    recall = tp_cum / tot_pos
    denom = tp_cum + fp_cum
    prec = np.where(denom > 0.0, tp_cum / np.maximum(denom, 1.0), 1.0)
    # Step AP integral: sum over bins where TP increased
    d_rec = np.diff(recall, prepend=0.0)
    return float(np.sum(d_rec * prec))


def safe_spearman_pearson(x: np.ndarray, y: np.ndarray):
    if len(x) < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from scipy.stats import pearsonr, spearmanr
            sp = float(spearmanr(x, y).statistic)
            pr = float(pearsonr(x, y).statistic)
        except ImportError:
            from .native_engine import spearman_corr
            sp = float(spearman_corr(x, y))
            pr = float(np.corrcoef(x, y)[0, 1])
    return sp, pr


@torch.no_grad()
def evaluate_dataset(
    models,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_bins: int = 100000,
    eval_usage: bool = True,
    max_batches: int = 0,
):
    """
    Streaming memory-bounded evaluation (supports 1 model or multi-seed list ensemble):
      - Site classification PR-AUC (`acceptor`, `donor`, `macro`) across ALL positions,
        PLUS annotated-only subset (`source != 1`);
      - 15-head usage Spearman rho / Pearson r / MAE across covered empirical variable sites
        (`0.05 <= U <= 0.95`) and all covered sites;
      - Inter-tissue and developmental empirical differential Delta-U correlation.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    base_models = [unwrap_model(m).eval() for m in models]
    inv_m = 1.0 / len(base_models)

    neg_hist_acc = np.zeros(n_bins, dtype=np.int64)
    neg_hist_don = np.zeros(n_bins, dtype=np.int64)
    pos_acc_all, pos_don_all = [], []
    pos_acc_ann, pos_don_ann = [], []

    head_pred = [[] for _ in range(15)]
    head_true = [[] for _ in range(15)]
    head_src = [[] for _ in range(15)]
    diff_pred = [[] for _ in range(len(CONTRAST_PAIRS))]
    diff_true = [[] for _ in range(len(CONTRAST_PAIRS))]

    use_amp = device.type == "cuda"
    oh_dtype = amp_dtype if use_amp else torch.float32

    for b_idx, batch in enumerate(loader):
        if max_batches > 0 and b_idx >= max_batches:
            break
        seq_u8, cls_y, use_y, msk_y, src_y = batch
        seq_u8 = seq_u8.to(device, non_blocking=True)
        probs_acc = None
        u_acc = None
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            x_oh = uint8_to_onehot_gpu(seq_u8, oh_dtype)
            for bm in base_models:
                s_log, u_log = bm(x_oh)
                p_b = F.softmax(s_log.float(), dim=1)
                probs_acc = p_b if probs_acc is None else (probs_acc + p_b)
                if eval_usage:
                    u_b = torch.sigmoid(u_log.float())
                    u_acc = u_b if u_acc is None else (u_acc + u_b)

        probs_np = (probs_acc * inv_m).cpu().numpy()
        cls_np = cls_y.numpy()
        src_np = src_y.numpy()

        m_neg = cls_np == 0
        p_acc_neg = probs_np[:, 1, :][m_neg]
        p_don_neg = probs_np[:, 2, :][m_neg]
        neg_hist_acc += np.bincount(np.clip((p_acc_neg * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins)
        neg_hist_don += np.bincount(np.clip((p_don_neg * n_bins).astype(np.int32), 0, n_bins - 1), minlength=n_bins)

        m_acc = cls_np == 1
        if m_acc.any():
            pa = probs_np[:, 1, :][m_acc]
            pos_acc_all.append(pa)
            sa = src_np[m_acc]
            if (sa != 1).any():
                pos_acc_ann.append(pa[sa != 1])

        m_don = cls_np == 2
        if m_don.any():
            pd_s = probs_np[:, 2, :][m_don]
            pos_don_all.append(pd_s)
            sd = src_np[m_don]
            if (sd != 1).any():
                pos_don_ann.append(pd_s[sd != 1])

        if eval_usage:
            u_pred_np = (u_acc * inv_m).cpu().numpy()
            u_true_np = use_y.numpy()
            msk_np = msk_y.numpy() > 0.5
            for h in range(15):
                mh = msk_np[:, h, :]
                if mh.any():
                    head_pred[h].append(u_pred_np[:, h, :][mh].astype(np.float32))
                    head_true[h].append(u_true_np[:, h, :][mh].astype(np.float32))
                    head_src[h].append(src_np[mh])
            emp_flag = src_np != 2
            for pi, (ha, hb, _cname) in enumerate(CONTRAST_PAIRS):
                both = msk_np[:, ha, :] & msk_np[:, hb, :] & emp_flag
                if both.any():
                    diff_pred[pi].append((u_pred_np[:, ha, :][both] - u_pred_np[:, hb, :][both]).astype(np.float32))
                    diff_true[pi].append((u_true_np[:, ha, :][both] - u_true_np[:, hb, :][both]).astype(np.float32))

    p_acc_a = np.concatenate(pos_acc_all) if pos_acc_all else np.empty(0, dtype=np.float32)
    p_don_a = np.concatenate(pos_don_all) if pos_don_all else np.empty(0, dtype=np.float32)
    p_acc_n = np.concatenate(pos_acc_ann) if pos_acc_ann else np.empty(0, dtype=np.float32)
    p_don_n = np.concatenate(pos_don_ann) if pos_don_ann else np.empty(0, dtype=np.float32)

    ap_acc_all = ap_from_hist_and_pos(p_acc_a, neg_hist_acc, n_bins)
    ap_don_all = ap_from_hist_and_pos(p_don_a, neg_hist_don, n_bins)
    ap_acc_ann = ap_from_hist_and_pos(p_acc_n, neg_hist_acc, n_bins)
    ap_don_ann = ap_from_hist_and_pos(p_don_n, neg_hist_don, n_bins)

    res = {
        "site_prauc_all": {
            "acceptor": round(ap_acc_all, 6) if np.isfinite(ap_acc_all) else None,
            "donor": round(ap_don_all, 6) if np.isfinite(ap_don_all) else None,
            "macro": round(0.5 * (ap_acc_all + ap_don_all), 6) if np.isfinite(ap_acc_all) and np.isfinite(ap_don_all) else None,
            "n_acceptor": int(len(p_acc_a)),
            "n_donor": int(len(p_don_a)),
        },
        "site_prauc_annotated_only": {
            "acceptor": round(ap_acc_ann, 6) if np.isfinite(ap_acc_ann) else None,
            "donor": round(ap_don_ann, 6) if np.isfinite(ap_don_ann) else None,
            "macro": round(0.5 * (ap_acc_ann + ap_don_ann), 6) if np.isfinite(ap_acc_ann) and np.isfinite(ap_don_ann) else None,
            "n_acceptor": int(len(p_acc_n)),
            "n_donor": int(len(p_don_n)),
        },
        "usage_per_head": {},
        "inter_tissue_diff": {},
    }

    if eval_usage:
        main_var_rhos = []
        for h, hname in enumerate(HEAD_NAMES):
            if not head_pred[h]:
                res["usage_per_head"][hname] = {"n_total": 0}
                continue
            xp = np.concatenate(head_pred[h])
            xt = np.concatenate(head_true[h])
            xs = np.concatenate(head_src[h])
            sp_all, pr_all = safe_spearman_pearson(xp, xt)
            m_var = (xs != 2) & (xt >= 0.05) & (xt <= 0.95)
            sp_var, pr_var = safe_spearman_pearson(xp[m_var], xt[m_var])
            mae_var = float(np.mean(np.abs(xp[m_var] - xt[m_var]))) if m_var.any() else float("nan")
            m_emp = xs != 2
            sp_emp, pr_emp = safe_spearman_pearson(xp[m_emp], xt[m_emp])
            res["usage_per_head"][hname] = {
                "n_total": int(len(xt)),
                "n_empirical": int(m_emp.sum()),
                "n_variable_005_095": int(m_var.sum()),
                "spearman_all": round(sp_all, 6) if np.isfinite(sp_all) else None,
                "spearman_empirical": round(sp_emp, 6) if np.isfinite(sp_emp) else None,
                "spearman_var_005_095": round(sp_var, 6) if np.isfinite(sp_var) else None,
                "pearson_var_005_095": round(pr_var, 6) if np.isfinite(pr_var) else None,
                "mae_var_005_095": round(mae_var, 6) if np.isfinite(mae_var) else None,
            }
            if 1 <= h <= 12 and np.isfinite(sp_var):
                main_var_rhos.append(sp_var)
        res["mean_main_heads_var_spearman"] = round(float(np.mean(main_var_rhos)), 6) if main_var_rhos else None

        for pi, (_ha, _hb, cname) in enumerate(CONTRAST_PAIRS):
            if not diff_pred[pi]:
                res["inter_tissue_diff"][cname] = {"n_sites": 0}
                continue
            dp = np.concatenate(diff_pred[pi])
            dt = np.concatenate(diff_true[pi])
            sp_a, _ = safe_spearman_pearson(dp, dt)
            dm = np.abs(dt) >= 0.15
            n_dm = int(dm.sum())
            sp_d, _ = safe_spearman_pearson(dp[dm], dt[dm])
            s_acc = float(((dp[dm] > 0) == (dt[dm] > 0)).mean()) if n_dm > 0 else float("nan")
            res["inter_tissue_diff"][cname] = {
                "n_sites": int(len(dt)),
                "spearman_all": round(sp_a, 6) if np.isfinite(sp_a) else None,
                "n_diff_015": n_dm,
                "spearman_diff_015": round(sp_d, 6) if np.isfinite(sp_d) else None,
                "sign_acc_diff_015": round(s_acc, 6) if np.isfinite(s_acc) else None,
            }

    return res


def get_lr_lambda(step: int, warmup_steps: int, total_steps: int, min_frac: float = 0.01):
    if step < warmup_steps:
        return max(0.01, float(step + 1) / max(1, warmup_steps))
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_frac + (1.0 - min_frac) * cosine


def train_job(args):
    t0 = time.time()
    job_dir = Path(args.runs_dir) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "DONE.json"
    if args.resume == "auto" and done_path.exists() and not args.force:
        print(f"[SKIP] {args.job_id} already DONE ({done_path})", flush=True)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)

    genome_dict, labels_npz, folds = load_pack_arrays(Path(args.pack_dir))
    arch_norm = _normalize_arch_name(args.arch)
    ann_only = (arch_norm == "siteonly")
    head_type = "site_only" if ann_only else "residual_mlp"
    tr_mask, va_mask, te_mask = build_window_masks(labels_npz, folds, args.fold, args.arch)

    ds_tr = SheepCompactDataset(genome_dict, labels_npz, tr_mask, annotated_only=ann_only, return_source=(not ann_only))
    ds_va = SheepCompactDataset(genome_dict, labels_npz, va_mask, annotated_only=ann_only, return_source=True)
    n_workers = 0 if (os.name == "nt" or device.type == "cpu" or args.num_workers == 0) else args.num_workers
    dl_kw = {"num_workers": n_workers, "pin_memory": (device.type == "cuda")}
    if n_workers > 0:
        dl_kw["persistent_workers"] = True

    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False, **dl_kw)
    va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kw)

    base_model = OviSplice15Head(head_type=head_type).to(device)
    init_tensors = 0
    if arch_norm in ("ovisplice", "ovisplice_full") and args.pangolin_ckpt:
        init_tensors = base_model.load_pangolin_trunk(args.pangolin_ckpt)
        assert init_tensors == 236, f"Expected 236 Pangolin trunk tensors, got {init_tensors}"

    model = torch.compile(base_model) if (args.compile and device.type == "cuda") else base_model
    use_trunk_mult = (arch_norm in ("ovisplice", "ovisplice_full")) and bool(args.pangolin_ckpt)
    trunk_lr = args.lr * (args.trunk_lr_mult if use_trunk_mult else 1.0)
    raw_model = unwrap_model(model)
    trunk_params, head_params = [], []
    for name, p in raw_model.named_parameters():
        if name.startswith(("conv1.", "skip.", "resblocks.", "convs.")):
            trunk_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": trunk_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    total_steps = max(1, args.epochs * max(1, len(tr_loader)))
    warmup_steps = min(500, max(1, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda st: get_lr_lambda(st, warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    start_epoch = 1
    best_score = -1.0
    last_ckpt = job_dir / "last.pt"
    best_ckpt = job_dir / "best.pt"
    if args.resume == "auto" and last_ckpt.exists():
        st = torch.load(last_ckpt, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(st["model_state"])
        opt.load_state_dict(st["opt_state"])
        sched.load_state_dict(st["sched_state"])
        if "scaler_state" in st and amp_dtype == torch.float16:
            scaler.load_state_dict(st["scaler_state"])
        start_epoch = int(st["epoch"]) + 1
        best_score = float(st["best_score"])
        print(f"[RESUME] {args.job_id} from epoch {start_epoch} (best={best_score:.4f})", flush=True)

    hist_path = job_dir / "history.jsonl"
    use_amp = device.type == "cuda"
    oh_dtype = amp_dtype if use_amp else torch.float32

    for ep in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        sum_loss, sum_cls, sum_use, n_steps = 0.0, 0.0, 0.0, 0

        for step_idx, batch in enumerate(tr_loader):
            if args.max_steps > 0 and step_idx >= args.max_steps:
                break
            if ann_only:
                seq_u8, cls_y, use_y, msk_y = batch
                src_y = None
            else:
                seq_u8, cls_y, use_y, msk_y, src_y = batch
                src_y = src_y.to(device, non_blocking=True)
            seq_u8 = seq_u8.to(device, non_blocking=True)
            cls_y = cls_y.to(device, non_blocking=True)
            use_y = use_y.to(device, non_blocking=True)
            msk_y = msk_y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_oh = uint8_to_onehot_gpu(seq_u8, oh_dtype)
                s_log, u_log = model(x_oh)
                loss, l_cls, l_use = compute_loss(
                    s_log, u_log, cls_y, use_y, msk_y,
                    src_y=src_y,
                    annotation_only=ann_only,
                    cls_scale=args.cls_scale,
                    lambda_usage=args.lambda_usage,
                    lambda_emp=args.lambda_emp,
                    lambda_diff=args.lambda_diff,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            sum_loss += float(loss.detach())
            sum_cls += float(l_cls)
            sum_use += float(l_use)
            n_steps += 1

        val_res = evaluate_dataset(
            model, va_loader, device, amp_dtype,
            eval_usage=(not ann_only), max_batches=args.max_val_batches,
        )
        va_ap = val_res["site_prauc_annotated_only"]["macro"] if ann_only else val_res["site_prauc_all"]["macro"]
        va_ap = 0.0 if va_ap is None else float(va_ap)
        va_rho = val_res.get("mean_main_heads_var_spearman")
        va_rho = 0.0 if va_rho is None else float(va_rho)
        score = va_ap if ann_only else (va_ap + 1.0 * va_rho)

        is_best = score > best_score
        if is_best:
            best_score = score
            atomic_save({
                "epoch": ep,
                "arch": args.arch,
                "head_type": head_type,
                "fold": args.fold,
                "seed": args.seed,
                "best_score": best_score,
                "model_state": unwrap_model(model).state_dict(),
            }, best_ckpt)

        atomic_save({
            "epoch": ep,
            "arch": args.arch,
            "head_type": head_type,
            "best_score": best_score,
            "model_state": unwrap_model(model).state_dict(),
            "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
            "scaler_state": scaler.state_dict(),
        }, last_ckpt)

        rec = {
            "epoch": ep,
            "train_loss": round(sum_loss / max(1, n_steps), 5),
            "train_cls": round(sum_cls / max(1, n_steps), 5),
            "train_use": round(sum_use / max(1, n_steps), 5),
            "val_macro_prauc": round(va_ap, 5),
            "val_mean_var_rho": round(va_rho, 5),
            "val_score": round(score, 5),
            "is_best": is_best,
            "lr": round(opt.param_groups[-1]["lr"], 7),
            "epoch_s": round(time.time() - ep_t0, 2),
        }
        with open(hist_path, "a", encoding="utf-8") as w:
            w.write(json.dumps(rec) + "\n")
        print(f"[{args.job_id}] ep={ep}/{args.epochs} {rec}", flush=True)

    # Final held-out fold evaluation using best.pt (both paralog-free and all)
    test_metrics = {}
    if args.fold > 0 and te_mask.any() and not args.skip_test_eval:
        best_st = torch.load(best_ckpt, map_location=device, weights_only=True)
        unwrap_model(model).load_state_dict(best_st["model_state"])
        w_para = labels_npz["win_paralog"]
        bit = 1 << (args.fold - 1)
        for subset_name, sub_mask in (
            ("paralog_free", te_mask & ((w_para & bit) == 0)),
            ("all_test_windows", te_mask),
        ):
            ds_te = SheepCompactDataset(genome_dict, labels_npz, sub_mask, annotated_only=False, return_source=True)
            te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, **dl_kw)
            test_metrics[subset_name] = evaluate_dataset(
                model, te_loader, device, amp_dtype, eval_usage=(not ann_only), max_batches=args.max_val_batches,
            )

    summary = {
        "job_id": args.job_id,
        "arch": args.arch,
        "head_type": head_type,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "init_pangolin_tensors": init_tensors,
        "best_val_score": round(best_score, 6),
        "test_metrics": test_metrics,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(job_dir / "test_eval.json", "w", encoding="utf-8") as w:
        json.dump(summary, w, indent=2)
    with open(done_path, "w", encoding="utf-8") as w:
        json.dump({"done": True, "job_id": args.job_id, "best_val_score": best_score}, w, indent=2)


def eval_fold_cmd(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)

    genome_dict, labels_npz, folds = load_pack_arrays(Path(args.pack_dir))
    ckpts = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    models = []
    arch = args.arch
    for cp in ckpts:
        m, st = load_ovisplice_ckpt(cp, device)
        arch = st.get("arch", arch)
        models.append(m)

    _, _, te_mask = build_window_masks(labels_npz, folds, args.fold, arch)
    w_para = labels_npz["win_paralog"]
    bit = 1 << (args.fold - 1)
    out = {"fold": args.fold, "arch": arch, "ckpt": args.ckpt, "n_models": len(models)}
    for subset_name, sub_mask in (
        ("paralog_free", te_mask & ((w_para & bit) == 0)),
        ("all_test_windows", te_mask),
    ):
        ds_te = SheepCompactDataset(genome_dict, labels_npz, sub_mask, annotated_only=False, return_source=True)
        loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)
        out[subset_name] = evaluate_dataset(
            models, loader, device, amp_dtype, eval_usage=(not _is_siteonly_arch(arch)), max_batches=args.max_val_batches,
        )
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as w:
            json.dump(out, w, indent=2)
    print(json.dumps(out, indent=2), flush=True)


@torch.no_grad()
def score_variants_cmd(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    use_amp = device.type == "cuda"
    oh_dtype = amp_dtype if use_amp else torch.float32

    ckpts = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    models = []
    for cp in ckpts:
        m, _st = load_ovisplice_ckpt(cp, device)
        models.append(m)

    pack_dir = Path(args.pack_dir)
    if args.species == "sheep":
        g_npz = np.load(pack_dir / "sheep_genome_uint8.npz", allow_pickle=False)
        genome = {k: g_npz[k] for k in g_npz.files}
        b = np.load(pack_dir / "bench_sheep.npz", allow_pickle=False)
        idx = np.flatnonzero(b["fold"] == args.fold) if args.fold > 0 else np.arange(len(b["pos"]))
    else:
        b = np.load(pack_dir / "cattle_eval_windows.npz", allow_pickle=False)
        idx = np.arange(len(b["pos"]))

    if args.max_variants > 0:
        idx = idx[: args.max_variants]

    win_half = 5500
    win_len = 2 * win_half + 1  # 11,001 -> output 1,001 (+-500 bp around index 500)
    out_center = 500
    out_records = []

    for start in range(0, len(idx), args.batch_size):
        sub_i = idx[start : start + args.batch_size]
        B = len(sub_i)
        ref_mat = np.empty((B, win_len), dtype=np.uint8)
        alt_mat = np.empty((B, win_len), dtype=np.uint8)

        if args.species == "sheep":
            for j, vi in enumerate(sub_i):
                g = genome[str(int(b["chrom_id"][vi]))]
                g0 = int(b["pos"][vi]) - 1
                s0, s1 = g0 - win_half, g0 + win_half + 1
                if s0 >= 0 and s1 <= len(g):
                    wseq = g[s0:s1].copy()
                else:
                    wseq = np.full(win_len, 4, dtype=np.uint8)
                    a0, a1 = max(0, s0), min(len(g), s1)
                    if a1 > a0:
                        wseq[a0 - s0:a1 - s0] = g[a0:a1]
                r_u8 = int(b["ref_u8"][vi])
                a_u8 = int(b["alt_u8"][vi])
                if int(b["strand"][vi]) < 0:
                    wseq = COMP_U8[wseq[::-1]]
                    r_u8 = int(COMP_U8[r_u8])
                    a_u8 = int(COMP_U8[a_u8])
                assert int(wseq[win_half]) == r_u8, f"Sheep REF mismatch at {b['snp_id'][vi]}"
                ref_mat[j] = wseq
                wseq_alt = wseq.copy()
                wseq_alt[win_half] = a_u8
                alt_mat[j] = wseq_alt
        else:
            ref_mat[:] = b["win_u8"][sub_i]
            alt_mat[:] = ref_mat
            alt_mat[:, win_half] = b["alt_tx_u8"][sub_i]

        t_ref = torch.from_numpy(ref_mat).to(device)
        t_alt = torch.from_numpy(alt_mat).to(device)

        p_ref_acc = np.zeros((B, 2, 1001), dtype=np.float32)
        p_alt_acc = np.zeros((B, 2, 1001), dtype=np.float32)
        u_ref_acc = np.zeros((B, 15, 1001), dtype=np.float32)
        u_alt_acc = np.zeros((B, 15, 1001), dtype=np.float32)

        for m in models:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                sr, ur = m(uint8_to_onehot_gpu(t_ref, oh_dtype))
                sa, ua = m(uint8_to_onehot_gpu(t_alt, oh_dtype))
            p_ref_acc += F.softmax(sr.float(), dim=1)[:, 1:3, :].cpu().numpy()
            p_alt_acc += F.softmax(sa.float(), dim=1)[:, 1:3, :].cpu().numpy()
            u_ref_acc += torch.sigmoid(ur.float()).cpu().numpy()
            u_alt_acc += torch.sigmoid(ua.float()).cpu().numpy()

        inv_m = 1.0 / len(models)
        p_ref_acc *= inv_m
        p_alt_acc *= inv_m
        u_ref_acc *= inv_m
        u_alt_acc *= inv_m

        dp = p_alt_acc - p_ref_acc                      # [B, 2, 1001]
        du = u_alt_acc - u_ref_acc                      # [B, 15, 1001]
        p_any_max = np.maximum(p_ref_acc.sum(axis=1), p_alt_acc.sum(axis=1))  # [B, 1001]
        mod_du = du * p_any_max[:, None, :]             # probability-gated usage delta
        mod_signed = (u_alt_acc * p_alt_acc.sum(axis=1)[:, None, :]) - (u_ref_acc * p_ref_acc.sum(axis=1)[:, None, :])

        c50_lo, c50_hi = out_center - 50, out_center + 51
        for j, vi in enumerate(sub_i):
            h = int(b["head_idx"][vi]) if args.species == "sheep" else 1  # SkeletalMuscle for cattle
            st_v = int(b["strand"][vi])
            pos_v = int(b["pos"][vi])
            i_s = int(b["i_s"][vi])
            i_e = int(b["i_e"][vi])

            rel_s = out_center + ((i_s - pos_v) if st_v > 0 else (pos_v - i_s))
            rel_e = out_center + ((i_e - pos_v) if st_v > 0 else (pos_v - i_e))
            d_u_vals, d_mod_vals = [], []
            for rp in (rel_s, rel_e) if i_s != i_e else (rel_s,):
                if 0 <= rp < 1001:
                    d_u_vals.append(float(du[j, h, rp]))
                    d_mod_vals.append(float(mod_signed[j, h, rp]))

            out_records.append({
                "snp_id": str(b["snp_id"][vi]),
                "set_type": str(b["set_type"][vi]),
                "label": int(b["label"][vi]),
                "fold": int(b["fold"][vi]) if args.species == "sheep" else 0,
                "head_idx": h,
                "ds_site_50": round(float(np.abs(dp[j, :, c50_lo:c50_hi]).max()), 6),
                "ds_site_500": round(float(np.abs(dp[j]).max()), 6),
                "ds_usage_matched_50": round(float(np.abs(mod_du[j, h, c50_lo:c50_hi]).max()), 6),
                "ds_usage_matched_500": round(float(np.abs(mod_du[j, h]).max()), 6),
                "ds_usage_max12_50": round(float(np.abs(mod_du[j, 1:13, c50_lo:c50_hi]).max()), 6),
                "ds_usage_max12_500": round(float(np.abs(mod_du[j, 1:13]).max()), 6),
                "du_raw_matched_50": round(float(np.abs(du[j, h, c50_lo:c50_hi]).max()), 6),
                "du_raw_matched_500": round(float(np.abs(du[j, h]).max()), 6),
                "du_raw_max12_50": round(float(np.abs(du[j, 1:13, c50_lo:c50_hi]).max()), 6),
                "du_raw_max12_500": round(float(np.abs(du[j, 1:13]).max()), 6),
                "delta_intron_u": round(float(np.mean(d_u_vals)), 6) if d_u_vals else float("nan"),
                "delta_intron_mod": round(float(np.mean(d_mod_vals)), 6) if d_mod_vals else float("nan"),
                "beta": float(b["beta"][vi]),
                "dist": int(b["dist"][vi]),
            })

    import pandas as pd
    df_out = pd.DataFrame(out_records)
    if args.out_tsv:
        out_p = Path(args.out_tsv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        comp = "gzip" if str(out_p).endswith(".gz") else None
        df_out.to_csv(out_p, sep="\t", index=False, compression=comp)
    print(f"[SCORE:{args.species}] scored={len(df_out)} elapsed={time.time() - t0:.2f}s", flush=True)


def run_queue_cmd(args):
    with open(args.jobs_file, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            job_id = row["job_id"].strip()
            if not job_id or job_id.startswith("#"):
                continue
            j_args = argparse.Namespace(
                job_id=job_id,
                arch=row["arch"].strip(),
                fold=int(row["fold"]),
                seed=int(row["seed"]),
                epochs=int(row["epochs"]),
                batch_size=int(row["batch_size"]),
                lr=float(row.get("lr", 1e-3)),
                trunk_lr_mult=float(row.get("trunk_lr_mult", 0.3)),
                cls_scale=float(row.get("cls_scale", 500.0)),
                lambda_usage=float(row.get("lambda_usage", 1.0)),
                lambda_emp=float(row.get("lambda_emp", 1.0)),
                lambda_diff=float(row.get("lambda_diff", 4.0)),
                compile=row["compile"].strip().lower() in ("1", "true", "yes"),
                pack_dir=args.pack_dir,
                runs_dir=args.runs_dir,
                pangolin_ckpt=args.pangolin_ckpt,
                resume="auto",
                force=False,
                cpu=args.cpu,
                num_workers=args.num_workers,
                max_steps=args.max_steps,
                max_val_batches=args.max_val_batches,
                skip_test_eval=False,
            )
            train_job(j_args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "train", "queue", "eval_fold", "score_variants"], default="smoke")
    ap.add_argument("--pack_dir", type=str, default="data/pack")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--jobs_file", type=str, default="")
    ap.add_argument("--job_id", type=str, default="debug_job")
    ap.add_argument("--arch", choices=["siteonly", "scratch", "ovisplice", "ovisplice_full"], default="ovisplice")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trunk_lr_mult", type=float, default=0.3)
    ap.add_argument("--cls_scale", type=float, default=500.0)
    ap.add_argument("--lambda_usage", type=float, default=1.0)
    ap.add_argument("--lambda_emp", type=float, default=1.0)
    ap.add_argument("--lambda_diff", type=float, default=4.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--pangolin_ckpt", type=str, default="")
    ap.add_argument("--resume", type=str, default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    ap.add_argument("--skip_test_eval", action="store_true")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--species", choices=["sheep", "cattle"], default="sheep")
    ap.add_argument("--out_tsv", type=str, default="")
    ap.add_argument("--max_variants", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "smoke":
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = OviSplice15Head().to(device)
        x = torch.randint(0, 4, (2, 15000), device=device, dtype=torch.uint8)
        s_log, u_log = model(uint8_to_onehot_gpu(x, torch.float32))
        print(json.dumps({"ok": True, "s_log": list(s_log.shape), "u_log": list(u_log.shape)}))
    elif args.mode == "train":
        train_job(args)
    elif args.mode == "queue":
        run_queue_cmd(args)
    elif args.mode == "eval_fold":
        eval_fold_cmd(args)
    elif args.mode == "score_variants":
        score_variants_cmd(args)


if __name__ == "__main__":
    main()
