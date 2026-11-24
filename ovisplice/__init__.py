# -*- coding: utf-8 -*-
"""
OviSplice: Dual-Species Multi-Tissue Deep Learning and Ruminant SpliceMap Framework.
"""
__version__ = "1.0.0"

from .native_engine import (
    HEAD_NAMES,
    CATTLE_TISSUE_MAP,
    BASE_MAP,
    COMP_U8,
    load_ensemble,
    load_pytorch_ckpt_numpy,
    seq_to_u8,
    u8_to_seq,
    tie_invariant_auroc_auprc,
    paired_delong_test,
)
from .splicemap import (
    build_splicemap_index,
    query_pure_splicemap,
    fit_cv_splicemap,
)

__all__ = [
    "__version__",
    "HEAD_NAMES",
    "CATTLE_TISSUE_MAP",
    "BASE_MAP",
    "COMP_U8",
    "load_ensemble",
    "load_pytorch_ckpt_numpy",
    "seq_to_u8",
    "u8_to_seq",
    "tie_invariant_auroc_auprc",
    "paired_delong_test",
    "build_splicemap_index",
    "query_pure_splicemap",
    "fit_cv_splicemap",
]
