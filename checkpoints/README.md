# Pre-trained OviSplice Checkpoints

This directory bundles the **3-seed full pre-trained OviSplice ensemble** (`ovisplice_full_seed42.pt`, `ovisplice_full_seed1042.pt`, `ovisplice_full_seed2042.pt`; 2.9 MB each, `719,410` parameters) used for out-of-the-box cross-species and genome-wide variant scoring (`--ensemble full`).

To keep the GitHub software repository lightweight (~9.0 MB), the complete **19-checkpoint archive** (`ovisplice_all_19_checkpoints_v1.0.tar.gz` — including all 10 paralog-shielded 5-fold LOCO sheep weights `ovisplice_fold1..5_seed42/1042.pt`, the 5 single-task `OviSplice-SiteOnly` weights [`696,739` parameters each], and the `OviSplice-Scratch` random-initialization weight) is deposited on Zenodo (`<INSERT_PUBLIC_ZENODO_DOI>`). Unpack those `.pt` files into this `checkpoints/` directory to run LOCO fold and ablation presets (`--ensemble sheep_fold1` .. `sheep_fold5`, `siteonly_fold1` .. `siteonly_fold5`, `scratch_fold1`).
