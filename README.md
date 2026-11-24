# OviSplice

**OviSplice** is a multi-tissue deep learning and empirical tissue-prior (**`Ruminant SpliceMap`**) software package for predicting pre-mRNA splice-site probabilities, 15-head ruminant organ/developmental splice usage, and non-coding variant splicing effects across sheep (*Ovis aries*, `ARS-UI_Ramb_v2.0`) and cattle (*Bos taurus*, `ARS-UCD1.2`).

---

## Overview

- **10,001-bp Dilated Residual Architecture (`719,410` parameters):** 16 one-dimensional dilated residual blocks (`[(11,1)]x4 + [(11,4)]x4 + [(21,10)]x4 + [(41,25)]x4`) coupled to a 3-class splice-site classifier (`neither / 3'ss / 5'ss`) and a 15-head residual usage module (`PanTissue` + 12 ruminant organs including **Rumen** and **Abomasum** + Prenatal/Postnatal Skeletal Muscle; `OviSplice-SiteOnly` ablation: `696,739` parameters).
- **Zero-PyTorch C++17 / OpenMP / AVX2 Inference Engine + Pure-NumPy Fallback:** Batch-normalization-folded light-cone convolutions (`ovisplice/ovi_NativeEngine.cpp`) match `float32` PyTorch within `< 2e-5` without requiring a GPU or PyTorch installation, automatically falling back to pure NumPy when `g++` is absent.
- **Distance-Free `Ruminant SpliceMap` (`ovisplice/splicemap.py`):** Zero-retraining post-hoc integration of sequence perturbation scores with four strictly distance-free empirical target-tissue splicing priors (`has_active_500_t`, `psi_ref_t`, `psi_het_t`, `psi_spec_t`).

---

## Installation

```bash
git clone https://github.com/Ruminant-Genomics/OviSplice.git
cd OviSplice
pip install -r requirements.txt
# Optional editable CLI installation:
pip install -e .
```
*(PyTorch `>= 2.1` is optional and only required if retraining models via `ovisplice/model.py`.)*

---

## Quick Start

### 1. Run Self-Test on Synthetic Example Variants

```bash
python ovisplice_predict.py demo
```

### 2. Predict Base-Resolution Splice Sites & 15-Head Usage on FASTA Sequences

```bash
python ovisplice_predict.py fasta \
  --input examples/example_sequences.fa \
  --output example_predictions.npz \
  --ensemble full
```

### 3. Score Genetic Variants (TSV + Indexed Reference FASTA)

```bash
# Score a TSV table (columns: chrom, pos, ref, alt, strand) against an indexed reference FASTA (.fa + .fa.fai):
python ovisplice_predict.py variants \
  --tsv my_variants.tsv \
  --fasta ARS-UCD1.2_genomic.fna \
  --radius 500 \
  --ensemble full \
  --output my_variants_scored.tsv

# Or score pre-extracted 11,001-bp 5'->3' windows directly:
python ovisplice_predict.py variants \
  --tsv examples/example_variants.tsv \
  --seq-col seq_11001bp_5to3 \
  --alt-col alt_5to3 \
  --radius 500 \
  --output example_scored.tsv
```
*(Use `--radius 50` when scoring short-insert episomal minigene assays such as Vex-seq.)*

---

## Prediction Head Index (`0..14`)

| Index | Head Name | Organ / Developmental Scope |
| :---: | :--- | :--- |
| `0` | `PanTissue` | All-sample ruminant reference head (also fallback for Embryo & Mammary Gland) |
| `1` | `SkeletalMuscle` | Bulk skeletal muscle |
| `2` | `Liver` | Liver |
| `3` | `Adipose` | Adipose tissue |
| `4` | `SmallIntestine` | Small intestine |
| `5` | `CNS` | Brain / central nervous system |
| `6` | `Rumen` | Foregut fermentation epithelium (rumen) |
| `7` | `Pituitary` | Pituitary gland |
| `8` | `Lung` | Lung |
| `9` | `Abomasum` | Glandular stomach (abomasum) |
| `10` | `Gonad` | Testis and ovary |
| `11` | `Heart` | Heart |
| `12` | `Skin` | Skin and wool follicle |
| `13` | `SkeletalMuscle_Prenatal` | Prenatal skeletal muscle |
| `14` | `SkeletalMuscle_Postnatal` | Postnatal skeletal muscle |

---

## Repository Layout & External Archives

```text
OviSplice/
├── ovisplice_predict.py       # Standalone CLI (demo / fasta / variants)
├── setup.py                   # Package installer
├── ovisplice/                 # Core model, C++17/AVX2 engine, NumPy fallback & Ruminant SpliceMap
├── checkpoints/               # Default 3-seed pre-trained OviSplice ensemble (8.7 MB)
├── examples/                  # Synthetic 11,001-bp test FASTA & variant TSV (~24 KB)
├── scripts/                   # Preprocessing, training, benchmarking & TE evolution pipelines
└── docs/PUBLIC_DATA_SOURCES.md # Official accessions/DOIs for all public input datasets
```

- **Full 19-Checkpoint LOCO/Ablation Archive & 624,445-SNV Sheep Splicing Score Atlas:** Hosted on Zenodo (`<INSERT_PUBLIC_ZENODO_DOI>`).
- **Public Input Datasets:** Listed with official accessions and DOIs in [`docs/PUBLIC_DATA_SOURCES.md`](docs/PUBLIC_DATA_SOURCES.md).

---

## License

Released under the [MIT License](LICENSE).
