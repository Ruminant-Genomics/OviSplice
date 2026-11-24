# Public Input Data Sources & Official Accessions

Following standard computational biology data-sharing practices, **OviSplice** does not redistribute third-party public sequencing matrices or consortium databases inside this software repository. All input datasets analyzed in the study are publicly accessible from their original repositories using the accessions and DOIs below:

| Dataset / Resource | Assembly / Scope | Official Repository & Accession / DOI |
| :--- | :--- | :--- |
| **Sheep Reference Genome & Annotation** | `ARS-UI_Ramb_v2.0` (`NCBI RefSeq 104` + `Ensembl 113`) | NCBI Assembly `GCF_016772045.1` (`https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_016772045.1/`) and Ensembl Release 113 FTP (`https://ftp.ensembl.org/pub/release-113/gtf/ovis_aries_rambouillet/`) |
| **Cattle Reference Genome & Annotation** | `ARS-UCD1.2` (`bosTau9`, `NCBI RefSeq`) | NCBI Assembly `GCF_002263795.1` (`https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_002263795.1/`) |
| **SheepGTEx Consortium Atlas** | 8,148 training RNA-seq junction matrices, genotypes (`MAF >= 0.05`), fine-mapped sQTLs & GWAS colocalizations | SheepGTEx Portal (`https://sheepgtex.org/` / Gong et al., *Nature Genetics*, `doi:10.1038/s41588-026-02825-2`) |
| **Independent Sheep Muscle Developmental Blind Test** | 12 raw RNA-seq BAMs across 4 prenatal/postnatal stages | NCBI GEO `GSE224958` / BioProject `PRJNA933184` |
| **CattleGTEx Phase 1 Consortium Atlas** | 7,483 zero-shot RNA-seq junction matrices (12 organs), 28-tissue fine-mapped sQTLs & GWAS colocalizations | Zenodo `doi:10.5281/zenodo.7560235` and `doi:10.5281/zenodo.7559568` (Liu et al., *Nature Genetics* 2022) |
| **Bovine Vex-seq MPRA Wet-Lab Benchmark** | 304 cattle variants (`121 Var.P + 183 Var.GWAS`) assayed in `MAC-T` & `HEK293T` cells | Charles et al., *Nature Communications* 2025 (`doi:10.1038/s41467-025-58970-5`, Supplementary Data) |
| **OMIA Mendelian Splice-Disrupting Mutations** | Curated wet-lab causal splicing mutations in cattle (`9913`) and sheep (`9940`) | Online Mendelian Inheritance in Animals (`https://omia.org/`) |
| **UCSC Pairwise Syntenic Alignment Chains** | Pairwise genome alignment chains (`bosTau9`, `oviAri4`/`ARS-UI_Ramb_v2.0`, `hg38`) | UCSC Genome Browser Downloads (`https://hgdownload.soe.ucsc.edu/downloads.html`) |

## Author-Generated Model Checkpoints & Genome-Wide Prediction Scores (Zenodo)

All 19 frozen OviSplice PyTorch checkpoints (5-fold LOCO ensemble, full 3-seed cross-species ensemble, and controlled single-task / random-initialization ablations; `ovisplice_ckpts_*_v1.0.tar.gz`) together with the self-generated **624,445-SNV sheep 15-head splicing prediction atlas** (`ovisplice_sheep_624k_snv_atlas.npz`) are archived on Zenodo ([`doi:10.5281/zenodo.24503626`](https://doi.org/10.5281/zenodo.24503626)).
