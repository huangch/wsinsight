# WSInsight — Remote Data & TCGA Cohorts

Reference material for `SKILL.md`. Read this only when the task involves
building a TCGA/GDC cohort or joining WSInsight outputs to clinical endpoints.
Day-to-day operation needs nothing from this file.

- [1. Acquiring a TCGA slide manifest via the GDC API](#1-acquiring-a-tcga-slide-manifest-via-gdc-api)
- [2. Acquiring TCGA clinical & molecular data](#2-acquiring-tcga-clinical--molecular-data)

---

## 1. Acquiring a TCGA Slide Manifest via GDC API

The NCI Genomic Data Commons (GDC) hosts all TCGA, TARGET, and other NCI
program data.  Its REST API can return **manifest files** directly — no
`gdc-client` binary is needed.  The manifest is a TSV listing slide UUIDs and
filenames; WSInsight then downloads the actual slides on demand during
inference via the GDC data endpoint (`https://api.gdc.cancer.gov/data/{uuid}`)
with automatic retries and MD5 verification.

### 1.1 Generating a Manifest

POST to `https://api.gdc.cancer.gov/files` with `return_type=manifest`:

```bash
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-BRCA"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "Diagnostic Slide"
                }
            }
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-dx-manifest.tsv
```

**Key parameters:**

| Parameter       | Value              | Purpose                                          |
| --------------- | ------------------ | ------------------------------------------------ |
| `return_type`   | `manifest`         | Returns TSV manifest instead of JSON metadata    |
| `size`          | `99999`            | Max results (default is 10 — always override)    |
| `filters`       | JSON object        | GDC filter syntax (see below)                    |

### 1.2 Filter Fields for Slide Images

| Field                            | Values                                              | Notes                                   |
| -------------------------------- | --------------------------------------------------- | --------------------------------------- |
| `cases.project.project_id`       | `TCGA-BRCA`, `TCGA-LUAD`, etc.                     | Required — selects the cohort           |
| `data_type`                      | `Slide Image`                                       | Required — filters to WSI files         |
| `experimental_strategy`          | `Diagnostic Slide` or `Tissue Slide`                | Diagnostic = formalin-fixed; Tissue = frozen section |
| `data_format`                    | `SVS`                                               | Optional — all TCGA slides are SVS      |
| `cases.submitter_id`             | `TCGA-A7-A0CE`, ...                                | Optional — filter to specific cases     |

Filters use the GDC query DSL with operators: `=`, `!=`, `in`, `and`, `or`.
Nested filters are combined with `"op": "and"` at the top level.

### 1.3 Common TCGA Project IDs

| Cancer Type                  | Project ID    |
| ---------------------------- | ------------- |
| Breast invasive carcinoma    | `TCGA-BRCA`   |
| Lung adenocarcinoma          | `TCGA-LUAD`   |
| Lung squamous cell carcinoma | `TCGA-LUSC`   |
| Prostate adenocarcinoma      | `TCGA-PRAD`   |
| Pancreatic adenocarcinoma    | `TCGA-PAAD`   |
| Colon adenocarcinoma         | `TCGA-COAD`   |
| Rectum adenocarcinoma        | `TCGA-READ`   |
| Glioblastoma multiforme      | `TCGA-GBM`    |
| Ovarian serous cystadenocarcinoma | `TCGA-OV` |
| Uterine corpus endometrial   | `TCGA-UCEC`   |
| Kidney renal clear cell      | `TCGA-KIRC`   |
| Head and neck squamous cell  | `TCGA-HNSC`   |
| Liver hepatocellular         | `TCGA-LIHC`   |
| Stomach adenocarcinoma       | `TCGA-STAD`   |
| Bladder urothelial           | `TCGA-BLCA`   |
| Skin cutaneous melanoma      | `TCGA-SKCM`   |

### 1.4 Manifest Format

The GDC API returns a TSV with these columns:

```text
id	filename	md5	size	state
UUID-1	TCGA-A7-A0CE-01Z-00-DX1.svs	abc123...	234567890	released
UUID-2	TCGA-A7-A13E-01Z-00-DX1.svs	def456...	345678901	released
```

WSInsight's `URIPath` reads this natively — it looks for `id`/`file_id` and
`filename`/`file_name` columns, plus optional `md5` for checksum verification.

### 1.5 Access Control

- **TCGA diagnostic and tissue slides are open-access** — no token needed.
- For controlled-access data (e.g. some TARGET projects), obtain an
  authentication token from the [GDC Data Portal](https://portal.gdc.cancer.gov/).
  There is no CLI flag for the token; pass it programmatically via the `token`
  (or `token_path`) keyword argument on `URIPath`.  For typical TCGA workflows
  this is not required.

### 1.6 Combining Filters

To select only specific cases within a project, use `"op": "in"`:

```bash
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": ["TCGA-A7-A0CE", "TCGA-A7-A13E", "TCGA-BH-A0B3"]
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "Diagnostic Slide"
                }
            }
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-subset-manifest.tsv
```

### 1.7 End-to-End Example

```bash
# 1. Download manifest for all TCGA-BRCA diagnostic slides
curl --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "filters": {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}}
        ]
    },
    "return_type": "manifest",
    "size": "99999"
  }' \
  'https://api.gdc.cancer.gov/files' \
  > tcga-brca-dx-manifest.tsv

# 2. Verify slide count (header + data lines)
wc -l tcga-brca-dx-manifest.tsv

# 3. Run WSInsight on the manifest
wsinsight run \
  --wsi-dir "gdc-manifest://$(pwd)/tcga-brca-dx-manifest.tsv" \
  --results-dir results-brca/ \
  --model breast-tumor-resnet34.tcga-brca \
  --batch-size 32
```

WSInsight downloads each slide on demand via `https://api.gdc.cancer.gov/data/{uuid}`,
caches it locally under the directory set by `WSINSIGHT_REMOTE_CACHE_DIR`
(defaults to `~/.cache/wsinsight` via `platformdirs.user_cache_dir`), and
processes it.

---

## 2. Acquiring TCGA Clinical & Molecular Data

WSInsight produces per-slide morphological outputs.  Linking them to clinical
endpoints (survival, treatment, subtypes) requires external clinical tables.
TCGA slide filenames encode the patient barcode:

```text
TCGA-A7-A0CE-01Z-00-DX1.svs
└─── patient ───┘
```

Extract the first 12 characters (`TCGA-A7-A0CE`) as the join key.

### 2.1 GDC `/cases` API — Demographics, Staging & Treatment

```bash
curl 'https://api.gdc.cancer.gov/cases?filters={"op":"=","content":{"field":"cases.project.project_id","value":"TCGA-BRCA"}}&expand=diagnoses,diagnoses.treatments,demographic&size=99999&format=TSV' \
  > tcga-brca-clinical.tsv
```

Fields returned via `expand=`:

| Expand path              | Key fields                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `demographic`            | `vital_status`, `days_to_death`, `days_to_birth`, `gender`, `race`, `ethnicity`                |
| `diagnoses`              | `ajcc_pathologic_stage`, `ajcc_pathologic_t/n/m`, `primary_diagnosis`, `age_at_diagnosis`, `days_to_last_follow_up`, `days_to_recurrence`, `progression_or_recurrence`, `tumor_grade` |
| `diagnoses.treatments`   | `treatment_type` (Surgery / Radiation / Pharmaceutical), `therapeutic_agents`, `treatment_intent_type` (Adjuvant / First-Line) |

Join key: `submitter_id` in the TSV (e.g. `TCGA-A7-A0CE`).

### 2.2 Curated Survival — Liu et al. 2018 (Recommended)

The GDC raw fields (`days_to_death`, `days_to_last_follow_up`) require manual
curation.  **Liu et al.** already did this for all 33 TCGA cancer types:

> Liu J et al. "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive
> High-Quality Survival Outcome Analytics." *Cell* 173(2):400-416, 2018.
> DOI: [10.1016/j.cell.2018.02.052](https://doi.org/10.1016/j.cell.2018.02.052)

Download **Supplementary Table 1** (Excel).  Columns:

| Column      | Meaning                                |
| ----------- | -------------------------------------- |
| `bcr_patient_barcode` | Patient ID (e.g. `TCGA-A7-A0CE`) |
| `OS`, `OS.time`       | Overall Survival (event + days)  |
| `PFI`, `PFI.time`     | Progression-Free Interval        |
| `DFI`, `DFI.time`     | Disease-Free Interval            |
| `DSS`, `DSS.time`     | Disease-Specific Survival        |

This is the gold-standard source for survival analysis on TCGA data.

### 2.3 Molecular Subtypes & Biomarkers

| Data type               | Best source                                | Join key           | Notes                                                        |
| ----------------------- | ------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| **PAM50** (BRCA)        | TCGA BRCA paper supplementary or cBioPortal | `PATIENT_ID`       | Luminal A/B, HER2-enriched, Basal-like, Normal-like          |
| **MSI-H / MSS**         | TCGA PanCanAtlas or MANTIS/MSISensor scores | `PATIENT_ID`       | Relevant for COAD, READ, STAD, UCEC                          |
| **Immune subtypes**     | Thorsson et al. 2018 (*Immunity*)          | `TCGA Participant Barcode` | C1–C6 pan-cancer immune subtypes                     |
| **ER / PR / HER2**      | cBioPortal clinical data tab               | `PATIENT_ID`       | Receptor status for breast cancer                            |
| **Mutations / CNA**     | cBioPortal or GDC MAF files                | `PATIENT_ID`       | TP53, PIK3CA, BRAF, KRAS, etc.                               |

### 2.4 cBioPortal — One-Stop Download

[cBioPortal](https://www.cbioportal.org/) aggregates clinical, molecular, and
genomic data into downloadable TSVs.  Example for TCGA-BRCA:

```text
https://www.cbioportal.org/study/clinicalData?id=brca_tcga_pan_can_atlas_2018
```

Download the "Clinical Data" tab as TSV — it includes PAM50, ER/PR/HER2
status, survival, and staging in a single table keyed by `PATIENT_ID`.

### 2.5 Joining Clinical Data with WSInsight Outputs

```python
import pandas as pd
from pathlib import Path

# Load WSInsight per-slide output
slide_csv = Path("results/model-outputs-csv/TCGA-A7-A0CE-01Z-00-DX1.csv")
df = pd.read_csv(slide_csv)

# Extract patient barcode from filename
patient_id = slide_csv.stem.rsplit("-", 3)[0]   # "TCGA-A7-A0CE"

# Load clinical table (e.g. Liu et al. or cBioPortal)
clinical = pd.read_csv("tcga-brca-clinical.tsv", sep="\t")

# Join
patient_row = clinical[clinical["bcr_patient_barcode"] == patient_id]
print(patient_row[["OS", "OS.time", "PFI", "PFI.time"]].iloc[0])
```

For cohort-level analysis, iterate over all CSVs in `model-outputs-csv/`,
extract each patient barcode, and merge into a single DataFrame.

### 2.6 Example: cross-cohort biomarker-landscape screen

A reference end-to-end pattern (TCGA-BRCA + TCGA-CRC) lives at
`experiments/biomarker-landscape.ipynb`. It uses only two WSInsight artefacts
per slide — `model-outputs-csv/<slide>.csv` and `graphs/<slide>.h5` (both
produced by the stable `infer` + `ncomp` pipeline) — and builds per-slide
features stratified into three region strata (`all` / `tumor` / `nontumor`)
from the cached Delaunay graph:

```python
import h5py, numpy as np
with h5py.File("results/graphs/SLIDE.h5", "r") as fh:
    simplices = fh["simplices"][()]           # (M, 3) int32
    centers   = fh["cell_centers"][()]        # (N, 2) int32

# Prune simplices by µm edge length (25 µm default, matches ncomp).
EDGE_LIMIT_UM = 25.0
SPACING_UM_PX = 0.25                          # 40x slides
edge_limit_px = EDGE_LIMIT_UM / SPACING_UM_PX
coords = centers[simplices]
max_edge = np.linalg.norm(coords[:, [0, 1, 0]] - coords[:, [1, 2, 2]], axis=2).max(axis=1)
simp_ok = simplices[max_edge <= edge_limit_px]
```

Slide features are averaged to the patient level (first-12-char barcode) and
screened univariately against every available clinical score per cohort —
binary outcomes (receptor status, stage, MSI-H, 5-year OS event) via
two-sided Mann–Whitney *U* / AUC, continuous outcomes (tumor purity, CYT,
GZMA, PRF1, Leukocyte Fraction, Thorsson immune scores, age) via Spearman *ρ*,
with Benjamini–Hochberg FDR correction per (cohort, score). Survival is
handled with `lifelines` (KM + Cox HR).

The notebook ships with a BRCA pretreatment filter
(`history_of_neoadjuvant_treatment == "No"`, 1085/1099 patients) so
morphology-derived biomarkers are not confounded by neoadjuvant therapy.
Cohort artefacts land in `biomarker_landscape_results/`
(`slide_features_<cohort>.csv`, `biomarker_landscape_screen.csv`,
`biomarker_landscape_best.csv`, `km_best_per_cohort.{png,svg}`, …).
Adding a new cohort only requires a `COHORTS[...]` entry and a clinical-score
assembly function that emits a long-form
`(patient_barcode, score_name, score_type, value)` table.

> Any features in this notebook that go beyond what `ncomp` emits (e.g.
> triad-level summaries computed from the raw `simplices` dataset) are
> *user-side* derivations. They are not part of the stable WSInsight surface
> and will not track schema changes in the experimental `tcomp` command.
