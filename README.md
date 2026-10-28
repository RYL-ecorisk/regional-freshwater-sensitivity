# Cross-Species Toxicity Prediction and Regional SSD Analysis

Reproducibility code for the study of freshwater assemblage sensitivity to contaminants with diverse modes of action. The repository uses a simple root-level script layout: all analysis scripts are directly in the project root, all input data are in `data/`, and generated outputs are written to `results/`.

## Repository Structure

```text
.
├── README.md
├── jhm_toxicity_prediction.Rproj
├── r_dependencies.txt
├── python_dependencies.txt
├── 01_train_six_models.R
├── 02_export_xgboost_model.R
├── 03_analyze_shap_moa.R
├── 04_analyze_applicability_domain.R
├── 05_prepare_prediction_matrix.R
├── 06_predict_cross_species_toxicity.R
├── 07_select_ssd_distribution.R
├── 08_estimate_regional_hc5.R
├── 09_analyze_toxicity_gradients.R
├── 10_assess_protection_thresholds_and_risk.R
├── data/
│   ├── training_data.csv
│   ├── region_taxon_membership.csv
│   ├── taxon_traits_prediction.csv
│   ├── external_validation_raw.xlsx
│   ├── moa_cas_reference.xlsx
│   ├── supplementary_tables_reference.xlsx
│   └── README.md
└── results/
    └── generated models, predictions, checkpoints, and analysis outputs
```

Open the `.Rproj` file in RStudio or set the working directory to the repository root before running any script. All paths are relative to this root; no user-specific absolute paths are required.

## Six-Model Training

All six model-training workflows are combined in **one script**, `01_train_six_models.R`:

1. XGBoost
2. Ridge regression
3. Elastic Net
4. Random Forest
5. LightGBM
6. FT-Transformer

The script contains coarse search, TPE-based Bayesian optimization, local refinement, parameter-boundary checks, final fitting, and the FT-Transformer grid search. The five classical models use five-repeat, five-fold taxon-blocked cross-validation; FT-Transformer uses the shared fold assignments. A run-specific master seed is generated and saved on the first run. The seed, fold assignments, and checkpoints are retained so an interrupted run can resume.

The R dependencies are listed in `r_dependencies.txt`. FT-Transformer also requires the Python packages in `python_dependencies.txt`, a configured `RETICULATE_PYTHON`, and a working CUDA-enabled PyTorch environment. Training is computationally intensive and may require substantial time and disk space.

## Analysis Workflow

Run the scripts from the repository root in the following order.

### 1. Train and compare all six models

```r
source("01_train_six_models.R")
```

This writes model-specific tuning records, fitted model bundles, audit files, and the consolidated six-model comparison to `results/model_training/`.

### 2. Export the selected XGBoost model for downstream analyses

```r
source("02_export_xgboost_model.R")
```

### 3. Analyze SHAP contributions and MOA patterns

```r
source("03_analyze_shap_moa.R")
```

### 4. Calculate the applicability domain (analysis only)

```r
source("04_analyze_applicability_domain.R")
```

This script contains no plotting code. It calculates an equal-weight, 24-variable Manhattan-style applicability-domain score: 13 chemical descriptors, five ecological traits, five taxonomic variables, and one MOA variable. Numeric contributions use training-set standardization; categorical contributions use training-set category prevalence; each original variable contributes a weight of `1/24`. The 95th percentile of the training scores (R quantile type 7) defines the threshold.

The final screened external-validation dataset must be provided as `data/strict_external_validation.csv`. The included workbook `data/external_validation_raw.xlsx` is the raw source and is not assumed to be equivalent to the final screened 437-record dataset. The script checks the expected training size, external size, training P95, and number of external records within the domain.

### 5. Prepare the chemical–taxon prediction matrix

```r
source("05_prepare_prediction_matrix.R")
```

### 6. Predict cross-species toxicity

```r
source("06_predict_cross_species_toxicity.R")
```

### 7. Select the SSD distribution

```r
source("07_select_ssd_distribution.R")
```

### 8. Estimate regional HC5 values

```r
source("08_estimate_regional_hc5.R")
```

### 9. Analyze toxicity gradients and regional sensitivity variation

```r
source("09_analyze_toxicity_gradients.R")
```

### 10. Compare protection thresholds and prioritize risks

```r
source("10_assess_protection_thresholds_and_risk.R")
```

The nine-site Taihu pesticide-risk analysis additionally requires `data/taihu_monitoring_data.xlsx`, which was not included in the supplied source files. That analysis section cannot be reproduced until the original workbook is supplied; the code does not fabricate a substitute.

## File and Output Policy

- `data/` contains source and reference inputs only.
- The numbered `.R` files in the repository root contain the analysis workflows.
- `results/` contains generated outputs, fitted models, checkpoints, and audit tables.
- The scripts do not use machine-specific absolute paths. Run them from the project root.
- Do not delete checkpoints while a long-running training or SSD analysis is active.

## Reproducibility Note

The package has been reorganized and checked for path references and archive integrity. The complete R/Python workflow has not been executed end-to-end in the current environment. Before public release, run the workflow using the final study inputs and record the R, Python, package, CUDA, and hardware versions.
