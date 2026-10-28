# JHM Cross-Species Toxicity Prediction: Reproducibility Package

This package contains the model-training workflow and downstream analysis scripts for the manuscript. The applicability-domain (AD) script is analysis-only: it calculates the equal-weight 24-variable Manhattan-style AD score and exports CSV results. It contains no plotting code.

## Contents

- `01_model_training.R`: five-model training and tuning for XGBoost, Ridge, Elastic Net, Random Forest, and LightGBM.
- `01b_ft_transformer_tuning.R`: optional FT-Transformer grid search.
- `01c_export_xgboost_compat.R`: export the selected XGBoost model and preprocessing metadata.
- `01d_combine_model_comparison.R`: combine five R-model results with FT-Transformer results.
- `00_prepare_prediction_matrix.R`: construct the chemical-by-taxon prediction matrix.
- `02_shap_moa.R`: SHAP and MOA analysis.
- `03_external_validation_and_applicability_domain.R`: analysis-only equal-weight 24-variable Manhattan-style AD calculation; no model fitting or plotting.
- `04_species_toxicity_prediction.R`: cross-species toxicity prediction.
- `05_ssd_distribution_selection.R`: SSD distribution comparison.
- `06_regional_ssd_hc5.R`: regional HC5 estimation and Monte Carlo uncertainty propagation.
- `07_toxicity_gradients_and_regional_sensitivity_variation.R`: toxicity gradients and regional sensitivity comparisons.
- `08_protection_thresholds_and_risk_prioritization.R`: protection-threshold comparison and risk prioritization.
- `data/`: training data and supporting data files supplied with this package.

Standalone figure-only scripts are not included. The AD calculation script does not generate figures.

## Software requirements

- R and the packages listed in `R-packages.txt`.
- XGBoost for R version 3.0.0 or later.
- For FT-Transformer only: a Python environment configured through `RETICULATE_PYTHON`, with the packages listed in `requirements-ft.txt`.

Install the required R packages as needed, for example:

```r
install.packages(c("data.table", "glmnet", "ranger", "xgboost", "ggplot2", "patchwork", "MASS", "reticulate", "readxl"))
```

LightGBM and `ssdtools` may require installation steps specific to the local R environment. Confirm package versions before a formal reproducibility run.

## Recommended run order

Run scripts from the package root.

1. Run `source("01_model_training.R")` to tune and evaluate the five R-based models.
2. Optionally configure `RETICULATE_PYTHON` and run `source("01b_ft_transformer_tuning.R")` for FT-Transformer.
3. If FT-Transformer was run, use `source("01d_combine_model_comparison.R")` to combine model-comparison results.
4. Run `source("01c_export_xgboost_compat.R")` after the XGBoost tuning run has completed.
5. Run `source("00_prepare_prediction_matrix.R")`, then the relevant downstream scripts (`02_shap_moa.R`, `04_species_toxicity_prediction.R`, `05_ssd_distribution_selection.R`, `06_regional_ssd_hc5.R`, `07_toxicity_gradients_and_regional_sensitivity_variation.R`, and `08_protection_thresholds_and_risk_prioritization.R`) in dependency order.
6. For the AD analysis, place the exact final screened external dataset at `data/STRICT_EXTERNAL.csv`, then run `source("03_external_validation_and_applicability_domain.R")`. This script is independent of the model-fitting stage and uses the training predictors to define the AD reference distribution.

## Model tuning and reproducibility

The five R-based models use a three-stage search: coarse exploration, TPE-based optimization, and local refinement, followed by parameter-boundary checks. The current search budgets are defined by model name in `01_model_training.R`. Candidate evaluation uses shared five-repeat, five-fold taxon-blocked cross-validation. Final candidate eligibility also applies the configured per-repeat training-versus-OOF R-squared gap constraint; the exact selection logic is implemented in the script and recorded in the output audit files.

No fixed master seed is hard-coded. On the first run, the training script generates a run-specific seed and saves it to `results/model_tuning_run06/00_RUN_SEED.txt`. Fold assignments, tuning checkpoints, parameter-boundary audits, and run metadata are saved under `results/`. Re-running the same working directory resumes the recorded run. For reproducibility, archive the seed, fold assignments, input files, package versions, and output audit tables together.

## Applicability-domain method

The AD analysis uses 24 original predictors: 13 chemical descriptors, five ecological traits, five taxonomic variables, and one MOA variable. Each variable contributes weight `1/24`.

- Numeric predictors are standardized using the training-set mean and standard deviation; their contribution is the absolute standardized deviation divided by 24.
- Categorical predictors use training-set category prevalence; the contribution is `(1 - training prevalence) / 24`. A category not seen in training receives a contribution of `1/24` for that variable.
- The training-set 95th percentile (R quantile type 7) defines the AD threshold.
- All numeric reference parameters and categorical frequencies are estimated from the training set only.

The script checks the manuscript reference values by default: 7,043 training records, 437 screened external records, training P95 approximately `1.057963`, and 396/437 external records within the AD. If these checks fail, the script stops rather than silently changing the input data or method. Do not disable the checks merely to force a run to complete.

The script writes four CSV files to `results/applicability_domain_equal24_manhattan/`: training AD scores, external-record AD scores and classifications, an external AD summary, and the reference parameters used in the calculation.

## Required inputs not included in this archive

- `data/STRICT_EXTERNAL.csv`: the final screened 437-record external dataset used in the manuscript. `External_validation_raw.xlsx` is provided as raw source material, but the final screened CSV is intentionally not reconstructed automatically because the original screening and grouping rules must be preserved.
- `data/Taihu_9Sites_10OP_Original_LOD2_Simple.xlsx`: required by `08_protection_thresholds_and_risk_prioritization.R`; it was not present among the supplied files when this archive was assembled.
- `data/Pollutants_13_descriptors_by_category.csv`: required only for the separate ChemExpo chemical-space workflow; that figure-specific workflow is not included in this analysis package.

Supply these files from the exact study records before running the corresponding analyses. Do not substitute an unprocessed source file unless the manuscript's original screening rules have been reproduced.

## Scope and validation status

The package includes the updated TPE training script and the analysis-only 24-variable AD script. Static file and path checks were performed when assembling the archive. A complete end-to-end R execution was not performed in this environment; verify execution with the specified R/Python packages and exact study inputs before releasing the repository publicly.
