# Regional Freshwater Sensitivity to Contaminants

This repository contains the reproducible training workflow and supporting data for cross-species freshwater toxicity prediction and regional assemblage sensitivity analysis.

## Six-model training workflow

Run the complete four-stage training workflow from the repository root:

```r
source("01_model_training.R")
```

The entry script runs the five R-native models first and starts FT-Transformer only after all five have produced eligible final models. The two backend scripts are kept separate to isolate the R-native and CUDA/Python environments; `01_model_training.R` is the single entry point for the full six-model workflow.

### Models and predictors

The six model families are:

1. XGBoost
2. Ridge regression
3. Elastic Net
4. Random Forest (`ranger`)
5. LightGBM
6. FT-Transformer

The corrected training dataset is expected at `data/Training_DATABASE_MOA_RECHECKED.csv`. It contains 7,043 chemical–taxon records for 2,109 unique chemicals. The model matrix contains 24 predictors: 13 chemical descriptors, five ecological traits, five taxonomic variables, and one broad mode-of-action (MOA) variable. Specific-MOA fields are retained for post-hoc analyses and audit exports; they are **not** model predictors.

The target is `log10_Geomean_ug_L`. Taxon is used for blocked cross-validation and is not a predictor.

### Four-stage tuning design

Every candidate is evaluated using the complete fixed **five-repeat, five-fold Taxon-blocked cross-validation** (25 train/validation fits per candidate). The same fold assignments are shared across all six model families. Preprocessing is fitted within each training partition.

| Stage | Five R-native models | FT-Transformer |
|---|---|---|
| 1. Coarse exploration | Discrete, space-filling exploration of each model's declared parameter space | 15-candidate coarse grid across embedding dimension and learning rate |
| 2. Global/local search | TPE-style Bayesian search using completed candidate scores | 15-candidate fine grid around the Stage 1 optimum, varying the full declared FT search-parameter set |
| 3. Local refinement or rescue | Local refinement around strong candidates | Up to five targeted candidates that probe selected parameter boundaries |
| 4. Boundary expansion and audit | Adaptive boundary expansion, repeat-level gap checks, and a full audit of every selected parameter's boundary status | 12 fresh, targeted boundary candidates followed by final ranking and repeat-level eligibility checks |

For the five R-native models, the planned Stage 1–3 candidate counts are XGBoost 90, Ridge 60, Elastic Net 70, Random Forest 80, and LightGBM 90 (390 candidates in total before Stage 4). Stage 4 evaluates additional candidates in batches of 12; it may stop when the declared stopping criteria are met, or be resumed after the per-invocation search budget is reached.

For FT-Transformer, the planned maximum is 47 candidates: 15 in Stage 1, 15 in Stage 2, up to five in Stage 3, and 12 in Stage 4. The number of Stage 3 rescue candidates can be lower when no new boundary candidate is needed.

### Tuned hyperparameters

All hyperparameters explicitly defined in the search spaces are optimized. The parameter domains and every tested candidate are exported to the output folders.

| Model | Tuned hyperparameters |
|---|---|
| Ridge | Regularization strength (`lambda`) |
| Elastic Net | Elastic-net mixing parameter (`alpha`) and regularization strength (`lambda`) |
| Random Forest | Number of trees, number of candidate predictors per split (`mtry`), minimum node size, sample fraction, maximum depth |
| XGBoost | Learning rate, maximum depth, minimum child weight, row subsampling, column subsampling, L1 and L2 regularization, split penalty (`gamma`), maximum categorical split threshold |
| LightGBM | Learning rate, number of leaves, maximum depth, minimum data per leaf, feature fraction, bagging fraction, L2 regularization, categorical smoothing, minimum split gain |
| FT-Transformer | Input embedding dimension, number of attention heads, number of attention blocks, dropout, learning rate, weight decay, batch size, and training epochs |

For FT-Transformer, the epoch count is derived from the learning rate for the coarse/fine grids and is explicitly perturbed in Stage 4. The optimizer family (AdamW) and feed-forward hidden multiplier are fixed in the current implementation; they are not claimed to have been searched.

### Model-selection rule

A candidate is eligible only when the maximum training–cross-validation \(R^2\) gap is no greater than 0.20 in **each of the five repeats**. Among eligible candidates, the model-specific selection routine prioritizes mean pooled out-of-fold (OOF) cross-validation \(R^2\), with RMSE and the boundary audit used as tie-breakers or eligibility checks. A model is not marked final-eligible if the gap constraint or required boundary checks remain unresolved.

The repeated Taxon-blocked CV results are internal cross-validation, not independent external validation.

## Requirements

### R environment

Use a recent R installation with these packages available:

- `data.table`
- `glmnet`
- `ranger`
- `xgboost` (version 3.0.0 or later; native categorical support is required)
- `lightgbm`
- `reticulate`

Package installation for `lightgbm` and XGBoost may require platform-specific setup.

### FT-Transformer Python environment

FT-Transformer requires a Python environment containing:

- `torch` with CUDA available
- `pytorch_tabular`
- `numpy`
- `pandas`

Set `ECOTOX_FT_PYTHON` to the Python executable in that environment before running the workflow. For example, in R:

```r
Sys.setenv(ECOTOX_FT_PYTHON = "C:/path/to/your/cuda-env/python.exe")
source("01_model_training.R")
```

The exact path is machine-specific. The FT-Transformer script stops if the configured Python executable is missing or CUDA is unavailable; the five R-native models can be run only after installing their R dependencies.

### Data-path override

The default input is `data/Training_DATABASE_MOA_RECHECKED.csv`. If the corrected dataset is stored elsewhere, set `ECOTOX_DATA_FILE` to its absolute path:

```r
Sys.setenv(ECOTOX_DATA_FILE = "D:/project/data/Training_DATABASE_MOA_RECHECKED.csv")
source("01_model_training.R")
```

The script checks for the audited MOA columns and verifies key dataset counts and selected MOA corrections before fitting. Do not substitute an older or differently processed training table without updating and reviewing those checks.

## Outputs and resumability

The workflow writes new outputs to versioned directories so that older model checkpoints and scores are not silently reused:

- `01_SHARED_FOLDS_MOA_RECHECKED_v13_FULL4STAGE/` — common fold assignments
- `02_5MODELS_MOA_RECHECKED_v13_FULL4STAGE/` — candidate definitions and scores, per-repeat metrics, OOF predictions, selected parameters, final fitted models, and boundary audits for the five R-native models
- `03_FT_TRANSFORMER_MOA_RECHECKED_v13_FULL4STAGE/` — FT-Transformer candidates, checkpoints, metrics, final model, and parameter summaries

The five-model backend has an 18-hour soft time budget per invocation. If it stops at the time budget or any model remains unresolved, rerun:

```r
source("01_model_training.R")
```

Completed, validated candidate-repeat checkpoints are resumed. The workflow will not start FT-Transformer until the five-model completion audit reports `ALL_5_MODELS_FINAL_ELIGIBLE`.

Review the final completion-status files, per-repeat gap metrics, selected-parameter tables, and boundary audits before reporting results. A completed run is not, by itself, evidence of independent external validation.

## Supporting datasets

1. `freshwater_macroinvertebrate_external_validation_toxicity_dataset.csv`: Used for independent external validation.
2. `taihu_surface_water_organophosphate_concentrations_9_sites.csv`: Used for the Taihu pesticide risk quotient (RQ) analysis and contaminant prioritization.

