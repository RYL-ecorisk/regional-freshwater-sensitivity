# Reproducibility Package

## Regional variation in freshwater assemblage sensitivity across chemical toxicity gradients and modes of action

### Overview

This package contains the analysis scripts and selected input data used for cross-species toxicity prediction and region-specific chemical hazard assessment in the associated manuscript. The workflow compares six machine-learning algorithms, evaluates predictive performance and applicability domain, interprets predictor contributions using SHAP, predicts toxicity for freshwater macroinvertebrate assemblages in seven regions, derives species sensitivity distributions (SSDs) and hazardous concentrations for 5% of species (HC5), and evaluates the implications for protection thresholds and chemical risk prioritization.

The training dataset contains 7,043 chemical–taxon pairs involving 2,109 chemicals and 690 taxa. The regional prediction matrix covers 2,109 chemicals and 1,317 taxa.

### Package contents

- `data/`: Training data, taxonomic and ecological-trait information, regional taxon membership, and supporting reference files.
- `R-packages.txt`: R package requirements. XGBoost version 3.0.0 or later is required.
- `requirements-ft.txt`: Python dependencies for the FT-Transformer analysis.
- `01_model_training.R`: Hyperparameter tuning and model fitting for XGBoost, ridge regression, elastic net, random forest, and LightGBM.
- `01b_ft_transformer_tuning.R`: FT-Transformer grid search and boundary checks.
- `01c_export_xgboost_compat.R`: Exports the selected XGBoost model, preprocessing metadata, and out-of-fold (OOF) predictions for downstream analyses.
- `01d_combine_model_comparison.R`: Combines the five R-based models and FT-Transformer for model comparison.
- `00_prepare_prediction_matrix.R`: Builds the chemical–taxon prediction matrix for the seven regional assemblages.
- `02_shap_moa.R`: SHAP interpretation across modes of action.
- `03_external_validation_and_applicability_domain.R`: Independent external validation and applicability-domain assessment.
- `04_species_toxicity_prediction.R`: Toxicity prediction for all chemical–taxon combinations in the regional prediction matrix.
- `05_ssd_distribution_selection.R`: Comparison of candidate SSD distributions using AICc.
- `06_regional_ssd_hc5.R`: Regional HC5 estimation and uncertainty propagation.
- `07_toxicity_gradients_and_regional_sensitivity_variation.R`: Analysis of toxicity gradients, modes of action, and regional HC5 variation.
- `08_protection_thresholds_and_risk_prioritization.R`: Comparison of protection thresholds and chemical risk prioritization.
- `09_fig4c_chemical_space.R`: Optional chemical-space analysis for Fig. 4c.

### Software requirements

Install the R packages listed in `R-packages.txt`. The model-training script requires XGBoost 3.0.0 or later. The FT-Transformer analysis additionally requires Python with the packages listed in `requirements-ft.txt`, accessed through the R package `reticulate`.

Run the scripts from the package root directory. File paths are relative to this directory.

To select the Python environment used by `reticulate`, set `RETICULATE_PYTHON` before running the FT-Transformer script. For example, on macOS or Linux:

```bash
export RETICULATE_PYTHON=/path/to/python
Rscript 01b_ft_transformer_tuning.R
```

On Windows, set `RETICULATE_PYTHON` to the full path of the intended `python.exe` before starting the R session.

### Recommended workflow

1. **Tune the five R-based models.** Run:

   ```bash
   Rscript 01_model_training.R
   ```

   The script performs coarse screening, Bayesian optimization using a tree-structured Parzen estimator (TPE), local refinement, and parameter-boundary checks. All five models use the same five repeats of fivefold taxon-blocked cross-validation. Preprocessing is fitted within each training partition. For these five models, the implemented selection rule requires the maximum Train R2–OOF CV R2 gap across repeats to be no greater than 0.20; among eligible candidates, mean CV R2 is the primary selection criterion. The search can be computationally intensive and writes checkpoints to support resumption.

2. **Tune FT-Transformer, if required for the six-model comparison.** After the shared folds and R-model tuning are available, configure `RETICULATE_PYTHON` and run:

   ```bash
   Rscript 01b_ft_transformer_tuning.R
   ```

   After the FT-Transformer run has completed, combine the results:

   ```bash
   Rscript 01d_combine_model_comparison.R
   ```

3. **Export the selected XGBoost model.** Once XGBoost tuning is complete, run:

   ```bash
   Rscript 01c_export_xgboost_compat.R
   ```

4. **Build the regional prediction matrix.** Run:

   ```bash
   Rscript 00_prepare_prediction_matrix.R
   ```

5. **Run downstream analyses in dependency order.** Run `02_shap_moa.R` and, when its required input is available, `03_external_validation_and_applicability_domain.R`. Then run `04_species_toxicity_prediction.R`, followed by `05_ssd_distribution_selection.R`, `06_regional_ssd_hc5.R`, and the relevant analyses in `07_toxicity_gradients_and_regional_sensitivity_variation.R` and `08_protection_thresholds_and_risk_prioritization.R`.

6. **Run the optional Fig. 4c analysis.** `09_fig4c_chemical_space.R` is separate from model training and requires an additional pollutant-category dataset; see *Additional inputs* below.

### Reproducibility and checkpoint management

The first model-training run generates a run-specific random seed and saves it in `results/model_tuning_run06/00_RUN_SEED.txt`. Shared fold assignments, run manifests, software-session information, candidate-level results, parameter records, boundary audits, OOF predictions, and model artifacts are saved under `results/model_tuning_run06/`.

Keep the seed file, fold assignments, run manifest, and checkpoint directories together. If a run stops at the time limit, rerun the same script to resume. Do not combine checkpoints from different runs or reuse an existing checkpoint directory after changing the input data or search configuration. A new run may select different hyperparameters; report the parameters and performance metrics from the run actually used for the manuscript.

### Additional inputs required for selected analyses

The following files are not included in this archive and must be supplied separately for the corresponding analyses:

- `data/STRICT_EXTERNAL.csv`: Required by `03_external_validation_and_applicability_domain.R`. The archive includes `External_validation_raw.xlsx`, but the strict external-validation dataset is not generated automatically by this package. Apply the eligibility and exclusion criteria documented for the study before running the validation script.
- `data/Pollutants_13_descriptors_by_category.csv`: Required by the optional `09_fig4c_chemical_space.R` analysis.
- `data/Taihu_9Sites_10OP_Original_LOD2_Simple.xlsx`: Required for the Taihu monitoring-data risk-ranking section of `08_protection_thresholds_and_risk_prioritization.R`. If this file is absent, the script can still run the threshold comparison, but the monitoring-data analysis is skipped.

### Outputs and reporting

Analysis outputs are written to subdirectories under `results/`. The tuning outputs include candidate rankings, selected parameters, boundary audits, OOF predictions, model artifacts, and software-session information. Use the outputs from the final completed run when preparing the model-performance results and supplementary hyperparameter tables. Because tuning is stochastic, the selected parameters from a new run may differ from previously reported values.

### Citation and data reuse

This package supports the manuscript titled *Regional variation in freshwater assemblage sensitivity across chemical toxicity gradients and modes of action*. Please cite the associated article using its final bibliographic details when available. Consult the original sources and their terms of use before redistributing or reusing third-party datasets.
