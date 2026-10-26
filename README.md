# Regional freshwater assemblage sensitivity

R scripts for cross-species toxicity prediction and regional freshwater hazard assessment.

## Analyses

| Script | Purpose |
|---|---|
| `01_model_training.R` | Train and compare six toxicity prediction models. |
| `02_shap_and_moa_analysis.R` | Analyze SHAP contributions across modes of action with bootstrap. |
| `03_external_validation_and_applicability_domain.R` | Validate predictions and assess the 24-variable Manhattan applicability domain. |
| `04_cross_species_toxicity_prediction.R` | Predict toxicity for chemical–taxon combinations. |
| `05_ssd_distribution_selection.R` | Select SSD distributions using AICc. |
| `06_regional_ssd_hc5.R` | Estimate regional HC5 values and Monte Carlo uncertainty. |
| `07_toxicity_gradients_and_regional_variation.R` | Analyze chemical toxicity gradients and regional sensitivity variation. |
| `08_protection_thresholds_and_risk_prioritization.R` | Compare protection thresholds and chemical risk rankings. |

## Data

- `training_data.csv`: curated data for model training and analysis.
- `regional_prediction_taxon_traits.csv`: taxonomy and trait predictors for the 1,317 unique taxa pooled from seven regional assemblages; used in cross-species toxicity prediction (script 04).
- `regional_taxa.csv`: region-to-taxon membership used to construct regional SSDs (scripts 05–06).

## Run

Run scripts 01–08 in order from the repository root. Add `external_validation.csv` for script 03 and `taihu_monitoring.xlsx` for the monitoring analysis in script 08. Results are written to `results/`.
