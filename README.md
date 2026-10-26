# Regional freshwater assemblage sensitivity

R scripts for cross-species toxicity prediction and region-specific freshwater hazard assessment.

## Analyses

| Script | Analysis |
|---|---|
| `01_model_training.R` | Compare six toxicity prediction models. |
| `02_shap_and_moa_analysis.R` | Assess SHAP contributions across modes of action and bootstrap uncertainty. |
| `03_external_validation_and_applicability_domain.R` | Perform independent external validation and equal-weight, 24-variable Manhattan applicability domain assessment. |
| `04_cross_species_toxicity_prediction.R` | Predict toxicity across chemical–taxon combinations. |
| `05_ssd_distribution_selection.R` | Compare five SSD distributions using AICc. |
| `06_regional_ssd_hc5.R` | Estimate HC5 across seven regions with Monte Carlo uncertainty. |
| `07_toxicity_gradients_and_regional_variation.R` | Examine toxicity gradients, modes of action, and regional HC5 variation. |
| `08_protection_thresholds_and_risk_prioritization.R` | Compare protection thresholds and assess chemical risk priorities. |

## Data

- `training_data.csv`: curated chemical–taxon toxicity data and model predictors.
- `regional_prediction_taxon_traits.csv`: taxonomy and ecological traits for 1,317 taxa in the seven regional assemblages, used for regional toxicity prediction.
- `regional_taxa.csv`: taxon membership of the seven study regions, used for regional SSDs.
- `external_validation.csv`: independent external toxicity data for validation and applicability domain assessment.
- `Moa_reference_database.xlsx`: chemical mode-of-action reference database.
- `taihu_monitoring.xlsx`: Taihu pesticide monitoring data for the risk assessment.

## Run

Run the scripts in numerical order from the repository root. Outputs are saved in `results/`.
