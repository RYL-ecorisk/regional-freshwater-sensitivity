# Input Data

All supplied source and reference data are stored in this directory. Generated models, predictions, checkpoints, and analysis outputs are written to `results/`; input files are not overwritten by the analysis scripts.

| File | Description |
|---|---|
| `training_data.csv` | Curated chemical–taxon toxicity training records, model predictors, and reviewed MOA fields. |
| `region_taxon_membership.csv` | Taxon membership for the seven formal study regions. |
| `taxon_traits_prediction.csv` | Taxonomy and ecological-trait records for the prediction taxa. |
| `external_validation_raw.xlsx` | Raw external-validation source workbook; not the final screened 437-record dataset. |
| `moa_cas_reference.xlsx` | MOA/CAS reference workbook. |
| `supplementary_tables_reference.xlsx` | Reference copy of the study supplementary tables. |

The following two inputs were not present in the supplied source materials and have not been fabricated:

- `strict_external_validation.csv`: the exact final screened 437-record external-validation dataset required by `04_analyze_applicability_domain.R`.
- `taihu_monitoring_data.xlsx`: the nine-site Taihu pesticide-monitoring workbook required for the monitoring-data section of `10_assess_protection_thresholds_and_risk.R`.

Place these files here only when the original final study records are available. Do not substitute raw source data unless the original screening and processing steps have been reproduced.
