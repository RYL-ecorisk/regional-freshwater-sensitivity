# JHM model tuning package

本包将原来的固定参数训练脚本替换为完整调参流程：五个 R 模型采用 coarse search、TPE 和 local refinement；FT-Transformer 单独进行网格搜索。所有模型使用同一套 5 × 5 taxon-blocked folds。

## 运行顺序

1. 在本目录作为 working directory，运行 `source("01_model_training.R")`。训练使用 `data/Training_DATABASE_MOA_RECHECKED.csv`。
2. XGBoost 完成后，如需 FT-Transformer 对比，设置 `RETICULATE_PYTHON` 指向含 `torch`、`pytorch_tabular`、`numpy` 和 `pandas` 的 Python 环境，再运行 `source("01b_ft_transformer_tuning.R")`。
3. 如 FT-Transformer 已完成，运行 `source("01d_combine_model_comparison.R")` 汇总六个模型；不运行 FT 时，五模型比较结果仍保留在调参输出目录。
4. 运行 `source("01c_export_xgboost_compat.R")`，导出后续分析需要的 XGBoost 模型、预处理信息和 OOF 预测。
5. 运行 `source("00_prepare_prediction_matrix.R")`，生成区域分类单元预测矩阵。
6. 按需运行 `02_shap_moa.R`、`03_external_validation_and_applicability_domain.R`、`04_species_toxicity_prediction.R`、`05_ssd_distribution_selection.R`、`06_regional_ssd_hc5.R`、`07_toxicity_gradients_and_regional_sensitivity_variation.R` 和 `08_protection_thresholds_and_risk_prioritization.R`。

## 运行说明

- 不含硬编码的固定随机种子。首次运行时生成一个 run-specific seed，保存到 `results/model_tuning_run06/00_RUN_SEED.txt`，并供五模型、FT 和随机抽样步骤共用；fold assignments 也会保存，便于中断后续跑。
- 五模型调参可能耗时较长。若达到软时间上限，重新运行 `01_model_training.R` 即可续跑；不要删除或混合已有 checkpoint。
- `01_model_training.R` 保留逐 repeat 的 Train R² − OOF CV R² ≤ 0.20 约束。若这是最终采用的筛选规则，Methods 可补一句：“Candidates were retained only when the Train R² minus OOF CV R² was ≤ 0.20 in every repeat; among eligible candidates, the configuration with the highest mean CV R² was selected.”
- `03_external_validation_and_applicability_domain.R` 需要预先整理好的 `data/STRICT_EXTERNAL.csv`。本包包含原始 `External_validation_raw.xlsx`，但不自动重建严格筛选后的验证集，以免改变既有纳入/排除规则。
- `09_fig4c_chemical_space.R` 是可选脚本，需要另行准备 `data/Pollutants_13_descriptors_by_category.csv`；该文件不在本次提供的附件中。
- `data/SI_Tables_reference.xlsx` 保留为参数范围和报告结果的对照材料。重新运行随机调参后，最优参数可能与已报告的 Table S4 不完全相同；应以生成的搜索记录和实际最终模型为准。

## 文件

- `01_model_training.R`：五个模型的 coarse/TPE/local 调参与最终模型拟合。
- `01b_ft_transformer_tuning.R`：FT-Transformer 网格搜索。
- `01c_export_xgboost_compat.R`：为下游脚本导出兼容文件。
- `01d_combine_model_comparison.R`：汇总五个 R 模型与 FT-Transformer 的比较结果。
- `00_prepare_prediction_matrix.R`：生成 2,109 种化学品 × 1,317 个区域分类单元的预测矩阵。
- `02`–`09`：SHAP、外部验证、毒性预测、SSD/HC5、风险评价和 Fig. 4c 脚本。
