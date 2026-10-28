# =============================================================================
# 01_model_training.R
# Full four-stage hyperparameter tuning for all six ECOTOX prediction models.
#
# Models:
#   1. XGBoost
#   2. Ridge
#   3. Elastic Net
#   4. Random Forest
#   5. LightGBM
#   6. FT-Transformer
#
# All candidates use the same 5 repeats x 5 folds Taxon-blocked CV.
# Stage 4 is model-specific boundary expansion/rescue and a full boundary audit.
# Specific MOA is retained as post-hoc metadata; Broad MOA is a predictor.
# =============================================================================

run_ecotox_training <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  script_file <- if (length(file_arg)) {
    sub("^--file=", "", file_arg[[1L]])
  } else {
    ofiles <- vapply(
      sys.frames(),
      function(frame) {
        if (is.null(frame$ofile)) NA_character_ else as.character(frame$ofile)
      },
      character(1)
    )
    ofiles <- ofiles[!is.na(ofiles) & nzchar(ofiles)]
    if (length(ofiles)) tail(ofiles, 1L) else "01_model_training.R"
  }

  script_file <- normalizePath(script_file, winslash = "/", mustWork = TRUE)
  project_root <- dirname(script_file)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(project_root)

  Sys.setenv(ECOTOX_PROJECT_DIR = project_root)

  # An explicit absolute ECOTOX_DATA_FILE takes precedence. Otherwise use the
  # corrected MOA-audited data file expected in data/.
  data_override <- Sys.getenv("ECOTOX_DATA_FILE", unset = "")
  data_candidates <- unique(c(
    if (nzchar(data_override)) data_override,
    file.path(project_root, "data", "Training_DATABASE_MOA_RECHECKED.csv"),
    file.path(project_root, "Training_DATABASE_MOA_RECHECKED.csv"),
    file.path(project_root, "data", "Training_DATABASE_MOA_RECHECKED_WITH_BARRON_SPECIFIC_20261014.csv"),
    file.path(project_root, "Training_DATABASE_MOA_RECHECKED_WITH_BARRON_SPECIFIC_20261014.csv")
  ))
  data_candidates <- data_candidates[file.exists(data_candidates) & !dir.exists(data_candidates)]

  if (!length(data_candidates)) {
    stop(
      "Corrected training data not found. Expected data/Training_DATABASE_MOA_RECHECKED.csv ",
      "or set ECOTOX_DATA_FILE to the corrected MOA-audited CSV."
    )
  }
  data_file <- normalizePath(data_candidates[[1L]], winslash = "/", mustWork = TRUE)
  Sys.setenv(ECOTOX_DATA_FILE = data_file)

  five_model_script <- file.path(project_root, "scripts", "01a_five_models_full_tuning.R")
  ft_script <- file.path(project_root, "scripts", "01b_ft_transformer_full_tuning.R")
  if (!file.exists(five_model_script) || !file.exists(ft_script)) {
    stop(
      "The full-tuning backend scripts are missing. Expected:\n",
      five_model_script, "\n", ft_script
    )
  }

  cat("\n============================================================\n")
  cat("ECOTOX FULL SIX-MODEL TRAINING | FOUR-STAGE HYPERPARAMETER TUNING\n")
  cat("Project root: ", project_root, "\n", sep = "")
  cat("Training data: ", data_file, "\n", sep = "")
  cat("Shared CV: 5 repeats x 5 Taxon-blocked folds\n")
  cat("Constraint: maximum Train R2 - pooled OOF CV R2 <= 0.20 in every repeat\n")
  cat("============================================================\n\n")

  # Stages 1-4 for the five R-native models:
  # coarse exploration -> TPE -> local refinement -> boundary expansion/audit.
  # A separate environment protects the orchestrator from rm(list = ls()) in
  # the legacy full-training implementation.
  cat("\n[1/2] Running full four-stage tuning for XGBoost, Ridge, Elastic Net, RF and LightGBM.\n")
  five_env <- new.env(parent = globalenv())
  source(five_model_script, local = five_env, chdir = FALSE)

  five_status_file <- file.path(
    project_root,
    "02_5MODELS_MOA_RECHECKED_v13_FULL4STAGE",
    "FINAL_ALL5_COMPLETION_STATUS.txt"
  )
  if (!file.exists(five_status_file)) {
    stop(
      "Five-model completion status was not produced. Check the five-model output directory:\n",
      dirname(five_status_file)
    )
  }
  five_status <- trimws(readLines(five_status_file, warn = FALSE)[1L])
  if (!identical(five_status, "ALL_5_MODELS_FINAL_ELIGIBLE")) {
    stop(
      "The five-model stage is not yet complete and eligible (status: ",
      five_status,
      "). Do not start FT-Transformer yet. Re-run source('01_model_training.R') to resume ",
      "the saved five-model checkpoints; completed candidate folds will be reused."
    )
  }

  # Stage 1 coarse grid -> Stage 2 fine/local grid -> Stage 3 boundary rescue
  # -> Stage 4 12 targeted boundary candidates for FT-Transformer.
  # The FT backend requires the CUDA-enabled Python environment configured below.
  cat("\n[2/2] All five R-native models are eligible. Running FT-Transformer.\n")
  ft_env <- new.env(parent = globalenv())
  source(ft_script, local = ft_env, chdir = FALSE)

  ft_done_file <- file.path(
    project_root,
    "03_FT_TRANSFORMER_MOA_RECHECKED_v13_FULL4STAGE",
    "02_FT_Transformer",
    "DONE.txt"
  )
  if (!file.exists(ft_done_file)) {
    stop(
      "FT-Transformer did not write DONE.txt. Check its checkpoints and Python/CUDA logs:\n",
      dirname(ft_done_file)
    )
  }

  cat("\nAll six model families have completed the configured four-stage workflow.\n")
  cat("Review each model's final parameter audit, repeat metrics and boundary status before reporting results.\n")
  invisible(TRUE)
}

run_ecotox_training()
