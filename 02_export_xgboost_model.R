# Export the tuned XGBoost model and OOF predictions for downstream analyses.
suppressPackageStartupMessages(library(data.table))
ROOT <- normalizePath(".", winslash="/", mustWork=TRUE)
RUN_ROOT <- file.path(ROOT,"results","model_training")
MODEL_DIR <- file.path(RUN_ROOT,"classical_models","02_MODELS","XGBoost")
FINAL_DIR <- file.path(MODEL_DIR,"FINAL_MODEL")
DONE <- file.path(MODEL_DIR,"DONE.txt")
MODEL_FILE <- file.path(FINAL_DIR,"MODEL.json")
BUNDLE_FILE <- file.path(FINAL_DIR,"MODEL_BUNDLE.rds")
OOF_FILE <- file.path(MODEL_DIR,"10_BEST_OOF.csv")
if (!all(file.exists(c(DONE,MODEL_FILE,BUNDLE_FILE,OOF_FILE))))
  stop("XGBoost tuning is incomplete. Run 01_train_six_models.R until the XGBoost DONE.txt file exists.")

bundle <- readRDS(BUNDLE_FILE)
if (!identical(as.character(bundle$model),"XGBoost")) stop("The selected artifact is not XGBoost.")
out <- file.path(ROOT,"results","xgboost_model")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if (!file.copy(MODEL_FILE,file.path(out,"xgboost_model.json"),overwrite=TRUE))
  stop("Could not export the XGBoost model.")
prep <- list(
  encoder=list(median=bundle$encoder$med, levels=bundle$encoder$lev),
  numeric=bundle$numeric,
  categorical=bundle$categorical,
  target="log10_Geomean_ug_L"
)
saveRDS(prep,file.path(out,"xgboost_preprocessing.rds"))
oof <- fread(OOF_FILE)
stopifnot(all(c("Repeat","Taxon","Observed","Predicted") %in% names(oof)))
fwrite(oof[,.(Repeat,Taxon,Observed,Predicted)],file.path(out,"xgboost_oof.csv"))
comparison <- file.path(RUN_ROOT,"classical_models","MASTER_MODEL_COMPARISON.csv")
if (file.exists(comparison)) file.copy(comparison,file.path(out,"model_comparison.csv"),overwrite=TRUE)
cat("Exported tuned XGBoost model, preprocessing metadata and OOF predictions to ",out,"\n",sep="")
