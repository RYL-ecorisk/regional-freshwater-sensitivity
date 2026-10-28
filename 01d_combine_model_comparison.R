# Combine the five R-model results with the FT-Transformer result for Fig. 2.
suppressPackageStartupMessages(library(data.table))
ROOT <- normalizePath(".", winslash="/", mustWork=TRUE)
RUN_ROOT <- file.path(ROOT,"results","model_tuning_run06")
R_FILE <- file.path(RUN_ROOT,"five_model_tuning","MASTER_MODEL_COMPARISON.csv")
FT_FILE <- file.path(RUN_ROOT,"ft_transformer_tuning","02_FT_Transformer","13_MODEL_RESULT.csv")
OUT <- file.path(RUN_ROOT,"MODEL_COMPARISON_ALL6.csv")
if (!file.exists(R_FILE)) stop("Missing five-model results: ",R_FILE)
if (!file.exists(FT_FILE)) stop("Missing FT-Transformer results: ",FT_FILE)
r <- fread(R_FILE)
ft <- fread(FT_FILE)
if (!"Model" %in% names(r)) r[,Model := NA_character_]
if (!"Model" %in% names(ft)) ft[,Model := "FT_Transformer"]
ft[,Model := "FT_Transformer"]
metrics <- c("Mean_CV_R2","Mean_CV_RMSE","Mean_CV_MAE","Mean_CV_nRMSE",
             "SD_CV_R2","Mean_Train_R2","Mean_R2_Gap")
get_col <- function(d, target) {
  candidates <- switch(target,
    Mean_CV_R2=c("Mean_CV_R2","Final_Mean_CV_R2","CV_R2"),
    Mean_CV_RMSE=c("Mean_CV_RMSE","Final_Mean_CV_RMSE","CV_RMSE"),
    Mean_CV_MAE=c("Mean_CV_MAE","Final_Mean_CV_MAE","CV_MAE"),
    Mean_CV_nRMSE=c("Mean_CV_nRMSE","Final_Mean_CV_nRMSE","CV_nRMSE"),
    SD_CV_R2=c("SD_CV_R2","Final_SD_CV_R2"),
    Mean_Train_R2=c("Mean_Train_R2","Final_Mean_Train_R2"),
    Mean_R2_Gap=c("Mean_R2_Gap","Final_Mean_R2_Gap"))
  hit <- intersect(candidates,names(d))
  if(length(hit)) as.numeric(d[[hit[1L]]]) else rep(NA_real_,nrow(d))
}
extract <- function(d) {
  out <- data.table(Model=as.character(d$Model))
  for (m in metrics) out[,(m):=get_col(d,m)]
  out
}
all6 <- rbindlist(list(extract(r),extract(ft)),fill=TRUE)
all6 <- unique(all6,by="Model")
setorder(all6,-Mean_CV_R2)
fwrite(all6,OUT)
print(all6)
cat("Saved: ",OUT,"\n",sep="")
