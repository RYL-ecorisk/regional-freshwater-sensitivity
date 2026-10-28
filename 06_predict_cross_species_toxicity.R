# Species toxicity prediction using the final XGBoost model.

INPUT <- "results/prediction_matrix.csv"
MODEL_DIR <- "results/xgboost_model"
OUT <- "results/species_toxicity_prediction"
CHUNK <- 100000L

for (p in c("data.table", "xgboost"))
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
library(data.table)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

prep <- readRDS(file.path(MODEL_DIR, "xgboost_preprocessing.rds"))
model <- xgboost::xgb.load(file.path(MODEL_DIR, "xgboost_model.json"))
NUM <- prep$numeric
CAT <- prep$categorical
FEATURES <- c(NUM, CAT)
enc <- prep$encoder

raw <- fread(INPUT, na.strings = c("", "NA", "NaN"), check.names = FALSE)
stopifnot(all(c("DTXSID", "Taxon", FEATURES) %in% names(raw)))
raw[, `:=`(DTXSID = trimws(as.character(DTXSID)),
           Taxon = trimws(as.character(Taxon)))]
if (anyNA(raw[, .(DTXSID, Taxon)]) ||
    anyDuplicated(raw, by = c("DTXSID", "Taxon")))
  stop("Missing or duplicated chemical–taxon identifiers")

clean_cat <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x == ""] <- "__MISSING__"
  x
}
make_xgb <- function(d) {
  x <- setNames(lapply(NUM, function(v) {
    z <- suppressWarnings(as.numeric(d[[v]]))
    z[!is.finite(z)] <- enc$median[[v]]
    z
  }), NUM)
  for (v in CAT) {
    z <- clean_cat(d[[v]])
    z[!z %in% enc$levels[[v]]] <- "__NOVEL__"
    x[[v]] <- factor(replace(z, z == "__NOVEL__", NA_character_),
                     levels = enc$levels[[v]])
  }
  as.data.frame(x, check.names = FALSE, optional = TRUE)
}

result_file <- file.path(OUT, "predicted_toxicity.csv")
if (file.exists(result_file)) file.remove(result_file)
starts <- seq.int(1L, nrow(raw), by = CHUNK)
for (i in seq_along(starts)) {
  ix <- starts[i]:min(nrow(raw), starts[i] + CHUNK - 1L)
  x <- xgboost::xgb.DMatrix(make_xgb(raw[ix]))
  pred <- as.numeric(predict(model, x))
  stopifnot(all(is.finite(pred)))
  z <- data.table(DTXSID = raw$DTXSID[ix], Taxon = raw$Taxon[ix],
                  Pred_log10_ug_L = pred)
  fwrite(z, result_file, append = i > 1L, col.names = i == 1L)
  cat(sprintf("Predicted %d / %d rows\n", max(ix), nrow(raw)))
}

cat("Saved:", result_file, "\n")
