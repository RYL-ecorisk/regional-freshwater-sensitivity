# Cross-species toxicity prediction for taxa from seven regional assemblages.
# Taxon traits: pooled regional prediction taxa; regional membership is in regional_taxa.csv.

TRAIN <- "data/training_data.csv"
TAXA <- "data/regional_prediction_taxon_traits.csv"
MODEL_DIR <- "results/01_model_training"
OUT <- "results/04_cross_species_toxicity_prediction"
CHEMICALS_PER_BATCH <- 50L

library(data.table)
if (packageVersion("xgboost") < "3.0.0") stop("Install xgboost >= 3.0")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

prep <- readRDS(file.path(MODEL_DIR, "xgboost_preprocessing.rds"))
model <- xgboost::xgb.load(file.path(MODEL_DIR, "xgboost_model.json"))
NUM <- prep$numeric
CAT <- prep$categorical
enc <- prep$encoder
CHEM <- c("logS", "logKow", "logD7_4", "MW", "TPSA", "log_Henry",
          "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
          "AromaticRingCount", "RotatableBondCount", "BertzCT")
BIO <- setdiff(c(NUM, CAT), c(CHEM, "Model_MOA_final"))

training <- fread(TRAIN, select = c("DTXSID", CHEM, "Model_MOA_final"))
chemical <- unique(training)
if (anyDuplicated(chemical, by = "DTXSID")) stop("Inconsistent chemical descriptors")
chemical[, DTXSID := as.character(DTXSID)]
rm(training)

taxon <- unique(fread(TAXA, select = c("Taxon", BIO)))
if (anyDuplicated(taxon, by = "Taxon")) stop("Duplicated taxa")
taxon[, Taxon := as.character(Taxon)]

clean_cat <- function(x) {
  z <- trimws(as.character(x))
  z[is.na(z) | z == ""] <- "__MISSING__"
  z
}
make_xgb <- function(d) {
  z <- setNames(lapply(NUM, function(v) {
    a <- suppressWarnings(as.numeric(d[[v]]))
    a[!is.finite(a)] <- enc$median[[v]]
    a
  }), NUM)
  for (v in CAT) {
    a <- clean_cat(d[[v]])
    a[!a %in% enc$levels[[v]]] <- "__NOVEL__"
    z[[v]] <- factor(replace(a, a == "__NOVEL__", NA_character_),
                     levels = enc$levels[[v]])
  }
  as.data.frame(z, check.names = FALSE, optional = TRUE)
}

result_file <- file.path(OUT, "predicted_toxicity.csv")
if (file.exists(result_file)) file.remove(result_file)
starts <- seq.int(1L, nrow(chemical), by = CHEMICALS_PER_BATCH)
for (i in seq_along(starts)) {
  ids <- chemical$DTXSID[starts[i]:min(nrow(chemical), starts[i] + CHEMICALS_PER_BATCH - 1L)]
  pairs <- CJ(DTXSID = ids, Taxon = taxon$Taxon)
  pairs <- merge(pairs, chemical, by = "DTXSID", sort = FALSE)
  pairs <- merge(pairs, taxon, by = "Taxon", sort = FALSE)
  pred <- as.numeric(predict(model, xgboost::xgb.DMatrix(make_xgb(pairs))))
  stopifnot(length(pred) == nrow(pairs), all(is.finite(pred)))
  fwrite(data.table(DTXSID = pairs$DTXSID, Taxon = pairs$Taxon,
                    Pred_log10_ug_L = pred), result_file,
         append = i > 1L, col.names = i == 1L)
  cat(sprintf("Predicted %d/%d chemicals\n", min(nrow(chemical), starts[i] + CHEMICALS_PER_BATCH - 1L), nrow(chemical)))
}
cat("Saved:", result_file, "\n")
