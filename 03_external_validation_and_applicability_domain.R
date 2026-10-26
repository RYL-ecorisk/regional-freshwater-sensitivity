# Independent external validation and equal-weight 24-variable Manhattan AD.

TRAIN <- "data/training_data.csv"
EXTERNAL <- "data/external_validation.csv"
MODEL_DIR <- "results/01_model_training"
OUT <- "results/03_external_validation_and_applicability_domain"

library(data.table)
library(xgboost)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

info <- readRDS(file.path(MODEL_DIR, "xgboost_preprocessing.rds"))
model <- xgboost::xgb.load(file.path(MODEL_DIR, "xgboost_model.json"))
NUM <- info$numeric
CAT <- info$categorical
TARGET <- info$target
stopifnot(length(NUM) == 18L, length(CAT) == 6L)

d <- unique(fread(TRAIN, na.strings = c("", "NA", "NaN")))
e <- fread(EXTERNAL, na.strings = c("", "NA", "NaN"))
d <- d[is.finite(as.numeric(get(TARGET))) & !is.na(Taxon) & nzchar(trimws(Taxon))]
needed <- c("DTXSID", "Taxon", TARGET, NUM, CAT)
stopifnot(all(needed %in% names(d)), all(needed %in% names(e)))
stopifnot(all(c("External_record_ID", "Four_way_group_RECHECKED") %in% names(e)))

clean_cat <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x == ""] <- "__MISSING__"
  x
}

make_xgb <- function(dat) {
  x <- setNames(lapply(NUM, function(v) {
    z <- suppressWarnings(as.numeric(dat[[v]]))
    z[!is.finite(z)] <- info$encoder$median[[v]]
    z
  }), NUM)
  for (v in CAT) {
    z <- clean_cat(dat[[v]])
    z[!z %in% info$encoder$levels[[v]]] <- "__NOVEL__"
    x[[v]] <- factor(replace(z, z == "__NOVEL__", NA_character_),
                     levels = info$encoder$levels[[v]])
  }
  as.data.frame(x, check.names = FALSE)
}

# External validation across four novelty groups.
e[, `:=`(Observed = as.numeric(get(TARGET)),
         Predicted = as.numeric(predict(model, xgboost::xgb.DMatrix(make_xgb(e)))))]
stopifnot(all(is.finite(e$Observed)), all(is.finite(e$Predicted)))
groups <- c("Both_seen_new_pair", "Seen_chemical_Unseen_species",
            "Unseen_chemical_Seen_species", "Both_unseen")
stopifnot(!anyNA(e$Four_way_group_RECHECKED),
          all(e$Four_way_group_RECHECKED %in% groups))
metrics <- rbindlist(lapply(groups, function(g) {
  z <- e[Four_way_group_RECHECKED == g]
  stopifnot(nrow(z) >= 3L)
  y <- z$Observed; p <- z$Predicted
  sst <- sum((y - mean(y))^2)
  data.table(Group = g, N = nrow(z),
             R2 = if (sst > 0) 1 - sum((y - p)^2) / sst else NA_real_,
             Pearson_r = cor(y, p),
             Spearman_rho = cor(y, p, method = "spearman"))
}))
fwrite(metrics, file.path(OUT, "external_validation_metrics.csv"))
fwrite(e[, .(External_record_ID, DTXSID, Taxon, Four_way_group_RECHECKED,
             Observed, Predicted)],
       file.path(OUT, "external_validation_predictions.csv"))

# Manhattan distance to the training centroid, equally weighted over 24 predictors.
for (v in NUM) {
  d[, (v) := suppressWarnings(as.numeric(get(v)))]
  e[, (v) := suppressWarnings(as.numeric(get(v)))]
  stopifnot(all(is.finite(d[[v]])), all(is.finite(e[[v]])))
}
for (v in CAT) {
  d[, (v) := clean_cat(get(v))]
  e[, (v) := clean_cat(get(v))]
}
mu <- vapply(NUM, function(v) mean(d[[v]]), numeric(1))
sdv <- vapply(NUM, function(v) sd(d[[v]]), numeric(1))
sdv[!is.finite(sdv) | sdv < 1e-12] <- 1
freq <- setNames(lapply(CAT, function(v) {
  tab <- prop.table(table(d[[v]]))
  setNames(as.numeric(tab), names(tab))
}), CAT)

manhattan_ad <- function(dat) {
  z <- data.table(Chemical = numeric(nrow(dat)),
                  Ecological_traits = numeric(nrow(dat)),
                  Taxonomy = numeric(nrow(dat)), MOA = numeric(nrow(dat)))
  unseen <- integer(nrow(dat))
  for (v in NUM) {
    part <- abs((dat[[v]] - mu[[v]]) / sdv[[v]]) / 24
    block <- if (v %in% NUM[1:13]) "Chemical" else "Ecological_traits"
    z[[block]] <- z[[block]] + part
  }
  for (v in CAT) {
    p <- unname(freq[[v]][as.character(dat[[v]])])
    missing_level <- is.na(p)
    unseen <- unseen + as.integer(missing_level)
    part <- (1 - p) / 24
    part[missing_level] <- 1 / 24
    block <- if (v == "Model_MOA_final") "MOA" else "Taxonomy"
    z[[block]] <- z[[block]] + part
  }
  z[, `:=`(Manhattan_AD_score = Chemical + Ecological_traits + Taxonomy + MOA,
            Unknown_level_count = unseen)]
  z
}

train_score <- manhattan_ad(d)$Manhattan_AD_score
threshold <- as.numeric(quantile(train_score, 0.95, type = 7))
rec <- cbind(e[, .(External_record_ID, DTXSID, Taxon)], manhattan_ad(e))
rec[, `:=`(Training_P95 = threshold,
           AD_status = ifelse(Manhattan_AD_score <= threshold,
                              "Within AD", "Outside AD"))]
summary <- data.table(Training_N = nrow(d), External_N = nrow(e),
                      Training_P95 = threshold,
                      Within_AD_N = sum(rec$AD_status == "Within AD"),
                      Outside_AD_N = sum(rec$AD_status == "Outside AD"),
                      Within_AD_percent = 100 * mean(rec$AD_status == "Within AD"))
fwrite(rec, file.path(OUT, "external_AD_records.csv"))
fwrite(summary, file.path(OUT, "external_AD_summary.csv"))
print(metrics)
print(summary)
