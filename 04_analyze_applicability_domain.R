# JHM | Equal-weight 24-variable Manhattan-style applicability-domain analysis
# Analysis only: no figures, plotting dependencies, or model fitting.
# Run from the package root after placing the final screened external dataset at:
#   data/strict_external_validation.csv

options(stringsAsFactors = FALSE, warn = 1, scipen = 9)
suppressPackageStartupMessages(library(data.table))

ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)
DATA_DIR <- file.path(ROOT, "data")
TRAIN_FILE <- file.path(DATA_DIR, "training_data.csv")
EXTERNAL_FILE <- file.path(DATA_DIR, "strict_external_validation.csv")
OUT_DIR <- file.path(ROOT, "results", "applicability_domain")

# Manuscript reference checks. Keep enabled for the submitted JHM analysis.
CHECK_MANUSCRIPT_REFERENCE <- TRUE
EXPECTED_TRAIN_N <- 7043L
EXPECTED_EXTERNAL_N <- 437L
EXPECTED_TRAINING_P95 <- 1.057963
EXPECTED_WITHIN_AD_N <- 396L
P95_TOLERANCE <- 0.00006

CHEM <- c(
  "logS", "logKow", "logD7_4", "MW", "TPSA", "log_Henry",
  "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
  "AromaticRingCount", "RotatableBondCount", "BertzCT"
)
ECO <- c(
  "Log Body size", "Feeding_score", "Respiration_score",
  "Locomotion_score", "Body_armor_score"
)
TAX <- c("Phylum", "Class", "Order", "Family", "Genus")
MOA <- "Model_MOA_final"
NUM_AD <- c(CHEM, ECO)
CAT_AD <- c(TAX, MOA)
AD_FEATURES <- c(NUM_AD, CAT_AD)

stopifnot(length(CHEM) == 13L, length(ECO) == 5L,
          length(TAX) == 5L, length(MOA) == 1L,
          length(NUM_AD) == 18L, length(CAT_AD) == 6L,
          length(AD_FEATURES) == 24L)

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop('Install the required R package with install.packages("data.table").')
}
if (!file.exists(TRAIN_FILE)) stop("Training data not found: ", TRAIN_FILE)
if (!file.exists(EXTERNAL_FILE)) {
  stop(
    "The final screened external dataset is required: ", EXTERNAL_FILE,
    "\nProvide the exact 437-record dataset used in the manuscript. Do not substitute the raw Excel file unless the original screening rules have been reproduced."
  )
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

clean_cat <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | !nzchar(x)] <- "__MISSING__"
  x
}

# Read and apply the same training-row filters used by the training workflow.
train <- fread(TRAIN_FILE, na.strings = c("", "NA", "NaN"), encoding = "UTF-8")
if (!all(c("DTXSID", "Taxon", "log10_Geomean_ug_L", AD_FEATURES) %in% names(train))) {
  stop("Training data are missing required columns: ",
       paste(setdiff(c("DTXSID", "Taxon", "log10_Geomean_ug_L", AD_FEATURES), names(train)),
             collapse = ", "))
}
train[, log10_Geomean_ug_L := suppressWarnings(as.numeric(log10_Geomean_ug_L))]
train[, Taxon := trimws(as.character(Taxon))]
train <- train[is.finite(log10_Geomean_ug_L) & !is.na(Taxon) & nzchar(Taxon)]
if (CHECK_MANUSCRIPT_REFERENCE && nrow(train) != EXPECTED_TRAIN_N) {
  stop("Unexpected training sample size after the manuscript filters: ", nrow(train),
       " (expected ", EXPECTED_TRAIN_N, ").")
}

ext <- fread(EXTERNAL_FILE, na.strings = c("", "NA", "NaN"), encoding = "UTF-8")
if (!all(c("DTXSID", "Taxon", AD_FEATURES) %in% names(ext))) {
  stop("External data are missing required columns: ",
       paste(setdiff(c("DTXSID", "Taxon", AD_FEATURES), names(ext)), collapse = ", "))
}
if (CHECK_MANUSCRIPT_REFERENCE && nrow(ext) != EXPECTED_EXTERNAL_N) {
  stop("Expected ", EXPECTED_EXTERNAL_N, " screened external records; found ", nrow(ext), ".")
}
if (!"External_record_ID" %in% names(ext)) ext[, External_record_ID := seq_len(.N)]
ext[, Taxon := trimws(as.character(Taxon))]
ext[, External_row := seq_len(.N)]

# Validate and standardize data types without changing the source values.
for (v in NUM_AD) {
  train[, (v) := suppressWarnings(as.numeric(get(v)))]
  ext[, (v) := suppressWarnings(as.numeric(get(v)))]
  if (any(!is.finite(train[[v]]))) stop("Training data contain non-finite numeric predictor: ", v)
  if (any(!is.finite(ext[[v]]))) stop("External data contain non-finite numeric predictor: ", v)
}
for (v in CAT_AD) {
  train[, (v) := clean_cat(get(v))]
  ext[, (v) := clean_cat(get(v))]
}

# Numeric reference parameters are estimated from the training data only.
train_mean <- vapply(NUM_AD, function(v) mean(train[[v]]), numeric(1))
train_sd <- vapply(NUM_AD, function(v) sd(train[[v]]), numeric(1))
train_sd[!is.finite(train_sd) | train_sd < 1e-12] <- 1

# Empirical category frequencies are estimated from the training data only.
category_probabilities <- lapply(CAT_AD, function(v) {
  p <- prop.table(table(train[[v]]))
  setNames(as.numeric(p), names(p))
})
names(category_probabilities) <- CAT_AD

# Each of the 24 original variables receives weight 1/24.
# Numeric contribution: abs((x - training mean) / training SD) / 24.
# Categorical contribution: (1 - training prevalence) / 24; unseen levels receive 1/24.
# The total is the manuscript's equal-weight 24-variable Manhattan-style AD score.
score_components <- function(dat) {
  n <- nrow(dat)
  chemical <- numeric(n)
  ecology <- numeric(n)
  taxonomy <- numeric(n)
  moa <- numeric(n)

  for (v in CHEM) {
    chemical <- chemical + abs((dat[[v]] - train_mean[[v]]) / train_sd[[v]]) / 24
  }
  for (v in ECO) {
    ecology <- ecology + abs((dat[[v]] - train_mean[[v]]) / train_sd[[v]]) / 24
  }
  for (v in TAX) {
    p <- unname(category_probabilities[[v]][as.character(dat[[v]])])
    contribution <- (1 - p) / 24
    contribution[is.na(p)] <- 1 / 24
    taxonomy <- taxonomy + contribution
  }
  p_moa <- unname(category_probabilities[[MOA]][as.character(dat[[MOA]])])
  moa_contribution <- (1 - p_moa) / 24
  moa_contribution[is.na(p_moa)] <- 1 / 24
  moa <- moa + moa_contribution

  data.table(
    Chemical_descriptor_contribution = chemical,
    Ecological_trait_contribution = ecology,
    Taxonomy_contribution = taxonomy,
    MOA_contribution = moa,
    Manhattan_AD_score = chemical + ecology + taxonomy + moa
  )
}

train_scores <- score_components(train)
training_p95 <- as.numeric(quantile(train_scores$Manhattan_AD_score,
                                    probs = 0.95, type = 7, names = FALSE))
if (CHECK_MANUSCRIPT_REFERENCE &&
    abs(training_p95 - EXPECTED_TRAINING_P95) > P95_TOLERANCE) {
  stop("Training AD P95 does not match the manuscript reference: ",
       sprintf("%.6f", training_p95), " (expected ",
       sprintf("%.6f", EXPECTED_TRAINING_P95), "). Check the exact training data and preprocessing.")
}

external_scores <- score_components(ext)
within_ad <- external_scores$Manhattan_AD_score <= training_p95
if (CHECK_MANUSCRIPT_REFERENCE && sum(within_ad) != EXPECTED_WITHIN_AD_N) {
  stop("External within-AD count does not match the manuscript reference: ",
       sum(within_ad), "/", nrow(ext), " (expected ",
       EXPECTED_WITHIN_AD_N, "/", EXPECTED_EXTERNAL_N,
       "). Check the exact screened external data and predictor preprocessing.")
}

# Save analysis outputs only; no figures or plotting code are included.
training_output <- data.table(
  Training_row = seq_len(nrow(train)),
  DTXSID = as.character(train$DTXSID),
  Taxon = as.character(train$Taxon)
)
training_output <- cbind(training_output, train_scores)
training_output[, Training_P95 := training_p95]
fwrite(training_output, file.path(OUT_DIR, "training_AD_scores.csv"))

id_cols <- intersect(c("External_row", "External_record_ID", "DTXSID", "Taxon",
                       "Four_way_group_RECHECKED"), names(ext))
external_output <- cbind(copy(ext[, ..id_cols]), external_scores)
external_output[, `:=`(
  Training_P95 = training_p95,
  Within_AD = within_ad,
  AD_status = ifelse(within_ad, "Within AD", "Outside AD")
)]
fwrite(external_output, file.path(OUT_DIR, "external_AD_records.csv"))

summary_output <- data.table(
  Method = "EQUAL_WEIGHT_24_VARIABLE_MANHATTAN_STYLE",
  Training_N = nrow(train),
  External_N = nrow(ext),
  Number_of_variables = length(AD_FEATURES),
  Numeric_variables = length(NUM_AD),
  Categorical_variables = length(CAT_AD),
  Training_P95 = training_p95,
  Within_AD_N = sum(within_ad),
  Outside_AD_N = sum(!within_ad),
  Within_AD_percent = 100 * mean(within_ad),
  Outside_AD_percent = 100 * mean(!within_ad),
  Quantile_method = "R type 7"
)
fwrite(summary_output, file.path(OUT_DIR, "external_AD_summary.csv"))

# Save the training-only reference parameters used to calculate the scores.
num_parameters <- data.table(
  Variable = NUM_AD,
  Variable_type = ifelse(NUM_AD %in% CHEM, "Chemical descriptor", "Ecological trait"),
  Training_mean = as.numeric(train_mean[NUM_AD]),
  Training_SD = as.numeric(train_sd[NUM_AD]),
  Category_level = NA_character_,
  Training_frequency = NA_real_
)
cat_parameters <- rbindlist(lapply(CAT_AD, function(v) {
  p <- category_probabilities[[v]]
  data.table(
    Variable = v,
    Variable_type = if (v %in% TAX) "Taxonomy" else "MOA",
    Training_mean = NA_real_,
    Training_SD = NA_real_,
    Category_level = names(p),
    Training_frequency = as.numeric(p)
  )
}))
reference_parameters <- rbindlist(list(num_parameters, cat_parameters), use.names = TRUE, fill = TRUE)
fwrite(reference_parameters, file.path(OUT_DIR, "AD_reference_parameters.csv"))

cat("Applicability-domain analysis completed.\n")
cat("Output directory: ", OUT_DIR, "\n", sep = "")
cat(sprintf("Training N = %d; P95 = %.6f; external N = %d; within AD = %d (%.2f%%); outside AD = %d (%.2f%%).\n",
            nrow(train), training_p95, nrow(ext), sum(within_ad),
            100 * mean(within_ad), sum(!within_ad), 100 * mean(!within_ad)))
