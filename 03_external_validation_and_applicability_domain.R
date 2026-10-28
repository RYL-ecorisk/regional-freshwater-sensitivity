# External validation and applicability domain assessment (Fig. 4a-b)
# Run after 01_model_training.R; AD uses 10-NN mixed Gower distance.

TRAIN <- "data/Training_DATABASE_MOA_RECHECKED.csv"
EXTERNAL <- "data/STRICT_EXTERNAL.csv"
MODEL_DIR <- "results/01_model_training"
OUT <- "results/03_external_validation_and_applicability_domain"
KNN <- 10L

for (p in c("data.table", "xgboost", "ggplot2")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
}
if (packageVersion("xgboost") < "3.0.0") stop("Requires xgboost >= 3.0")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
library(data.table)
library(ggplot2)

d <- unique(fread(TRAIN, na.strings = c("", "NA", "NaN")))
e <- fread(EXTERNAL, na.strings = c("", "NA", "NaN"))
enc_info <- readRDS(file.path(MODEL_DIR, "xgboost_preprocessing.rds"))
enc <- enc_info$encoder
NUM <- enc_info$numeric
CAT <- enc_info$categorical
TARGET <- enc_info$target
model <- xgboost::xgb.load(file.path(MODEL_DIR, "xgboost_model.json"))
d <- d[is.finite(as.numeric(get(TARGET))) & !is.na(Taxon) & nzchar(trimws(Taxon))]
required <- c("DTXSID", "Taxon", TARGET, NUM, CAT)
if (!all(required %in% names(d)) || !all(required %in% names(e)))
  stop("Missing training or external prediction columns")
if (!all(c("Four_way_group_RECHECKED", "External_record_ID") %in% names(e)))
  stop("Missing external validation group or record ID")

# 1. Independent external predictions
clean_cat <- function(x) {
  x <- trimws(as.character(x))
  x[is.na(x) | x == ""] <- "__MISSING__"
  x
}
make_xgb <- function(dat) {
  z <- setNames(lapply(NUM, function(v) {
    a <- suppressWarnings(as.numeric(dat[[v]]))
    a[!is.finite(a)] <- enc$median[[v]]
    a
  }), NUM)
  for (v in CAT) {
    a <- clean_cat(dat[[v]])
    a[!a %in% enc$levels[[v]]] <- "__NOVEL__"
    z[[v]] <- factor(replace(a, a == "__NOVEL__", NA_character_),
                     levels = enc$levels[[v]])
  }
  as.data.frame(z, check.names = FALSE, optional = TRUE)
}
e[, `:=`(Observed = as.numeric(get(TARGET)),
         Predicted = as.numeric(predict(model, xgboost::xgb.DMatrix(make_xgb(e)))))]
if (any(!is.finite(e$Observed)) || any(!is.finite(e$Predicted)))
  stop("Non-finite external observations or predictions")

GROUPS <- c("Both_seen_new_pair", "Seen_chemical_Unseen_species",
            "Unseen_chemical_Seen_species", "Both_unseen")
LABELS <- c("Both seen: new pair", "Unseen species", "Unseen chemical", "Both unseen")
if (anyNA(e$Four_way_group_RECHECKED) ||
    !all(e$Four_way_group_RECHECKED %in% GROUPS))
  stop("Unrecognized external validation group")
metrics <- rbindlist(lapply(seq_along(GROUPS), function(i) {
  z <- e[Four_way_group_RECHECKED == GROUPS[i]]
  if (nrow(z) < 3L) stop("Too few external observations in: ", GROUPS[i])
  sst <- sum((z$Observed - mean(z$Observed))^2)
  data.table(Group = GROUPS[i], Label = LABELS[i], N = nrow(z),
             R2 = if (sst > 0) 1 - sum((z$Predicted - z$Observed)^2) / sst else NA_real_,
             Pearson_r = cor(z$Observed, z$Predicted, method = "pearson"),
             Spearman_rho = cor(z$Observed, z$Predicted, method = "spearman"))
}))
fwrite(metrics, file.path(OUT, "external_validation_metrics.csv"))
fwrite(e[, .(External_record_ID, DTXSID, Taxon, Group = Four_way_group_RECHECKED, Observed, Predicted)],
       file.path(OUT, "external_validation_predictions.csv"))

# 2. Mixed Gower distance: 10 nearest training neighbors
AD_NUM <- c("Log Body size", "logS", "logKow", "logD7_4", "MW", "TPSA",
            "log_Henry", "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
            "AromaticRingCount", "RotatableBondCount", "BertzCT")
AD_CAT <- c("Feeding", "Respiration", "Locomotion", "Body armor",
            "Phylum", "Class", "Order", "Family", "Genus", "Model_MOA_final")
AD_FEATURES <- c(AD_NUM, AD_CAT)
if (!all(AD_FEATURES %in% names(d)) || !all(AD_FEATURES %in% names(e)))
  stop("Missing raw applicability-domain columns: ",
       paste(setdiff(AD_FEATURES, intersect(names(d), names(e))), collapse = ", "))

make_ad <- function(dat, minimum, span) {
  num <- sapply(AD_NUM, function(v) suppressWarnings(as.numeric(dat[[v]])))
  if (is.vector(num)) num <- matrix(num, ncol = length(AD_NUM))
  num <- sweep(sweep(num, 2L, minimum, "-"), 2L, span, "/")
  cat <- as.matrix(dat[, ..AD_CAT])
  storage.mode(cat) <- "character"
  if (any(!is.finite(num)) || anyNA(cat) || any(trimws(cat) == ""))
    stop("Missing or invalid applicability-domain variables")
  list(num = num, cat = cat)
}
minimum <- vapply(d[, ..AD_NUM], function(x) min(as.numeric(x), na.rm = TRUE), numeric(1))
maximum <- vapply(d[, ..AD_NUM], function(x) max(as.numeric(x), na.rm = TRUE), numeric(1))
span <- pmax(maximum - minimum, 1e-12)
a <- make_ad(d, minimum, span)
b <- make_ad(e, minimum, span)

gower_knn <- function(q, ref, k = 10L, leave_one_out = FALSE, batch = 48L) {
  n <- nrow(q$num); m <- nrow(ref$num)
  if (m <= k) stop("Not enough training records for nearest-neighbor AD")
  result <- numeric(n)
  for (first in seq.int(1L, n, by = batch)) {
    ix <- first:min(n, first + batch - 1L)
    dst <- matrix(0, length(ix), m)
    for (j in seq_len(ncol(q$num)))
      dst <- dst + pmin(abs(outer(q$num[ix, j], ref$num[, j], "-")), 1)
    for (j in seq_len(ncol(q$cat)))
      dst <- dst + outer(q$cat[ix, j], ref$cat[, j], "!=")
    dst <- dst / (ncol(q$num) + ncol(q$cat))
    if (leave_one_out) dst[cbind(seq_along(ix), ix)] <- Inf
    result[ix] <- apply(dst, 1L, function(z) mean(sort(z, partial = k)[seq_len(k)]))
  }
  result
}

train_dist <- gower_knn(a, a, KNN, leave_one_out = TRUE)
ext_dist <- gower_knn(b, a, KNN)
threshold <- as.numeric(quantile(train_dist, 0.95))
e[, `:=`(Gower_10NN = ext_dist,
         AD_status = ifelse(ext_dist <= threshold, "Within AD", "Outside AD"))]
ad_summary <- data.table(Training_N = nrow(d), External_N = nrow(e),
                         K = KNN, Training_P95 = threshold,
                         Within_AD_N = sum(e$AD_status == "Within AD"),
                         Within_AD_percent = 100 * mean(e$AD_status == "Within AD"))
fwrite(e[, .(External_record_ID, DTXSID, Taxon, Gower_10NN, AD_status)],
       file.path(OUT, "external_AD_records.csv"))
fwrite(ad_summary, file.path(OUT, "external_AD_summary.csv"))

# 3. Figure 4a-b
long <- melt(metrics, id.vars = c("Group", "Label", "N"),
             measure.vars = c("R2", "Pearson_r", "Spearman_rho"),
             variable.name = "Metric", value.name = "Value")
long[, Label := factor(Label, levels = rev(LABELS))]
fig_a <- ggplot(long, aes(Label, Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(width = 0.75), vjust = -0.25, size = 3) +
  coord_flip() + labs(x = NULL, y = "Metric value") + theme_classic(12)
ggsave(file.path(OUT, "Fig4a_external_validation.png"), fig_a,
       width = 8, height = 5, dpi = 300)

bins <- sort(unique(c(seq(0, max(ext_dist) + 0.016, by = 0.008), threshold)))
counts <- hist(ext_dist, breaks = bins, plot = FALSE)
h <- data.table(xmin = head(counts$breaks, -1L), xmax = tail(counts$breaks, -1L),
                Percent = 100 * counts$counts / nrow(e))
h[, AD := ifelse(xmax <= threshold, "Within AD", "Outside AD")]
cdf <- data.table(Distance = sort(ext_dist), Cumulative = 100 * seq_along(ext_dist) / length(ext_dist))
scale_factor <- max(h$Percent) / 100
fig_b <- ggplot() +
  geom_rect(data = h, aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = Percent, fill = AD)) +
  geom_step(data = cdf, aes(Distance, Cumulative * scale_factor), color = "#2F96D2") +
  geom_vline(xintercept = threshold, linetype = "dashed") +
  scale_fill_manual(values = c("Within AD" = "#37B8AE", "Outside AD" = "#E27D9F")) +
  scale_y_continuous("Frequency (%)", sec.axis = sec_axis(~ . / scale_factor, name = "Cumulative (%)")) +
  labs(x = "Gower distance", fill = NULL,
       subtitle = sprintf("Training P95 = %.3f  |  External within AD = %.1f%%",
                          threshold, ad_summary$Within_AD_percent)) + theme_classic(12)
ggsave(file.path(OUT, "Fig4b_applicability_domain.png"), fig_b,
       width = 8, height = 5, dpi = 300)

print(metrics)
print(ad_summary)
cat("External validation and applicability domain results: ", OUT, "\n", sep = "")
