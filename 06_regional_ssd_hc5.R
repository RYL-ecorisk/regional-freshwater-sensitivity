# Regional SSD HC5: fixed log-Gumbel and 10,000 paired Monte Carlo simulations.

PRED <- "results/04_species_toxicity_prediction/predicted_toxicity.csv"
SELECTION <- "results/05_ssd_distribution_selection/ssd_best_fit.csv"
ROSTER <- "data/SEVEN_REGION_TAXON_MEMBERSHIP_20261012.csv"
TRAIN <- "data/Training_DATABASE_MOA_RECHECKED.csv"
OOF <- "results/01_model_training/xgboost_oof.csv"
OUT <- "results/06_regional_ssd_hc5"
N_MC <- 10000L
MC_BATCH <- 500L
WORKERS <- 4L
SEED_FILE <- file.path("results","model_tuning_run06","00_RUN_SEED.txt")
if (!file.exists(SEED_FILE)) stop("Run 01_model_training.R first to create the run seed.")
SEED <- as.integer(readLines(SEED_FILE, n=1L))
REGIONS <- c("Taihu Basin", "Liao River", "Songhua River",
             "Pearl River (Guangzhou segment)", "England",
             "Elbe River", "Danube River")

if (!requireNamespace("data.table", quietly = TRUE)) stop("Install data.table")
library(data.table)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "checkpoints"), showWarnings = FALSE)

selected <- fread(SELECTION)
frequency <- selected[, .N, by = Distribution][order(-N)]
stopifnot(nrow(frequency) > 0L)
print(frequency)

p <- fread(PRED, select = c("DTXSID", "Taxon", "Pred_log10_ug_L"))
setorder(p, DTXSID, Taxon)
chems <- unique(p$DTXSID)
taxa <- unique(p$Taxon)
if (nrow(p) != length(chems) * length(taxa) ||
    anyDuplicated(p, by = c("DTXSID", "Taxon")) ||
    any(!is.finite(p$Pred_log10_ug_L))) stop("Incomplete prediction matrix")
X <- matrix(p$Pred_log10_ug_L, nrow = length(taxa))
rm(p)

roster <- fread(ROSTER)
if (!"Taxon" %in% names(roster) && "Species" %in% names(roster))
  setnames(roster, "Species", "Taxon")
stopifnot(all(c("Region", "Taxon") %in% names(roster)))
roster <- unique(roster[, .(Region = trimws(as.character(Region)),
                            Taxon = trimws(as.character(Taxon)))])
roster[Region %in% c("Pearl River (Guangzhou Segment)",
                     "Pearl River, Guangzhou segment"),
       Region := "Pearl River (Guangzhou segment)"]
stopifnot(setequal(roster$Region, REGIONS), setequal(roster$Taxon, taxa))
idx <- setNames(lapply(REGIONS, function(v)
  match(roster[Region == v, Taxon], taxa)), REGIONS)
stopifnot(all(vapply(idx, function(x) length(x) >= 6L && !anyNA(x), logical(1))))

# Five-repeat out-of-fold residuals, averaged by original training row.
d <- unique(fread(TRAIN, na.strings = c("", "NA", "NaN")))
d[, log10_Geomean_ug_L := as.numeric(log10_Geomean_ug_L)]
d <- d[is.finite(log10_Geomean_ug_L) & !is.na(Taxon) & nzchar(trimws(Taxon))]
d[, Taxon := trimws(as.character(Taxon))]
oof <- fread(OOF)
stopifnot(all(c("Repeat", "Taxon", "Observed", "Predicted") %in% names(oof)))
oof[, Row_ID := seq_len(.N), by = Repeat]
a <- oof[, .(Taxon = first(Taxon), Observed = first(Observed),
             Predicted = mean(Predicted), N = .N), by = Row_ID][order(Row_ID)]
stopifnot(nrow(a) == nrow(d), all(a$N == uniqueN(oof$Repeat)),
          identical(a$Taxon, d$Taxon),
          max(abs(a$Observed - d$log10_Geomean_ug_L)) < 1e-8)
residuals <- a$Observed - a$Predicted
weights <- 1 / as.numeric(table(a$Taxon)[a$Taxon])
weights <- weights / sum(weights)
stopifnot(all(is.finite(residuals)), all(is.finite(weights)))
rm(oof, a, d)

# Profile maximum-likelihood fit for the Gumbel distribution of ln(concentration).
fit_gumbel <- function(x, center) {
  x <- as.matrix(x)
  n <- nrow(x)
  y <- (x - center) * log(10)
  ymin <- apply(y, 2L, min)
  z <- sweep(y, 2L, ymin, "-")
  m1 <- colMeans(z)
  m2 <- colMeans(z * z)
  s <- sqrt(pmax(m2 - m1 * m1, 0)) * sqrt(6) / pi
  s[s <= 0] <- 1e-6
  for (iter in seq_len(35L)) {
    w <- exp(-sweep(z, 2L, s, "/"))
    sw <- colSums(w)
    e1 <- colSums(w * z) / sw
    e2 <- colSums(w * z * z) / sw
    g <- s - m1 + e1
    if (max(abs(g) / (1 + s)) < 1e-11) break
    next_s <- s - g / (1 + pmax(0, e2 - e1 * e1) / (s * s))
    bad <- !is.finite(next_s) | next_s <= 0
    next_s[bad] <- s[bad] / 2
    s <- next_s
  }
  w <- exp(-sweep(z, 2L, s, "/"))
  score <- abs(s - m1 + colSums(w * z) / colSums(w)) / (1 + s)
  ok <- is.finite(s) & s > 0 & is.finite(score) & score < 1e-8
  result <- rep(NA_real_, ncol(x))
  if (any(ok)) {
    mu <- ymin[ok] - s[ok] * log(colSums(w)[ok] / n)
    result[ok] <- center + (mu - s[ok] * log(-log(0.05))) / log(10)
  }
  if (any(!ok)) for (j in which(!ok)) {
    one <- y[, j]
    logmeanexp <- function(v) { m <- max(v); m + log(mean(exp(v - m))) }
    f <- function(log_s) {
      scale <- exp(log_s)
      n * (log(scale) + mean(one) / scale + logmeanexp(-one / scale) + 1)
    }
    s0 <- max(stats::sd(one), 1e-6)
    opt <- optimize(f, log(s0) + c(-10, 10))
    sj <- exp(opt$minimum)
    mu <- -sj * logmeanexp(-one / sj)
    result[j] <- center + (mu - sj * log(-log(0.05))) / log(10)
  }
  if (any(!is.finite(result))) stop("Invalid log-Gumbel HC5")
  result
}

# Identical simulated taxon response is used wherever a taxon occurs regionally.
fit_one <- function(k) {
  file <- file.path(OUT, "checkpoints", sprintf("chem_%04d.rds", k))
  if (file.exists(file)) return(k)
  set.seed(SEED + k)
  n_taxa <- length(taxa)
  point <- vapply(REGIONS, function(r) {
    v <- X[idx[[r]], k]
    fit_gumbel(matrix(v, ncol = 1L), median(v))[1L]
  }, numeric(1))
  mc <- matrix(NA_real_, nrow = length(REGIONS), ncol = N_MC)
  for (start in seq.int(1L, N_MC, by = MC_BATCH)) {
    end <- min(N_MC, start + MC_BATCH - 1L)
    errors <- matrix(sample(residuals, n_taxa * (end - start + 1L),
                            replace = TRUE, prob = weights), nrow = n_taxa)
    for (j in seq_along(REGIONS)) {
      v <- X[idx[[j]], k]
      sim <- sweep(errors[idx[[j]], , drop = FALSE], 1L, v, "+")
      mc[j, start:end] <- fit_gumbel(sim, median(v))
    }
  }
  med <- apply(mc, 1L, median)
  lower <- apply(mc, 1L, quantile, probs = 0.025, names = FALSE)
  upper <- apply(mc, 1L, quantile, probs = 0.975, names = FALSE)
  z <- data.table(DTXSID = chems[k], Region = REGIONS,
                  N_species = vapply(idx, length, integer(1)),
                  Distribution = "lgumbel", N_MC = N_MC,
                  HC5_point_log10_ug_L = point,
                  HC5_MC_median_log10_ug_L = med,
                  HC5_MC_lower95_log10_ug_L = lower,
                  HC5_MC_upper95_log10_ug_L = upper)
  saveRDS(z, file)
  k
}

files <- file.path(OUT, "checkpoints", sprintf("chem_%04d.rds", seq_along(chems)))
manifest <- list(pred = unname(tools::md5sum(PRED)),
                 roster = unname(tools::md5sum(ROSTER)),
                 oof = unname(tools::md5sum(OOF)),
                 train = unname(tools::md5sum(TRAIN)), n_mc = N_MC,
                 mc_batch = MC_BATCH, seed = SEED, distribution = "lgumbel")
mfile <- file.path(OUT, "run_manifest.rds")
if (file.exists(mfile)) {
  stopifnot(identical(manifest, readRDS(mfile)))
} else {
  stopifnot(!any(file.exists(files)))
  saveRDS(manifest, mfile)
}
pending <- which(!file.exists(files))
if (length(pending)) {
  WORKERS <- min(WORKERS, length(pending))
  if (WORKERS == 1L) {
    for (k in pending) fit_one(k)
  } else {
    cl <- parallel::makeCluster(WORKERS)
    tryCatch({
      parallel::clusterEvalQ(cl, library(data.table))
      parallel::clusterExport(cl, c("OUT", "SEED", "taxa", "chems", "X",
        "idx", "REGIONS", "N_MC", "MC_BATCH", "residuals", "weights",
        "fit_gumbel", "fit_one"), envir = .GlobalEnv)
      blocks <- split(pending, ceiling(seq_along(pending) / (WORKERS * 4L)))
      for (b in seq_along(blocks)) {
        parallel::parLapplyLB(cl, blocks[[b]], fit_one)
        cat("Monte Carlo batch", b, "/", length(blocks), "complete\n")
      }
    }, finally = parallel::stopCluster(cl))
  }
}

result <- rbindlist(lapply(files, readRDS))
for (v in c("HC5_point", "HC5_MC_median", "HC5_MC_lower95", "HC5_MC_upper95"))
  result[, (paste0(v, "_ug_L")) := 10^get(paste0(v, "_log10_ug_L"))]
fwrite(result, file.path(OUT, "regional_hc5.csv"))

variation <- result[, .(
  HC5_max_min_fold = 10^(max(HC5_MC_median_log10_ug_L) -
                         min(HC5_MC_median_log10_ug_L))), by = DTXSID]
fwrite(variation, file.path(OUT, "regional_hc5_fold_range.csv"))

wide <- dcast(result, DTXSID ~ Region, value.var = "HC5_MC_median_log10_ug_L")
M <- as.matrix(wide[, ..REGIONS])
ft <- friedman.test(M)
fwrite(data.table(Statistic = unname(ft$statistic), df = unname(ft$parameter),
                  p = ft$p.value), file.path(OUT, "friedman_test.csv"))
pairs <- rbindlist(lapply(combn(seq_along(REGIONS), 2L, simplify = FALSE), function(k) {
  test <- suppressWarnings(wilcox.test(M[, k[1]], M[, k[2]],
                                       paired = TRUE, exact = FALSE))
  data.table(Region_1 = REGIONS[k[1]], Region_2 = REGIONS[k[2]],
             p = test$p.value)
}))
pairs[, p_Holm := p.adjust(p, method = "holm")]
fwrite(pairs, file.path(OUT, "regional_pairwise_tests.csv"))
cat("Saved", nrow(result), "regional HC5 estimates in", OUT, "\n")
