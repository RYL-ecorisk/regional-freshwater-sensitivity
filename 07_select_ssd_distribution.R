# Five SSD families compared by AICc across the seven regional assemblages.

PRED <- "results/species_toxicity_prediction/predicted_toxicity.csv"
ROSTER <- "data/region_taxon_membership.csv"
OUT <- "results/ssd_distribution_selection"
WORKERS <- 4L
DISTS <- c("lnorm", "llogis", "gamma", "weibull", "lgumbel")
REGIONS <- c("Taihu Basin", "Liao River", "Songhua River",
             "Pearl River (Guangzhou segment)", "England",
             "Elbe River", "Danube River")

for (p in c("data.table", "ssdtools"))
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
library(data.table)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "checkpoints"), showWarnings = FALSE)

p <- fread(PRED, select = c("DTXSID", "Taxon", "Pred_log10_ug_L"))
r <- fread(ROSTER)
if (!"Taxon" %in% names(r) && "Species" %in% names(r))
  setnames(r, "Species", "Taxon")
stopifnot(all(c("Region", "Taxon") %in% names(r)))
r <- unique(r[, .(Region = trimws(as.character(Region)),
                  Taxon = trimws(as.character(Taxon)))])
r[Region %in% c("Pearl River (Guangzhou Segment)",
                "Pearl River, Guangzhou segment"),
  Region := "Pearl River (Guangzhou segment)"]
stopifnot(setequal(r$Region, REGIONS))
setorder(p, DTXSID, Taxon)
chems <- unique(p$DTXSID)
taxa <- unique(p$Taxon)
if (nrow(p) != length(chems) * length(taxa) ||
    anyDuplicated(p, by = c("DTXSID", "Taxon")) ||
    !setequal(r$Taxon, taxa) || any(!is.finite(p$Pred_log10_ug_L)))
  stop("Check prediction matrix and regional taxon membership")
X <- matrix(p$Pred_log10_ug_L, nrow = length(taxa))
idx <- setNames(lapply(REGIONS, function(v) match(r[Region == v, Taxon], taxa)),
                REGIONS)
stopifnot(all(vapply(idx, function(x) length(x) >= 6L && !anyNA(x), logical(1))))
rm(p, r)

# Normalize concentrations before fitting to improve numerical stability.
fit_ssd <- function(id, region, x) {
  shift <- median(x)
  d <- data.frame(Conc = 10^(x - shift))
  result <- data.table(DTXSID = id, Region = region, Distribution = DISTS,
                       N_species = length(x), AICc = NA_real_,
                       HC5_log10_ug_L = NA_real_)
  extract <- function(distributions) {
    fit <- tryCatch(suppressWarnings(ssdtools::ssd_fit_dists(
      d, dists = distributions, nrow = 6L, all_dists = FALSE,
      at_boundary_ok = FALSE, silent = TRUE)), error = function(e) NULL)
    if (is.null(fit)) return(NULL)
    gof <- tryCatch(as.data.table(ssdtools::ssd_gof(fit, wt = TRUE)),
                    error = function(e) data.table())
    hc <- tryCatch(as.data.table(ssdtools::ssd_hc(
      fit, proportion = 0.05, average = FALSE, ci = FALSE)),
      error = function(e) data.table())
    list(gof = gof, hc = hc)
  }
  z <- extract(DISTS)
  if (is.null(z)) z <- list(gof = data.table(), hc = data.table())
  for (dist in DISTS) {
    gi <- match(dist, as.character(z$gof$dist))
    hi <- match(dist, as.character(z$hc$dist))
    gg <- if (is.na(gi)) data.table() else z$gof[gi]
    hh <- if (is.na(hi)) data.table() else z$hc[hi]
    if (!nrow(gg) || !nrow(hh)) {
      single <- extract(dist)
      if (is.null(single)) next
      gg <- single$gof
      hh <- single$hc
    }
    if (!nrow(gg) || !nrow(hh)) next
    i <- match(dist, result$Distribution)
    result$AICc[i] <- as.numeric(gg$aicc[1L])
    hc5 <- as.numeric(hh$est[1L])
    if (is.finite(hc5) && hc5 > 0)
      result$HC5_log10_ug_L[i] <- shift + log10(hc5)
  }
  result[, Best := FALSE]
  valid <- which(is.finite(result$AICc) & is.finite(result$HC5_log10_ug_L))
  if (length(valid)) result$Best[valid[which.min(result$AICc[valid])]] <- TRUE
  result
}

# One checkpoint per chemical keeps a long analysis resumable.
files <- file.path(OUT, "checkpoints", sprintf("chem_%04d.rds", seq_along(chems)))
manifest <- list(pred = unname(tools::md5sum(PRED)),
                 roster = unname(tools::md5sum(ROSTER)), dists = DISTS)
mfile <- file.path(OUT, "run_manifest.rds")
if (file.exists(mfile)) {
  stopifnot(identical(manifest, readRDS(mfile)))
} else {
  stopifnot(!any(file.exists(files)))
  saveRDS(manifest, mfile)
}
fit_one <- function(k) {
  if (file.exists(files[k])) return(k)
  z <- rbindlist(lapply(REGIONS, function(region)
    fit_ssd(chems[k], region, X[idx[[region]], k])))
  saveRDS(z, files[k])
  k
}
pending <- which(!file.exists(files))
if (length(pending)) {
  WORKERS <- min(WORKERS, length(pending))
  if (WORKERS == 1L) {
    for (k in pending) fit_one(k)
  } else {
    cl <- parallel::makeCluster(WORKERS)
    tryCatch({
      parallel::clusterEvalQ(cl, { library(data.table); library(ssdtools); NULL })
      parallel::clusterExport(cl, c("DISTS", "REGIONS", "X", "idx",
        "chems", "files", "fit_ssd", "fit_one"), envir = .GlobalEnv)
      blocks <- split(pending, ceiling(seq_along(pending) / (WORKERS * 4L)))
      for (b in seq_along(blocks)) {
        parallel::parLapplyLB(cl, blocks[[b]], fit_one)
        cat("SSD fitting batch", b, "/", length(blocks), "complete\n")
      }
    }, finally = parallel::stopCluster(cl))
  }
}
all_fits <- rbindlist(lapply(files, readRDS))
best <- all_fits[Best == TRUE]
fwrite(all_fits, file.path(OUT, "ssd_five_distribution_aicc.csv"))
fwrite(best, file.path(OUT, "ssd_best_fit.csv"))
fwrite(best[, .N, by = Distribution][order(-N)],
       file.path(OUT, "ssd_selection_frequency.csv"))
cat("Saved SSD distribution comparisons for", length(chems), "chemicals.\n")
