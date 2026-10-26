# Chemical toxicity gradients, modes of action, and regional HC5 variation (Fig. 6).

PRED <- "results/04_cross_species_toxicity_prediction/predicted_toxicity.csv"
HC5 <- "results/06_regional_ssd_hc5/regional_hc5.csv"
TRAIN <- "data/training_data.csv"
OUT <- "results/07_toxicity_gradients_and_regional_variation"

for (pkg in c("data.table", "ggplot2"))
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install R package: ", pkg)
library(data.table)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

p <- fread(PRED, select = c("DTXSID", "Taxon", "Pred_log10_ug_L"),
           colClasses = list(character = c("DTXSID", "Taxon")))
h <- fread(HC5, select = c("DTXSID", "Region", "HC5_MC_median_log10_ug_L"),
           colClasses = list(character = c("DTXSID", "Region")))
t <- fread(TRAIN, select = c("DTXSID", "Model_MOA_final", "Barron_Specific_MOA"),
           colClasses = list(character = "DTXSID"))
stopifnot(!anyDuplicated(p, by = c("DTXSID", "Taxon")),
          !anyDuplicated(h, by = c("DTXSID", "Region")),
          all(is.finite(p$Pred_log10_ug_L)),
          all(is.finite(h$HC5_MC_median_log10_ug_L)))

# Overall chemical toxicity: median predicted log10 effect concentration over all taxa.
toxicity <- p[, .(N_taxa = .N,
                  Median_log10_ug_L = median(Pred_log10_ug_L)), by = DTXSID]
stopifnot(uniqueN(toxicity$N_taxa) == 1L)
toxicity[, Toxicity_percentile := 100 * (frank(Median_log10_ug_L,
                      ties.method = "average") - 1) / (.N - 1)]

# Regional variability: maximum / minimum HC5 among seven regions.
variation <- h[, .(N_regions = .N,
                   Regional_log10_ratio = max(HC5_MC_median_log10_ug_L) -
                                          min(HC5_MC_median_log10_ug_L)), by = DTXSID]
stopifnot(all(variation$N_regions == 7L))
variation[, HC5_max_min_fold := 10^Regional_log10_ratio]

# Chemical-level broad and specific MOA labels; specific MOA is post-hoc metadata.
moa <- t[, .(N_records = .N, Broad_MOA = first(Model_MOA_final),
             Specific_MOA = first(Barron_Specific_MOA),
             N_broad = uniqueN(Model_MOA_final, na.rm = TRUE),
             N_specific = uniqueN(Barron_Specific_MOA, na.rm = TRUE)), by = DTXSID]
stopifnot(all(moa$N_broad == 1L), all(moa$N_specific <= 1L),
          !anyDuplicated(moa$DTXSID))
moa[, Specific_MOA := trimws(as.character(Specific_MOA))]
moa[is.na(Specific_MOA) | !nzchar(Specific_MOA), Specific_MOA := NA_character_]
moa[Broad_MOA == "Mitochondrial energy disruption", Broad_MOA := "Energy disruption"]
moa[Broad_MOA == "Other / Unclassified", Broad_MOA := "Other / unclassified"]

z <- merge(merge(toxicity, variation, by = "DTXSID"), moa, by = "DTXSID")
stopifnot(nrow(z) == nrow(toxicity), nrow(z) == nrow(variation))
fwrite(z, file.path(OUT, "chemical_toxicity_and_regional_variation.csv"))

# Correlation and toxicity deciles.
correlation <- cor.test(z$Toxicity_percentile, z$Regional_log10_ratio,
                        method = "spearman", exact = FALSE)
fwrite(data.table(Spearman_rho = unname(correlation$estimate),
                  p = correlation$p.value, N_chemicals = nrow(z)),
       file.path(OUT, "toxicity_regional_variation_correlation.csv"))
z[, Toxicity_decile := pmin(10L, floor(Toxicity_percentile / 10) + 1L)]
deciles <- z[, .(N_chemicals = .N,
                  Median_HC5_fold = median(HC5_max_min_fold),
                  P10_HC5_fold = unname(quantile(HC5_max_min_fold, 0.10)),
                  P90_HC5_fold = unname(quantile(HC5_max_min_fold, 0.90)),
                  Pct_less_5fold = 100 * mean(HC5_max_min_fold < 5),
                  Pct_5_to_10fold = 100 * mean(HC5_max_min_fold >= 5 & HC5_max_min_fold < 10),
                  Pct_at_least_10fold = 100 * mean(HC5_max_min_fold >= 10)),
             by = Toxicity_decile][order(Toxicity_decile)]
fwrite(deciles, file.path(OUT, "toxicity_deciles.csv"))

# Compare the 14 specific MOAs (at least 20 toxicity records and 2 chemicals).
eligible <- moa[!is.na(Specific_MOA) & tolower(Specific_MOA) != "other",
                .(N_records = sum(N_records), N_chemicals = .N),
                by = .(Broad_MOA, Specific_MOA)][N_records >= 20 & N_chemicals >= 2]
sub <- z[eligible[, .(Broad_MOA, Specific_MOA)],
         on = .(Broad_MOA, Specific_MOA), nomatch = 0L]
if (uniqueN(sub$Specific_MOA) >= 2L) {
  sub[, MOA_group := paste(Broad_MOA, Specific_MOA, sep = " / ")]
  kw <- kruskal.test(Regional_log10_ratio ~ MOA_group, data = sub)
  fwrite(data.table(N_chemicals = nrow(sub), N_groups = uniqueN(sub$MOA_group),
                    Statistic = unname(kw$statistic), df = unname(kw$parameter),
                    p = kw$p.value), file.path(OUT, "moa_kruskal_test.csv"))
  fwrite(sub[, .(N_chemicals = .N, Median_HC5_fold = median(HC5_max_min_fold)),
             by = .(Broad_MOA, Specific_MOA)],
         file.path(OUT, "specific_moa_variation.csv"))
}

# Empirical 60% Mahalanobis ellipses for specific MOA groups.
make_ellipse <- function(d) {
  if (nrow(d) < 5L) return(NULL)
  xy <- as.matrix(d[, .(Toxicity_percentile, Regional_log10_ratio)])
  center <- colMeans(xy)
  covariance <- cov(xy)
  eigenfit <- eigen(covariance, symmetric = TRUE)
  if (any(!is.finite(eigenfit$values)) || min(eigenfit$values) <= 1e-10)
    return(NULL)
  radius <- sqrt(unname(quantile(mahalanobis(xy, center, covariance), 0.60)))
  angle <- seq(0, 2 * pi, length.out = 160)
  shape <- eigenfit$vectors %*% diag(sqrt(eigenfit$values), 2) %*%
           rbind(cos(angle), sin(angle))
  data.table(X = center[1L] + radius * shape[1L, ],
             Y = center[2L] + radius * shape[2L, ])
}
ellipses <- sub[, make_ellipse(.SD), by = .(Broad_MOA, Specific_MOA)]

# Fig. 6: chemical toxicity percentile vs. interregional HC5 variation.
fig <- ggplot2::ggplot(z, ggplot2::aes(Toxicity_percentile, Regional_log10_ratio)) +
  ggplot2::geom_point(size = 0.65, alpha = 0.35, color = "#9DBBD8") +
  ggplot2::geom_vline(xintercept = 50, linetype = 2, color = "gray50") +
  ggplot2::geom_hline(yintercept = median(z$Regional_log10_ratio),
                      linetype = 2, color = "gray50") +
  ggplot2::geom_path(data = ellipses,
                     ggplot2::aes(X, Y, group = Specific_MOA,
                                  colour = Specific_MOA),
                     inherit.aes = FALSE, linewidth = 0.8) +
  ggplot2::facet_wrap(~ Broad_MOA, ncol = 3) +
  ggplot2::labs(x = "Chemical toxicity percentile (lower = more toxic)",
                y = "Regional HC5 variation (log10 max/min)", colour = "Specific MOA") +
  ggplot2::theme_bw(base_size = 11)
ggplot2::ggsave(file.path(OUT, "figure6_toxicity_gradients.pdf"), fig,
                width = 12, height = 7)
print(fig)
cat("Saved Fig. 6 analysis:", OUT, "\n")
