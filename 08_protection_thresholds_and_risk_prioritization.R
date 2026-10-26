# Conventional versus regional protection thresholds and pesticide risk ranking (Fig. 7).

TRAIN <- "data/training_data.csv"
HC5 <- "results/06_regional_ssd_hc5/regional_hc5.csv"
MONITOR <- "data/taihu_monitoring.xlsx"
OUT <- "results/08_protection_thresholds_and_risk_prioritization"
AF <- 100
SSD_AF <- 5
ACUTE_TO_CHRONIC <- 10
MIN_SSD_TAXA <- 10L

for (pkg in c("data.table", "ggplot2", "readxl"))
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install R package: ", pkg)
library(data.table)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

obs <- fread(TRAIN, select = c("DTXSID", "CASRN", "Taxon", "log10_Geomean_ug_L"),
             colClasses = list(character = c("DTXSID", "CASRN", "Taxon")))
reg <- fread(HC5, select = c("DTXSID", "Region", "HC5_MC_median_log10_ug_L"),
             colClasses = list(character = c("DTXSID", "Region")))
stopifnot(!anyDuplicated(obs, by = c("DTXSID", "Taxon")),
          !anyDuplicated(reg, by = c("DTXSID", "Region")),
          all(is.finite(obs$log10_Geomean_ug_L)),
          all(is.finite(reg$HC5_MC_median_log10_ug_L)))

# General SSD: log-Gumbel maximum-likelihood HC5 on observed taxa.
fit_general_loggumbel <- function(log10_x) {
  if (length(log10_x) < MIN_SSD_TAXA || sd(log10_x) <= 0) return(NA_real_)
  center <- median(log10_x)
  x <- (log10_x - center) * log(10)
  lme <- function(v) { m <- max(v); m + log(mean(exp(v - m))) }
  objective <- function(log_s) {
    s <- exp(log_s)
    length(x) * (log(s) + mean(x) / s + lme(-x / s) + 1)
  }
  fit <- optimize(objective, log(sd(x)) + c(-10, 10))
  s <- exp(fit$minimum)
  mu <- -s * lme(-x / s)
  center + (mu - s * log(-log(0.05))) / log(10)
}

chem <- obs[, .(N_observed_taxa = .N,
                AF_PNEC_log10 = min(log10_Geomean_ug_L) -
                  log10(AF * ACUTE_TO_CHRONIC)), by = DTXSID]
ids <- chem[N_observed_taxa >= MIN_SSD_TAXA, DTXSID]
general <- obs[DTXSID %in% ids,
               .(General_HC5_log10 = fit_general_loggumbel(log10_Geomean_ug_L)),
               by = DTXSID]
general[, General_PNEC_log10 := General_HC5_log10 -
                                     log10(SSD_AF * ACUTE_TO_CHRONIC)]
chem <- merge(chem, general, by = "DTXSID", all.x = TRUE)
fwrite(chem, file.path(OUT, "conventional_pnecs.csv"))

# Relative thresholds and regional underprotection percentages (Fig. 7a,b).
z <- merge(reg, chem, by = "DTXSID")
z[, Regional_PNEC_log10 := HC5_MC_median_log10_ug_L - log10(ACUTE_TO_CHRONIC)]
z[, Gap_AF := AF_PNEC_log10 - Regional_PNEC_log10]
z[, Gap_General := General_PNEC_log10 - Regional_PNEC_log10]
gaps <- rbindlist(list(
  z[, .(DTXSID, Region, Method = "AF-based", Log10_threshold_ratio = Gap_AF)],
  z[is.finite(Gap_General),
    .(DTXSID, Region, Method = "General SSD", Log10_threshold_ratio = Gap_General)]))
summary <- gaps[, .(N_chemicals = .N,
                    N_potentially_underprotected = sum(Log10_threshold_ratio > 0),
                    Pct_potentially_underprotected = 100 * mean(Log10_threshold_ratio > 0),
                    Median_log10_ratio = median(Log10_threshold_ratio)),
                by = .(Method, Region)]
fwrite(gaps, file.path(OUT, "conventional_vs_regional_pnec_ratios.csv"))
fwrite(summary, file.path(OUT, "potential_underprotection_by_region.csv"))

fig_ab <- ggplot2::ggplot(gaps, ggplot2::aes(Log10_threshold_ratio, Region)) +
  ggplot2::geom_vline(xintercept = 0, linetype = 2, color = "gray40") +
  ggplot2::geom_boxplot(fill = "#8DB7D7", outlier.shape = NA) +
  ggplot2::facet_wrap(~Method) +
  ggplot2::labs(x = "log10(conventional PNEC / regional PNEC)", y = NULL) +
  ggplot2::theme_bw(base_size = 11)
ggplot2::ggsave(file.path(OUT, "figure7_threshold_comparison.pdf"),
                fig_ab, width = 12, height = 5)

# Taihu monitoring data: ten organophosphates from nine sites (Fig. 7c).
if (file.exists(MONITOR)) {
  mon <- as.data.table(readxl::read_excel(MONITOR, sheet = "10_OP_9_Sites"))
  conc_col <- grep("^LOD/2-adjusted", names(mon), value = TRUE, ignore.case = TRUE)
  stopifnot(length(conc_col) == 1L,
            all(c("Site ID", "Chemical") %in% names(mon)))
  setnames(mon, c("Site ID", conc_col), c("Site_ID", "Concentration_ug_L"))
  mon[, `:=`(Site_ID = trimws(as.character(Site_ID)),
             Chemical = trimws(as.character(Chemical)),
             Concentration_ug_L = suppressWarnings(as.numeric(Concentration_ug_L)))]
  # Remove the workbook's blank/footnote rows, keeping all monitoring records.
  mon <- mon[!(is.na(Chemical) | Chemical == "") |
             !(is.na(Site_ID) | Site_ID == "") |
             is.finite(Concentration_ug_L)]
  mon <- mon[!(is.na(Chemical) | Chemical == "")]
  stopifnot(nrow(mon) == 90L, uniqueN(mon$Site_ID) == 9L,
            uniqueN(mon$Chemical) == 10L,
            !anyDuplicated(mon, by = c("Site_ID", "Chemical")),
            all(is.finite(mon$Concentration_ug_L) & mon$Concentration_ug_L >= 0))

  key <- data.table(
    Chemical = c("Diazinon", "Dimethoate", "Chlorpyrifos", "Malathion",
                 "Dichlorvos", "Methyl parathion", "Parathion",
                 "Methidathion", "Triazophos", "Trichlorfon"),
    CASRN = c("333-41-5", "60-51-5", "2921-88-2", "121-75-5",
              "62-73-7", "298-00-0", "56-38-2", "950-37-8",
              "24017-47-8", "52-68-6"))
  stopifnot(setequal(mon$Chemical, key$Chemical))
  identity <- unique(obs[, .(CASRN, DTXSID)])
  key <- merge(key, identity, by = "CASRN", all.x = TRUE)
  stopifnot(nrow(key) == 10L, !anyNA(key$DTXSID), !anyDuplicated(key$DTXSID))
  taihu <- z[Region == "Taihu Basin",
             .(DTXSID, AF_PNEC_log10, General_PNEC_log10, Regional_PNEC_log10)]
  key <- merge(key, taihu, by = "DTXSID")
  stopifnot(nrow(key) == 10L)
  site <- merge(mon[, .(Chemical, Site_ID, Concentration_ug_L)],
                key[, .(Chemical, DTXSID, AF_PNEC_log10,
                         General_PNEC_log10, Regional_PNEC_log10)], by = "Chemical")
  rq <- melt(site,
             id.vars = c("Chemical", "DTXSID", "Site_ID", "Concentration_ug_L"),
             measure.vars = c("Regional_PNEC_log10", "AF_PNEC_log10",
                              "General_PNEC_log10"),
             variable.name = "Method", value.name = "PNEC_log10")
  rq[, Method := fifelse(Method == "Regional_PNEC_log10", "Regional HC5",
                         fifelse(Method == "AF_PNEC_log10", "AF-based", "General SSD"))]
  rq[, RQ := Concentration_ug_L / (10^PNEC_log10)]
  ranks <- rq[, .(Mean_RQ = if (all(is.na(RQ))) NA_real_ else mean(RQ, na.rm = TRUE),
                  SD_RQ = if (all(is.na(RQ))) NA_real_ else sd(RQ, na.rm = TRUE)),
              by = .(Chemical, Method)]
  ranks[, Priority_rank := as.integer(rank(-Mean_RQ, na.last = "keep",
                                           ties.method = "min")), by = Method]
  fwrite(rq, file.path(OUT, "taihu_site_risk_quotients.csv"))
  fwrite(ranks, file.path(OUT, "taihu_mean_rq_and_ranks.csv"))

  # Compare regional and conventional ranks over the same evaluable chemicals.
  regional_rq <- ranks[Method == "Regional HC5", .(Chemical, Regional_RQ = Mean_RQ)]
  shifts <- rbindlist(lapply(c("AF-based", "General SSD"), function(method) {
    x <- merge(ranks[Method == method & is.finite(Mean_RQ)], regional_rq,
               by = "Chemical")
    x[, Regional_rank := as.integer(rank(-Regional_RQ, ties.method = "min"))]
    x[, Rank_shift := Regional_rank - Priority_rank]
    x
  }))
  fwrite(shifts, file.path(OUT, "taihu_risk_rank_shifts.csv"))
  fig_c <- ggplot2::ggplot(ranks[is.finite(Mean_RQ) & Mean_RQ > 0],
                            ggplot2::aes(Chemical, Mean_RQ, fill = Method)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8)) +
    ggplot2::scale_y_log10() +
    ggplot2::labs(x = NULL, y = "Mean risk quotient") +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 40, hjust = 1))
  ggplot2::ggsave(file.path(OUT, "figure7_taihu_risk_quotients.pdf"),
                  fig_c, width = 12, height = 6)
  print(fig_c)
} else {
  message("Taihu monitoring workbook not found; threshold comparison saved without Fig. 7c.")
}
print(fig_ab)
cat("Saved Fig. 7 analysis:", OUT, "\n")
