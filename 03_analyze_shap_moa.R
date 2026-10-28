# Post-training TreeSHAP analysis for the selected XGBoost model (Fig. 3).

DATA <- "data/training_data.csv"
MODEL_DIR <- "results/xgboost_model"
OUT <- "results/shap_moa"
BOOTSTRAP_N <- 5000L
SEED_FILE <- file.path("results","model_training","00_RUN_SEED.txt")
if (!file.exists(SEED_FILE)) stop("Run 01_train_six_models.R first to create the run seed.")
SEED <- as.integer(readLines(SEED_FILE, n=1L))

for (p in c("data.table", "xgboost", "ggplot2")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
}
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
library(data.table)

CHEM <- c("logS", "logKow", "logD7_4", "MW", "TPSA", "log_Henry",
          "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
          "AromaticRingCount", "RotatableBondCount", "BertzCT")
TRAIT <- c("Log Body size", "Feeding_score", "Respiration_score",
           "Locomotion_score", "Body_armor_score")
TAX <- c("Phylum", "Class", "Order", "Family", "Genus")
CAT <- c(TAX, "Model_MOA_final")
NUM <- c(CHEM, TRAIT)
FEATURES <- c(NUM, CAT)
TARGET <- "log10_Geomean_ug_L"

# 1. Load the final model and its training-set encoding.
d <- unique(fread(DATA, na.strings = c("", "NA", "NaN")))
required <- c("DTXSID", "Taxon", TARGET, FEATURES, "Barron_Specific_MOA")
if (!all(required %in% names(d))) stop("Missing columns: ", paste(setdiff(required, names(d)), collapse = ", "))
for (v in c(TARGET, NUM)) set(d, j = v, value = suppressWarnings(as.numeric(d[[v]])))
d <- d[is.finite(get(TARGET)) & !is.na(Taxon) & nzchar(trimws(Taxon))]
d[, Taxon := trimws(as.character(Taxon))]
if (anyDuplicated(d, by = c("DTXSID", "Taxon"))) stop("Duplicate chemical-taxon combinations")
d[, Specific_MOA := trimws(as.character(Barron_Specific_MOA))]
d[is.na(Specific_MOA) | toupper(Specific_MOA) %in% c("", "NA", "N/A", "NONE", "NULL", "NAN"), Specific_MOA := NA_character_]

enc <- readRDS(file.path(MODEL_DIR, "xgboost_preprocessing.rds"))$encoder
model <- xgboost::xgb.load(file.path(MODEL_DIR, "xgboost_model.json"))
clean_cat <- function(x) {
  z <- trimws(as.character(x)); z[is.na(z) | z == ""] <- "__MISSING__"; z
}
make_xgb <- function(d) {
  z <- setNames(lapply(NUM, function(v) {
    a <- as.numeric(d[[v]]); a[!is.finite(a)] <- enc$median[[v]]; a
  }), NUM)
  for (v in CAT) {
    a <- clean_cat(d[[v]])
    a[!a %in% enc$levels[[v]]] <- "__NOVEL__"
    z[[v]] <- factor(replace(a, a == "__NOVEL__", NA_character_),
                     levels = enc$levels[[v]])
  }
  as.data.frame(z, check.names = FALSE, optional = TRUE)
}

# 2. Native TreeSHAP for all training observations.
x <- xgboost::xgb.DMatrix(make_xgb(d))
shap <- as.matrix(predict(model, x, predcontrib = TRUE))
stopifnot(ncol(shap) == length(FEATURES) + 1L, nrow(shap) == nrow(d))
S <- abs(shap[, seq_along(FEATURES), drop = FALSE])
colnames(S) <- FEATURES
blocks <- setNames(c(rep("Chemical", 13), rep("Ecological traits", 5),
                     rep("Taxonomy", 5), "MOA"), FEATURES)
global <- data.table(Feature = FEATURES, Block = unname(blocks),
                     MeanAbsSHAP = colMeans(S))
setorder(global, -MeanAbsSHAP)
fwrite(global, file.path(OUT, "global_shap.csv"))

# 3. Mean row-normalized SHAP shares per chemical and specific MOA.
total <- rowSums(S)
if (any(total <= 0 | !is.finite(total))) stop("Invalid SHAP contributions")
share <- data.table(DTXSID = d$DTXSID, Broad_MOA = d$Model_MOA_final,
                    Specific_MOA = d$Specific_MOA)
for (b in c("Chemical", "MOA", "Taxonomy", "Ecological traits")) {
  set(share, j = b, value = rowSums(S[, blocks == b, drop = FALSE]) / total)
}
chem <- share[, lapply(.SD, mean), by = .(DTXSID, Broad_MOA, Specific_MOA),
              .SDcols = c("Chemical", "MOA", "Taxonomy", "Ecological traits")]
setnames(chem, "Ecological traits", "Traits")
counts <- chem[, .(N_Chemicals = .N), by = .(Broad_MOA, Specific_MOA)]
row_counts <- share[, .(N_Rows = .N), by = .(Broad_MOA, Specific_MOA)]
counts <- merge(counts, row_counts, by = c("Broad_MOA", "Specific_MOA"))
counts[, Selected := !is.na(Specific_MOA) & tolower(Specific_MOA) != "other" &
         N_Rows >= 20L & N_Chemicals >= 2L]
fwrite(counts, file.path(OUT, "moa_group_counts.csv"))
selected <- counts[Selected == TRUE, .(Broad_MOA, Specific_MOA)]
if (nrow(selected) < 2L) stop("Need at least two specific MOA groups")
chem <- merge(chem, selected, by = c("Broad_MOA", "Specific_MOA"))
chem[, `:=`(Biology = Taxonomy + Traits,
            Group = paste(Broad_MOA, Specific_MOA, sep = " / "))]
chem[, `:=`(B_Organismal = Biology / (Chemical + Biology),
            C_Taxonomy = Taxonomy / Biology)]

# Fig. 3b: exclude broad-MOA SHAP from display; Fig. 3c: within biology.
moa <- chem[, .(N_Chemicals = .N, Chemical = mean(Chemical),
                MOA = mean(MOA), Taxonomy = mean(Taxonomy),
                Traits = mean(Traits)), by = .(Broad_MOA, Specific_MOA)]
moa[, `:=`(B_Chemical = Chemical / (Chemical + Taxonomy + Traits),
            B_Organismal = (Taxonomy + Traits) / (Chemical + Taxonomy + Traits),
            C_Taxonomy = Taxonomy / (Taxonomy + Traits),
            C_Traits = Traits / (Taxonomy + Traits))]
fwrite(moa, file.path(OUT, "moa_shap_contributions.csv"))
fwrite(chem, file.path(OUT, "chemical_shap_shares.csv"))

# 4. Kruskal-Wallis and Holm-adjusted pairwise Wilcoxon tests (chemical = unit).
tests <- list(Fig3b = "B_Organismal", Fig3c = "C_Taxonomy")
kw <- rbindlist(lapply(names(tests), function(panel) {
  z <- chem[, .(Group, Value = get(tests[[panel]]))]
  fit <- kruskal.test(Value ~ Group, data = z)
  data.table(Panel = panel, N_Chemicals = nrow(z), Groups = uniqueN(z$Group),
             H = unname(fit$statistic), df = unname(fit$parameter), p = fit$p.value)
}))
fwrite(kw, file.path(OUT, "moa_kruskal_wallis.csv"))
pairs <- rbindlist(lapply(names(tests), function(panel) {
  z <- chem[, .(Group, Value = get(tests[[panel]]))]
  groups <- sort(unique(z$Group))
  res <- rbindlist(lapply(combn(groups, 2, simplify = FALSE), function(g) {
    p <- suppressWarnings(wilcox.test(z[Group == g[1], Value],
                                      z[Group == g[2], Value], exact = FALSE)$p.value)
    data.table(Panel = panel, Group_1 = g[1], Group_2 = g[2], p = p)
  }))
  res[, p_Holm := p.adjust(p, method = "holm")]
  res
}))
fwrite(pairs, file.path(OUT, "moa_pairwise_wilcoxon.csv"))

# 5. Chemical-level bootstrap confidence intervals for Fig. 3b and 3c.
set.seed(SEED)
summary_shares <- function(z) {
  a <- colMeans(z)                  # Chemical, MOA, Taxonomy, Traits
  c(B_Chemical = unname(a[1] / sum(a[c(1, 3, 4)])),
    B_Organismal = unname(sum(a[3:4]) / sum(a[c(1, 3, 4)])),
    C_Taxonomy = unname(a[3] / sum(a[3:4])),
    C_Traits = unname(a[4] / sum(a[3:4])))
}
boot <- rbindlist(lapply(seq_len(nrow(moa)), function(i) {
  g <- moa[i]
  z <- as.matrix(chem[Broad_MOA == g$Broad_MOA & Specific_MOA == g$Specific_MOA,
                      .(Chemical, MOA, Taxonomy, Traits)])
  n <- nrow(z)
  estimate <- summary_shares(z)
  draws <- replicate(BOOTSTRAP_N, summary_shares(z[sample.int(n, n, replace = TRUE), , drop = FALSE]))
  ci <- t(apply(draws, 1, quantile, probs = c(0.025, 0.975), names = FALSE))
  data.table(Broad_MOA = g$Broad_MOA, Specific_MOA = g$Specific_MOA,
             N_Chemicals = n, Component = names(estimate), Estimate = unname(estimate),
             Lower_95 = ci[, 1], Upper_95 = ci[, 2])
}))
fwrite(boot, file.path(OUT, "moa_bootstrap_95CI.csv"))

# 6. Individual ecological trait shares within the organismal component.
trait_rows <- data.table(DTXSID = d$DTXSID, Broad_MOA = d$Model_MOA_final,
                         Specific_MOA = d$Specific_MOA)
for (v in TRAIT) set(trait_rows, j = v, value = S[, v] / total)
trait_chem <- trait_rows[, lapply(.SD, mean),
                         by = .(DTXSID, Broad_MOA, Specific_MOA), .SDcols = TRAIT]
trait_chem <- merge(trait_chem, selected, by = c("Broad_MOA", "Specific_MOA"))
trait_moa <- trait_chem[, lapply(.SD, mean), by = .(Broad_MOA, Specific_MOA), .SDcols = TRAIT]
trait_moa <- merge(trait_moa, moa[, .(Broad_MOA, Specific_MOA, Taxonomy, Traits)],
                   by = c("Broad_MOA", "Specific_MOA"))
for (v in TRAIT) trait_moa[[v]] <- 100 * trait_moa[[v]] / (trait_moa$Taxonomy + trait_moa$Traits)
fwrite(trait_moa[, setdiff(names(trait_moa), c("Taxonomy", "Traits")), with = FALSE],
       file.path(OUT, "trait_shap_by_moa.csv"))

# 7. Compact figure panels.
library(ggplot2)
p_a <- ggplot(global, aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP, fill = Block)) +
  geom_col() + coord_flip() + labs(x = NULL, y = "Mean absolute SHAP") + theme_classic(12)
ggsave(file.path(OUT, "Fig3a_global_shap.png"), p_a, width = 7, height = 7, dpi = 300)

plot_heatmap <- function(columns, labels, path, palette) {
  z <- copy(moa)
  z[, Label := paste(Specific_MOA, Broad_MOA, sep = "  |  ")]
  z <- melt(z, id.vars = "Label", measure.vars = columns,
            variable.name = "Component", value.name = "Share")
  z[, Component := factor(Component, levels = columns, labels = labels)]
  z[, Percent := ifelse(Component == labels[1], round(100 * Share),
                        100 - round(100 * (1 - Share)))]
  z[, Label := factor(Label, levels = rev(unique(Label)))]
  p <- ggplot(z, aes(Component, Label, fill = Share)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = paste0(Percent, "%")), size = 3) +
    scale_fill_gradient(low = "white", high = palette, limits = c(0, 1)) +
    labs(x = NULL, y = NULL) + theme_minimal(11) +
    theme(panel.grid = element_blank(), legend.position = "none")
  ggsave(file.path(OUT, path), p, width = 8.5, height = 6.5, dpi = 300)
}
plot_heatmap(c("B_Chemical", "B_Organismal"),
             c("Chemical descriptors", "Taxonomy and traits"), "Fig3b_moa_contributions.png", "#9CCBFF")
plot_heatmap(c("C_Taxonomy", "C_Traits"),
             c("Taxonomy", "Ecological traits"), "Fig3c_taxonomy_traits.png", "#97D19F")
cat("SHAP and MOA analysis finished: ", OUT, "\n", sep = "")
