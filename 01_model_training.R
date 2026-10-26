# Six models | 5 x 5 taxon-blocked CV | final XGBoost
# Run from repository root. Set RETICULATE_PYTHON for the FT-Transformer environment.

DATA <- "data/training_data.csv"
OUT <- "results/01_model_training"
SEED <- 263131L
QUICK_TEST <- FALSE                       # TRUE: 2 folds x 1 repeat, fewer epochs
RUN_FT <- TRUE                            # FALSE: run only five R models
K <- if (QUICK_TEST) 2L else 5L
REPEATS <- if (QUICK_TEST) 1L else 5L
MODELS <- c("Ridge", "Elastic_Net", "Random_Forest", "XGBoost", "LightGBM")
if (RUN_FT) MODELS <- c(MODELS, "FT_Transformer")

for (p in c("data.table", "glmnet", "ranger", "xgboost", "lightgbm", if (RUN_FT) "reticulate")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("Install R package: ", p)
}
if (packageVersion("xgboost") < "3.0.0") stop("Requires xgboost >= 3.0 for categorical features")
if (!file.exists(DATA)) stop("Place the curated training CSV at: ", DATA)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)

CHEM <- c("logS", "logKow", "logD7_4", "MW", "TPSA", "log_Henry",
          "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
          "AromaticRingCount", "RotatableBondCount", "BertzCT")
TRAIT <- c("Log Body size", "Feeding_score", "Respiration_score",
           "Locomotion_score", "Body_armor_score")
CAT <- c("Phylum", "Class", "Order", "Family", "Genus", "Model_MOA_final")
NUM <- c(CHEM, TRAIT)
TARGET <- "log10_Geomean_ug_L"
FEATURES <- c(NUM, CAT)
stopifnot(length(FEATURES) == 24L)

d <- unique(data.table::fread(DATA))
if (!all(c("DTXSID", "Taxon", TARGET, FEATURES) %in% names(d)))
  stop("Missing required training columns")
for (v in c(TARGET, NUM)) data.table::set(d, j = v, value = suppressWarnings(as.numeric(d[[v]])))
d <- d[is.finite(get(TARGET)) & !is.na(Taxon) & nzchar(trimws(Taxon))]
d[, Taxon := trimws(as.character(Taxon))]
if (anyDuplicated(d, by = c("DTXSID", "Taxon")))
  stop("Repeated chemical-taxon IDs in curated input")
if (anyNA(d$Model_MOA_final) || any(!nzchar(trimws(d$Model_MOA_final))))
  stop("Missing broad MOA classifications")
cat(sprintf("Training: %d rows, %d chemicals, %d taxa\n",
            nrow(d), data.table::uniqueN(d$DTXSID), data.table::uniqueN(d$Taxon)))

clean_cat <- function(x) {
  x <- trimws(as.character(x)); x[is.na(x) | x == ""] <- "__MISSING__"; x
}
encoder <- function(tr) list(
  median = setNames(lapply(NUM, function(v) {
    z <- tr[[v]]; z <- z[is.finite(z)]; if (length(z)) median(z) else 0
  }), NUM),
  levels = setNames(lapply(CAT, function(v)
    unique(c(sort(unique(clean_cat(tr[[v]]))), "__NOVEL__"))), CAT)
)
features <- function(x, enc, type) {
  z <- setNames(lapply(NUM, function(v) {
    a <- as.numeric(x[[v]]); a[!is.finite(a)] <- enc$median[[v]]; a
  }), NUM)
  for (v in CAT) {
    a <- clean_cat(x[[v]]); a[!a %in% enc$levels[[v]]] <- "__NOVEL__"
    z[[v]] <- switch(type,
      linear = a, rf = factor(a, levels = enc$levels[[v]]),
      xgb = factor(replace(a, a == "__NOVEL__", NA_character_), levels = enc$levels[[v]]),
      lgb = match(a, enc$levels[[v]]) - 1L,
      ft = a)
  }
  if (type == "linear") {
    parts <- lapply(CAT, function(v) {
      a <- sapply(enc$levels[[v]], function(level) as.numeric(z[[v]] == level))
      colnames(a) <- paste0(v, "_", make.names(enc$levels[[v]], unique = TRUE)); a
    })
    return(do.call(cbind, c(list(as.matrix(as.data.frame(z[NUM], check.names = FALSE))), parts)))
  }
  as.data.frame(z, check.names = FALSE, optional = TRUE)
}

# FT-Transformer (PyTorch Tabular via reticulate)
if (RUN_FT) for (p in c("torch", "pytorch_tabular", "numpy", "pandas")) {
  if (!reticulate::py_module_available(p)) stop("Missing Python module: ", p)
}
if (RUN_FT) reticulate::py_run_string("
import gc, random
import numpy as np
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig

def fit_ft(train, valid, nums, cats, target, seed, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dc = DataConfig(target=[target], continuous_cols=list(nums), categorical_cols=list(cats),
                    normalize_continuous_features=True, handle_unknown_categories=True,
                    handle_missing_values=True, num_workers=0)
    tc = TrainerConfig(batch_size=256, max_epochs=epochs, min_epochs=epochs,
                       accelerator='auto', devices=1, early_stopping=None,
                       checkpoints=None, load_best=False, progress_bar='none', seed=seed)
    mc = FTTransformerConfig(task='regression', input_embed_dim=64, num_heads=8,
                             num_attn_blocks=3, attn_dropout=0.15,
                             add_norm_dropout=0.15, ff_dropout=0.15,
                             learning_rate=0.0005, seed=seed)
    model = TabularModel(data_config=dc, trainer_config=tc,
                         optimizer_config=OptimizerConfig(optimizer='AdamW',
                                                          optimizer_params={'weight_decay': 0.00003}),
                         model_config=mc, verbose=False, suppress_lightning_logger=True)
    model.fit(train=train, seed=seed)
    def pred(df):
        p = model.predict(df.drop(columns=[target]), progress_bar=None)
        return p[target + '_prediction'].to_numpy(dtype=float)
    a, b = pred(train), pred(valid)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {'train': a, 'valid': b}
")

fit_predict <- function(model, tr, va, enc, seed) {
  y <- tr[[TARGET]]
  if (model %in% c("Ridge", "Elastic_Net")) {
    a <- features(tr, enc, "linear"); b <- features(va, enc, "linear")
    keep <- apply(a, 2, function(z) sd(z) > 0)
    a <- a[, keep, drop = FALSE]; b <- b[, keep, drop = FALSE]
    alpha <- if (model == "Ridge") 0 else 0.5
    m <- glmnet::glmnet(a, y, alpha = alpha, lambda = 0.1, standardize = TRUE)
    return(list(train = as.numeric(predict(m, a, s = 0.1)),
                valid = as.numeric(predict(m, b, s = 0.1))))
  }
  if (model == "Random_Forest") {
    a <- features(tr, enc, "rf"); b <- features(va, enc, "rf")
    m <- ranger::ranger(x = a, y = y, num.trees = 1200, mtry = 8,
                        min.node.size = 8, respect.unordered.factors = "order", seed = seed)
    return(list(train = as.numeric(predict(m, data = a)$predictions),
                valid = as.numeric(predict(m, data = b)$predictions)))
  }
  if (model == "XGBoost") {
    a <- xgboost::xgb.DMatrix(features(tr, enc, "xgb"), label = y)
    b <- xgboost::xgb.DMatrix(features(va, enc, "xgb"))
    m <- xgboost::xgb.train(params = list(objective = "reg:squarederror", tree_method = "hist",
           max_cat_to_onehot = 1L, eta = 0.03, max_depth = 5L,
           min_child_weight = 4, subsample = 0.8, colsample_bytree = 0.8,
           lambda = 10, alpha = 3, seed = seed, nthread = 2L),
           data = a, nrounds = if (QUICK_TEST) 20L else 500L, verbose = 0)
    return(list(train = as.numeric(predict(m, a)), valid = as.numeric(predict(m, b))))
  }
  if (model == "LightGBM") {
    a <- as.matrix(features(tr, enc, "lgb")); b <- as.matrix(features(va, enc, "lgb"))
    ds <- lightgbm::lgb.Dataset(a, label = y, categorical_feature = CAT)
    m <- lightgbm::lgb.train(params = list(objective = "regression", verbosity = -1L,
           learning_rate = 0.03, num_leaves = 31L, min_data_in_leaf = 30L,
           feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1L,
           lambda_l2 = 10, seed = seed, num_threads = 2L), data = ds,
           nrounds = if (QUICK_TEST) 20L else 500L, verbose = -1)
    return(list(train = as.numeric(predict(m, a)), valid = as.numeric(predict(m, b))))
  }
  a <- features(tr, enc, "ft"); b <- features(va, enc, "ft")
  a[[TARGET]] <- y; b[[TARGET]] <- va[[TARGET]]
  pr <- reticulate::py$fit_ft(a, b, NUM, CAT, TARGET,
                             as.integer(seed), if (QUICK_TEST) 2L else 65L)
  list(train = as.numeric(pr$train), valid = as.numeric(pr$valid))
}

# Each taxon appears in exactly one validation fold per repeat.
make_folds <- function(repeat_id) {
  set.seed(SEED + 1000L * repeat_id)
  counts <- d[, .N, by = Taxon]
  counts[, tie := runif(.N)]
  data.table::setorder(counts, -N, tie)
  loads <- integer(K); fold <- integer(nrow(counts))
  for (i in seq_len(nrow(counts))) {
    f <- sample(which(loads == min(loads)), 1L)
    fold[i] <- f; loads[f] <- loads[f] + counts$N[i]
  }
  setNames(fold, counts$Taxon)
}
score <- function(obs, pred) c(R2 = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2),
                               RMSE = sqrt(mean((obs - pred)^2)),
                               nRMSE = 100 * sqrt(mean((obs - pred)^2)) / diff(range(d[[TARGET]])))

rows <- list(); xgb_oof <- list()
for (r in seq_len(REPEATS)) {
  fold_map <- make_folds(r); fold_id <- unname(fold_map[d$Taxon])
  for (model in MODELS) {
    oof <- rep(NA_real_, nrow(d)); train_obs <- train_pred <- numeric()
    for (f in seq_len(K)) {
      iv <- which(fold_id == f); it <- which(fold_id != f)
      tr <- d[it]; va <- d[iv]; enc <- encoder(tr)
      pr <- fit_predict(model, tr, va, enc, SEED + r * 100 + f)
      oof[iv] <- pr$valid
      train_obs <- c(train_obs, tr[[TARGET]]); train_pred <- c(train_pred, pr$train)
    }
    stopifnot(all(is.finite(oof)))
    rows[[length(rows) + 1L]] <- data.table::data.table(
      Model = model, Repeat = r,
      Train_R2 = unname(score(train_obs, train_pred)["R2"]),
      CV_R2 = unname(score(d[[TARGET]], oof)["R2"]),
      CV_RMSE = unname(score(d[[TARGET]], oof)["RMSE"]),
      CV_nRMSE = unname(score(d[[TARGET]], oof)["nRMSE"]))
    if (model == "XGBoost") xgb_oof[[r]] <- data.table::data.table(
      Repeat = r, Row_ID = seq_len(nrow(d)), Taxon = d$Taxon,
      Observed = d[[TARGET]], Predicted = oof)
    cat(sprintf("%s | repeat %d | CV R2 %.3f\n", model, r, rows[[length(rows)]]$CV_R2))
  }
}
metrics <- data.table::rbindlist(rows)
data.table::fwrite(metrics, file.path(OUT, "cv_by_repeat.csv"))
data.table::fwrite(metrics[, lapply(.SD, mean), by = Model, .SDcols = names(metrics)[-(1:2)]],
                   file.path(OUT, "model_comparison.csv"))
data.table::fwrite(data.table::rbindlist(xgb_oof), file.path(OUT, "xgboost_oof.csv"))

# Final XGBoost fit on all curated training observations (used by later modules).
enc <- encoder(d)
x <- xgboost::xgb.DMatrix(features(d, enc, "xgb"), label = d[[TARGET]])
final <- xgboost::xgb.train(params = list(objective = "reg:squarederror", tree_method = "hist",
        max_cat_to_onehot = 1L, eta = 0.03, max_depth = 5L, min_child_weight = 4,
        subsample = 0.8, colsample_bytree = 0.8, lambda = 10, alpha = 3,
        seed = SEED, nthread = 2L), data = x,
        nrounds = if (QUICK_TEST) 20L else 500L, verbose = 0)
xgboost::xgb.save(final, file.path(OUT, "xgboost_model.json"))
saveRDS(list(encoder = enc, numeric = NUM, categorical = CAT, target = TARGET),
        file.path(OUT, "xgboost_preprocessing.rds"))
cat("Finished. Results: ", OUT, "\n")
