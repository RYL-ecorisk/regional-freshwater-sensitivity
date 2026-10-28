# FT-Transformer grid search using the shared 5 x 5 taxon-blocked folds.
# Run after 01_model_training.R; set RETICULATE_PYTHON to the intended Python environment.

rm(list = ls()); gc()
options(stringsAsFactors = FALSE, warn = 1)
RUN_START <- Sys.time()
SCRIPT_VERSION <- "MOA_v12_FT_GRID_24FEATURES_SPECIFIC_METADATA"
PROJECT_DIR <- normalizePath(".", winslash="/", mustWork=TRUE)
RUN_ROOT <- file.path(PROJECT_DIR,"results","model_tuning_run06")
SEED_FILE <- file.path(RUN_ROOT,"00_RUN_SEED.txt")
if (!file.exists(SEED_FILE)) stop("Run 01_model_training.R first to create the shared run seed and folds.")
MASTER_SEED <- suppressWarnings(as.integer(readLines(SEED_FILE,n=1L)))
if (length(MASTER_SEED)!=1L || is.na(MASTER_SEED)) stop("Invalid saved run seed.")
RNGkind("L'Ecuyer-CMRG"); set.seed(MASTER_SEED)

DATA_CANDIDATES <- file.path(PROJECT_DIR,"data",c(
  "Training_DATABASE_MOA_RECHECKED.csv",
  "Training_DATABASE_MOA_RECHECKED_WITH_BARRON_SPECIFIC_20261014.csv"
))
MAIN_FOLD_FILE <- file.path(RUN_ROOT,"shared_folds","FOLD_ASSIGNMENTS.csv")
OUT_DIR <- file.path(RUN_ROOT,"ft_transformer_tuning")
XGB_FIRST_DONE <- file.path(RUN_ROOT,"five_model_tuning","02_MODELS","XGBoost","DONE.txt")
if (!file.exists(XGB_FIRST_DONE))
  stop("Complete the XGBoost tuning stage in 01_model_training.R before running FT-Transformer.")
if (!file.exists(MAIN_FOLD_FILE))
  stop("Shared fold file not found. Run 01_model_training.R first.")

DIR_CONFIG <- file.path(OUT_DIR, "00_CONFIG")
DIR_FOLDS <- file.path(OUT_DIR, "01_FOLDS")
DIR_MODEL <- file.path(OUT_DIR, "02_FT_Transformer")

for (d in c(
  OUT_DIR,
  DIR_CONFIG,
  DIR_FOLDS,
  DIR_MODEL,
  file.path(DIR_MODEL, "checkpoints")
)) {
  dir.create(
    d,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

K_FOLDS <- 5L
N_REPEATS <- 5L

ENABLE_STAGE2_BOUNDARY_RESCUE <- TRUE
MAX_RESCUE_CANDIDATES <- 5L

FT_STAGE1_BUDGET <- 15L
FT_STAGE2_BUDGET <- 15L
FT_PLANNED_TOTAL <- 30L

TARGET <- "log10_Geomean_ug_L"
GROUP_COL <- "Taxon"

CHEM_NUM <- c(
  "logS", "logKow", "logD7_4", "MW", "TPSA", "log_Henry",
  "log_VaporPressure", "MeltingPoint", "HBA", "HBD",
  "AromaticRingCount", "RotatableBondCount", "BertzCT"
)

TRAIT_NUM <- c(
  "Log Body size",
  "Feeding_score",
  "Respiration_score",
  "Locomotion_score",
  "Body_armor_score"
)

TAX_CAT <- c(
  "Phylum", "Class", "Order", "Family", "Genus"
)

MOA_CAT <- "Model_MOA_final"

NUM_VARS <- c(
  CHEM_NUM,
  TRAIT_NUM
)

CAT_VARS <- c(
  TAX_CAT,
  MOA_CAT
)

SCIENCE_PREDICTORS <- c(
  NUM_VARS,
  CAT_VARS
)

FT_PYTHON <- Sys.getenv("RETICULATE_PYTHON", unset = "")
TORCH_WHEEL <- Sys.getenv("TORCH_WHEEL_PATH", unset = "")

# ==============================================================================
# 1. R + PYTHON PACKAGES
# ==============================================================================

required_r <- c(
  "data.table",
  "reticulate"
)

missing_r <- required_r[
  !vapply(
    required_r,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_r)) {
  stop(
    "Missing R package(s): ",
    paste(
      missing_r,
      collapse = ", "
    )
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(reticulate)
})

if (!nzchar(FT_PYTHON) || !file.exists(FT_PYTHON)) {
  stop(
    "FT Python environment not found:\n",
    FT_PYTHON
  )
}

# MUST happen before Python is initialized.
Sys.setenv(
  RETICULATE_PYTHON = FT_PYTHON
)

reticulate::use_python(
  FT_PYTHON,
  required = TRUE
)

py_config_txt <- capture.output(
  reticulate::py_config()
)

writeLines(
  py_config_txt,
  file.path(
    DIR_CONFIG,
    "PYTHON_CONFIG.txt"
  )
)

py_modules <- c(
  "torch",
  "pytorch_tabular",
  "numpy",
  "pandas"
)

missing_py <- py_modules[
  !vapply(
    py_modules,
    reticulate::py_module_available,
    logical(1)
  )
]

if (length(missing_py)) {
  stop(
    "Missing Python package(s) in fttransformer environment: ",
    paste(
      missing_py,
      collapse = ", "
    ),
    "\nExpected Python: ",
    FT_PYTHON,
    "\nIf torch alone is missing, local wheel is available at:\n",
    TORCH_WHEEL
  )
}

torch <- reticulate::import(
  "torch"
)

cat("\n============================================================\n")
cat("FT-TRANSFORMER PYTHON ENVIRONMENT\n")
cat("============================================================\n")
cat("Python : ", FT_PYTHON, "\n", sep = "")
cat("PyTorch: ", as.character(torch$`__version__`), "\n", sep = "")
cat("CUDA   : ", torch$cuda$is_available(), "\n", sep = "")

if (!torch$cuda$is_available()) {
  stop(
    "CUDA is unavailable in the fttransformer environment."
  )
}

cat(
  "GPU    : ",
  torch$cuda$get_device_name(0L),
  "\n",
  sep = ""
)
cat("============================================================\n\n")

# ==============================================================================
# 2. INPUT DATA
# ==============================================================================

DATA_CANDIDATES <- DATA_CANDIDATES[
  file.exists(DATA_CANDIDATES) & !dir.exists(DATA_CANDIDATES)
]
data_path <- DATA_CANDIDATES[1]

if (
  length(data_path) == 0L ||
    is.na(data_path)
) {
  stop(
    "Training data not found. Expected one of:\n",
    paste(
      DATA_CANDIDATES,
      collapse = "\n"
    )
  )
}

DT <- data.table::fread(
  data_path,
  na.strings = c(
    "",
    "NA",
    "NaN"
  ),
  encoding = "UTF-8"
)

# Verify the corrected 10.14 input identity and preserved Barron Specific MOA.
mandatory_moa_cols <- c('DTXSID','CASRN','Model_MOA_final',
                        'Barron_Specific_MOA','Specific_MOA_fixed')
if (!all(mandatory_moa_cols %in% names(DT)))
  stop('Expected updated training data with Barron Specific MOA columns; missing: ',
       paste(setdiff(mandatory_moa_cols,names(DT)),collapse=', '))
if (!identical(unique(as.character(DT[DTXSID=='DTXSID8031865',Model_MOA_final])),
               'Mitochondrial energy disruption'))
  stop('PFOA automatic MOA label differs from the reviewed input; check dataset.')
if (!identical(unique(as.character(DT[DTXSID=='DTXSID6032437',Barron_Specific_MOA])),
               'Nonpolar'))
  stop('Naptalam Barron Specific MOA should be Nonpolar; check dataset.')

required_cols <- unique(
  c(
    TARGET,
    GROUP_COL,
    NUM_VARS,
    CAT_VARS
  )
)

missing_cols <- setdiff(
  required_cols,
  names(DT)
)

if (length(missing_cols)) {
  stop(
    "Training data missing required columns: ",
    paste(
      missing_cols,
      collapse = ", "
    )
  )
}

for (v in c(
  TARGET,
  NUM_VARS
)) {
  data.table::set(
    DT,
    j = v,
    value = suppressWarnings(
      as.numeric(
        DT[[v]]
      )
    )
  )
}

DT <- DT[
  is.finite(get(TARGET)) &
    !is.na(get(GROUP_COL)) &
    trimws(
      as.character(
        get(GROUP_COL)
      )
    ) != ""
]

DT[, (GROUP_COL) :=
     trimws(
       as.character(
         get(GROUP_COL)
       )
     )]

DT[, Row_ID := .I]

cat(
  "Rows: ",
  nrow(DT),
  " | Taxa: ",
  uniqueN(DT[[GROUP_COL]]),
  " | Predictors: 24\n\n",
  sep = ""
)

# ==============================================================================
# 3. HELPERS
# ==============================================================================

clean_cat <- function(x) {
  x <- as.character(x)
  x <- gsub(
    "\u00A0",
    " ",
    x,
    fixed = TRUE
  )
  x <- trimws(x)

  x[
    is.na(x) |
      x == ""
  ] <- "__MISSING__"

  x
}

metric_set <- function(
    obs,
    pred
) {
  ok <- is.finite(obs) &
    is.finite(pred)

  obs <- obs[ok]
  pred <- pred[ok]

  if (length(obs) < 3L) {
    return(
      c(
        R2 = NA_real_,
        Pearson_R2 = NA_real_,
        RMSE = NA_real_,
        MAE = NA_real_,
        nRMSE = NA_real_
      )
    )
  }

  sse <- sum(
    (obs - pred)^2
  )

  sst <- sum(
    (obs - mean(obs))^2
  )

  r2 <- if (sst > 0) {
    1 - sse / sst
  } else {
    NA_real_
  }

  pr2 <- suppressWarnings(
    stats::cor(
      obs,
      pred,
      use = "complete.obs"
    )^2
  )

  rmse <- sqrt(
    mean(
      (obs - pred)^2
    )
  )

  mae <- mean(
    abs(
      obs - pred
    )
  )

  obs_sd <- stats::sd(
    obs
  )

  nrmse <- if (
    is.finite(obs_sd) &&
      obs_sd > 0
  ) {
    rmse / obs_sd
  } else {
    NA_real_
  }

  c(
    R2 = r2,
    Pearson_R2 = pr2,
    RMSE = rmse,
    MAE = mae,
    nRMSE = nrmse
  )
}

summarize_candidate <- function(
    metrics_dt
) {
  metrics_dt[, .(
    Mean_Train_R2 = mean(
      Train_R2,
      na.rm = TRUE
    ),
    SD_Train_R2 = stats::sd(
      Train_R2,
      na.rm = TRUE
    ),
    Mean_CV_R2 = mean(
      CV_R2,
      na.rm = TRUE
    ),
    SD_CV_R2 = stats::sd(
      CV_R2,
      na.rm = TRUE
    ),
    Mean_Train_RMSE = mean(
      Train_RMSE,
      na.rm = TRUE
    ),
    Mean_CV_RMSE = mean(
      CV_RMSE,
      na.rm = TRUE
    ),
    Mean_Train_MAE = mean(
      Train_MAE,
      na.rm = TRUE
    ),
    Mean_CV_MAE = mean(
      CV_MAE,
      na.rm = TRUE
    ),
    Mean_Train_nRMSE = mean(
      Train_nRMSE,
      na.rm = TRUE
    ),
    Mean_CV_nRMSE = mean(
      CV_nRMSE,
      na.rm = TRUE
    ),
    Mean_R2_Gap = mean(
      R2_gap,
      na.rm = TRUE
    ),
    SD_R2_Gap = stats::sd(
      R2_gap,
      na.rm = TRUE
    ),
    Max_R2_Gap = max(
      R2_gap,
      na.rm = TRUE
    )
  )]
}

rank_summary <- function(x) {
  z <- copy(x)

  z[
    !is.finite(Mean_CV_R2),
    Mean_CV_R2 := -Inf
  ]

  setorder(
    z,
    -Mean_CV_R2,
    Mean_CV_RMSE,
    Mean_R2_Gap,
    Candidate
  )

  z
}

safe_dir <- function(path) {
  dir.create(
    path,
    recursive = TRUE,
    showWarnings = FALSE
  )

  path
}

candidate_dir_name <- function(
    candidate_id
) {
  sprintf(
    "candidate_%04d",
    as.integer(
      candidate_id
    )
  )
}

make_balanced_taxon_folds <- function(
    dt,
    k,
    seed
) {
  set.seed(seed)

  cnt <- dt[
    ,
    .N,
    by = Taxon
  ]

  cnt[, tie := runif(.N)]

  setorder(
    cnt,
    -N,
    tie
  )

  loads <- rep(
    0L,
    k
  )

  fold_id <- integer(
    nrow(cnt)
  )

  for (
    i in seq_len(
      nrow(cnt)
    )
  ) {
    candidates <- which(
      loads ==
        min(loads)
    )

    chosen <- if (
      length(candidates) == 1L
    ) {
      candidates
    } else {
      sample(
        candidates,
        1L
      )
    }

    fold_id[i] <- chosen
    loads[chosen] <-
      loads[chosen] +
      cnt$N[i]
  }

  cnt[, Fold := fold_id]
  cnt[, tie := NULL]

  cnt
}

fit_encoder <- function(
    train_dt
) {
  enc <- list()

  enc$medians <- setNames(
    vapply(
      NUM_VARS,
      function(v) {
        z <- suppressWarnings(
          as.numeric(
            train_dt[[v]]
          )
        )

        z <- z[
          is.finite(z)
        ]

        if (!length(z)) {
          0
        } else {
          stats::median(z)
        }
      },
      numeric(1)
    ),
    NUM_VARS
  )

  enc$cat_levels <- setNames(
    lapply(
      CAT_VARS,
      function(v) {
        lev <- sort(
          unique(
            clean_cat(
              train_dt[[v]]
            )
          )
        )

        unique(
          c(
            lev,
            "__NOVEL__"
          )
        )
      }
    ),
    CAT_VARS
  )

  enc
}

make_numeric_block <- function(
    dt,
    enc
) {
  x <- matrix(
    NA_real_,
    nrow = nrow(dt),
    ncol = length(NUM_VARS),
    dimnames = list(
      NULL,
      NUM_VARS
    )
  )

  for (
    j in seq_along(
      NUM_VARS
    )
  ) {
    v <- NUM_VARS[j]

    z <- suppressWarnings(
      as.numeric(
        dt[[v]]
      )
    )

    bad <- !is.finite(z)

    if (any(bad)) {
      z[bad] <-
        enc$medians[[v]]
    }

    x[, j] <- z
  }

  x
}

make_cat_strings <- function(
    dt,
    enc
) {
  out <- vector(
    "list",
    length(CAT_VARS)
  )

  names(out) <-
    CAT_VARS

  for (v in CAT_VARS) {
    z <- clean_cat(
      dt[[v]]
    )

    lev <-
      enc$cat_levels[[v]]

    z[
      !z %in% lev
    ] <- "__NOVEL__"

    out[[v]] <- z
  }

  as.data.frame(
    out,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

make_ft_dataframe <- function(
    dt,
    enc,
    include_target = TRUE
) {
  num <- as.data.frame(
    make_numeric_block(
      dt,
      enc
    ),
    check.names = FALSE
  )

  cats <- make_cat_strings(
    dt,
    enc
  )

  x <- data.frame(
    num,
    cats,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  if (include_target) {
    x[[TARGET]] <-
      as.numeric(
        dt[[TARGET]]
      )
  }

  x
}

# 4. Shared 5 x 5 taxon-blocked folds

local_fold_file <- file.path(DIR_FOLDS,"FOLD_ASSIGNMENTS.csv")

# ===========================================================================
# RUN06 shared 5x5 Taxon-blocked folds under:
# results/model_tuning_run06/shared_folds/FOLD_ASSIGNMENTS.csv
# Both tuning scripts use the same dataset, saved run seed and fold assignments.
# Run either script first; the other must reuse the same assignment file.
# ===========================================================================
FOLD_SEED <- MASTER_SEED
create_shared_folds <- function(dt, path, k=5L, repeats=5L) {
  if (!file.exists(path)) {
    dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
    pieces <- lapply(seq_len(repeats), function(r) {
      set.seed(FOLD_SEED + 1000L * r)
      cnt <- dt[, .N, by=Taxon]
      cnt[,tie:=runif(.N)]; data.table::setorder(cnt,-N,tie)
      load <- integer(k)
      assigned <- integer(nrow(cnt))
      for (i in seq_len(nrow(cnt))) {
        opts <- which(load==min(load))
        f <- if (length(opts)==1L) opts else sample(opts,1L)
        assigned[i] <- f
        load[f] <- load[f]+cnt$N[i]
      }
      cnt[,`:=`(Fold=assigned,Repeat=r,tie=NULL)]
      data.table::setcolorder(cnt,c('Repeat','Fold','Taxon','N'))
      cnt
    })
    dat <- data.table::rbindlist(pieces)
    # Never overwrite an existing fold file; same path is shared by both runs.
    if (!file.exists(path)) data.table::fwrite(dat,path)
  }
  fa <- data.table::fread(path)
  if (!all(c('Repeat','Fold','Taxon','N') %in% names(fa)) ||
      nrow(fa)!=repeats*data.table::uniqueN(dt$Taxon) ||
      anyDuplicated(fa,by=c('Repeat','Taxon')) ||
      !setequal(fa$Repeat,seq_len(repeats)) ||
      !setequal(fa$Fold,seq_len(k)))
    stop('Shared fold file is missing/corrupt/incompatible: ',path)
  counts <- dt[, .N, by=Taxon]
  for (r in seq_len(repeats)) {
    rr <- fa[Repeat==r]
    if (!setequal(rr$Taxon,counts$Taxon) ||
        any(rr$N!=counts$N[match(rr$Taxon,counts$Taxon)]) ||
        any(tabulate(rr$Fold,nbins=k)==0L))
      stop('Fold counts/taxa changed in repeat ',r,': ',path)
  }
  fa
}

# Identical full-data and fold membership for BOTH training scripts.
stopifnot(nrow(DT)==7043L, data.table::uniqueN(DT$DTXSID)==2109L,
          length(SCIENCE_PREDICTORS)==24L)
if (any(DT[, .(n=uniqueN(Model_MOA_final)), by=DTXSID]$n != 1L))
  stop("Conflicting Broad MOA assignments within a DTXSID.")
FOLDS <- create_shared_folds(DT,MAIN_FOLD_FILE,k=K_FOLDS,repeats=N_REPEATS)
# Verify even a pre-existing shared file contains RUN06 seed-specific fold memberships.
expected_run05 <- data.table::rbindlist(lapply(seq_len(N_REPEATS), function(r) {
  set.seed(FOLD_SEED + 1000L * r)
  cc <- DT[, .N, by=Taxon]
  cc[, tie := runif(.N)]; data.table::setorder(cc, -N, tie)
  loads <- integer(K_FOLDS); ff <- integer(nrow(cc))
  for (i in seq_len(nrow(cc))) {
    possible <- which(loads == min(loads))
    pick <- if(length(possible)==1L) possible else sample(possible, 1L)
    ff[i] <- pick; loads[pick] <- loads[pick] + cc$N[i]
  }
  cc[, `:=`(Repeat=r, Fold=ff, tie=NULL)]
  cc[, .(Repeat,Fold,Taxon,N)]
}))
check_fold <- function(z) {
  zz <- data.table::copy(z)[, .(Repeat,Fold,Taxon,N)]
  data.table::setorder(zz, Repeat, Taxon)
  zz
}
if (!isTRUE(all.equal(as.data.frame(check_fold(FOLDS)),
                     as.data.frame(check_fold(expected_run05)),
                     check.attributes=FALSE)))
  stop('Shared folds do not match the saved run seed. ',
       'Do not mix previous runs; check ', MAIN_FOLD_FILE)
rm(expected_run05,check_fold)
fwrite(FOLDS,local_fold_file)
cat("Shared FT/5-model fold assignments: ",MAIN_FOLD_FILE,"\n",sep="")
# Resume guard: do not reuse checkpoints after any data/fold/feature change.
manifest_path <- file.path(DIR_CONFIG,'V12_FT_RETRAIN_MANIFEST.rds')
manifest <- list(
  version=SCRIPT_VERSION,
  data_md5=unname(tools::md5sum(data_path)),
  fold_md5=unname(tools::md5sum(MAIN_FOLD_FILE)),
  predictors=SCIENCE_PREDICTORS,
  master_seed=MASTER_SEED,
  fold_seed=FOLD_SEED,
  specific_moa_used_as_predictor=FALSE,
  folds=K_FOLDS,
  repeats=N_REPEATS,
  budget=c(FT_STAGE1_BUDGET,FT_STAGE2_BUDGET,12L),
  python=FT_PYTHON
)
if (file.exists(manifest_path)) {
  if (!identical(readRDS(manifest_path),manifest))
    stop('Training data, CV folds or FT settings changed; create a NEW output directory.')
} else saveRDS(manifest,manifest_path)

# ============================================================================
# RUN06 AUDIT EXPORTS: ACTUAL 24-PREDICTOR MATRIX + SPECIFIC MOA METADATA
# These files are not the fold-specific imputed/encoded GPU tensors.
# Fold preprocessing is fitted on the training partition only (no CV leakage).
# ============================================================================
MATRIX_DIR <- file.path(OUT_DIR, "00_TRAINING_MATRIX_AND_SPECIFIC_MOA")
dir.create(MATRIX_DIR,recursive=TRUE,showWarnings=FALSE)
SPECIFIC_META_COLS <- intersect(c(
  'Specific_MOA_fixed','Barron_Specific_MOA','Barron_Specific_MOA_source_raw',
  'Specific_MOA_Fig2d_14','Specific_MOA_AnalysisScope',
  'Broad_MOA_Barron_raw','Barron_Broad_MOA_source_raw',
  'Barron_Source_CAS','Barron_Source_Chemical_Name',
  'Barron_Match_Resolution','MOA_Source_recomputed',
  'MOA_Match_QA','MOA_Match_Flags'
),names(DT))
stopifnot(all(c('Barron_Specific_MOA','Specific_MOA_fixed') %in% SPECIFIC_META_COLS))
RUN06_MATRIX <- DT[, c('Row_ID',TARGET,SCIENCE_PREDICTORS),with=FALSE]
stopifnot(nrow(RUN06_MATRIX)==7043L, ncol(RUN06_MATRIX)==26L)
RUN06_MATRIX_AUDIT <- DT[, unique(c(
  'Row_ID','DTXSID','CASRN','Chemical_Name','Taxon', TARGET,
  SCIENCE_PREDICTORS,SPECIFIC_META_COLS
)),with=FALSE]
RUN06_CHEM_MOA <- unique(DT[, unique(c(
  'DTXSID','CASRN','Chemical_Name','Model_MOA_final',SPECIFIC_META_COLS
)),with=FALSE])
if(nrow(RUN06_CHEM_MOA)!=2109L || anyDuplicated(RUN06_CHEM_MOA$DTXSID))
  stop('Chemical-level Specific MOA audit must have exactly 2109 unique DTXSIDs.')
data.table::fwrite(RUN06_MATRIX,file.path(MATRIX_DIR,'RUN06_TRAINING_MATRIX_24_PREDICTORS.csv'))
data.table::fwrite(RUN06_MATRIX_AUDIT,file.path(MATRIX_DIR,'RUN06_TRAINING_MATRIX_WITH_SPECIFIC_MOA_METADATA.csv'))
data.table::fwrite(RUN06_CHEM_MOA,file.path(MATRIX_DIR,'RUN06_CHEMICAL_SPECIFIC_MOA_2109.csv'))
writeLines(c(
  paste0('SOURCE=',normalizePath(data_path,winslash='/',mustWork=TRUE)),
  paste0('INPUT_MD5=',unname(tools::md5sum(data_path))),
  paste0('FOLDS_MD5=',unname(tools::md5sum(MAIN_FOLD_FILE))),
  paste0('MASTER_SEED=',MASTER_SEED),paste0('FOLD_SEED=',FOLD_SEED),
  paste0('N_TRAIN_ROWS=',nrow(DT)),paste0('N_UNIQUE_DTXSID=',uniqueN(DT$DTXSID)),
  paste0('N_PREDICTORS=',length(SCIENCE_PREDICTORS)),
  paste0('PREDICTOR_COLS=',paste(SCIENCE_PREDICTORS,collapse=' | ')),
  paste0('SPECIFIC_METADATA_COLS=',paste(SPECIFIC_META_COLS,collapse=' | ')),
  'Specific MOA fields are retained ONLY for post hoc/SHAP subgroup auditing.',
  'Specific MOA is NOT a model predictor; do not add it to CAT_VARS.',
  'No missing Specific MOA values were inferred or filled.',
  'Matrix values are raw; fold-wise imputation and categorical encoding happen separately.'
),file.path(MATRIX_DIR,'RUN06_MATRIX_FEATURE_MANIFEST.txt'))
stopifnot(identical(SCIENCE_PREDICTORS,c(NUM_VARS,CAT_VARS)))
cat('RUN06 matrices and Barron Specific MOA exported: ',MATRIX_DIR,'\n',sep='')

if (
  !all(
    c(
      "Repeat",
      "Fold",
      "Taxon"
    ) %in%
      names(FOLDS)
  )
) {
  stop(
    "Fold file lacks Repeat/Fold/Taxon."
  )
}

for (
  r in seq_len(
    N_REPEATS
  )
) {
  fa <- FOLDS[
    Repeat == r
  ]

  if (
    !setequal(
      fa$Taxon,
      unique(
        DT[[GROUP_COL]]
      )
    )
  ) {
    stop(
      "Fold file taxa do not match current training data in Repeat ",
      r
    )
  }

  for (
    f in seq_len(
      K_FOLDS
    )
  ) {
    va_taxa <- fa[
      Fold == f,
      Taxon
    ]

    tr_taxa <- fa[
      Fold != f,
      Taxon
    ]

    if (
      length(
        intersect(
          va_taxa,
          tr_taxa
        )
      )
    ) {
      stop(
        "Taxon leakage detected in Repeat ",
        r,
        ", Fold ",
        f
      )
    }
  }
}

cat(
  "Taxon leakage check: PASS\n\n"
)

# ==============================================================================
# 5. FT-TRANSFORMER PYTHON FUNCTIONS
# ==============================================================================

reticulate::py_run_string("
import os
import gc
import random
import shutil
import numpy as np
import torch

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _pred_col(df, target):
    expected = target + '_prediction'
    if expected in df.columns:
        return expected

    cols = [
        c for c in df.columns
        if c.endswith('_prediction')
    ]

    if not cols:
        raise RuntimeError(
            'FT prediction column not found.'
        )

    return cols[0]


def _prep(df, cat_cols):
    df = df.copy()

    for c in cat_cols:
        df[c] = (
            df[c]
            .fillna('__MISSING__')
            .astype(str)
        )

    return df


def _build_model(
    num_cols,
    cat_cols,
    target,
    embed,
    heads,
    blocks,
    dropout,
    lr,
    wd,
    batch,
    epochs,
    seed
):
    dc = DataConfig(
        target=[target],
        continuous_cols=num_cols,
        categorical_cols=cat_cols,
        normalize_continuous_features=True,
        handle_unknown_categories=True,
        handle_missing_values=True,
        num_workers=0,
    )

    tc = TrainerConfig(
        batch_size=int(batch),
        max_epochs=int(epochs),
        min_epochs=int(epochs),
        accelerator='gpu',
        devices=1,
        early_stopping=None,
        checkpoints=None,
        load_best=False,
        progress_bar='none',
        seed=int(seed),
        trainer_kwargs={
            'enable_model_summary': False
        },
    )

    oc = OptimizerConfig(
        optimizer='AdamW',
        optimizer_params={
            'weight_decay': float(wd)
        },
    )

    mc = FTTransformerConfig(
        task='regression',
        input_embed_dim=int(embed),
        num_heads=int(heads),
        num_attn_blocks=int(blocks),
        attn_dropout=float(dropout),
        add_norm_dropout=float(dropout),
        ff_dropout=float(dropout),
        ff_hidden_multiplier=4,
        learning_rate=float(lr),
        seed=int(seed),
    )

    return TabularModel(
        data_config=dc,
        trainer_config=tc,
        optimizer_config=oc,
        model_config=mc,
        verbose=False,
        suppress_lightning_logger=True,
    )


def ecotox_ft_fit_predict(
    train_df,
    valid_df,
    num_cols,
    cat_cols,
    target,
    seed,
    embed,
    heads,
    blocks,
    dropout,
    lr,
    wd,
    batch,
    epochs
):
    num_cols = _as_list(num_cols)
    cat_cols = _as_list(cat_cols)

    train_df = _prep(
        train_df,
        cat_cols
    )

    valid_df = _prep(
        valid_df,
        cat_cols
    )

    target = str(target)
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = _build_model(
        num_cols=num_cols,
        cat_cols=cat_cols,
        target=target,
        embed=embed,
        heads=heads,
        blocks=blocks,
        dropout=dropout,
        lr=lr,
        wd=wd,
        batch=batch,
        epochs=epochs,
        seed=seed,
    )

    model.fit(
        train=train_df,
        seed=seed
    )

    ptr = model.predict(
        train_df.drop(
            columns=[target]
        ),
        progress_bar=None
    )

    pva = model.predict(
        valid_df.drop(
            columns=[target]
        ),
        progress_bar=None
    )

    pc = _pred_col(
        ptr,
        target
    )

    out = {
        'train':
            ptr[pc]
            .to_numpy()
            .astype(float),
        'valid':
            pva[pc]
            .to_numpy()
            .astype(float),
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return out


def ecotox_ft_fit_save(
    train_df,
    num_cols,
    cat_cols,
    target,
    seed,
    embed,
    heads,
    blocks,
    dropout,
    lr,
    wd,
    batch,
    epochs,
    save_dir
):
    num_cols = _as_list(num_cols)
    cat_cols = _as_list(cat_cols)

    train_df = _prep(
        train_df,
        cat_cols
    )

    target = str(target)
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = _build_model(
        num_cols=num_cols,
        cat_cols=cat_cols,
        target=target,
        embed=embed,
        heads=heads,
        blocks=blocks,
        dropout=dropout,
        lr=lr,
        wd=wd,
        batch=batch,
        epochs=epochs,
        seed=seed,
    )

    model.fit(
        train=train_df,
        seed=seed
    )

    ptr = model.predict(
        train_df.drop(
            columns=[target]
        ),
        progress_bar=None
    )

    pc = _pred_col(
        ptr,
        target
    )

    if os.path.exists(save_dir):
        shutil.rmtree(
            save_dir,
            ignore_errors=True
        )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    try:
        model.save_model(
            save_dir,
            inference_only=True
        )
    except TypeError:
        model.save_model(
            save_dir
        )

    out = {
        'train':
            ptr[pc]
            .to_numpy()
            .astype(float),
        'model_dir':
            save_dir,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return out
")

# ==============================================================================
# 6. INTEGER-CODED COARSE + FINE FT GRIDS
# ==============================================================================

ft_epochs_from_lr <- function(
    lr_x1e6
) {
  lr_x1e6 <- as.integer(
    lr_x1e6
  )

  if (lr_x1e6 <= 200L) {
    return(90L)
  }

  if (lr_x1e6 <= 400L) {
    return(80L)
  }

  if (lr_x1e6 <= 800L) {
    return(65L)
  }

  55L
}

add_meta <- function(
    z,
    stage,
    start_id,
    reason
) {
  z <- unique(
    as.data.table(z)
  )

  if (!nrow(z)) {
    return(z)
  }

  z[, Candidate :=
      seq.int(
        start_id,
        length.out = .N
      )]

  z[, `:=`(
    Stage = stage,
    Generation_Reason = reason
  )]

  setcolorder(
    z,
    c(
      "Candidate",
      "Stage",
      "Generation_Reason",
      setdiff(
        names(z),
        c(
          "Candidate",
          "Stage",
          "Generation_Reason"
        )
      )
    )
  )

  z
}

ft_param_cols <- c(
  "embed_dim",
  "heads",
  "blocks",
  "dropout_pct",
  "lr_x1e6",
  "weight_decay_x1e6",
  "batch_size",
  "epochs"
)


# ------------------------------------------------------------------------------
# INTEGER-CODED FT SEARCH HELPERS
# ------------------------------------------------------------------------------
# Search-table columns are integer-valued:
#   dropout_pct       = 20  -> dropout = 0.20
#   lr_x1e6           = 300 -> learning rate = 0.000300
#   weight_decay_x1e6 = 30  -> weight decay = 0.000030
#
# Candidate IDs:
#   Stage 1 coarse : 1, 2, ...
#   Stage 2 fine   : 1001, 1002, ...
#   Boundary rescue: 2001, 2002, ...

format_ft_integer_params <- function(pars) {
  paste(
    vapply(
      ft_param_cols,
      function(p) {
        paste0(
          p,
          "=",
          as.character(
            as.integer(
              round(pars[[p]][1])
            )
          )
        )
      },
      character(1)
    ),
    collapse = ", "
  )
}

assert_ft_integer_grid <- function(dt) {
  for (p in ft_param_cols) {
    x <- dt[[p]]

    if (!is.numeric(x) && !is.integer(x)) {
      stop(
        "FT search parameter is not numeric: ",
        p
      )
    }

    if (
      any(
        is.finite(x) &
          abs(x - round(x)) > 1e-12
      )
    ) {
      stop(
        "FT search grid is not integer-coded: ",
        p
      )
    }
  }

  invisible(TRUE)
}

make_stage1_ft <- function() {
  # FORMAL Stage 1: 3 embedding dimensions x 5 learning rates = 15 candidates.
  z <- CJ(
    embed_dim = c(
      64L, 96L, 128L
    ),
    lr_x1e6 = c(
      150L, 300L, 500L, 800L, 1200L
    )
  )

  z[, `:=`(
    heads = 8L,
    blocks = 3L,
    dropout_pct = 15L,
    weight_decay_x1e6 = 30L,
    batch_size = 256L,
    epochs = vapply(
      lr_x1e6,
      ft_epochs_from_lr,
      integer(1)
    )
  )]

  setcolorder(
    z,
    ft_param_cols
  )

  stopifnot(
    nrow(z) ==
      FT_STAGE1_BUDGET
  )

  assert_ft_integer_grid(z)

  add_meta(
    z,
    "Stage1_Coarse",
    1L,
    "FORMAL: 3 embed x 5 learning-rate = 15 candidates"
  )
}

ft_signature <- function(z) {
  do.call(
    paste,
    c(
      z[, ..ft_param_cols],
      sep = "|"
    )
  )
}

remove_existing_ft <- function(
    proposed,
    existing
) {
  proposed <- unique(
    proposed,
    by = ft_param_cols
  )

  if (!nrow(existing)) {
    return(proposed)
  }

  ps <- ft_signature(
    proposed
  )

  es <- ft_signature(
    existing
  )

  proposed[
    !ps %in% es
  ]
}


select_ft_local_grid <- function(
    pool,
    best,
    existing,
    n_take
) {
  pool <- remove_existing_ft(
    pool,
    existing
  )

  if (!nrow(pool)) {
    return(pool)
  }

  scales <- c(
    embed_dim = 32,
    heads = 4,
    blocks = 1,
    dropout_pct = 5,
    lr_x1e6 = max(
      100,
      as.numeric(best$lr_x1e6) / 2
    ),
    weight_decay_x1e6 = 30,
    batch_size = 256,
    epochs = 20
  )

  score <- rep(0, nrow(pool))

  for (p in names(scales)) {
    score <- score +
      abs(
        as.numeric(pool[[p]]) -
          as.numeric(best[[p]][1])
      ) / as.numeric(scales[[p]])
  }

  pool[, Fine_Distance := score]

  data.table::setorderv(
    pool,
    c("Fine_Distance", ft_param_cols),
    c(1L, rep(1L, length(ft_param_cols)))
  )

  if (nrow(pool) < n_take) {
    stop(
      "FT fine-grid pool has only ",
      nrow(pool),
      " new candidates but budget requires ",
      n_take
    )
  }

  out <- pool[seq_len(n_take)]
  out[, Fine_Distance := NULL]

  out
}


make_stage2_ft <- function(
    best,
    existing
) {
  e <- as.integer(best$embed_dim)
  b <- as.integer(best$blocks)
  d <- as.integer(best$dropout_pct)
  lr <- as.integer(best$lr_x1e6)
  wd <- as.integer(best$weight_decay_x1e6)

  embed_levels <- unique(
    pmax(
      32L,
      e + c(-64L, -32L, 0L, 32L, 64L)
    )
  )

  head_levels <- c(4L, 8L)

  block_levels <- unique(
    pmax(
      2L,
      pmin(
        6L,
        b + c(-1L, 0L, 1L)
      )
    )
  )

  dropout_levels <- unique(
    pmax(
      5L,
      pmin(
        35L,
        d + c(-10L, -5L, 0L, 5L, 10L)
      )
    )
  )

  lr_levels <- unique(
    pmax(
      50L,
      pmin(
        2500L,
        round(
          lr * c(0.50, 0.75, 1.00, 1.50, 2.00)
        )
      )
    )
  )

  wd_levels <- unique(
    pmax(
      1L,
      pmin(
        300L,
        c(
          max(1L, round(wd / 3)),
          wd,
          round(wd * 3)
        )
      )
    )
  )

  batch_levels <- c(256L, 512L)

  pool <- CJ(
    embed_dim = as.integer(embed_levels),
    heads = as.integer(head_levels),
    blocks = as.integer(block_levels),
    dropout_pct = as.integer(dropout_levels),
    lr_x1e6 = as.integer(lr_levels),
    weight_decay_x1e6 = as.integer(wd_levels),
    batch_size = as.integer(batch_levels)
  )

  pool <- pool[
    embed_dim %% heads == 0L
  ]

  pool[, epochs := vapply(
    lr_x1e6,
    ft_epochs_from_lr,
    integer(1)
  )]

  setcolorder(
    pool,
    ft_param_cols
  )

  z <- select_ft_local_grid(
    pool,
    best,
    existing,
    FT_STAGE2_BUDGET
  )

  if (nrow(z) != FT_STAGE2_BUDGET) {
    stop(
      "FT Stage-2 budget mismatch: expected ",
      FT_STAGE2_BUDGET,
      ", got ",
      nrow(z)
    )
  }

  assert_ft_integer_grid(z)

  add_meta(
    z,
    "Stage2_Fine",
    1001L,
    paste0(
      "FORMAL local FT fine grid: exactly ",
      FT_STAGE2_BUDGET,
      " NEW candidates"
    )
  )
}

make_ft_rescue <- function(
    best,
    all_candidates
) {
  if (
    !ENABLE_STAGE2_BOUNDARY_RESCUE
  ) {
    return(
      data.table()
    )
  }

  props <- list()
  ii <- 0L

  add_one <- function(
      changes,
      reason
  ) {
    ii <<- ii + 1L

    z <- copy(
      best[
        1,
        ..ft_param_cols
      ]
    )

    for (
      nm in names(changes)
    ) {
      z[, (nm) :=
          changes[[nm]]]
    }

    z[, Rescue_Reason :=
        reason]

    props[[ii]] <<- z
  }

  e <- as.integer(
    best$embed_dim
  )

  b <- as.integer(
    best$blocks
  )

  d <- as.integer(
    best$dropout_pct
  )

  lr <- as.integer(
    best$lr_x1e6
  )

  if (
    e == min(
      all_candidates$embed_dim
    ) &&
      e > 24L
  ) {
    add_one(
      list(
        embed_dim =
          as.integer(
            floor(
              (e - 16L) / 8L
            ) * 8L
          )
      ),
      "embed lower boundary"
    )
  }

  if (
    e == max(
      all_candidates$embed_dim
    ) &&
      e < 256L
  ) {
    add_one(
      list(
        embed_dim =
          as.integer(
            ceiling(
              (e + 32L) / 8L
            ) * 8L
          )
      ),
      "embed upper boundary"
    )
  }

  if (
    b == max(
      all_candidates$blocks
    ) &&
      b < 6L
  ) {
    add_one(
      list(
        blocks =
          b + 1L
      ),
      "blocks upper boundary"
    )
  }

  if (
    d == min(
      all_candidates$dropout_pct
    ) &&
      d > 0L
  ) {
    add_one(
      list(
        dropout_pct =
          max(
            0L,
            d - 5L
          )
      ),
      "dropout lower boundary"
    )
  }

  if (
    d == max(
      all_candidates$dropout_pct
    ) &&
      d < 40L
  ) {
    add_one(
      list(
        dropout_pct =
          min(
            40L,
            d + 5L
          )
      ),
      "dropout upper boundary"
    )
  }

  if (
    lr == min(
      all_candidates$lr_x1e6
    ) &&
      lr > 50L
  ) {
    new_lr <- max(
      50L,
      as.integer(
        floor(
          lr / 2
        )
      )
    )

    add_one(
      list(
        lr_x1e6 = new_lr,
        epochs =
          ft_epochs_from_lr(
            new_lr
          )
      ),
      "learning-rate lower boundary"
    )
  }

  if (
    lr == max(
      all_candidates$lr_x1e6
    ) &&
      lr < 2500L
  ) {
    new_lr <- min(
      2500L,
      as.integer(
        ceiling(
          lr * 1.5
        )
      )
    )

    add_one(
      list(
        lr_x1e6 = new_lr,
        epochs =
          ft_epochs_from_lr(
            new_lr
          )
      ),
      "learning-rate upper boundary"
    )
  }

  if (!length(props)) {
    return(
      data.table()
    )
  }

  z <- rbindlist(
    props,
    fill = TRUE
  )

  z[, Rescue_Reason := NULL]

  z <- remove_existing_ft(
    z,
    all_candidates
  )

  z <- head(
    z,
    MAX_RESCUE_CANDIDATES
  )

  if (!nrow(z)) {
    return(z)
  }

  assert_ft_integer_grid(z)

  add_meta(
    z,
    "Stage2_Boundary_Rescue",
    2001L,
    "One-pass targeted FT boundary rescue"
  )
}

decode_ft <- function(z) {
  z <- copy(
    as.data.table(z)
  )

  z[, dropout_actual :=
      dropout_pct / 100]

  z[, learning_rate_actual :=
      lr_x1e6 / 1e6]

  z[, weight_decay_actual :=
      weight_decay_x1e6 / 1e6]

  z
}

# ==============================================================================
# 7. FIT/PREDICT ONE FT CANDIDATE
# ==============================================================================

fit_predict_ft <- function(
    pars,
    tr_dt,
    va_dt,
    enc,
    seed
) {
  tr_ft <- make_ft_dataframe(
    tr_dt,
    enc,
    include_target = TRUE
  )

  va_ft <- make_ft_dataframe(
    va_dt,
    enc,
    include_target = TRUE
  )

  pr <- reticulate::py$ecotox_ft_fit_predict(
    train_df = tr_ft,
    valid_df = va_ft,
    num_cols = NUM_VARS,
    cat_cols = CAT_VARS,
    target = TARGET,
    seed = as.integer(seed),
    embed = as.integer(
      pars$embed_dim
    ),
    heads = as.integer(
      pars$heads
    ),
    blocks = as.integer(
      pars$blocks
    ),
    dropout = as.numeric(
      pars$dropout_pct
    ) / 100,
    lr = as.numeric(
      pars$lr_x1e6
    ) / 1e6,
    wd = as.numeric(
      pars$weight_decay_x1e6
    ) / 1e6,
    batch = as.integer(
      pars$batch_size
    ),
    epochs = as.integer(
      pars$epochs
    )
  )

  list(
    train = as.numeric(
      pr$train
    ),
    valid = as.numeric(
      pr$valid
    )
  )
}

# ==============================================================================
# 8. REPEAT-LEVEL CHECKPOINTED EVALUATION
# ==============================================================================

evaluate_candidate <- function(
    pars
) {
  cid <- as.integer(
    pars$Candidate
  )

  cdir <- safe_dir(
    file.path(
      DIR_MODEL,
      "checkpoints",
      candidate_dir_name(
        cid
      )
    )
  )

  combined_metrics_file <- file.path(
    cdir,
    "repeat_metrics.csv"
  )

  summary_file <- file.path(
    cdir,
    "summary.csv"
  )

  if (
    file.exists(
      combined_metrics_file
    ) &&
      file.exists(
        summary_file
      )
  ) {
    cat(
      "    candidate ",
      cid,
      " already complete -> skip\n",
      sep = ""
    )

    return(
      fread(
        summary_file
      )
    )
  }

  repeat_metrics <- vector(
    "list",
    N_REPEATS
  )

  for (
    r in seq_len(
      N_REPEATS
    )
  ) {
    metric_file <- file.path(
      cdir,
      sprintf(
        "repeat_%02d_metrics.csv",
        r
      )
    )

    oof_file <- file.path(
      cdir,
      sprintf(
        "repeat_%02d_oof.rds",
        r
      )
    )

    if (
      file.exists(metric_file) &&
        file.exists(oof_file)
    ) {
      repeat_metrics[[r]] <-
        fread(metric_file)

      cat(
        "      Repeat ",
        r,
        "/",
        N_REPEATS,
        " checkpoint -> skip\n",
        sep = ""
      )

      next
    }

    fa <- FOLDS[
      Repeat == r
    ]

    oof_repeat <- vector(
      "list",
      K_FOLDS
    )

    train_repeat <- vector(
      "list",
      K_FOLDS
    )

    for (
      f in seq_len(
        K_FOLDS
      )
    ) {
      va_taxa <- fa[
        Fold == f,
        Taxon
      ]

      idx_va <- which(
        DT[[GROUP_COL]] %in%
          va_taxa
      )

      idx_tr <- setdiff(
        seq_len(
          nrow(DT)
        ),
        idx_va
      )

      tr_dt <- DT[
        idx_tr
      ]

      va_dt <- DT[
        idx_va
      ]

      if (
        length(
          intersect(
            unique(
              tr_dt[[GROUP_COL]]
            ),
            unique(
              va_dt[[GROUP_COL]]
            )
          )
        )
      ) {
        stop(
          "Taxon leakage in FT candidate ",
          cid,
          ", Repeat ",
          r,
          ", Fold ",
          f
        )
      }

      enc <- fit_encoder(
        tr_dt
      )

      model_seed <-
        MASTER_SEED +
        100000L * r +
        1000L * f +
        10L * cid

      pr <- fit_predict_ft(
        pars = pars,
        tr_dt = tr_dt,
        va_dt = va_dt,
        enc = enc,
        seed = model_seed
      )

      ytr <- as.numeric(
        tr_dt[[TARGET]]
      )

      yva <- as.numeric(
        va_dt[[TARGET]]
      )

      oof_repeat[[f]] <-
        data.table(
          Row_ID = idx_va,
          Repeat = r,
          Fold = f,
          Taxon =
            va_dt[[GROUP_COL]],
          Observed = yva,
          Predicted = pr$valid
        )

      train_repeat[[f]] <-
        data.table(
          Row_ID = idx_tr,
          Repeat = r,
          Fold = f,
          Taxon =
            tr_dt[[GROUP_COL]],
          Observed = ytr,
          Predicted = pr$train
        )

      rm(
        enc,
        pr,
        tr_dt,
        va_dt
      )

      invisible(
        gc(FALSE)
      )
    }

    oof_r <- rbindlist(
      oof_repeat
    )

    train_r <- rbindlist(
      train_repeat
    )

    if (
      nrow(oof_r) !=
        nrow(DT) ||
        anyDuplicated(
          oof_r$Row_ID
        )
    ) {
      stop(
        "FT OOF integrity check failed: candidate ",
        cid,
        ", repeat ",
        r
      )
    }

    m_cv <- metric_set(
      oof_r$Observed,
      oof_r$Predicted
    )

    m_tr <- metric_set(
      train_r$Observed,
      train_r$Predicted
    )

    metric_r <- data.table(
      Model = "FT_Transformer",
      Candidate = cid,
      Repeat = r,
      Train_R2 =
        unname(
          m_tr["R2"]
        ),
      Train_Pearson_R2 =
        unname(
          m_tr["Pearson_R2"]
        ),
      Train_RMSE =
        unname(
          m_tr["RMSE"]
        ),
      Train_MAE =
        unname(
          m_tr["MAE"]
        ),
      Train_nRMSE =
        unname(
          m_tr["nRMSE"]
        ),
      CV_R2 =
        unname(
          m_cv["R2"]
        ),
      CV_Pearson_R2 =
        unname(
          m_cv["Pearson_R2"]
        ),
      CV_RMSE =
        unname(
          m_cv["RMSE"]
        ),
      CV_MAE =
        unname(
          m_cv["MAE"]
        ),
      CV_nRMSE =
        unname(
          m_cv["nRMSE"]
        ),
      R2_gap =
        unname(
          m_tr["R2"] -
            m_cv["R2"]
        )
    )

    fwrite(
      metric_r,
      metric_file
    )

    saveRDS(
      oof_r,
      oof_file,
      compress = TRUE
    )

    repeat_metrics[[r]] <-
      metric_r

    cat(
      "      Repeat ",
      r,
      "/",
      N_REPEATS,
      " complete | CV R2 = ",
      sprintf(
        "%.4f",
        metric_r$CV_R2
      ),
      "\n",
      sep = ""
    )

    rm(
      oof_repeat,
      train_repeat,
      oof_r,
      train_r
    )

    invisible(
      gc(FALSE)
    )
  }

  metrics_dt <- rbindlist(
    repeat_metrics
  )

  sm <- summarize_candidate(
    metrics_dt
  )

  candidate_summary <- cbind(
    data.table(
      Model = "FT_Transformer"
    ),
    copy(pars),
    sm
  )

  fwrite(
    metrics_dt,
    combined_metrics_file
  )

  fwrite(
    candidate_summary,
    summary_file
  )

  candidate_summary
}

collect_completed <- function() {
  checkpoint_root <- file.path(
    DIR_MODEL,
    "checkpoints"
  )

  dirs <- list.dirs(
    checkpoint_root,
    recursive = FALSE,
    full.names = TRUE
  )

  sum_files <- file.path(
    dirs,
    "summary.csv"
  )

  metric_files <- file.path(
    dirs,
    "repeat_metrics.csv"
  )

  sums <- sum_files[
    file.exists(
      sum_files
    )
  ]

  mets <- metric_files[
    file.exists(
      metric_files
    )
  ]

  summary_dt <- if (
    length(sums)
  ) {
    rbindlist(
      lapply(
        sums,
        fread
      ),
      fill = TRUE
    )
  } else {
    data.table()
  }

  metrics_dt <- if (
    length(mets)
  ) {
    rbindlist(
      lapply(
        mets,
        fread
      ),
      fill = TRUE
    )
  } else {
    data.table()
  }

  if (nrow(summary_dt)) {
    summary_dt <- unique(
      summary_dt,
      by = c(
        "Model",
        "Candidate"
      )
    )
  }

  if (nrow(metrics_dt)) {
    metrics_dt <- unique(
      metrics_dt,
      by = c(
        "Model",
        "Candidate",
        "Repeat"
      )
    )
  }

  list(
    summaries = summary_dt,
    repeat_metrics = metrics_dt
  )
}

load_oof <- function(
    candidate_id
) {
  cdir <- file.path(
    DIR_MODEL,
    "checkpoints",
    candidate_dir_name(
      candidate_id
    )
  )

  files <- file.path(
    cdir,
    sprintf(
      "repeat_%02d_oof.rds",
      seq_len(
        N_REPEATS
      )
    )
  )

  if (
    !all(
      file.exists(files)
    )
  ) {
    stop(
      "Missing OOF checkpoint(s) for FT candidate ",
      candidate_id
    )
  }

  rbindlist(
    lapply(
      files,
      readRDS
    ),
    fill = TRUE
  )
}

eval_set <- function(
    z
) {
  if (!nrow(z)) {
    return(
      invisible(NULL)
    )
  }

  for (
    i in seq_len(
      nrow(z)
    )
  ) {
    pars <- z[i]

    cat(
      "\n  ",
      pars$Stage,
      " | Candidate ",
      as.integer(pars$Candidate),
      " | ",
      format_ft_integer_params(pars),
      " | ",
      i,
      "/",
      nrow(z),
      "\n",
      sep = ""
    )

    tryCatch(
      {
        evaluate_candidate(
          pars
        )
      },
      error = function(e) {
        cdir <- safe_dir(
          file.path(
            DIR_MODEL,
            "checkpoints",
            candidate_dir_name(
              pars$Candidate
            )
          )
        )

        writeLines(
          c(
            paste0(
              "TIME: ",
              Sys.time()
            ),
            paste0(
              "ERROR: ",
              conditionMessage(e)
            )
          ),
          file.path(
            cdir,
            "ERROR.txt"
          )
        )

        message(
          "FT candidate failed: ",
          pars$Candidate,
          " | ",
          conditionMessage(e)
        )
      }
    )

    cur <- collect_completed()

    if (nrow(cur$summaries)) {
      fwrite(
        rank_summary(
          cur$summaries
        ),
        file.path(
          DIR_MODEL,
          "06_ALL_PARAMETER_SUMMARY.csv"
        )
      )
    }
  }

  invisible(NULL)
}

# ==============================================================================
# 9. RUN STAGE 1 -> STAGE 2 -> RESCUE
# ==============================================================================

done_file <- file.path(
  DIR_MODEL,
  "DONE.txt"
)

candidate_file <- file.path(
  DIR_MODEL,
  "00_ALL_CANDIDATES.csv"
)

if (file.exists(done_file)) {
  cat(
    "FT-Transformer already DONE. Nothing to rerun.\n"
  )
} else {
  if (
    file.exists(candidate_file)
  ) {
    candidates <- fread(
      candidate_file
    )
  } else {
    candidates <- make_stage1_ft()

    fwrite(
      candidates,
      candidate_file
    )
  }

  # Stage 1
  stage1 <- candidates[
    Stage == "Stage1_Coarse"
  ]

  cat(
    "Formal FT candidate budget | Stage 1 = ",
    FT_STAGE1_BUDGET,
    " | Stage 2 = ",
    FT_STAGE2_BUDGET,
    " | planned total = ",
    FT_PLANNED_TOTAL,
    " (+ up to ",
    MAX_RESCUE_CANDIDATES,
    " rescue)\n",
    sep = ""
  )

  done_ids <- collect_completed()$summaries$Candidate

  if (is.null(done_ids)) {
    done_ids <- integer(0)
  }

  eval_set(
    stage1[
      !Candidate %in%
        done_ids
    ]
  )

  res <- collect_completed()

  s1 <- rank_summary(
    res$summaries[
      Stage ==
        "Stage1_Coarse"
    ]
  )

  if (!nrow(s1)) {
    stop(
      "All FT Stage-1 candidates failed."
    )
  }

  if(nrow(s1)!=FT_STAGE1_BUDGET) stop('FT Stage1 incomplete: expected all 15 candidates.')

  fwrite(
    s1,
    file.path(
      DIR_MODEL,
      "01_STAGE1_COARSE_SUMMARY.csv"
    )
  )

  s1best <- s1[1]

  fwrite(
    decode_ft(
      s1best
    ),
    file.path(
      DIR_MODEL,
      "02_STAGE1_BEST_PARAMETERS.csv"
    )
  )

  cat(
    "\nStage 1 best candidate ",
    s1best$Candidate,
    " | Mean CV R2 = ",
    sprintf(
      "%.4f",
      s1best$Mean_CV_R2
    ),
    "\n",
    sep = ""
  )

  # Stage 2
  stage2 <- candidates[
    Stage == "Stage2_Fine"
  ]

  if (!nrow(stage2)) {
    stage2 <- make_stage2_ft(
      s1best,
      candidates
    )

    candidates <- rbindlist(
      list(
        candidates,
        stage2
      ),
      fill = TRUE
    )

    fwrite(
      candidates,
      candidate_file
    )
  }

  fwrite(
    stage2,
    file.path(
      DIR_MODEL,
      "03_STAGE2_FINE_GRID.csv"
    )
  )

  done_ids <- collect_completed()$summaries$Candidate

  if (is.null(done_ids)) {
    done_ids <- integer(0)
  }

  eval_set(
    stage2[
      !Candidate %in%
        done_ids
    ]
  )

  res <- collect_completed()

  stage12 <- rank_summary(
    res$summaries[
      Stage %in%
        c(
          "Stage1_Coarse",
          "Stage2_Fine"
        )
    ]
  )

  fwrite(
    stage12[
      Stage ==
        "Stage2_Fine"
    ],
    file.path(
      DIR_MODEL,
      "04_STAGE2_FINE_SUMMARY.csv"
    )
  )

  best_before_rescue <- stage12[1]

  if(nrow(stage12[Stage=='Stage2_Fine']) != FT_STAGE2_BUDGET)
    stop('FT Stage2 incomplete: expected all 15 candidates.')

  # Boundary rescue
  rescue <- candidates[
    Stage ==
      "Stage2_Boundary_Rescue"
  ]

  if (
    ENABLE_STAGE2_BOUNDARY_RESCUE &&
      !nrow(rescue)
  ) {
    rescue <- make_ft_rescue(
      best_before_rescue,
      candidates
    )

    if (nrow(rescue)) {
      candidates <- rbindlist(
        list(
          candidates,
          rescue
        ),
        fill = TRUE
      )

      fwrite(
        candidates,
        candidate_file
      )
    }
  }

  fwrite(
    rescue,
    file.path(
      DIR_MODEL,
      "05_STAGE2_BOUNDARY_RESCUE_GRID.csv"
    )
  )

  if (nrow(rescue)) {
    done_ids <- collect_completed()$summaries$Candidate

    if (is.null(done_ids)) {
      done_ids <- integer(0)
    }

    eval_set(
      rescue[
        !Candidate %in%
          done_ids
      ]
    )
  }


  # ============================================================================
  # V12 STAGE 3: 12 targeted boundary candidates, freshly evaluated on current
  # training data and the same 5x5 CV folds. NEVER compare against old v4 OOF.
  # ============================================================================
  stage3 <- candidates[Stage=="Stage3_Boundary12"]
  if (!nrow(stage3)) {
    before12 <- rank_summary(collect_completed()$summaries)
    base <- candidates[Candidate == as.integer(before12$Candidate[1L])]
    if (nrow(base)!=1L) stop('Cannot locate FT stage-3 baseline configuration.')
    # User-provided 12 boundary experiments: regularization, batch, attention,
    # training length and coupled perturbations around the new-data winner.
    variations <- list(
      list(weight_decay_x1e6=100L),
      list(weight_decay_x1e6=300L),
      list(weight_decay_x1e6=1000L),
      list(weight_decay_x1e6=3000L),
      list(weight_decay_x1e6=10000L),
      list(batch_size=128L),
      list(heads=16L),
      list(epochs=120L),
      list(epochs=150L),
      list(weight_decay_x1e6=300L,batch_size=128L),
      list(weight_decay_x1e6=1000L,batch_size=128L),
      list(weight_decay_x1e6=300L,epochs=120L),
      # Fallbacks if an earlier search already tested a proposed point:
      list(weight_decay_x1e6=50L),
      list(weight_decay_x1e6=500L),
      list(weight_decay_x1e6=5000L),
      list(batch_size=192L),
      list(epochs=110L),
      list(epochs=180L),
      list(weight_decay_x1e6=500L,epochs=120L),
      list(heads=16L,weight_decay_x1e6=300L)
    )
    seen <- ft_signature(candidates)
    stage3_parts <- list()
    for (i in seq_along(variations)) {
      zz <- data.table::copy(base[, ..ft_param_cols])
      for (nm in names(variations[[i]])) data.table::set(zz,j=nm,value=as.integer(variations[[i]][[nm]]))
      if (zz$embed_dim %% zz$heads != 0L) next
      sig <- ft_signature(zz)
      if (sig %in% seen) next
      seen <- c(seen,sig)
      zz[,`:=`(Candidate=as.integer(3000L+length(stage3_parts)+1L),
               Stage='Stage3_Boundary12',
               Generation_Reason=paste(names(variations[[i]]),collapse='+'))]
      stage3_parts[[length(stage3_parts)+1L]] <- zz
      if (length(stage3_parts)==12L) break
    }
    if(length(stage3_parts)!=12L) stop('Fewer than 12 new unique FT boundary candidates.')
    stage3 <- data.table::rbindlist(stage3_parts,fill=TRUE)
    data.table::setcolorder(stage3,c('Candidate','Stage','Generation_Reason',ft_param_cols))
    candidates <- data.table::rbindlist(list(candidates,stage3),fill=TRUE)
    data.table::fwrite(candidates,candidate_file)
  }
  fwrite(stage3,file.path(DIR_MODEL,'05B_STAGE3_12_BOUNDARY_GRID.csv'))
  done_ids <- collect_completed()$summaries$Candidate
  if(is.null(done_ids)) done_ids <- integer(0)
  eval_set(stage3[!Candidate %in% done_ids])
  pre_final <- collect_completed()
  if (nrow(pre_final$summaries[Stage=='Stage3_Boundary12']) != 12L)
    stop('FT targeted boundary search incomplete; resume to finish 12 candidates.')
  fwrite(rank_summary(pre_final$summaries[Stage=='Stage3_Boundary12']),
         file.path(DIR_MODEL,'05C_STAGE3_12_BOUNDARY_SUMMARY.csv'))

  # Final selection
  res <- collect_completed()

  final_summary <- rank_summary(
    res$summaries
  )

  fwrite(
    final_summary,
    file.path(
      DIR_MODEL,
      "06_ALL_PARAMETER_SUMMARY.csv"
    )
  )

  fwrite(
    res$repeat_metrics,
    file.path(
      DIR_MODEL,
      "07_ALL_REPEAT_METRICS.csv"
    )
  )

  # Match the five-model feasibility rule; do not silently accept overfitting.
  eligible <- final_summary[is.finite(Max_R2_Gap) & Max_R2_Gap <= 0.20 + 1e-9]
  fwrite(eligible,file.path(DIR_MODEL,'06B_ELIGIBLE_STRICT_GAP020.csv'))
  if (!nrow(eligible)) {
    writeLines('No FT model satisfied the <=0.20 gap in all five repeats. Do not declare final model.',
               file.path(DIR_MODEL,'NO_GAP_FEASIBLE_MODEL.txt'))
    stop('No eligible FT candidate under strict per-repeat gap <=0.20; checkpoints saved.')
  }
  best <- eligible[1]

  best_id <- as.integer(
    best$Candidate
  )

  best_pars <- candidates[
    Candidate ==
      best_id
  ]

  fwrite(
    best_pars,
    file.path(
      DIR_MODEL,
      "08_BEST_PARAMETERS_CODED_INTEGER.csv"
    )
  )

  best_decoded <- decode_ft(
    best_pars
  )

  best_decoded[, `:=`(
    Final_Mean_Train_R2 =
      best$Mean_Train_R2,
    Final_Mean_CV_R2 =
      best$Mean_CV_R2,
    Final_SD_CV_R2 =
      best$SD_CV_R2,
    Final_Mean_R2_Gap =
      best$Mean_R2_Gap
  )]

  fwrite(
    best_decoded,
    file.path(
      DIR_MODEL,
      "08_BEST_PARAMETERS_DECODED.csv"
    )
  )

  best_repeat <- res$repeat_metrics[
    Candidate ==
      best_id
  ]

  fwrite(
    best_repeat,
    file.path(
      DIR_MODEL,
      "09_BEST_REPEAT_METRICS.csv"
    )
  )

  best_oof <- load_oof(
    best_id
  )

  best_oof[, `:=`(
    Model = "FT_Transformer",
    Candidate = best_id
  )]

  fwrite(
    best_oof,
    file.path(
      DIR_MODEL,
      "10_BEST_OOF_PREDICTIONS.csv"
    )
  )
  # OOF remains genuine held-out-fold prediction; MOA appended for post-hoc use.
  oof_metadata <- DT[, unique(c('Row_ID','DTXSID','CASRN',
    'Model_MOA_final',SPECIFIC_META_COLS)),with=FALSE]
  best_oof_audit <- merge(best_oof,oof_metadata,by='Row_ID',all.x=TRUE,sort=FALSE)
  if(nrow(best_oof_audit)!=nrow(DT)*N_REPEATS || anyNA(best_oof_audit$DTXSID))
    stop('OOF Specific MOA metadata merge failed.')
  fwrite(best_oof_audit,file.path(DIR_MODEL,
    '10_BEST_OOF_PREDICTIONS_WITH_SPECIFIC_MOA.csv'))

  # ============================================================================
  # 10. FINAL FULL-DATA FIT + BUNDLE
  # ============================================================================

  enc <- fit_encoder(
    DT
  )

  train_ft <- make_ft_dataframe(
    DT,
    enc,
    include_target = TRUE
  )

  ft_model_dir <- file.path(
    DIR_MODEL,
    "FINAL_MODEL_FILES",
    "FT_MODEL"
  )

  dir.create(
    dirname(
      ft_model_dir
    ),
    recursive = TRUE,
    showWarnings = FALSE
  )

  final_seed <-
    MASTER_SEED +
    909090L

  pr <- reticulate::py$ecotox_ft_fit_save(
    train_df = train_ft,
    num_cols = NUM_VARS,
    cat_cols = CAT_VARS,
    target = TARGET,
    seed = as.integer(
      final_seed
    ),
    embed = as.integer(
      best_pars$embed_dim
    ),
    heads = as.integer(
      best_pars$heads
    ),
    blocks = as.integer(
      best_pars$blocks
    ),
    dropout = as.numeric(
      best_pars$dropout_pct
    ) / 100,
    lr = as.numeric(
      best_pars$lr_x1e6
    ) / 1e6,
    wd = as.numeric(
      best_pars$weight_decay_x1e6
    ) / 1e6,
    batch = as.integer(
      best_pars$batch_size
    ),
    epochs = as.integer(
      best_pars$epochs
    ),
    save_dir = normalizePath(
      ft_model_dir,
      winslash = "/",
      mustWork = FALSE
    )
  )

  train_pred <- as.numeric(
    pr$train
  )
  if(length(train_pred)!=nrow(DT) || any(!is.finite(train_pred)))
    stop('Final FT training predictions are incomplete or non-finite.')
  fitted_audit <- DT[, unique(c('Row_ID','DTXSID','CASRN','Chemical_Name',
    'Taxon',TARGET,'Model_MOA_final',SPECIFIC_META_COLS)),with=FALSE]
  fitted_audit[, `:=`(Predicted=train_pred,
                       Residual=get(TARGET)-train_pred)]
  fwrite(fitted_audit,file.path(DIR_MODEL,
    '11_FINAL_FITTED_PREDICTIONS_WITH_SPECIFIC_MOA.csv'))

  train_metric <- metric_set(
    as.numeric(
      DT[[TARGET]]
    ),
    train_pred
  )

  full_train_metrics <- data.table(
    Model = "FT_Transformer",
    Train_R2 =
      unname(
        train_metric["R2"]
      ),
    Train_Pearson_R2 =
      unname(
        train_metric[
          "Pearson_R2"
        ]
      ),
    Train_RMSE =
      unname(
        train_metric["RMSE"]
      ),
    Train_MAE =
      unname(
        train_metric["MAE"]
      ),
    Train_nRMSE =
      unname(
        train_metric["nRMSE"]
      )
  )

  fwrite(
    full_train_metrics,
    file.path(
      DIR_MODEL,
      "11_FINAL_FULLTRAIN_METRICS.csv"
    )
  )

  bundle <- list(
    script_version =
      SCRIPT_VERSION,
    model_name =
      "FT_Transformer",
    target =
      TARGET,
    group_col =
      GROUP_COL,
    science_predictors =
      SCIENCE_PREDICTORS,
    specific_moa_metadata_cols =
      SPECIFIC_META_COLS,
    specific_moa_as_predictor =
      FALSE,
    run05_matrix_dir =
      MATRIX_DIR,
    fold_seed =
      FOLD_SEED,
    num_vars =
      NUM_VARS,
    cat_vars =
      CAT_VARS,
    encoder =
      enc,
    best_parameters_coded =
      as.list(
        best_pars
      ),
    best_parameters_decoded =
      as.list(
        best_decoded
      ),
    best_cv_summary =
      as.list(
        best
      ),
    full_training_metrics =
      as.list(
        full_train_metrics
      ),
    python_executable =
      FT_PYTHON,
    model_storage =
      "PyTorch Tabular save_model directory",
    model_path =
      ft_model_dir,
    python_loader =
      "pytorch_tabular.TabularModel.load_model",
    master_seed =
      MASTER_SEED,
    training_rows =
      nrow(DT),
    training_taxa =
      uniqueN(
        DT[[GROUP_COL]]
      ),
    created =
      as.character(
        Sys.time()
      )
  )

  saveRDS(
    bundle,
    file.path(
      DIR_MODEL,
      "12_FINAL_BUNDLE.rds"
    )
  )

  model_result <- copy(
    best
  )

  model_result[, `:=`(
    Best_Candidate =
      best_id,
    Stage1_Candidates =
      nrow(
        candidates[
          Stage ==
            "Stage1_Coarse"
        ]
      ),
    Stage2_Candidates =
      nrow(
        candidates[
          Stage ==
            "Stage2_Fine"
        ]
      ),
    Rescue_Candidates =
      nrow(
        candidates[
          Stage ==
            "Stage2_Boundary_Rescue"
        ]
      ),
    Targeted_Boundary12_Candidates = nrow(candidates[Stage=='Stage3_Boundary12']),
    Status =
      "DONE"
  )]

  fwrite(
    model_result,
    file.path(
      DIR_MODEL,
      "13_MODEL_RESULT.csv"
    )
  )

  writeLines(
    c(
      paste0(
        "DONE: ",
        Sys.time()
      ),
      paste0(
        "Best candidate: ",
        best_id
      ),
      paste0(
        "Best stage: ",
        best$Stage
      ),
      paste0(
        "Mean CV R2: ",
        sprintf(
          "%.6f",
          best$Mean_CV_R2
        )
      )
    ),
    done_file
  )

  cat(
    "\n============================================================\n"
  )
  cat(
    "FT-TRANSFORMER COMPLETE\n"
  )
  cat(
    "Best candidate: ",
    best_id,
    "\n",
    sep = ""
  )
  cat(
    "Best stage    : ",
    best$Stage,
    "\n",
    sep = ""
  )
  cat(
    "Mean CV R2    : ",
    sprintf(
      "%.4f",
      best$Mean_CV_R2
    ),
    "\n",
    sep = ""
  )
  cat(
    "============================================================\n"
  )
}

capture.output(
  sessionInfo(),
  file = file.path(
    DIR_CONFIG,
    "SESSION_INFO_END.txt"
  )
)

elapsed_hours <- as.numeric(
  difftime(
    Sys.time(),
    RUN_START,
    units = "hours"
  )
)

writeLines(
  c(
    paste0(
      "SCRIPT_VERSION=",
      SCRIPT_VERSION
    ),
    paste0(
      "START=",
      RUN_START
    ),
    paste0(
      "END=",
      Sys.time()
    ),
    paste0(
      "ELAPSED_HOURS=",
      sprintf(
        "%.3f",
        elapsed_hours
      )
    )
  ),
  file.path(
    OUT_DIR,
    "RUN_FINISHED.txt"
  )
)

cat(
  "\nElapsed hours: ",
  sprintf(
    "%.2f",
    elapsed_hours
  ),
  "\n",
  sep = ""
)
