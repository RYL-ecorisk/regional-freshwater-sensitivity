# Five-model tuning: coarse search, TPE, local refinement and boundary checks.
# Run from the package root. The generated run seed and fold assignments are saved for resumption.

rm(list=ls()); gc(); options(stringsAsFactors=FALSE,warn=1)
suppressPackageStartupMessages({ library(data.table) })
REQ <- c('glmnet','ranger','lightgbm','xgboost')
stopifnot(all(vapply(REQ,requireNamespace,logical(1),quietly=TRUE)))
if (packageVersion('xgboost') < '3.0.0') stop('XGBoost native factor input requires R xgboost >=3.0.0')

VERSION <- 'ecotox_5models_native24_gap020_MOA_RECHECKED_v12'
ROOT <- normalizePath('.', winslash='/', mustWork=TRUE)
RUN_ROOT <- file.path(ROOT,'results','model_tuning_run06')
OUT <- file.path(RUN_ROOT,'five_model_tuning')
FOLD_FILE <- file.path(RUN_ROOT,'shared_folds','FOLD_ASSIGNMENTS.csv')
SEED_FILE <- file.path(RUN_ROOT,'00_RUN_SEED.txt')
dir.create(RUN_ROOT, recursive=TRUE, showWarnings=FALSE)

DATA_CANDIDATES <- file.path(ROOT,'data',c(
  'Training_DATABASE_MOA_RECHECKED.csv',
  'Training_DATABASE_MOA_RECHECKED_WITH_BARRON_SPECIFIC_20261014.csv'
))
DATA_CANDIDATES <- DATA_CANDIDATES[file.exists(DATA_CANDIDATES) & !dir.exists(DATA_CANDIDATES)]
DATA_FILE <- if(length(DATA_CANDIDATES)) DATA_CANDIDATES[1L] else NA_character_
if(is.na(DATA_FILE)) stop('Training input not found in data/: expected Training_DATABASE_MOA_RECHECKED.csv')

if (file.exists(SEED_FILE)) {
  MASTER_SEED <- suppressWarnings(as.integer(readLines(SEED_FILE, n=1L)))
  if (length(MASTER_SEED)!=1L || is.na(MASTER_SEED)) stop('Invalid saved run seed: ',SEED_FILE)
} else {
  MASTER_SEED <- sample.int(.Machine$integer.max,1L)
  writeLines(as.character(MASTER_SEED),SEED_FILE)
}
K <- 5L; N_REPEATS <- 5L
RNGkind("L'Ecuyer-CMRG"); set.seed(MASTER_SEED)
THRESHOLD <- 0.001; EXPAND_BUDGET <- 12L
MAX_SEARCH_ROUNDS <- suppressWarnings(as.integer(Sys.getenv('ECOTOX_MAX_SEARCH_ROUNDS','25')))
if(is.na(MAX_SEARCH_ROUNDS)||MAX_SEARCH_ROUNDS<2L||MAX_SEARCH_ROUNDS>10000L) stop('Invalid ECOTOX_MAX_SEARCH_ROUNDS')
WALLTIME_HOURS <- suppressWarnings(as.numeric(Sys.getenv('ECOTOX_WALLTIME_HOURS','18')))
if(!is.finite(WALLTIME_HOURS)||WALLTIME_HOURS<=0) stop('Invalid ECOTOX_WALLTIME_HOURS')
RUN_STARTED <- Sys.time()
check_time_budget <- function() {
  elapsed <- as.numeric(difftime(Sys.time(),RUN_STARTED,units='hours'))
  if(elapsed >= WALLTIME_HOURS)
    stop(sprintf('ECOTOX_TIME_BUDGET: %.2f/%.2f hours; checkpoints are safe. Re-run this script to resume.',
                 elapsed,WALLTIME_HOURS),call.=FALSE)
  invisible(elapsed)
}

MODEL_NAMES <- c('XGBoost','Ridge','Elastic_Net','Random_Forest','LightGBM')
# Reordering training MUST NOT reassign the original tuning budgets.
# Resolve budgets by model name instead of positional c(...) vectors.
COARSE_BY_MODEL <- c(XGBoost=30L,Ridge=15L,Elastic_Net=20L,Random_Forest=25L,LightGBM=30L)
TPE_BY_MODEL    <- c(XGBoost=45L,Ridge=30L,Elastic_Net=35L,Random_Forest=40L,LightGBM=45L)
LOCAL_BY_MODEL  <- c(XGBoost=15L,Ridge=15L,Elastic_Net=15L,Random_Forest=15L,LightGBM=15L)
BUDGETS <- data.table(Model=MODEL_NAMES,
                      Coarse=as.integer(COARSE_BY_MODEL[MODEL_NAMES]),
                      TPE=as.integer(TPE_BY_MODEL[MODEL_NAMES]),
                      Local=as.integer(LOCAL_BY_MODEL[MODEL_NAMES]))
stopifnot(identical(BUDGETS$Model[1L], 'XGBoost'),
          identical(as.integer(unlist(BUDGETS[Model=='XGBoost', .(Coarse,TPE,Local)])),
                    c(30L,45L,15L)))
# Fresh planned candidates: 60 + 70 + 80 + 90 + 90 = 390 before edge probes.
# Each candidate: five repeats x five folds; all repeats evaluated anew.
TARGET <- 'log10_Geomean_ug_L'; GROUP <- 'Taxon'
CHEM <- c('logS','logKow','logD7_4','MW','TPSA','log_Henry',
          'log_VaporPressure','MeltingPoint','HBA','HBD',
          'AromaticRingCount','RotatableBondCount','BertzCT')
TRAIT <- c('Log Body size','Feeding_score','Respiration_score',
           'Locomotion_score','Body_armor_score')
CATS <- c('Phylum','Class','Order','Family','Genus','Model_MOA_final')
NUMS <- c(CHEM,TRAIT); FEATURES <- c(NUMS,CATS)
stopifnot(length(FEATURES)==24L,file.exists(DATA_FILE))
DT <- fread(DATA_FILE,na.strings=c('','NA','NaN'),encoding='UTF-8')
# Verify this is the MOA-rechecked source, not the older 10.13 training file.
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
stopifnot(all(c(TARGET,GROUP,FEATURES) %in% names(DT)))
for (v in c(TARGET,NUMS)) set(DT,j=v,value=suppressWarnings(as.numeric(DT[[v]])))
DT <- DT[is.finite(get(TARGET)) & !is.na(get(GROUP)) & nzchar(trimws(as.character(get(GROUP))))]
DT[,Taxon:=trimws(as.character(Taxon))];DT[,Row_ID:=seq_len(.N)]
stopifnot(nrow(DT)==7043L, uniqueN(DT$DTXSID)==2109L)
if (anyNA(DT$Model_MOA_final) || any(!nzchar(trimws(as.character(DT$Model_MOA_final)))))
  stop('Missing Model_MOA_final after MOA audit; refusing to train.')
if (any(DT[, .(n=uniqueN(Model_MOA_final)), by=DTXSID]$n != 1L))
  stop('Conflicting Broad MOA assignments within one DTXSID.')
# Extra Specific-MOA columns are preserved in metadata only, NEVER added to FEATURES.

# ===========================================================================
# Shared 5 x 5 taxon-blocked folds.
# Do not read fold assignments/checkpoints from earlier five-model/FT runs.
# Use the saved run-specific seed for shared folds.
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

FOLDS<-create_shared_folds(DT,FOLD_FILE,k=K,repeats=N_REPEATS)
stopifnot(all(c('Repeat','Fold','Taxon','N') %in% names(FOLDS)))
FOLDS[,Taxon:=trimws(as.character(Taxon))]
# One (Repeat, Taxon) allocation row: 5 repeats x N taxa, NOT 5 folds x 5 repeats x N taxa.
expected_fold_rows <- N_REPEATS * uniqueN(DT$Taxon)
if (nrow(FOLDS) != expected_fold_rows)
  stop(sprintf('Fold assignment rows: got %d; expected %d (repeats x taxa).',
               nrow(FOLDS), expected_fold_rows))
stopifnot(!anyNA(FOLDS[,.(Repeat,Fold,Taxon,N)]),
          setequal(FOLDS$Repeat,seq_len(N_REPEATS)),
          setequal(FOLDS$Fold,seq_len(K)),
          !anyDuplicated(FOLDS,by=c('Repeat','Taxon')))
for(r in 1:5){
  fa<-FOLDS[Repeat==r]
  stopifnot(setequal(fa$Taxon,DT$Taxon),
            all(DT[, .N,by=Taxon][fa,on='Taxon']$N==fa$N))
}
# Fail early if R XGBoost lacks real native categorical factor support.
tryCatch({
  xx<-data.frame(a=1:6,b=factor(c('a','b','a','c','b','a')))
  dm<-xgboost::xgb.DMatrix(xx,label=as.numeric(1:6))
  tt<-xgboost::xgb.train(params=list(objective='reg:squarederror',
         tree_method='hist',max_cat_to_onehot=1L,nthread=1L),
         data=dm,nrounds=2L,verbose=0)
  stopifnot(all(is.finite(predict(tt,dm))))
  rm(xx,dm,tt)
},error=function(e) stop('Native XGBoost compatibility test failed: ',conditionMessage(e)))
FOLD_IDX<-lapply(1:5,function(r){
 fa<-FOLDS[Repeat==r]
 lapply(1:5,function(f){
  val<-which(DT$Taxon %in% fa[Fold==f,Taxon]); tr<-setdiff(seq_len(nrow(DT)),val)
  stopifnot(length(val)>0L,length(tr)>0L,!length(intersect(DT$Taxon[tr],DT$Taxon[val])))
  list(tr=tr,va=val)
 })
})
# Auto CPU planning for Windows PSOCK (24-core workstation or other hardware).
# Parallelize ACROSS independent candidates; use modest native threads per fit.
# Preserve a few OS cores; override with ECOTOX_CPU_BUDGET/WORKERS/THREADS if needed.
LOGICAL<-parallel::detectCores(logical=TRUE)
if(is.na(LOGICAL)||LOGICAL<1L) LOGICAL<-8L
PHYSICAL<-parallel::detectCores(logical=FALSE)
if(is.na(PHYSICAL)||PHYSICAL<1L) PHYSICAL<-max(2L,LOGICAL%/%2L)
read_env_int<-function(v,default) {
 z<-suppressWarnings(as.integer(Sys.getenv(v,'')))
 if(is.na(z)||z<1L)as.integer(default) else z
}
CPU_BUDGET<-min(PHYSICAL,read_env_int('ECOTOX_CPU_BUDGET',max(1L,PHYSICAL-2L)))
hardware_plan<-function(model) {
 # Lighter glmnet fits benefit from many 1-thread workers; heavy tree fits from
 # fewer workers x 3-4 native threads. All active cores <= CPU_BUDGET.
 suggested_workers<-switch(model,Ridge=20L,Elastic_Net=20L,
     Random_Forest=7L,XGBoost=7L,LightGBM=5L)
 suggested_threads<-switch(model,Ridge=1L,Elastic_Net=1L,
     Random_Forest=3L,XGBoost=3L,LightGBM=4L)
 workers<-min(CPU_BUDGET,read_env_int('ECOTOX_WORKERS',
            min(suggested_workers,max(1L,CPU_BUDGET%/%suggested_threads))))
 threads<-min(read_env_int('ECOTOX_THREADS',
            min(suggested_threads,max(1L,CPU_BUDGET%/%workers))),
            max(1L,CPU_BUDGET%/%workers))
 list(workers=as.integer(workers),threads=as.integer(threads),
      active=as.integer(workers*threads))
}
HARDWARE_PLAN<-rbindlist(lapply(MODEL_NAMES,function(m){
 h<-hardware_plan(m)
 data.table(Model=m,Workers=h$workers,Threads_Per_Worker=h$threads,
            Active_Threads=h$active,CPU_Budget=CPU_BUDGET)
}))
# Workers are re-created between MODELS to avoid multiplying all cached folds.
N_WORKERS<-1L;N_THREADS<-1L
Sys.setenv(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='1')
# STRICT overfitting constraint across EVERY full-repeat OOF metric.
# Infeasible points are ranked by their WORST repeat violation, not by average gap.
# An unmet constraint is never silently relaxed, even in emergency fallback.
GAP_LIMIT <- 0.20
GAP_TOLERANCE <- 1e-9
rank_constrained <- function(x) {
  z <- data.table::copy(x)
  z[,Gap_Feasible:=as.logical(Gap_Feasible)]
  z[,Gap_Excess:=pmax(0,as.numeric(Max_R2_Gap)-GAP_LIMIT)]
  # Search utility uses CV R2 FIRST and SOFT gap excess penalty.
  # Crucial: RF with CV=.712, gap=.203 is NOT ignored in favor of CV=.67.
  # Formal winner selection remains STRICT (all 5 repeats Gap <= .20).
  z[,SearchUtility:=Mean_CV_R2-0.25*Gap_Excess]
  data.table::setorder(z,-Gap_Feasible,-Mean_CV_R2,Mean_CV_RMSE,Candidate)
  z
}

# Targets are conservative, but viable high-complexity candidates remain searchable.
# No model is forced to achieve an arbitrary train R2 level or a trivial CV R2.
regularized_domain <- function(space, model) {
  s<-space
  restrict <- function(nm, fn) {
    if(!is.null(s[[nm]])){
      v<-s[[nm]][fn(s[[nm]])]
      if(length(v))s[[nm]]<<-v
    }
  }
  if(model=='Ridge')restrict('lambda_q',function(z)z>=0)
  if(model=='Elastic_Net')restrict('lambda_q',function(z)z>=-4)
  if(model=='Random_Forest'){
    restrict('min_node',function(z)z>=8)
    restrict('max_depth',function(z)z>0 & z<=16)
    restrict('mtry_pct',function(z)z<=35)
    restrict('sample_pct',function(z)z<=90)
  }
  if(model=='XGBoost'){
    restrict('max_depth',function(z)z<=6)
    restrict('child_weight_q4',function(z)z>=16)
    restrict('gamma_milli',function(z)z>=100)
    restrict('lambda',function(z)z>=10)
    restrict('alpha',function(z)z>=3)
    restrict('subsample_pct',function(z)z<=85)
    restrict('colsample_pct',function(z)z<=90)
  }
  if(model=='LightGBM'){
    restrict('max_depth',function(z)z>0 & z<=8)
    restrict('num_leaves',function(z)z<=63)
    restrict('min_data_in_leaf',function(z)z>=30)
    restrict('lambda_l2',function(z)z>=10)
    restrict('min_split_gain_milli',function(z)z>=100)
    restrict('feature_pct',function(z)z<=85)
    restrict('bagging_pct',function(z)z<=90)
  }
  s
}
# v8: type-aware edges. Only a real mathematical/algorithmic endpoint may
# remain an endpoint without further probes. Other bounds are *artificial*.
is_legal_terminal <- function(model, p, side, x) {
  if (side=='Lower') {
    if(model=='Elastic_Net' && p=='alpha_pct' && x==0) return(TRUE)
    if(model=='Random_Forest' && p %in% c('min_node','mtry_pct') && x==1) return(TRUE)
    if(model=='XGBoost' && p %in% c('lambda','alpha','gamma_milli',
                                    'child_weight_q4') && x==0) return(TRUE)
    if(model=='XGBoost' && p %in% c('max_cat_threshold','max_depth') && x==1) return(TRUE)
    if(model=='LightGBM' && p %in% c('lambda_l2','cat_smooth',
          'min_split_gain_milli','min_data_in_leaf') && x==0) return(TRUE)
    if(model=='LightGBM' && p=='num_leaves' && x==2) return(TRUE)
  } else {
    if(model=='Elastic_Net' && p=='alpha_pct' && x==100) return(TRUE)
    if(model=='Random_Forest' && p %in% c('mtry_pct','sample_pct') && x==100)
      return(TRUE)
    if(model=='XGBoost' && p %in% c('subsample_pct','colsample_pct') && x==100)
      return(TRUE)
    if(model=='LightGBM' && p %in% c('feature_pct','bagging_pct') && x==100)
      return(TRUE)
  }
  FALSE
}

# Search-edge extension. For any artificial edge, the script adds a small
# outward value and makes an actual full 5x5 candidate, instead of assuming
# that a hit at an arbitrary cap is legitimate.  Terminal safety caps are
# honest stop/report markers, NEVER called physical optima.
extend_space <- function(space, best, tested, hard, types=NULL) {
  model <- as.character(best$Model[1L])
  edges <- data.table::data.table(Parameter=character(),Side=character(),
                                  Value=numeric())
  out <- space; expandable <- FALSE
  for(p in names(space)) {
    x<-as.numeric(best[[p]][1L]); v<-sort(unique(as.numeric(space[[p]])))
    lo<-as.numeric(hard[[p]][1L]); hi<-as.numeric(hard[[p]][2L])
    if(p=='max_depth' && (lo<0 || 0 %in% v)){
      unlimited<-if(lo<0)-1L else 0L
      if(x==unlimited){
        edges<-data.table::rbindlist(list(edges,data.table::data.table(
          Parameter=p,Side='SPECIAL_LEGAL_UNLIMITED_DEPTH',Value=x)))
        next
      }
      v<-v[v>=1L];lo<-1
    }
    for(side in c('Lower','Upper')) {
      edge<-if(side=='Lower')min(v)else max(v)
      if(abs(x-edge)>1e-9)next
      if(is_legal_terminal(model,p,side,x)) {
        edges<-data.table::rbindlist(list(edges,data.table::data.table(
          Parameter=p,Side=paste0(side,'_LEGAL_ENDPOINT'),Value=x)))
        next
      }
      limit<-if(side=='Lower')lo else hi
      if((side=='Lower' && edge<=limit) || (side=='Upper' && edge>=limit)) {
        edges<-data.table::rbindlist(list(edges,data.table::data.table(
          Parameter=p,Side=paste0(side,'_ARTIFICIAL_SAFETY_CAP'),Value=x)))
        next
      }
      # For percentage/rounding-coded domains, probe one modest outward step.
      step<-if(length(v)>1L)max(1,stats::median(diff(v)))else 1
      jump<-max(step,round(max(1,abs(edge))*0.25))
      newval<-if(side=='Lower') max(limit,edge-jump) else min(limit,edge+jump)
      newval<-as.integer(round(newval))
      if(newval==edge || newval %in% out[[p]])next
      out[[p]]<-sort(unique(c(as.integer(out[[p]]),newval)))
      expandable<-TRUE
      edges<-data.table::rbindlist(list(edges,data.table::data.table(
        Parameter=p,Side=paste0(side,'_EXPANDED'),Value=newval)))
    }
  }
  list(space=out,edges=edges,changed=expandable)
}

# Eligible final selection: strict GAP across all 5 repeats; no unprobed
# *artificial* domain edge and no absolute artificial safety-cap optimum.
# Safety cap hits are reported separately.  Never force a legal endpoint away.
acceptable_candidates <- function(scores,space,hard,param_names) {
  cand<-scores[Gap_Feasible==TRUE]
  if(!nrow(cand))return(cand)
  ok<-vapply(seq_len(nrow(cand)),function(i) {
    edge<-extend_space(space,cand[i],NULL,hard)
    !edge$changed && !any(grepl('ARTIFICIAL_SAFETY_CAP',edge$edges$Side))
  },logical(1))
  cand[ok]
}

# Exhaustive audit of *all* final hyperparameters, not just the collided ones.
boundary_audit <- function(model, best, space, hard, original) {
  data.table::rbindlist(lapply(names(space),function(p) {
    v<-sort(unique(as.numeric(space[[p]])))
    x<-as.numeric(best[[p]][1L]); lim<-as.numeric(hard[[p]])
    old<-as.numeric(original[[p]])
    sentinel<- p=='max_depth' && model %in% c('Random_Forest','LightGBM') &&
       x==(if(model=='Random_Forest')0 else -1)
    edge<-!sentinel && (abs(x-min(v))<1e-9 || abs(x-max(v))<1e-9)
    side<-if(sentinel)'Unlimited' else if(abs(x-min(v))<1e-9)'Lower' else
      if(abs(x-max(v))<1e-9)'Upper' else 'Interior'
    legit<-sentinel || (edge && is_legal_terminal(model,p,side,x))
    cap<-edge && !legit && (abs(x-lim[1L])<1e-9 || abs(x-lim[2L])<1e-9)
    state<-if(sentinel)'LEGAL_UNLIMITED_DEPTH' else if(legit)'LEGAL_ENDPOINT' else
       if(cap)'UNRESOLVED_ARTIFICIAL_SAFETY_CAP' else
       if(edge)'UNPROBED_ARTIFICIAL_SEARCH_EDGE' else 'INTERIOR'
    data.table(Model=model,Parameter=p,Winner=x,Initial_Min=min(SEARCH_SPACE[[model]][[p]]),
      Initial_Max=max(SEARCH_SPACE[[model]][[p]]),
      Old_Soft_Min=old[1L],Old_Soft_Max=old[2L],
      Final_Grid_Min=min(v),Final_Grid_Max=max(v),
      Final_Safety_Min=lim[1L],Final_Safety_Max=lim[2L],
      Boundary_State=state,Legitimate_Endpoint=legit)
  }))
}

# This second-run v12 retains the original formal search space for comparability.
# Same search space as v9 for Ridge, Elastic Net, RF and XGB.
# LightGBM: v9 winner had feature_fraction=0.37, so add 0.35, 0.37, 0.45,
# 0.55 below v9's initial 0.65 boundary. All are Stage1-onward values.
# This is a data-informed RETUNING of the same training CV, NOT independent
# validation and not a prospective/uninformed initial parameter range.
# Remaining parameters preserve the v9 formal grids; edge probes remain active.
SEARCH_SPACE<-list(
 Ridge=list(lambda_q=seq(-40L,32L,1L)),
 Elastic_Net=list(alpha_pct=c(0L,10L,30L,50L,70L,90L,100L),lambda_q=seq(-36L,32L,2L)),
 Random_Forest=list(
   num_trees=c(800L,1200L,1600L,1800L,2000L),
   mtry_pct=c(5L,10L,20L,35L,50L,70L,100L),
   min_node=c(1L,2L,4L,8L,16L,32L),
   sample_pct=c(70L,80L,90L,95L,100L),
   max_depth=c(0L,8L,16L,24L)),
 XGBoost=list(
   eta_milli=c(2L,3L,5L,7L,10L,15L,25L,40L,60L),
   max_depth=c(3L,5L,7L,9L,11L,12L,14L),
   child_weight_q4=c(0L,2L,4L,8L,16L,32L,64L),
   subsample_pct=c(50L,55L,60L,65L,75L,85L,95L,100L),
   colsample_pct=c(50L,55L,60L,70L,80L,90L,100L),
   lambda=c(0L,1L,5L,10L,20L,40L),
   alpha=c(0L,1L,3L,10L,20L,40L),
   gamma_milli=c(0L,25L,100L,300L,700L,1500L),
   max_cat_threshold=c(1L,2L,3L,4L,8L,16L,32L,64L,128L)),
 LightGBM=list(
   learning_rate_milli=c(2L,3L,5L,7L,10L,15L,30L,60L),
   num_leaves=c(15L,31L,63L,127L,255L,319L,399L),
   max_depth=c(-1L,4L,6L,8L,10L,12L),
   min_data_in_leaf=c(0L,1L,2L,4L,8L,15L,30L,60L,100L,140L),
   feature_pct=c(35L,37L,45L,55L,65L,75L,85L,95L,100L),
   bagging_pct=c(52L,60L,65L,70L,80L,90L,100L),
   lambda_l2=c(0L,1L,5L,10L,20L,40L,50L),
   cat_smooth=c(0L,5L,10L,20L,50L,75L,78L,98L),
   min_split_gain_milli=c(0L,25L,100L,300L,700L,1500L)))

HARD<-list(
 Ridge=list(lambda_q=c(-48L,40L)),
 Elastic_Net=list(alpha_pct=c(0L,100L),lambda_q=c(-48L,40L)),
 Random_Forest=list(num_trees=c(500L,2400L),mtry_pct=c(1L,100L),min_node=c(1L,64L),
    sample_pct=c(50L,100L),max_depth=c(0L,48L)),
 XGBoost=list(eta_milli=c(1L,120L),max_depth=c(2L,16L),child_weight_q4=c(0L,1024L),
    subsample_pct=c(50L,100L),colsample_pct=c(50L,100L),lambda=c(0L,160L),
     alpha=c(0L,640L),gamma_milli=c(0L,5000L),max_cat_threshold=c(1L,256L)),
 LightGBM=list(learning_rate_milli=c(1L,120L),num_leaves=c(7L,1023L),
     max_depth=c(-1L,20L),min_data_in_leaf=c(0L,240L),feature_pct=c(50L,100L),bagging_pct=c(50L,100L),
    lambda_l2=c(0L,160L),cat_smooth=c(0L,150L),min_split_gain_milli=c(0L,5000L)))

# Retune from scratch: old results must never be used as candidates/metrics.
BASE_V10_VALID <- FALSE
base_seed_candidate <- function(model) data.table::data.table()

# Safety caps are algorithm/practical guardrails, NOT claimed physical optima.
# Original v12 preserves the initial feature_pct down to 35; the former guardrail of
# 50 must be lowered to allow genuine outward edge probes, not false acceptance
# at an artificial cap. Here soft minimum=30, final safety floor=20.
# Other HARD limits and mathematically legitimate endpoints retain v9 rules.
SOFT_HARD <- HARD
SOFT_HARD$LightGBM$feature_pct <- c(30L,100L)
HARD$Ridge$lambda_q                <- c(-52L,44L)
HARD$Elastic_Net$lambda_q          <- c(-52L,44L)
HARD$Random_Forest$num_trees      <- c(500L,2800L)
HARD$Random_Forest$min_node       <- c(1L,80L)
HARD$Random_Forest$max_depth      <- c(0L,56L)
HARD$XGBoost$eta_milli            <- c(1L,140L)
HARD$XGBoost$max_depth           <- c(1L,18L)
HARD$XGBoost$child_weight_q4     <- c(0L,1280L)
HARD$XGBoost$lambda              <- c(0L,200L)
HARD$XGBoost$alpha               <- c(0L,800L)
HARD$XGBoost$gamma_milli         <- c(0L,6000L)
HARD$XGBoost$max_cat_threshold   <- c(1L,320L)
HARD$LightGBM$learning_rate_milli<- c(1L,140L)
HARD$LightGBM$num_leaves         <- c(2L,1279L)
HARD$LightGBM$max_depth          <- c(-1L,22L)
HARD$LightGBM$min_data_in_leaf   <- c(0L,300L)
HARD$LightGBM$lambda_l2          <- c(0L,200L)
HARD$LightGBM$cat_smooth         <- c(0L,200L)
HARD$LightGBM$min_split_gain_milli<-c(0L,6000L)
HARD$XGBoost$subsample_pct    <- c(35L,100L)
HARD$XGBoost$colsample_pct    <- c(35L,100L)
HARD$LightGBM$bagging_pct     <- c(35L,100L)
HARD$LightGBM$feature_pct     <- c(20L,100L)
HARD$Random_Forest$sample_pct <- c(40L,100L)
# No need to force endpoints such as gamma=0 or 100% feature sampling inward.
# Other unchanged bounds in HARD reflect fixed mathematical or practical limits.

# This directory is deliberately different from the first training run.
# Resuming is permitted ONLY when its manifest matches the present input, seeds, folds and configuration.
if (dir.exists(OUT) && length(list.files(OUT, all.files=FALSE)) > 0L &&
    !file.exists(file.path(OUT,'00_RUN_MANIFEST.rds')))
  stop('The independent output directory already contains unverified files: ',OUT,
       '. Choose another NEW output directory rather than mixing results.')
dir.create(OUT,recursive=TRUE,showWarnings=FALSE)
writeLines(capture.output(sessionInfo()),file.path(OUT,'SESSION_INFO.txt'))
MANIFEST<-list(version=VERSION, data=unname(tools::md5sum(DATA_FILE)),
               folds=unname(tools::md5sum(FOLD_FILE)), models=MODEL_NAMES,
               seed=MASTER_SEED, fold_seed=FOLD_SEED, feature_list=FEATURES, n_repeats=N_REPEATS,
               k=K,threshold=THRESHOLD,gap_limit=GAP_LIMIT,gap_rule='all_5_repeats',search_objective='CV_R2_soft_gap_penalty_0p25',
               budgets=BUDGETS, xgboost_version=as.character(packageVersion('xgboost')),
               regularization_gamma_search=TRUE,
               expand_batch_budget=EXPAND_BUDGET, edge_round_limit_default=25L,
               historical_seed_enabled=FALSE,
               historical_seed_verified=BASE_V10_VALID,
               stability_reference_run='NONE_NEW_DATA',
               stability_reference_seed=NA_integer_,
               parameter_domains=SEARCH_SPACE,original_soft_bounds=SOFT_HARD,
                hard_bounds=HARD,edge_policy='v12_probe_artificial_vs_legal_without_faking_optimum')
# Canonical, field-by-field comparison avoids data.table's internal attribute
# differences producing false manifest conflicts on Windows/R re-sourcing.
# Data hash, folds, features, domains, budgets, seed, and versions MUST match.
canonical_manifest <- function(x) {
  if (inherits(x, 'data.table') || is.data.frame(x))
    return(lapply(as.data.frame(x, stringsAsFactors=FALSE), canonical_manifest))
  if (is.list(x)) return(lapply(x, canonical_manifest))
  if (is.integer(x) || is.double(x)) return(unname(as.numeric(x)))
  if (is.character(x)) return(unname(as.character(x)))
  if (is.logical(x)) return(unname(x))
  x
}
MF<-file.path(OUT,'00_RUN_MANIFEST.rds')
if(file.exists(MF)){
 old<-readRDS(MF)
 if(!setequal(names(old),names(MANIFEST)))
   stop('Manifest field names differ: ',paste(setdiff(names(MANIFEST),names(old)),collapse=', '))
 different<-names(MANIFEST)[!vapply(names(MANIFEST),function(nm)
   identical(canonical_manifest(old[[nm]]),canonical_manifest(MANIFEST[[nm]])),logical(1))]
 if(length(different))stop('Critical run settings/data changed in: ',
   paste(different,collapse=', '),'. Use NEW output directory; do not delete checkpoints.')
}else saveRDS(MANIFEST,MF)
fwrite(FOLDS,file.path(OUT,'01_FOLD_ASSIGNMENTS_COPY.csv'))
writeLines(c(paste('Logical cores',LOGICAL,'Physical cores',PHYSICAL),
             paste('CPU budget',CPU_BUDGET),'Automatic per-model worker allocation:',
             capture.output(print(HARDWARE_PLAN))),file.path(OUT,'HARDWARE_PLAN.txt'))
fwrite(HARDWARE_PLAN,file.path(OUT,'HARDWARE_PLAN.csv'))
# Machine-readable v12 design; regenerated identically on resume.
DESIGN_GRID <- data.table::rbindlist(lapply(MODEL_NAMES,function(m) {
 data.table::rbindlist(lapply(names(SEARCH_SPACE[[m]]),function(p) {
  vals <- as.integer(SEARCH_SPACE[[m]][[p]])
  data.table(Model=m, Parameter=p, Initial_Grid_Min=min(vals),
             Initial_Grid_Max=max(vals), Grid_Values=paste(vals,collapse=';'),
             Soft_Safety_Min=SOFT_HARD[[m]][[p]][1L],
             Soft_Safety_Max=SOFT_HARD[[m]][[p]][2L],
             Hard_Safety_Min=HARD[[m]][[p]][1L],
             Hard_Safety_Max=HARD[[m]][[p]][2L])
 }))
}))
fwrite(DESIGN_GRID,file.path(OUT,'FINAL_V12_SEARCH_DESIGN.csv'))
fwrite(BUDGETS,file.path(OUT,'FINAL_V12_STAGE_BUDGETS.csv'))

# Frozen scientific protocol for a new-data, genuinely fresh retraining run.
writeLines(c(
  'V12 FRESH RETRAIN FROM MOA-CORRECTED TRAINING DATABASE',
  paste('Dataset:',DATA_FILE),paste('Dataset MD5:',unname(tools::md5sum(DATA_FILE))),
  paste('Shared fold MD5:',unname(tools::md5sum(FOLD_FILE))),
  paste('Master seed:',MASTER_SEED),
  paste('Fold seed:',FOLD_SEED),
  paste('Independent run output:',OUT),
  'Exactly 24 native predictors; only Model_MOA_final is the MOA predictor.',
  'Specific_MOA_* columns must NOT become training predictors.',
  'Five repeated Taxon-blocked folds, all 25 OOF fits per candidate.',
  'Five model families, stages and strict <=0.20 per-repeat R2 gap retained.',
  'No old-model weights, CV scores, OOF files or warm-start hyperparameters reused.',
  'This is model redevelopment on corrected training data, not external validation.'
),file.path(OUT,'V12_MOA_RETRAIN_PROVENANCE.txt'))

clean_cat<-function(x) { x<-trimws(gsub('\u00a0',' ',as.character(x),fixed=TRUE));x[is.na(x)|x=='']<-'__MISSING__';x }
fit_enc<-function(tr) {
 med<-setNames(vapply(NUMS,function(v){x<-tr[[v]];x<-x[is.finite(x)];if(length(x)) median(x) else 0},numeric(1)),NUMS)
 lev<-setNames(lapply(CATS,function(v)sort(unique(clean_cat(tr[[v]])))),CATS)
 list(med=med,lev=lev)
}
make_features<-function(dt,enc,mode) {
 z<-lapply(NUMS,function(v){x<-dt[[v]];x[!is.finite(x)]<-enc$med[[v]];as.numeric(x)});names(z)<-NUMS
 if(mode=='linear'){
  parts<-list(as.matrix(as.data.frame(z,check.names=FALSE)))
  for(v in CATS){
   lev<-c(enc$lev[[v]],'__NOVEL__');lev<-unique(lev)
   val<-clean_cat(dt[[v]]);val[!val %in% lev]<-'__NOVEL__'
   mat<-matrix(0,nrow(dt),length(lev),dimnames=list(NULL,paste0(v,'__',make.names(lev,unique=TRUE))))
   mat[cbind(seq_len(nrow(dt)),match(val,lev))]<-1
   parts[[length(parts)+1L]]<-mat
  }
  out<-do.call(cbind,parts);storage.mode(out)<-'double';return(out)
 }
 for(v in CATS){
  val<-clean_cat(dt[[v]]);new<-!val %in% enc$lev[[v]]
  if(mode=='xgb') {val[new]<-NA_character_;z[[v]]<-factor(val,levels=enc$lev[[v]])}
  if(mode=='rf') {val[new]<-'__NOVEL__';z[[v]]<-factor(val,levels=unique(c(enc$lev[[v]],'__NOVEL__')))}
  if(mode=='lgb') {val[new]<-NA_character_;z[[v]]<-as.numeric(match(val,enc$lev[[v]])-1L)}
 }
 out<-as.data.frame(z,check.names=FALSE,optional=TRUE)
 stopifnot(identical(names(out),FEATURES),ncol(out)==24L)
 if(mode=='lgb') return(as.matrix(out))
 out
}
# Cached fold data is reused by model workers for all candidate parameters.
CACHE<-NULL;ACTIVE_MODEL<-NULL
prepare_worker<-function(model) {
 ACTIVE_MODEL<<-model
 mode<-switch(model,Ridge='linear',Elastic_Net='linear',Random_Forest='rf',XGBoost='xgb',LightGBM='lgb')
 CACHE<<-lapply(seq_along(FOLD_IDX),function(r)lapply(seq_along(FOLD_IDX[[r]]),function(f){
  ix<-FOLD_IDX[[r]][[f]]; tr<-DT[ix$tr];va<-DT[ix$va]
  enc<-fit_enc(tr)
  # Linear matrices may exceed RAM when cached across all 25 folds.
  if(mode=='linear')return(list(ix=ix,enc=enc))
  xtr<-make_features(tr,enc,mode);xva<-make_features(va,enc,mode)
  if(mode=='xgb') {
   xtr<-xgboost::xgb.DMatrix(xtr,label=tr[[TARGET]])
   xva<-xgboost::xgb.DMatrix(xva)
  } else if(mode=='lgb') {
   raw_train<-xtr
   xtr<-lightgbm::lgb.Dataset(xtr,label=tr[[TARGET]],categorical_feature=CATS,
                              params=list(feature_pre_filter=FALSE),free_raw_data=FALSE)
   xtr$construct(); # share same correctly constructed binning across candidates
  }
  if(mode=='lgb')return(list(ix=ix,enc=enc,xtr=xtr,xva=xva,rawtr=raw_train))
  list(ix=ix,enc=enc,xtr=xtr,xva=xva)
 }))
 invisible(TRUE)
}
rounds_xgb<-function(e) { if(e<=2)2800L else if(e<=5)2200L else if(e<=10)1600L else if(e<=20)1100L else if(e<=35)850L else if(e<=50)650L else 500L }
rounds_lgb<-function(e) { if(e<=2)2600L else if(e<=5)2000L else if(e<=10)1500L else if(e<=20)1000L else if(e<=35)750L else if(e<=50)600L else 450L }
model_params<-function(model,p,seed) {
 if(model=='XGBoost')return(list(objective='reg:squarederror',eval_metric='rmse',
    tree_method='hist',max_cat_to_onehot=1L,max_cat_threshold=as.integer(p$max_cat_threshold),
    eta=p$eta_milli/1000,max_depth=as.integer(p$max_depth),
    min_child_weight=p$child_weight_q4/4,subsample=p$subsample_pct/100,
    colsample_bytree=p$colsample_pct/100,lambda=p$lambda,alpha=p$alpha,
    gamma=p$gamma_milli/1000,  # moderate split regularization
    nthread=N_THREADS,seed=as.integer(seed)))
 if(model=='LightGBM')return(list(objective='regression',metric='rmse',
    learning_rate=p$learning_rate_milli/1000,num_leaves=as.integer(p$num_leaves),
    max_depth=as.integer(p$max_depth),min_data_in_leaf=as.integer(p$min_data_in_leaf),
    feature_fraction=p$feature_pct/100,bagging_fraction=p$bagging_pct/100,
    bagging_freq=1L,lambda_l2=p$lambda_l2,cat_smooth=as.integer(p$cat_smooth),
    min_gain_to_split=p$min_split_gain_milli/1000, # split regularization
    feature_pre_filter=FALSE,verbosity=-1L,num_threads=N_THREADS,
    seed=as.integer(seed),feature_fraction_seed=as.integer(seed),
    bagging_seed=as.integer(seed),data_random_seed=as.integer(seed)))
 stop('Unexpected tree model')
}
fit_predict<-function(model,p, fold,rr,ff, full=FALSE) {
 ix<-fold$ix;tr<-DT[ix$tr];va<-DT[ix$va]
 seed<-MASTER_SEED+100000L*rr+1000L*ff+10L*match(model,MODEL_NAMES)
 if(model %in% c('Ridge','Elastic_Net')) {
  xtr<-make_features(tr,fold$enc,'linear');xva<-make_features(va,fold$enc,'linear')
  s<-apply(xtr,2L,sd);keep<-is.finite(s)&s>0
  xtr<-xtr[,keep,drop=FALSE];xva<-xva[,keep,drop=FALSE]
  lam<-10^(p$lambda_q/4); al<-if(model=='Ridge')0 else p$alpha_pct/100
  fit<-glmnet::glmnet(xtr,tr[[TARGET]],family='gaussian',alpha=al,
                      lambda=lam,standardize=TRUE,intercept=TRUE)
  prtr<-as.numeric(predict(fit,newx=xtr,s=lam));prva<-as.numeric(predict(fit,newx=xva,s=lam))
 } else if(model=='Random_Forest') {
  depth<-as.integer(p$max_depth)
  # ranger's formula parser rejects valid scientific names containing spaces.
  # x/y interface keeps the EXACT original 24 variables, including Factor columns.
  args<-list(x=fold$xtr,y=tr[[TARGET]],
    num.trees=as.integer(p$num_trees),
    mtry=max(1L,as.integer(round(24*p$mtry_pct/100))),
    min.node.size=as.integer(p$min_node),sample.fraction=p$sample_pct/100,
    replace=TRUE,num.threads=N_THREADS,seed=as.integer(seed),
    respect.unordered.factors='order',importance='none',verbose=FALSE)
  if(depth>0L) args$max.depth<-depth
  fit<-do.call(ranger::ranger,args)
  prtr<-as.numeric(predict(fit,data=fold$xtr,num.threads=N_THREADS)$predictions)
  prva<-as.numeric(predict(fit,data=fold$xva,num.threads=N_THREADS)$predictions)
 } else if(model=='XGBoost') {
  fit<-xgboost::xgb.train(model_params(model,p,seed),data=fold$xtr,
                          nrounds=rounds_xgb(p$eta_milli),verbose=0)
  prtr<-as.numeric(predict(fit,fold$xtr));prva<-as.numeric(predict(fit,fold$xva))
 } else if(model=='LightGBM') {
  fit<-lightgbm::lgb.train(params=model_params(model,p,seed),data=fold$xtr,
                           nrounds=rounds_lgb(p$learning_rate_milli),verbose=-1)
  prtr<-as.numeric(predict(fit,fold$rawtr))
  prva<-as.numeric(predict(fit,fold$xva))
 } else stop('Unknown model')
 list(tr=prtr,va=prva)
}
score<-function(y,p){
 stopifnot(length(y)==length(p),all(is.finite(p)))
 sse<-sum((y-p)^2);sst<-sum((y-mean(y))^2)
 c(R2=if(sst>0)1-sse/sst else NA_real_,RMSE=sqrt(mean((y-p)^2)),MAE=mean(abs(y-p)))
}
model_dir<-function(model)file.path(OUT,'02_MODELS',model)
param_cols<-function(model)names(SEARCH_SPACE[[model]])
param_key<-function(z,params)paste(vapply(params,function(nm)as.character(z[[nm]][1]),character(1)),collapse='|')
# Each saved checkpoint is validated before reuse. OOF is pooled per repeat.
evaluate_candidate<-function(p) {
 model<-ACTIVE_MODEL;id<-as.integer(p$Candidate)
 cdir<-file.path(model_dir(model),'checkpoints',sprintf('candidate_%04d',id))
 dir.create(cdir,recursive=TRUE,showWarnings=FALSE)
 signature<-param_key(p,param_cols(model));rmet<-vector('list',5)
 for(rr in 1:5) {
  cpfile<-file.path(cdir,sprintf('repeat_%02d.rds',rr))
  good<-FALSE
  if(file.exists(cpfile)) {
   cp<-tryCatch(readRDS(cpfile),error=function(e)NULL)
   good<-is.list(cp)&&identical(cp$key,signature)&&identical(cp$repeat_id,rr)&&
     nrow(cp$oof)==nrow(DT)&&!anyDuplicated(cp$oof$Row_ID)&&
     all(cp$oof$Row_ID %in% DT$Row_ID)&&nrow(cp$metric)==1L
  }
  if(good){rmet[[rr]]<-cp$metric;next}
  oof<-vector('list',5); trains<-vector('list',5)
  for(ff in 1:5){
   fold<-CACHE[[rr]][[ff]];ix<-fold$ix
   pr<-fit_predict(model,p,fold,rr,ff)
   oof[[ff]]<-data.table(Row_ID=ix$va,Repeat=rr,Fold=ff,Taxon=DT$Taxon[ix$va],
                         Observed=DT[[TARGET]][ix$va],Predicted=pr$va)
   trains[[ff]]<-data.table(Observed=DT[[TARGET]][ix$tr],Predicted=pr$tr)
  }
  oo<-rbindlist(oof);tt<-rbindlist(trains)
  stopifnot(nrow(oo)==nrow(DT),!anyDuplicated(oo$Row_ID))
  cv<-score(oo$Observed,oo$Predicted);train<-score(tt$Observed,tt$Predicted)
  mm<-data.table(Model=model,Candidate=id,Repeat=rr,Train_R2=train['R2'],
     CV_R2=cv['R2'],CV_RMSE=cv['RMSE'],CV_MAE=cv['MAE'],
     Train_RMSE=train['RMSE'],Train_MAE=train['MAE'],R2_gap=train['R2']-cv['R2'])
  cp<-list(key=signature,repeat_id=rr,metric=mm,oof=oo)
  tmp<-paste0(cpfile,'.tmp_',Sys.getpid());saveRDS(cp,tmp,compress=TRUE)
  if(file.exists(cpfile))unlink(cpfile)
  stopifnot(file.rename(tmp,cpfile))
  rmet[[rr]]<-mm
 }
 met<-rbindlist(rmet)
 out<-data.table(Model=model,Candidate=id,Stage=p$Stage,
  Mean_Train_R2=mean(met$Train_R2),Mean_CV_R2=mean(met$CV_R2),
  SD_CV_R2=sd(met$CV_R2),Mean_CV_RMSE=mean(met$CV_RMSE),
  Mean_CV_MAE=mean(met$CV_MAE),Mean_R2_Gap=mean(met$R2_gap),
  Max_R2_Gap=max(met$R2_gap),
  Gap_Feasible=all(is.finite(met$R2_gap) & met$R2_gap<=GAP_LIMIT+GAP_TOLERANCE))
 for(k in param_cols(model))out[,(k):=p[[k]][1]]
 fwrite(out,file.path(cdir,'summary.csv'))
 fwrite(met,file.path(cdir,'repeat_metrics.csv'))
 out
}
# Shared optimization functions inserted below.
# ---------- Shared compact, discrete TPE optimizer (no optimizer dependency) ------
# TPE = separate kernel-smoothed distributions for top 20% vs remaining trials;
# candidates maximize log l(x)/g(x). Batched ask/tell for Windows parallelism.
space_grid <- function(space, n, seed, exclude=character(), method=c('uniform','tpe','local'), history=NULL) {
  method <- match.arg(method)
  set.seed(as.integer(seed))
  nm <- names(space)
  key <- function(z) apply(as.matrix(z[,..nm]),1L,paste,collapse='|')
  if (!is.null(history) && nrow(history)) exclude <- unique(c(exclude,key(history)))
  if (method=='tpe' && (is.null(history) || nrow(history)<10L)) method <- 'uniform'
  cand <- vector('list', max(200L, 60L*n))
  if (method=='tpe') {
    h <- data.table::copy(history)
    data.table::setorder(h,-SearchUtility)
    ng <- max(4L,as.integer(ceiling(.20*nrow(h))))
    good <- h[seq_len(ng)]
    bad <- h[seq.int(ng+1L,nrow(h))]
    kernel_prob <- function(values, observed) {
      k <- length(values); where <- match(observed,values)
      where <- where[!is.na(where)]
      prior <- rep(1/k,k)
      if (!length(where)) return(prior)
      bw <- max(1, k*.12)
      probs <- vapply(seq_len(k),function(j) sum(exp(-.5*((j-where)/bw)^2)),numeric(1))
      probs <- probs/sum(probs)
      .85*probs+.15*prior
    }
    pg <- lapply(nm,function(p)kernel_prob(space[[p]],good[[p]]))
    pb <- lapply(nm,function(p)kernel_prob(space[[p]],bad[[p]]))
    names(pg)<-names(pb)<-nm
    for(i in seq_along(cand)) {
      vals <- lapply(nm, function(p) {
        v<-space[[p]]
        if(runif(1)<.14) sample(v,1) else sample(v,1,prob=pg[[p]])
      })
      cand[[i]]<-setNames(vals,nm)
    }
  } else if (method=='local') {
    h<-data.table::copy(history)
    data.table::setorder(h,-SearchUtility)
    elite<-h[seq_len(min(nrow(h),5L))]
    for(i in seq_along(cand)) {
      parent<-elite[sample(seq_len(nrow(elite)),1L)]
      vals<-lapply(nm,function(p){
        v<-space[[p]]; j<-match(parent[[p]][1],v)
        if(is.na(j)) j<-which.min(abs(v-parent[[p]][1]))
        if(runif(1)<.13) sample(v,1) else v[max(1L,min(length(v),j+sample(c(-2L,-1L,0L,1L,2L),1L,prob=c(.1,.25,.3,.25,.1))))]
      })
      cand[[i]]<-setNames(vals,nm)
    }
  } else {
    # Stratified Latin-hypercube-like marginal coverage, randomized per parameter.
    perms<-lapply(space,function(v) sample(rep(v,length.out=length(cand))))
    for(i in seq_along(cand)) cand[[i]]<-lapply(perms,function(v)v[[i]])
  }
  proposed<-unique(data.table::rbindlist(lapply(cand,data.table::as.data.table),fill=TRUE),by=nm)
  if(!is.null(history) && nrow(history)) {
    proposed <- proposed[!key(proposed) %in% exclude]
  } else proposed<-proposed[!key(proposed) %in% exclude]
  if (method=='tpe' && nrow(proposed)) {
    proposed[, acquisition := 0]
    for(p in nm) {
      ig<-match(proposed[[p]],space[[p]])
      proposed[,acquisition:=acquisition+log(pg[[p]][ig]+1e-12)-log(pb[[p]][ig]+1e-12)]
    }
    data.table::setorder(proposed,-acquisition)
    proposed[,acquisition:=NULL]
  }
  if (method=='local' && nrow(proposed)) {
    # Preserve proximity to diverse elite solutions.
    scores<-vapply(seq_len(nrow(proposed)),function(i){
      min(vapply(seq_len(nrow(elite)),function(j){
        sum(vapply(nm,function(p) abs(match(proposed[[p]][i],space[[p]])-match(elite[[p]][j],space[[p]])),numeric(1)))
      },numeric(1)))
    },numeric(1))
    proposed[,local_score:=scores];data.table::setorder(proposed,local_score)
    proposed[,local_score:=NULL]
  }
  proposed[seq_len(min(n,nrow(proposed)))]
}

# ---------------- Stages, repeat-safe checkpoints, and final model -----------
collect_scores<-function(model){
 root<-file.path(model_dir(model),'checkpoints')
 if(!dir.exists(root))return(data.table())
 fs<-list.files(root,pattern='^summary\\.csv$',recursive=TRUE,full.names=TRUE)
 if(!length(fs))return(data.table())
 z<-rbindlist(lapply(fs,fread),fill=TRUE)
 z<-unique(z,by='Candidate')
 z<-rank_constrained(z)
 z
}
collect_repeats<-function(model){
 root<-file.path(model_dir(model),'checkpoints')
 fs<-list.files(root,pattern='^repeat_metrics\\.csv$',recursive=TRUE,full.names=TRUE)
 if(length(fs))rbindlist(lapply(fs,fread),fill=TRUE)else data.table()
}

# Export the selected candidate's row-level OOF predictions for fitting/validation plots.
# Long OOF contains 5 predictions per training row (one per repeat); the mean file contains
# exactly one honest cross-validated prediction per Row_ID by averaging the 5 repeats.
export_best_oof<-function(model,best){
 dir<-model_dir(model)
 wdir<-file.path(dir,'checkpoints',sprintf('candidate_%04d',as.integer(best$Candidate)))
 cpfiles<-file.path(wdir,sprintf('repeat_%02d.rds',seq_len(N_REPEATS)))
 if(!all(file.exists(cpfiles)))
   stop('Missing best-candidate repeat checkpoint(s) for ',model,' candidate ',best$Candidate)
 oof<-rbindlist(lapply(cpfiles,function(f)readRDS(f)$oof),fill=TRUE)
 stopifnot(nrow(oof)==N_REPEATS*nrow(DT),
           all(oof$Row_ID %in% DT$Row_ID))
 oof[,`:=`(Residual=Observed-Predicted,
           Abs_Error=abs(Observed-Predicted),
           Squared_Error=(Observed-Predicted)^2)]
 setorder(oof,Row_ID,Repeat,Fold)
 fwrite(oof,file.path(dir,'10_BEST_OOF.csv'))

 oof_mean<-oof[,.(Observed=Observed[1L],
                  Predicted=mean(Predicted),
                  Prediction_SD=sd(Predicted),
                  Residual=Observed[1L]-mean(Predicted),
                  Abs_Error=abs(Observed[1L]-mean(Predicted)),
                  Squared_Error=(Observed[1L]-mean(Predicted))^2,
                  N_Repeats=.N),
               by=.(Row_ID,Taxon)]
 stopifnot(nrow(oof_mean)==nrow(DT),all(oof_mean$N_Repeats==N_REPEATS))
 setorder(oof_mean,Row_ID)
 fwrite(oof_mean,file.path(dir,'10_BEST_OOF_MEAN_BY_ROW.csv'))
 invisible(oof_mean)
}

# Build one ready-to-plot table containing both in-sample final-fit predictions and
# cross-validated OOF predictions. Use CV_OOF_5REPEAT_MEAN for an honest fitting plot.
export_fitting_plot_data<-function(model){
 dir<-model_dir(model)
 trfile<-file.path(dir,'FINAL_MODEL','TRAINING_FITTED_VALUES.csv')
 cvfile<-file.path(dir,'10_BEST_OOF_MEAN_BY_ROW.csv')
 if(!file.exists(trfile) || !file.exists(cvfile))return(invisible(FALSE))
 tr<-fread(trfile,select=c('Row_ID',GROUP,'Observed','Predicted','Residual'))
 cv<-fread(cvfile,select=c('Row_ID','Taxon','Observed','Predicted','Residual',
                           'Prediction_SD','N_Repeats'))
 trplot<-data.table(Model=model,Prediction_Type='TRAIN_FULL_FIT',
   Row_ID=tr$Row_ID,Taxon=tr[[GROUP]],Observed=tr$Observed,
   Predicted=tr$Predicted,Residual=tr$Residual,
   Prediction_SD=NA_real_,N_Repeats=NA_integer_)
 cvplot<-data.table(Model=model,Prediction_Type='CV_OOF_5REPEAT_MEAN',
   Row_ID=cv$Row_ID,Taxon=cv$Taxon,Observed=cv$Observed,
   Predicted=cv$Predicted,Residual=cv$Residual,
   Prediction_SD=cv$Prediction_SD,N_Repeats=cv$N_Repeats)
 fwrite(rbindlist(list(trplot,cvplot),use.names=TRUE,fill=TRUE),
        file.path(dir,'11_FITTING_PLOT_DATA.csv'))
 invisible(TRUE)
}
full_fit<-function(model,best){
 d<-file.path(model_dir(model),'FINAL_MODEL');dir.create(d,recursive=TRUE,showWarnings=FALSE)
 enc<-fit_enc(DT);y<-DT[[TARGET]];seed<-MASTER_SEED+909090L
 mode<-switch(model,Ridge='linear',Elastic_Net='linear',Random_Forest='rf',XGBoost='xgb',LightGBM='lgb')
 x<-make_features(DT,enc,mode)
 keep<-NULL; preds<-NULL
 if(mode=='linear'){
  s<-apply(x,2L,sd);keep<-colnames(x)[is.finite(s)&s>0]
  x<-x[,keep,drop=FALSE]
  lam<-10^(best$lambda_q/4);al<-if(model=='Ridge')0 else best$alpha_pct/100
  fit<-glmnet::glmnet(x,y,family='gaussian',alpha=al,lambda=lam,standardize=TRUE)
  preds<-as.numeric(predict(fit,x,s=lam))
  saveRDS(fit,file.path(d,'MODEL.rds'))
  coef_tab <- as.matrix(stats::coef(fit,s=lam))
  fwrite(data.table(Variable=rownames(coef_tab),
    Coefficient=as.numeric(coef_tab[,1L]),
    Importance=abs(as.numeric(coef_tab[,1L]))),
    file.path(d,'FEATURE_IMPORTANCE.csv'))
 } else if(model=='Random_Forest'){
  # Same x/y interface as CV; final full-data fit must also support spaces.
  args<-list(x=x,y=y,num.trees=as.integer(best$num_trees),
    mtry=max(1L,as.integer(round(24*best$mtry_pct/100))),
    min.node.size=as.integer(best$min_node),
    sample.fraction=best$sample_pct/100,replace=TRUE,
    num.threads=N_THREADS,seed=as.integer(seed),
    respect.unordered.factors='order',importance='impurity',verbose=FALSE)
  if(best$max_depth>0) args$max.depth<-as.integer(best$max_depth)
  fit<-do.call(ranger::ranger,args)
  preds<-as.numeric(predict(fit,data=x,num.threads=N_THREADS)$predictions)
  saveRDS(fit,file.path(d,'MODEL.rds'))
  fwrite(data.table(Variable=names(fit$variable.importance),
                    Importance=as.numeric(fit$variable.importance)),
         file.path(d,'FEATURE_IMPORTANCE.csv'))
 } else if(model=='XGBoost'){
  mat<-xgboost::xgb.DMatrix(x,label=y)
  fit<-xgboost::xgb.train(params=model_params(model,best,seed),data=mat,
                           nrounds=rounds_xgb(best$eta_milli),verbose=0)
  preds<-as.numeric(predict(fit,mat))
  xgboost::xgb.save(fit,file.path(d,'MODEL.json'))
  fwrite(as.data.table(xgboost::xgb.importance(model=fit)),file.path(d,'FEATURE_IMPORTANCE.csv'))
 } else if(model=='LightGBM'){
  mat<-lightgbm::lgb.Dataset(x,label=y,categorical_feature=CATS,
                             params=list(feature_pre_filter=FALSE),free_raw_data=FALSE)
  fit<-lightgbm::lgb.train(params=model_params(model,best,seed),data=mat,
                           nrounds=rounds_lgb(best$learning_rate_milli),verbose=-1)
  preds<-as.numeric(predict(fit,x))
  lightgbm::lgb.save(fit,file.path(d,'MODEL.txt'))
  fwrite(as.data.table(lightgbm::lgb.importance(fit)),file.path(d,'FEATURE_IMPORTANCE.csv'))
 }
 saveRDS(list(model=model,encoder=enc,feature_order=FEATURES,keep=keep,
      numeric=NUMS,categorical=CATS,best=best,seed=seed,
      data_md5=unname(tools::md5sum(DATA_FILE))),file.path(d,'MODEL_BUNDLE.rds'))
 stopifnot(length(preds)==nrow(DT),all(is.finite(preds)))
 # Persist every original training row plus final-model fitted prediction.
 # This supports fitting plots and downstream colouring/faceting without re-joining source data.
 fit_values<-copy(DT)
 fit_values[,`:=`(Observed=as.numeric(get(TARGET)),
                  Predicted=as.numeric(preds),
                  Residual=as.numeric(get(TARGET))-as.numeric(preds),
                  Abs_Error=abs(as.numeric(get(TARGET))-as.numeric(preds)),
                  Squared_Error=(as.numeric(get(TARGET))-as.numeric(preds))^2)]
 lead_cols<-unique(c('Row_ID',GROUP,TARGET,'Observed','Predicted','Residual',
                     'Abs_Error','Squared_Error'))
 setcolorder(fit_values,c(lead_cols,setdiff(names(fit_values),lead_cols)))
 fwrite(fit_values,file.path(d,'TRAINING_FITTED_VALUES.csv'))

 met<-score(y,preds)
 fwrite(data.table(Model=model,Train_R2=met['R2'],Train_RMSE=met['RMSE'],
                   Train_MAE=met['MAE']),file.path(d,'TRAIN_METRICS.csv'))
}
# --- Standalone prediction and native SHAP helpers (no training required) ---
ec_clean_cat <- function(x) {
  x <- trimws(gsub("\u00a0", " ", as.character(x), fixed=TRUE))
  x[is.na(x) | x == ""] <- "__MISSING__"
  x
}
ec_load <- function(path) {
  d <- normalizePath(path, winslash="/", mustWork=TRUE)
  bf <- file.path(d, "MODEL_BUNDLE.rds")
  if (!file.exists(bf)) stop("MODEL_BUNDLE.rds not found in: ", d)
  b <- readRDS(bf)
  mf <- switch(as.character(b$model), Ridge="MODEL.rds", Elastic_Net="MODEL.rds",
    Random_Forest="MODEL.rds", XGBoost="MODEL.json", LightGBM="MODEL.txt",
    stop("Unsupported model type: ", b$model))
  fn <- file.path(d, mf)
  if (!file.exists(fn)) stop("Trained model file is missing: ", fn)
  fit <- if (b$model %in% c("Ridge", "Elastic_Net", "Random_Forest")) readRDS(fn) else
    if (b$model == "XGBoost") xgboost::xgb.load(fn) else lightgbm::lgb.load(fn)
  list(metadata=b, fit=fit, path=d)
}
ec_encode <- function(raw_data, loaded) {
  b <- loaded$metadata
  raw <- as.data.frame(raw_data, check.names=FALSE, optional=TRUE)
  absent <- setdiff(b$feature_order, names(raw))
  if (length(absent)) stop("Missing exact native feature columns: ", paste(absent, collapse=", "))
  n <- nrow(raw)
  if (n < 1L) stop("Input data has no rows")
  vals <- lapply(b$numeric, function(v) {
    z <- suppressWarnings(as.numeric(as.character(raw[[v]])))
    z[!is.finite(z)] <- as.numeric(b$encoder$med[[v]])
    z
  })
  names(vals) <- b$numeric
  linear <- b$model %in% c("Ridge", "Elastic_Net")
  if (linear) {
    parts <- list(as.matrix(as.data.frame(vals,check.names=FALSE)))
    for (v in b$categorical) {
      lev <- unique(c(b$encoder$lev[[v]], "__NOVEL__"))
      z <- ec_clean_cat(raw[[v]])
      z[!z %in% lev] <- "__NOVEL__"
      mm <- matrix(0,n,length(lev),dimnames=list(NULL,paste0(v,"__",make.names(lev,unique=TRUE))))
      mm[cbind(seq_len(n),match(z,lev))] <- 1
      parts[[length(parts)+1L]] <- mm
    }
    x <- do.call(cbind,parts)
    storage.mode(x) <- "double"
    missing_kept <- setdiff(b$keep,colnames(x))
    if (length(missing_kept)) stop("Training design columns missing: ",paste(missing_kept,collapse=", "))
    return(x[,b$keep,drop=FALSE])
  }
  for (v in b$categorical) {
    z <- ec_clean_cat(raw[[v]])
    unseen <- !z %in% b$encoder$lev[[v]]
    if (b$model == "XGBoost") {
      z[unseen] <- NA_character_
      vals[[v]] <- factor(z, levels=b$encoder$lev[[v]])
    } else if (b$model == "Random_Forest") {
      z[unseen] <- "__NOVEL__"
      vals[[v]] <- factor(z,levels=unique(c(b$encoder$lev[[v]],"__NOVEL__")))
    } else if (b$model == "LightGBM") {
      z[unseen] <- NA_character_
      vals[[v]] <- as.numeric(match(z,b$encoder$lev[[v]])-1L)
    } else stop("Unknown tree model")
  }
  x <- as.data.frame(vals,check.names=FALSE,optional=TRUE)
  if (!identical(names(x),b$feature_order)) stop("Feature order mismatch")
  if (b$model == "LightGBM") return(as.matrix(x))
  x
}
ec_predict <- function(loaded, new_data) {
  b <- loaded$metadata
  x <- ec_encode(new_data, loaded)
  if (b$model %in% c("Ridge", "Elastic_Net"))
    return(as.numeric(predict(loaded$fit,newx=x,s=10^(b$best$lambda_q/4))))
  if (b$model == "Random_Forest")
    return(as.numeric(predict(loaded$fit,data=x,num.threads=1L)$predictions))
  if (b$model == "XGBoost")
    return(as.numeric(predict(loaded$fit,xgboost::xgb.DMatrix(x))))
  if (b$model == "LightGBM")
    return(as.numeric(predict(loaded$fit,x)))
  stop("Unknown model")
}
ec_prediction_table <- function(loaded, new_data) {
  p <- ec_predict(loaded,new_data)
  data.frame(Row_ID=seq_len(NROW(new_data)),
    Model=as.character(loaded$metadata$model),
    Predicted_log10_Geomean_ug_L=p,
    Backtransformed_ug_L=10^p,check.names=FALSE)
}
ec_shap <- function(loaded, new_data, check_additivity=TRUE) {
  b <- loaded$metadata
  if (!b$model %in% c("XGBoost", "LightGBM"))
    stop("Native TreeSHAP is implemented here only for XGBoost/LightGBM. Use a separate appropriate explainer for other models.")
  x <- ec_encode(new_data,loaded)
  as_shap_matrix <- function(z) {
    if (is.null(z)) return(NULL)
    nr <- NROW(new_data); nc <- length(b$feature_order)+1L
    if (is.null(dim(z)) && length(z)==nr*nc)
      return(matrix(z,nrow=nr,ncol=nc,byrow=TRUE))
    as.matrix(z)
  }
  if (b$model == "XGBoost") {
    v <- predict(loaded$fit,xgboost::xgb.DMatrix(x),predcontrib=TRUE)
  } else {
    # R LightGBM supports different prediction signatures across versions.
    v <- tryCatch(predict(loaded$fit,x,type="contrib"),error=function(e)NULL)
    test <- as_shap_matrix(v)
    if (is.null(test) || ncol(test) != length(b$feature_order)+1L)
      v <- predict(loaded$fit,x,predcontrib=TRUE)
  }
  v <- as_shap_matrix(v)
  if (nrow(v) != NROW(new_data) || ncol(v) != length(b$feature_order)+1L)
    stop("TreeSHAP output shape mismatch: expected 24 native features + bias")
  colnames(v) <- c(b$feature_order,"BIAS")
  if (check_additivity) {
    diff <- abs(rowSums(v) - ec_predict(loaded,new_data))
    if (any(!is.finite(diff)) || max(diff)>1e-4)
      stop("SHAP additivity check failed; inspect model/library prediction API compatibility")
  }
  as.data.frame(v,check.names=FALSE)
}

# --- Reproducible downstream artefacts and model export ---
export_downstream_artifacts <- function(model, best) {
  modelroot <- model_dir(model)
  d <- file.path(modelroot,"FINAL_MODEL")
  dir.create(d,recursive=TRUE,showWarnings=FALSE)
  meta <- readRDS(file.path(d,"MODEL_BUNDLE.rds"))
  stopifnot(identical(as.character(meta$model),as.character(model)),
    identical(as.integer(meta$best$Candidate),as.integer(best$Candidate)))
  rawcols <- unique(c("Row_ID",GROUP,TARGET,FEATURES))
  rawref <- DT[, ..rawcols]
  data.table::fwrite(rawref,file.path(d,"SHAP_REFERENCE_NATIVE24.csv"))
  mode <- switch(model,Ridge="linear",Elastic_Net="linear",Random_Forest="rf",
    XGBoost="xgb",LightGBM="lgb")
  # Original raw feature values for SHAP coloring, encoded X for the trained model.
  # This is the full-data final-fit input, NOT CV OOF or independent validation.
  xx <- make_features(DT,meta$encoder,mode)
  if(mode=="linear")xx <- xx[,meta$keep,drop=FALSE]
  stopifnot(NROW(xx)==nrow(DT))
  saveRDS(list(Model=model,Candidate=as.integer(best$Candidate),
    Row_ID=as.integer(DT$Row_ID),Taxon=as.character(DT[[GROUP]]),
    Observed_log10=as.numeric(DT[[TARGET]]),
    Encoded_X=xx, Native_Features=FEATURES,
    Categorical=CATS, Encoding_Mode=mode),
    file.path(d,"SHAP_ENCODED_TRAINING_INPUT.rds"),compress=TRUE)
  # Compact, explicit per-model path-independent metadata for inference.
  saveRDS(list(Format="ECOTOX_NATIVE24_V12",Model=model,
    Candidate=as.integer(best$Candidate),
    Model_File=if(model %in% c("Ridge","Elastic_Net","Random_Forest")) "MODEL.rds" else
       if(model=="XGBoost") "MODEL.json" else "MODEL.txt",
    Metadata_File="MODEL_BUNDLE.rds",Feature_Reference="SHAP_REFERENCE_NATIVE24.csv",
    Encoded_Training_Reference="SHAP_ENCODED_TRAINING_INPUT.rds",
    Prediction_Scale="log10_Geomean_ug_L",Feature_Names=FEATURES,
    Categorical_Feature_Names=CATS),file.path(d,"INFERENCE_MANIFEST.rds"))
  rel <- c("08_BEST_PARAMETERS.csv","09_WINNER_PARAMETER_AUDIT.csv",
    "09_WINNER_BOUNDARY_STATUS.csv", "10_BEST_OOF.csv",
    "10_BEST_OOF_MEAN_BY_ROW.csv","11_FITTING_PLOT_DATA.csv",
    "FINAL_MODEL/MODEL_BUNDLE.rds","FINAL_MODEL/MODEL.rds",
    "FINAL_MODEL/MODEL.json","FINAL_MODEL/MODEL.txt",
    "FINAL_MODEL/FEATURE_IMPORTANCE.csv",
    "FINAL_MODEL/TRAINING_FITTED_VALUES.csv",
    "FINAL_MODEL/TRAIN_METRICS.csv",
    "FINAL_MODEL/SHAP_REFERENCE_NATIVE24.csv",
    "FINAL_MODEL/SHAP_ENCODED_TRAINING_INPUT.rds",
    "FINAL_MODEL/INFERENCE_MANIFEST.rds")
  paths <- file.path(modelroot,rel)
  have <- file.exists(paths)
  paths <- paths[have]; rel <- rel[have]
  data.table::fwrite(data.table::data.table(File=rel,
    Size_Bytes=as.numeric(file.info(paths)$size),
    MD5=as.character(unname(tools::md5sum(paths)))),
    file.path(d,"ARTIFACT_FILE_MANIFEST.csv"))
  invisible(TRUE)
}

export_predict_helper <- function() {
  f <- file.path(OUT,"00_MODEL_PREDICT_SHAP_HELPERS.R")
  writeLines(c(
    "# ECOTOX v12: prediction and native TreeSHAP from saved final model. No training triggered.",
    "# Example:",
    "# source('results/model_tuning_run06/five_model_tuning/00_MODEL_PREDICT_SHAP_HELPERS.R')",
    "# m <- ec_load('results/model_tuning_run06/five_model_tuning/BEST_MODEL_EXPORT')",
    "# new_data <- data.table::fread('your_new_data.csv')  # exactly 24 native feature columns",
    "# predictions <- ec_prediction_table(m, new_data)",
    "# shap_values <- ec_shap(m, new_data)  # XGBoost or LightGBM only",
    "# Prediction values are log10_Geomean_ug_L. SHAP is for the FULL FIT model, not CV evidence.",
    "suppressPackageStartupMessages(library(data.table))",
    ""
  ),f,useBytes=TRUE)
  dump(c("ec_clean_cat","ec_load","ec_encode","ec_predict",
    "ec_prediction_table","ec_shap"),file=f,
    envir=environment(ec_load),append=TRUE)
  stopifnot(file.exists(f),file.info(f)$size>1000)
  invisible(f)
}

export_global_bundle <- function(global_best) {
  model <- as.character(global_best$Model[1L])
  modelroot <- model_dir(model)
  dest <- file.path(OUT,"BEST_MODEL_EXPORT")
  dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  model_file <- if (model %in% c("Ridge","Elastic_Net","Random_Forest"))
    "MODEL.rds" else if (model=="XGBoost") "MODEL.json" else "MODEL.txt"
  needed <- c(
    "FINAL_MODEL/MODEL_BUNDLE.rds",
    paste0("FINAL_MODEL/",model_file),
    "FINAL_MODEL/INFERENCE_MANIFEST.rds",
    "FINAL_MODEL/SHAP_REFERENCE_NATIVE24.csv",
    "FINAL_MODEL/SHAP_ENCODED_TRAINING_INPUT.rds",
    "FINAL_MODEL/ARTIFACT_FILE_MANIFEST.csv",
    "FINAL_MODEL/FEATURE_IMPORTANCE.csv",
    "FINAL_MODEL/TRAINING_FITTED_VALUES.csv",
    "FINAL_MODEL/TRAIN_METRICS.csv",
    "08_BEST_PARAMETERS.csv","09_WINNER_PARAMETER_AUDIT.csv",
    "09_WINNER_BOUNDARY_STATUS.csv",
    "10_BEST_OOF.csv","10_BEST_OOF_MEAN_BY_ROW.csv",
    "11_FITTING_PLOT_DATA.csv")
  # Only remove known old model filenames; never remove user files inside export.
  for (stale in c("MODEL.rds","MODEL.json","MODEL.txt")) {
    fn <- file.path(dest,stale)
    if(file.exists(fn) && stale!=model_file)unlink(fn)
  }
  from <- file.path(modelroot,needed)
  if (!all(file.exists(from)))stop("Cannot assemble full best-model export; missing: ",
    paste(needed[!file.exists(from)],collapse=", "))
  to <- file.path(dest,basename(needed))
  copied <- file.copy(from,to,overwrite=TRUE,copy.mode=TRUE)
  if (!all(copied))stop("Could not copy best-model artifacts: ",
    paste(needed[!copied],collapse=", "))
  extra <- c("BEST_MODEL_SUMMARY.csv","MASTER_MODEL_COMPARISON.csv",
    "FINAL_ALL5_COMPLETENESS_AND_BOUNDARY_REVIEW.csv",
    "FINAL_ALL5_PARAMETER_BOUNDARY_AUDIT.csv",
    "01_FOLD_ASSIGNMENTS_COPY.csv",
    "00_MODEL_PREDICT_SHAP_HELPERS.R",
    "V12_MOA_RETRAIN_PROVENANCE.txt",
    "SESSION_INFO.txt")
  fromextra <- file.path(OUT,extra)
  missing_extra <- extra[!file.exists(fromextra)]
  if(length(missing_extra)) stop("Required final audit/helper files missing BEFORE export: ",
    paste(missing_extra,collapse=", "),"; verify the run output under: ",OUT)
  copied_extra <- file.copy(fromextra,file.path(dest,extra),overwrite=TRUE)
  if(!all(copied_extra))stop("Unable to copy existing final audit/helper files: ",
    paste(extra[!copied_extra],collapse=", "))
  writeLines(c(
    paste("FINAL WINNER:", model, "candidate", as.integer(global_best$Candidate[1])),
    "This folder alone contains the final trained model, its exact preprocessing encoder and feature order,",
    "raw native-24 and encoded full-training references, scatter-ready OOF and in-sample predictions,",
    "parameter and boundary audits, R inference helper, environment information and fold assignment.",
    "Re-source 00_MODEL_PREDICT_SHAP_HELPERS.R only to PREDICT / SHAP; do not source the main training script.",
    "R example:",
    "  source('00_MODEL_PREDICT_SHAP_HELPERS.R')",
    "  m <- ec_load('.')   # working directory is BEST_MODEL_EXPORT",
    "  newdata <- data.table::fread('your_new_data.csv')",
    "  pred <- ec_prediction_table(m,newdata)",
    "  shap <- ec_shap(m,newdata)   # LightGBM/XGBoost only",
    "Make OOF scatter plots from 11_FITTING_PLOT_DATA.csv where Prediction_Type == CV_OOF_5REPEAT_MEAN.",
    "SHAP explains the final FULL-DATA fit; CV OOF data evaluates model development performance.",
    "All predictions are log10_Geomean_ug_L; not independently validated."
  ),file.path(dest,"README_PREDICTION_SHAP_SCATTER.txt"))
  exported <- list.files(dest,full.names=TRUE)
  data.table::fwrite(data.table::data.table(
      File=basename(exported),
      Size_Bytes=as.numeric(file.info(exported)$size),
      MD5=as.character(unname(tools::md5sum(exported)))),
    file.path(OUT,"BEST_MODEL_EXPORT_FILE_MANIFEST.csv"))
  cat("BEST-MODEL scatter/SHAP/inference bundle exported: ",dest,"\n",sep="")
  invisible(dest)
}


# Export XGBoost as soon as its own formal search/final fit has completed.
# This is NOT a declaration that XGBoost wins against all five models.
export_xgboost_first <- function() {
  modelroot <- model_dir('XGBoost')
  d <- file.path(modelroot, 'FINAL_MODEL')
  dest <- file.path(OUT, 'XGBOOST_FIRST_EXPORT')
  artifacts <- c('MODEL_BUNDLE.rds','MODEL.json','SHAP_REFERENCE_NATIVE24.csv',
    'SHAP_ENCODED_TRAINING_INPUT.rds','INFERENCE_MANIFEST.rds',
    'TRAINING_FITTED_VALUES.csv','TRAIN_METRICS.csv','FEATURE_IMPORTANCE.csv')
  extra <- c('08_BEST_PARAMETERS.csv','09_WINNER_PARAMETER_AUDIT.csv',
    '09_SEARCH_STOP_REASON.txt','10_BEST_OOF_MEAN_BY_ROW.csv',
    '11_FITTING_PLOT_DATA.csv')
  sources <- c(file.path(d,artifacts),file.path(modelroot,extra),
    file.path(OUT,'00_MODEL_PREDICT_SHAP_HELPERS.R'))
  if(!all(file.exists(sources)))
    stop('XGBoost first export is missing files: ',paste(sources[!file.exists(sources)],collapse='; '))
  dir.create(dest,recursive=TRUE,showWarnings=FALSE)
  copied <- file.copy(sources,file.path(dest,basename(sources)),overwrite=TRUE)
  if(!all(copied)) stop('XGBoost-first export failed to copy some files.')
  writeLines(c(
    'XGBOOST-FIRST RUN06: independently completed XGBoost model only.',
    'This folder contains frozen model, encoder, feature order and native TreeSHAP helper.',
    'Five-model global winner is selected ONLY after all five models complete.',
    'Model development CV is not independent external validation.',
    'source("00_MODEL_PREDICT_SHAP_HELPERS.R")',
    'm <- ec_load(".")'),file.path(dest,'README_XGB_FIRST.txt'))
  invisible(dest)
}

run_all<-function(){
 # Windows: model-specific PSOCK pool, candidates in parallel, cache folds only once per worker.
 score_master<-list()
 for (model in MODEL_NAMES){
  # Do not allocate 20/7/5 workers or reprocess folds for DONE models.
  finished_file <- file.path(model_dir(model),'DONE.txt')
  params_file <- file.path(model_dir(model),'08_BEST_PARAMETERS.csv')
  bundle_file <- file.path(model_dir(model),'FINAL_MODEL','MODEL_BUNDLE.rds')
   audit_file <- file.path(model_dir(model),'09_WINNER_PARAMETER_AUDIT.csv')
   status_file <- file.path(model_dir(model),'09_SEARCH_STOP_REASON.txt')
   if (file.exists(finished_file) && file.exists(params_file) &&
       file.exists(bundle_file) && file.exists(audit_file) && file.exists(status_file)) {
     prior <- fread(params_file)
     saved_audit <- fread(audit_file)
     saved_status <- readLines(status_file, n=1L, warn=FALSE)
     if (nrow(prior)==1L && length(saved_status)==1L &&
         grepl('^FEASIBLE',saved_status) &&
         nrow(saved_audit)==length(SEARCH_SPACE[[model]]) &&
         all(saved_audit$Boundary_State %in% c('INTERIOR','LEGAL_ENDPOINT','LEGAL_UNLIMITED_DEPTH')) &&
         isTRUE(as.logical(prior$Gap_Feasible[1L])) &&
         is.finite(prior$Max_R2_Gap[1L]) &&
         prior$Max_R2_Gap[1L] <= GAP_LIMIT+GAP_TOLERANCE &&
         identical(as.integer(readRDS(bundle_file)$best$Candidate),
                   as.integer(prior$Candidate))) {
      # Restore missing plots only from valid checkpoints inside THIS seed-specific run.
      export_best_oof(model,prior)
      fitted_file<-file.path(model_dir(model),'FINAL_MODEL','TRAINING_FITTED_VALUES.csv')
      if(!file.exists(fitted_file)){
        hw<-hardware_plan(model);N_THREADS<<-hw$threads
        Sys.setenv(OMP_NUM_THREADS=as.character(N_THREADS),
                   OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cat('BACKFILL FITTING VALUES ',model,' using saved winner candidate ',
            prior$Candidate,'\n',sep='')
        full_fit(model,prior)
      }
      export_fitting_plot_data(model)
      export_downstream_artifacts(model,prior)
      if (identical(model,'XGBoost')) export_xgboost_first()
      score_master[[model]]<-copy(prior)[,Selection_Status:=saved_status]
      cat('SKIP COMPLETED ',model,' (checkpoint verified; fitting data available)\n',sep='')
      next
    }
  }
  check_time_budget()
  hw<-hardware_plan(model)
  N_WORKERS<<-hw$workers;N_THREADS<<-hw$threads
  Sys.setenv(OMP_NUM_THREADS=as.character(N_THREADS),
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  cat('\n==================================================\nMODEL: ',model,
      '  | PSOCK workers=',N_WORKERS,' threads each=',N_THREADS,
      ' (active=',hw$active,' of ',CPU_BUDGET,')\n',sep='')
  cl<-if(N_WORKERS>1L)parallel::makePSOCKcluster(N_WORKERS)else NULL
  # Guaranteed cleanup even on model error or an interrupted fit.
  tryCatch({
  exported<-c('DT','FOLD_IDX','TARGET','GROUP','NUMS','CATS','FEATURES','MODEL_NAMES',
    'MASTER_SEED','N_THREADS','OUT','SEARCH_SPACE','fit_enc','clean_cat',
    'make_features','prepare_worker','rounds_xgb','rounds_lgb','model_params',
    'fit_predict','score','model_dir','param_cols','param_key','evaluate_candidate',
    'CACHE','ACTIVE_MODEL','GAP_LIMIT','GAP_TOLERANCE')
  if(!is.null(cl)){
   parallel::clusterExport(cl,exported,envir=.GlobalEnv)
   parallel::clusterEvalQ(cl,{
     library(data.table)
     data.table::setDTthreads(1L) # stop nested OpenMP parallelism in PSOCK
     Sys.setenv(OMP_NUM_THREADS=as.character(N_THREADS),
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');NULL
   })
  }
  dir<-model_dir(model);dir.create(file.path(dir,'checkpoints'),recursive=TRUE,showWarnings=FALSE)
  if(is.null(cl)){prepare_worker(model)}else{
   parallel::clusterCall(cl,prepare_worker,model)
  }
  candidate_file<-file.path(dir,'00_ALL_CANDIDATES.csv')
  CAND<-if(file.exists(candidate_file))fread(candidate_file)else data.table()
  sp<-SEARCH_SPACE[[model]]
  # Restart reconstructs any expanded domains from already written candidates.
  if(nrow(CAND))for(p in names(sp))sp[[p]]<-sort(unique(c(sp[[p]],as.integer(CAND[[p]]))))
  evaluate_batch<-function(z){
    if(!nrow(z))return(invisible(NULL))
    check_time_budget() # safe batch-level stop: do not kill a 25-fold fit
    rows<-lapply(seq_len(nrow(z)),function(i)z[i])
    if(is.null(cl)){
      for(p in rows)evaluate_candidate(p)
    }else parallel::parLapplyLB(cl,rows,evaluate_candidate)
    cur<-collect_scores(model)
    if(nrow(cur)) {
      # Primary comparison table; no large console spam or extra diagnostics per fit.
      fwrite(cur,file.path(dir,'06_CANDIDATE_RANKING.csv'))
      fwrite(data.table(Model=model,Completed_Candidates=nrow(cur),
        Elapsed_Hours=as.numeric(difftime(Sys.time(),RUN_STARTED,units='hours')),
        Overall_Best_CV_R2=max(cur$Mean_CV_R2),
        Gap_Feasible_Best_CV_R2=if(any(cur$Gap_Feasible))
             max(cur[Gap_Feasible==TRUE]$Mean_CV_R2) else NA_real_),
        file.path(dir,'00_RUNTIME_PROGRESS.csv'))
    }
    invisible(NULL)
  }
  b<-BUDGETS[Model==model]
  stages<-data.table(Stage=c('Stage1_Explore','Stage2_TPE','Stage3_Local'),
                     Budget=c(b$Coarse,b$TPE,b$Local),
                     Base=c(1L,101L,201L),
                     Method=c('uniform','tpe','local'))
  run_stage<-function(stage,budget,base,method,forced=NULL,domain=NULL){
    repeat{
     z<-if(nrow(CAND))CAND[Stage==stage]else data.table()
     if(nrow(z)>budget)stop('Saved candidate count exceeds budget: ',stage)
     scores<-collect_scores(model)
     done<-if(nrow(scores))scores$Candidate else integer()
     pending<-if(nrow(z))z[!Candidate %in% done]else data.table()
     if(nrow(pending)){
       evaluate_batch(pending[seq_len(min(nrow(pending),max(1L,
         if(method=='tpe') min(N_WORKERS,10L) else N_WORKERS)))])
       next
     }
     if(nrow(z)==budget)break
     batchsize<-min(max(1L,if(method=='tpe') min(N_WORKERS,10L) else N_WORKERS),budget-nrow(z))
     if(!is.null(forced)&&nrow(forced)){
        pool<-forced[seq_len(min(nrow(forced),batchsize))]
        forced<-forced[-seq_len(nrow(pool))]
     } else pool<-data.table()
     nneed<-batchsize-nrow(pool)
     if(nneed>0L){
       drawn<-space_grid(if(is.null(domain))sp else domain,nneed,
         seed=MASTER_SEED+match(model,MODEL_NAMES)*100000L+base+nrow(z),
         exclude=if(nrow(CAND))apply(as.matrix(CAND[,names(sp),with=FALSE]),1L,paste,collapse='|')else character(),
         method=method,history=if(nrow(scores))scores else NULL)
       if(nrow(pool))drawn<-drawn[!apply(as.matrix(drawn[,names(sp),with=FALSE]),1L,paste,collapse='|') %in%
          apply(as.matrix(pool[,names(sp),with=FALSE]),1L,paste,collapse='|')]
       if(nrow(drawn))pool<-rbindlist(list(pool,drawn),fill=TRUE)
     }
     if(!nrow(pool)) stop('Candidate space exhausted for ',model,' stage ',stage)
     pool<-pool[seq_len(min(nrow(pool),budget-nrow(z)))]
     pool[,`:=`(Candidate=base+nrow(z)+seq_len(.N)-1L,Stage=stage)]
     CAND<<-rbindlist(list(CAND,pool),fill=TRUE)
     fwrite(CAND,candidate_file) # checkpoint candidate definitions BEFORE fitting
     evaluate_batch(pool)
    }
    invisible(TRUE)
  }
  for(i in seq_len(nrow(stages))) {
    st<-stages[i]
     # Reuse prior SCIENTIFIC PARAMETER DESIGN as one starting point,
     # but NEVER reuse prior-run CV metrics or checkpoints in this fresh v12 run.
     use_incumbent <- st$Stage=='Stage1_Explore' &&
       (!nrow(CAND) || !any(CAND$Stage=='Stage1_Explore'))
     forced <- if(use_incumbent) base_seed_candidate(model) else data.table()
     run_stage(st$Stage,st$Budget,st$Base,st$Method,forced=forced)
    tab<-collect_scores(model)
    cat(sprintf('Stage %-18s overall CV R2=%.6f | feasible best=%s | n=%d\n',
      st$Stage,max(tab$Mean_CV_R2),
      if(any(tab$Gap_Feasible))sprintf('%.6f',max(tab[Gap_Feasible==TRUE]$Mean_CV_R2)) else 'NONE',
      nrow(tab)))
  }
  # Search can stop EARLY after all three planned stages when there is no
  # measurable >=0.001 upside, no higher-performing near-feasible model, and
  # chosen winner does not sit on an EXPANDABLE boundary.
  scores<-collect_scores(model)
  eligible0<-acceptable_candidates(scores,sp,HARD[[model]],names(sp))
  prior<-scores[Stage!='Stage3_Local']
  improvement3<-if(nrow(prior))max(scores$Mean_CV_R2)-max(prior$Mean_CV_R2) else Inf
  feasible_max<-if(nrow(eligible0))max(eligible0$Mean_CV_R2) else -Inf
  tradeoff<-max(scores$Mean_CV_R2)-feasible_max
  # Probe even a *near-equivalent* better boundary candidate, rather than
  # silently selecting the weaker interior candidate to avoid work.
  feas0<-scores[Gap_Feasible==TRUE]
  best_feas0<-if(nrow(feas0))feas0[which.max(Mean_CV_R2)] else data.table()
  best_feas_can_expand<-nrow(best_feas0) &&
    extend_space(sp,best_feas0,CAND[,names(sp),with=FALSE],HARD[[model]])$changed
  # CV-centric target helps bring excellent near-feasible candidates (e.g., RF
  # max gap .203 vs .20) into the valid region instead of discarding them.
  previous_rounds<-file.path(dir,'09_EXPANSION_LOG.csv')
  oldlog<-if(file.exists(previous_rounds))fread(previous_rounds) else data.table()
  small_rounds<-if(nrow(oldlog))as.integer(tail(oldlog$SmallRounds,1L)) else 0L
  status<-'SEARCHING'
  if(nrow(eligible0) && improvement3<THRESHOLD && tradeoff<THRESHOLD &&
     !best_feas_can_expand && !nrow(oldlog))
    status<-'FEASIBLE_GAP020_INTERIOR_NO_EXTENSION_REQUIRED'
  if(status=='SEARCHING') for(r in seq_len(MAX_SEARCH_ROUNDS)){
    scores<-collect_scores(model)
    eligible<-acceptable_candidates(scores,sp,HARD[[model]],names(sp))
    # Probe the actual best feasible model's boundary, even if an interior
    # model is within 0.001 CV R2.  Only otherwise focus on bringing a markedly
    # better gap-infeasible candidate into the allowable region.
    feasible_rows<-scores[Gap_Feasible==TRUE]
    best_feasible<-if(nrow(feasible_rows))feasible_rows[which.max(Mean_CV_R2)] else data.table()
    if(nrow(best_feasible) && max(scores$Mean_CV_R2) <=
          best_feasible$Mean_CV_R2+THRESHOLD) {
      target<-best_feasible
    } else {
      scores_for_search<-copy(scores)
      setorder(scores_for_search,-SearchUtility)
      target<-scores_for_search[1]
    }
    top_edge<-extend_space(sp,target,CAND[,names(sp),with=FALSE],HARD[[model]])
    stage<-sprintf('Stage4_Expand_%03d',r)
    stored<-if(nrow(CAND))CAND[Stage==stage]else data.table()
    # Existing complete expansion rounds are already evaluated and logged.
    if(nrow(stored)==EXPAND_BUDGET && nrow(oldlog[Round==r]))next
    # Avoid prolonged search after three consecutive gains <0.001, but first
    # make sure the best feasible target is NO LONGER on an unprobed artificial
    # edge; a legal endpoint does not block stopping.
    best_gap_ok <- nrow(best_feasible) &&
      !extend_space(sp,best_feasible,CAND[,names(sp),with=FALSE],
                    HARD[[model]])$changed
    if(nrow(eligible) && small_rounds>=3L && !nrow(stored) &&
       best_gap_ok && max(scores$Mean_CV_R2)<=max(eligible$Mean_CV_R2)+THRESHOLD) {
      status <- 'FEASIBLE_GAP020_PLATEAU_AFTER_EDGE_PROBES';break
    }
    # Expand actual legal search-domain bounds; never search illegal values.
    sp<-top_edge$space
    forced<-data.table()
    if(nrow(top_edge$edges))for(j in seq_len(nrow(top_edge$edges))){
      ed<-top_edge$edges[j]
      if(!grepl('_EXPANDED$',ed$Side))next
      z<-as.data.table(as.list(target[1,names(sp),with=FALSE]))
      z[,(ed$Parameter):=as.integer(ed$Value)]
      forced<-rbindlist(list(forced,z),fill=TRUE)
    }
    if(nrow(forced)){
      forced<-unique(forced,by=names(sp))
      ks<-apply(as.matrix(CAND[,names(sp),with=FALSE]),1L,paste,collapse='|')
      forced<-forced[!apply(as.matrix(forced[,names(sp),with=FALSE]),1L,paste,collapse='|')%in%ks]
    }
    feasible_now<-any(scores$Gap_Feasible)
    # If no feasible candidate exists, favor stronger regularization. Otherwise
    # use TPE near promising settings to improve CV and meet <=0.20.
    domain<-if(feasible_now)sp else regularized_domain(sp,model)
    method<-if(feasible_now)'local' else 'tpe'
    before<-max(scores$Mean_CV_R2)
    run_stage(stage,EXPAND_BUDGET,301L+EXPAND_BUDGET*(r-1L),method,forced,domain)
    after<-collect_scores(model)
    gain<-max(after$Mean_CV_R2)-before
    small_rounds<-if(gain>=THRESHOLD)0L else small_rounds+1L
    el<-acceptable_candidates(after,sp,HARD[[model]],names(sp))
    logrow<-data.table(Round=as.integer(r),Before_CV=before,
      After_CV=max(after$Mean_CV_R2),Delta_CV_R2=gain,
      FeasibleCandidates=sum(after$Gap_Feasible),
      EligibleInteriorCandidates=nrow(el),
      Highest_CV_Gap=after[which.max(Mean_CV_R2)]$Max_R2_Gap,
      SmallRounds=small_rounds)
    logold<-if(file.exists(previous_rounds))fread(previous_rounds)else data.table()
    if(nrow(logold))logold<-logold[Round!=r]
    fwrite(rbindlist(list(logold,logrow),fill=TRUE),previous_rounds)
    cat(sprintf('%s expansion %d/%d | highest CV=%.5f feasible best=%s | +%.5f\n',
      model,r,MAX_SEARCH_ROUNDS,max(after$Mean_CV_R2),
      if(any(after$Gap_Feasible))sprintf('%.5f',max(after[Gap_Feasible==TRUE]$Mean_CV_R2)) else 'NONE',
      gain))
    if(r==MAX_SEARCH_ROUNDS)status<-'SAFETY_BUDGET_REACHED_CAN_RESUME'
  }
  scores<-collect_scores(model)
  eligible<-acceptable_candidates(scores,sp,HARD[[model]],names(sp))
  fwrite(scores,file.path(dir,'06_CANDIDATE_RANKING.csv'))
  fwrite(collect_repeats(model),file.path(dir,'07_ALL_REPEAT_METRICS.csv'))
  fwrite(if(any(scores$Gap_Feasible)) scores[Gap_Feasible==TRUE][1] else
        scores[which.max(Mean_CV_R2)],file.path(dir,'08_TOP_CONSTRAINED_CANDIDATE.csv'))
  fwrite(scores[which.max(Mean_CV_R2)],file.path(dir,'08_BEST_UNCONSTRAINED_CV.csv'))
  # Honest fallback: if the highest gap-qualified candidate lands on a
  # safety cap, do not claim a physical optimum or replace it by a much worse
  # interior candidate merely to make a cosmetically clean boundary table.
  feasible_final <- scores[Gap_Feasible==TRUE]
  best_feasible_score <- if(nrow(feasible_final))max(feasible_final$Mean_CV_R2)else -Inf
  eligible_score <- if(nrow(eligible))max(eligible$Mean_CV_R2)else -Inf
  too_much_cv_loss <- nrow(eligible)>0L && is.finite(best_feasible_score) &&
    best_feasible_score-eligible_score>THRESHOLD
  if(nrow(eligible)){
    prewinner <- eligible[which.max(Mean_CV_R2)]
    fwrite(boundary_audit(model,prewinner,sp,HARD[[model]],SOFT_HARD[[model]]),
           file.path(dir,'09_WINNER_PARAMETER_AUDIT.csv'))
  }
  if(!nrow(eligible) || too_much_cv_loss){
    if(sum(scores$Gap_Feasible)==0L) status<-'NO_GAP_FEASIBLE_CANDIDATE_YET'
    else if(too_much_cv_loss) status<-'NONARTIFICIAL_WINNER_COST_EXCEEDS_R2_TOLERANCE'
    else status<-'NO_NONARTIFICIAL_EDGE_FEASIBLE_CANDIDATE_YET'
    if(nrow(feasible_final)) {
      bf<-feasible_final[which.max(Mean_CV_R2)]
      fwrite(boundary_audit(model,bf,sp,HARD[[model]],SOFT_HARD[[model]]),
             file.path(dir,'09_HIGHEST_FEASIBLE_BOUNDARY_AUDIT.csv'))
    }
    writeLines(c(status,'Do NOT call this a fitted final model.',
      paste('To continue, raise ECOTOX_MAX_SEARCH_ROUNDS above',MAX_SEARCH_ROUNDS),
      'The run stays resumable without repeating completed folds.'),
      file.path(dir,'09_SEARCH_STOP_REASON.txt'))
    fwrite(extend_space(sp,scores[1],CAND[,names(sp),with=FALSE],HARD[[model]])$edges,
      file.path(dir,'09_WINNER_BOUNDARY_STATUS.csv'))
    score_master[[model]]<-copy(scores[which.max(Mean_CV_R2)])[,Selection_Status:=status]
    cat('NOT ELIGIBLE ',model,': ',status,'; review, then resume.\n',sep='')
  } else {
    # Interior/non-expandable is a PREDECLARED condition; never conceal
    # a higher CV model with excessive gap or unresolved expandable edges.
    setorder(eligible,-Mean_CV_R2,Mean_CV_RMSE,Candidate)
    winner<-eligible[1]
    eb<-extend_space(sp,winner,CAND[,names(sp),with=FALSE],HARD[[model]])
    stopifnot(isTRUE(winner$Gap_Feasible),winner$Max_R2_Gap<=GAP_LIMIT+GAP_TOLERANCE,
              !eb$changed)
    edges<-eb$edges
    stopifnot(!any(grepl('ARTIFICIAL_SAFETY_CAP',edges$Side)))
    search_hit_round_budget <- identical(status,'SAFETY_BUDGET_REACHED_CAN_RESUME')
    if(nrow(edges))status<-'FEASIBLE_GAP020_WITH_LEGAL_ENDPOINTS'
    else if(!status %in% c('FEASIBLE_GAP020_INTERIOR_AND_PLATEAU',
                               'FEASIBLE_GAP020_INTERIOR_NO_EXTENSION_REQUIRED',
                               'FEASIBLE_GAP020_PLATEAU_AFTER_EDGE_PROBES'))
      status<-'FEASIBLE_INTERIOR_BUDGET_REACHED'
    if (search_hit_round_budget && !grepl('BUDGET_REACHED',status))
      status <- paste0(status,'_SEARCH_BUDGET_REACHED')
    fwrite(data.table(Constrained_CV_R2=winner$Mean_CV_R2,
      Unconstrained_CV_R2=max(scores$Mean_CV_R2),
      CV_R2_Cost=max(scores$Mean_CV_R2)-winner$Mean_CV_R2,
      Constrained_Max_Repeat_Gap=winner$Max_R2_Gap),
      file.path(dir,'08_GAP_CONSTRAINT_TRADEOFF.csv'))
    fwrite(winner,file.path(dir,'08_BEST_PARAMETERS.csv'))
    fwrite(boundary_audit(model,winner,sp,HARD[[model]],SOFT_HARD[[model]]),
           file.path(dir,'09_WINNER_PARAMETER_AUDIT.csv'))
    fwrite(data.table(Model=model,Best_Feasible_CV=best_feasible_score,
      Selected_CV=winner$Mean_CV_R2,Selection_R2_Difference=
      best_feasible_score-winner$Mean_CV_R2,Max_R2_Gap=winner$Max_R2_Gap),
      file.path(dir,'09_FINAL_SELECTION_PROVENANCE.csv'))
    fwrite(edges,file.path(dir,'09_WINNER_BOUNDARY_STATUS.csv'))
    writeLines(status,file.path(dir,'09_SEARCH_STOP_REASON.txt'))
    export_best_oof(model,winner)
    donepath<-file.path(dir,'DONE.txt')
    need_fit<-TRUE
    if(file.exists(donepath) && file.exists(file.path(dir,'FINAL_MODEL','MODEL_BUNDLE.rds'))){
      bnd<-readRDS(file.path(dir,'FINAL_MODEL','MODEL_BUNDLE.rds'))
      need_fit<-as.integer(bnd$best$Candidate)!=as.integer(winner$Candidate)
    }
    if(need_fit || !file.exists(file.path(dir,'FINAL_MODEL','TRAINING_FITTED_VALUES.csv')))
      full_fit(model,winner)
    export_fitting_plot_data(model)
    export_downstream_artifacts(model,winner)
    writeLines(c('DONE',paste('Candidate',winner$Candidate),
      paste('Mean CV R2',winner$Mean_CV_R2),
      paste('Maximum repeat Gap',winner$Max_R2_Gap),status),donepath)
    if (identical(model,'XGBoost')) export_xgboost_first()
    score_master[[model]]<-copy(winner)[,Selection_Status:=status]
    cat('FINISHED ',model,' valid winner=',winner$Candidate,' R2=',winner$Mean_CV_R2,
      ' max repeat gap=',winner$Max_R2_Gap,' status=',status,'\n',sep='')
  }
  if(is.null(cl)){CACHE<-NULL;ACTIVE_MODEL<-NULL}else parallel::clusterEvalQ(cl,{CACHE<-NULL;ACTIVE_MODEL<-NULL;gc();NULL})
  },finally={if(!is.null(cl))parallel::stopCluster(cl)})
 }
 master<-rbindlist(score_master,fill=TRUE)
 fwrite(master,file.path(OUT,'MASTER_MODEL_COMPARISON.csv'))

 # Consolidated FIVE-MODEL final audit; not merely a single best-model result.
 audit_all <- data.table::rbindlist(lapply(MODEL_NAMES,function(m) {
   f <- file.path(model_dir(m),'09_WINNER_PARAMETER_AUDIT.csv')
   if (!file.exists(f)) return(data.table())
   fread(f)
 }),fill=TRUE)
 if (nrow(audit_all)) fwrite(audit_all,file.path(OUT,'FINAL_ALL5_PARAMETER_BOUNDARY_AUDIT.csv'))

 review <- data.table::rbindlist(lapply(MODEL_NAMES,function(m) {
   score <- master[Model==m]
   f <- file.path(model_dir(m),'09_WINNER_PARAMETER_AUDIT.csv')
   au <- if (file.exists(f)) fread(f) else data.table()
   unsafe <- !nrow(au) || any(!au$Boundary_State %in%
     c('INTERIOR','LEGAL_ENDPOINT','LEGAL_UNLIMITED_DEPTH'))
   complete <- nrow(score)==1L && grepl('^FEASIBLE',score$Selection_Status[1]) &&
     isTRUE(as.logical(score$Gap_Feasible[1L])) &&
     is.finite(score$Max_R2_Gap[1L]) &&
     score$Max_R2_Gap[1L] <= GAP_LIMIT+GAP_TOLERANCE && !unsafe &&
     file.exists(file.path(model_dir(m),'DONE.txt')) &&
     file.exists(file.path(model_dir(m),'FINAL_MODEL','MODEL_BUNDLE.rds'))
   data.table(Model=m, Final_Eligible=complete,
     Candidate=if(nrow(score))as.integer(score$Candidate[1]) else NA_integer_,
     CV_R2=if(nrow(score))as.numeric(score$Mean_CV_R2[1]) else NA_real_,
     Max_Repeat_Gap=if(nrow(score))as.numeric(score$Max_R2_Gap[1]) else NA_real_,
     Artificial_Edge_Issue=unsafe,
     Stop_Status=if(nrow(score))as.character(score$Selection_Status[1]) else 'MISSING')
 }))
 fwrite(review,file.path(OUT,'FINAL_ALL5_COMPLETENESS_AND_BOUNDARY_REVIEW.csv'))
 all_final <- nrow(review)==length(MODEL_NAMES) && all(review$Final_Eligible)
 writeLines(if(all_final) 'ALL_5_MODELS_FINAL_ELIGIBLE' else 'NOT_ALL_5_MODELS_FINAL_ELIGIBLE',
            file.path(OUT,'FINAL_ALL5_COMPLETION_STATUS.txt'))

 # Never publish a "global final winner" if some algorithm is still unresolved.
 valid_master<-master[grepl('^FEASIBLE',Selection_Status)]
 if(all_final && nrow(valid_master)==length(MODEL_NAMES)){
   setorder(valid_master,-Mean_CV_R2,Mean_CV_RMSE,Model)
   global_best<-valid_master[1]
   fwrite(global_best,file.path(OUT,'BEST_MODEL_SUMMARY.csv'))
   gmodel<-as.character(global_best$Model[1L])
   gdir<-model_dir(gmodel)
   tr<-fread(file.path(gdir,'FINAL_MODEL','TRAINING_FITTED_VALUES.csv'))
   cv<-fread(file.path(gdir,'10_BEST_OOF_MEAN_BY_ROW.csv'))
   plotdat<-fread(file.path(gdir,'11_FITTING_PLOT_DATA.csv'))
   tr[,Model:=gmodel];setcolorder(tr,c('Model',setdiff(names(tr),'Model')))
   cv[,Model:=gmodel];setcolorder(cv,c('Model',setdiff(names(cv),'Model')))
   fwrite(tr,file.path(OUT,'BEST_MODEL_TRAINING_FITTED_VALUES.csv'))
   fwrite(cv,file.path(OUT,'BEST_MODEL_OOF_MEAN_BY_ROW.csv'))
   fwrite(plotdat,file.path(OUT,'BEST_MODEL_FITTING_PLOT_DATA.csv'))
   export_global_bundle(global_best)
   cat('GLOBAL BEST MODEL: ',gmodel,' | CV R2=',global_best$Mean_CV_R2,
       ' | fitting data exported at run root.\n',sep='')
 }

 if (all_final) {
   cat('\nALL FIVE FINAL-ELIGIBLE. Outputs: ',OUT,'\n')
 } else {
   cat('\nFINAL AUDIT INCOMPLETE: Check FINAL_ALL5_COMPLETENESS_AND_BOUNDARY_REVIEW.csv;\n',
       ' unresolved models are NOT declared finished.\n',sep='')
 }
}

export_predict_helper()

# This file always executes FORMAL TRAINING, ignoring prior quick-check state.
cat("\nSTARTING v12 FRESH MOA-CORRECTED AUTO-EDGE TPE FORMAL 5x5 CV (no preflight; gap<=0.20; 18h soft budget)\n")
cat("Output directory: ", OUT, "\n", sep="")
tryCatch(run_all(), error=function(e){
  msg<-conditionMessage(e)
  if(grepl('ECOTOX_TIME_BUDGET:',msg,fixed=TRUE)){
    cat('\nSAFE SOFT TIME-BUDGET STOP: ',msg,'\n',sep='')
    writeLines(msg,file.path(OUT,'TIME_BUDGET_STOP.txt'))
    cat('Re-run this SAME RUN06 script to continue. No completed repeats are recomputed.\n')
    invisible(NULL)
  }else stop(e)
})
