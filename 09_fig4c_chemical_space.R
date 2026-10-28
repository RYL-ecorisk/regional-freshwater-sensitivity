# =============================================================================
# JHM RUN06 | Figure 4c | EQUAL-WEIGHT MANHATTAN CHEMICAL SPACE | LARGE-TYPE EDITION
# Full RECALCULATION of 13-descriptor chemical space for the original 8
# pollutant categories, with 2,109 UNIQUE training chemicals.
#
# CORE:       chemical Manhattan distance <= unique-training P90
# PERIPHERAL: unique-training P90 < distance <= unique-training P95
# OUTSIDE:    chemical Manhattan distance > unique-training P95
#
# Score = mean(abs((x - training chemical means) / training chemical SDs))
#          (each of 13 standardized chemical descriptors gets weight 1/13).
# PCA and 95% PCA ellipse are visual only and NOT the 13D AD boundary.
# Preserves the user's V26 chart layout, colors, inset, and category bars.
# Does NOT recalculate Fig 4a / 4b or retrain toxicity prediction model.
# =============================================================================
options(stringsAsFactors=FALSE, warn=1)
ROOT <- normalizePath('.', winslash='/', mustWork=TRUE)
DATA_ROOT <- file.path(ROOT,'data')
OUT <- file.path(ROOT,'results','fig4c_chemical_space')
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

for (pkg in c('ggplot2', 'patchwork', 'MASS')) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop('Install required R package: ', pkg)
}
suppressPackageStartupMessages({library(ggplot2); library(patchwork)})

# Optional analysis: requires the separately prepared ChemExpo category inventory.
poll_file <- file.path(DATA_ROOT, 'Pollutants_13_descriptors_by_category.csv')
if (!file.exists(poll_file)) {
  stop(paste0(
    'MISSING REQUIRED INPUT CSV: ', poll_file, '\n',
    'Place Pollutants_13_descriptors_by_category.csv in: ', DATA_ROOT, '\n',
    'The input must contain eight categories with chemical descriptors.\n',
    'Do not substitute a different pollutant, validation or training CSV.'
  ), call. = FALSE)
}
# Use the chemical-identified RUN06 model training database if available;
# fall back to the exact chemical reference used in the user's V26 plot.
# Never substitute the 437-record external validation set for training chemistry.
read_csv <- function(f, nrows=-1L) read.csv(
  f, nrows=nrows, check.names=FALSE, stringsAsFactors=FALSE,
  fileEncoding='UTF-8-BOM')
DESCR <- c('logS','logKow','logD7_4','MW','TPSA','log_Henry',
           'log_VaporPressure','MeltingPoint','HBA','HBD',
           'AromaticRingCount','RotatableBondCount','BertzCT')

preferred_train <- file.path(DATA_ROOT,'Training_DATABASE_MOA_RECHECKED.csv')
fallback_train <- file.path(DATA_ROOT, 'Training_data_Trait_scored(20261009-083028).csv')
other_train <- list.files(DATA_ROOT, pattern='^Training_data_Trait_scored.*[.]csv$',
                          full.names=TRUE, ignore.case=TRUE)
if (length(other_train)) {
  other_train <- other_train[order(file.info(other_train)$mtime, decreasing=TRUE)]
}
train_candidates <- unique(c(preferred_train, fallback_train, other_train))
train_candidates <- train_candidates[file.exists(train_candidates)]
valid_reference <- function(f) {
  header <- tryCatch(names(read_csv(f, nrows=0L)), error=function(e) character())
  all(c('DTXSID', DESCR) %in% header)
}
valid <- if (length(train_candidates)) {
  vapply(train_candidates, valid_reference, logical(1))
} else {
  logical()
}
if (!any(valid)) {
  stop(paste0(
    'No valid chemical-identified training CSV found.\n',
    'Fig. 4c needs a training file containing DTXSID and all 13 chemical descriptors.\n',
    'Expected canonical RUN06 file: ', preferred_train, '\n',
    'Checked: ', paste(train_candidates, collapse='; ')), call.=FALSE)
}
train_file <- train_candidates[which(valid)[1L]]
message('FIG4C INPUT | pollutant inventory: ', normalizePath(poll_file))
message('FIG4C INPUT | chemical reference:   ', normalizePath(train_file))
train <- read_csv(train_file)
poll <- read_csv(poll_file)
required_train <- c('DTXSID', DESCR)
required_poll <- c('DTXSID', 'Source_file', DESCR)
if (length(setdiff(required_train, names(train)))) {
  stop('Wrong training CSV: missing columns: ',
       paste(setdiff(required_train, names(train)), collapse=', '), call.=FALSE)
}
if (length(setdiff(required_poll, names(poll)))) {
  stop('Wrong pollutant CSV: missing columns: ',
       paste(setdiff(required_poll, names(poll)), collapse=', '), call.=FALSE)
}
if (any(!vapply(train[DESCR],is.numeric,logical(1))) ||
    any(!vapply(poll[DESCR],is.numeric,logical(1)))) {
  stop('Descriptor columns must be numeric in both input CSVs.', call.=FALSE)
}


# One training instance per DTXSID; one category entry per source file and DTXSID.
if (any(duplicated(train$DTXSID)) &&
    nrow(unique(train[,c('DTXSID',DESCR),drop=FALSE])) > length(unique(train$DTXSID))) {
  stop('Conflicting 13 chemical descriptors for the same training DTXSID.')
}
train <- train[!duplicated(train$DTXSID), , drop=FALSE]
train <- train[complete.cases(train[, DESCR, drop=FALSE]), , drop=FALSE]
poll <- poll[complete.cases(poll[, c('DTXSID','Source_file',DESCR),drop=FALSE]),,drop=FALSE]
poll <- poll[!duplicated(poll[,c('Source_file','DTXSID')]),,drop=FALSE]
# A chemical occurring in several source categories must have the same descriptors.
# This audit prevents the same DTXSID acquiring different chemical-space scores.
if (nrow(unique(poll[, c('DTXSID', DESCR), drop=FALSE])) >
    length(unique(poll$DTXSID)))
  stop('Inconsistent descriptors for the same target DTXSID across pollutant categories.')
poll$Category <- sub('[.]xlsx$', '', poll$Source_file, ignore.case=TRUE)
message('FIG4C COUNTS | Training unique DTXSIDs: ', nrow(train),
        ' | Pollutant category entries: ', nrow(poll),
        ' | Distinct pollutants: ', length(unique(poll$DTXSID)),
        ' | Original pollutant classes: ', length(unique(poll$Source_file)))
if (nrow(train) != 2109L)
  stop('Expected 2,109 unique RUN06 training chemicals. Found ', nrow(train),
       '. Do not proceed with a different training set.')
if (nrow(poll) != 45839L || length(unique(poll$DTXSID)) != 40805L ||
    length(unique(poll$Source_file)) != 8L)
  stop('Original v20 pollutant inventory counts changed. Recheck source input.')

# -----------------------------------------------------------------------------
# CHEMICAL SPACE: 13-D EQUAL-WEIGHT MANHATTAN DISTANCE TO TRAINING MEAN
#
# * Training CHEMICALS, not rows: exactly 2,109 unique DTXSIDs.
# * Standardization uses 2,109 unique-training chemical means and sample SDs.
# * Each standardized chemical descriptor receives the same weight (1/13).
# * P90 and P95 are computed on the SAME unique training chemical distances.
# * P95 is the chemical-space AD boundary; P90 separates core/peripheral.
# * Fig 4b uses 24 predictors and 7,043 records and therefore a DIFFERENT
#   Manhattan reference space and numerical training-P95 threshold.
# -----------------------------------------------------------------------------
if (anyNA(train$DTXSID) || any(!nzchar(trimws(as.character(train$DTXSID)))))
  stop('Blank training DTXSID.')
if (anyNA(poll$DTXSID) || anyNA(poll$Source_file))
  stop('Missing pollutant DTXSID or source category.')
if (any(!is.finite(as.matrix(train[, DESCR, drop=FALSE]))) ||
    any(!is.finite(as.matrix(poll[, DESCR, drop=FALSE]))))
  stop('Missing or nonfinite chemical descriptors.')

n_train <- nrow(train)
if (n_train != 2109L) stop('Expected exactly 2,109 unique training chemicals.')
Xtr <- as.matrix(train[, DESCR, drop=FALSE])
storage.mode(Xtr) <- 'double'
ctr <- colMeans(Xtr)
sc <- apply(Xtr, 2L, stats::sd)
if (any(!is.finite(sc)) || any(sc < 1e-12))
  stop('Zero-variance training chemical descriptor. Check the input data.')

# RowMeans(|Z|) is precisely the equal-weighted Manhattan distance (1/13
# for each chemical descriptor) to the standardized training centroid (0).
manhattan_for <- function(M) {
  if (is.data.frame(M)) M <- as.matrix(M)
  storage.mode(M) <- 'double'
  if (ncol(M) != length(DESCR)) stop('Wrong chemical-descriptor count.')
  Z <- sweep(sweep(M, 2L, ctr, '-'), 2L, sc, '/')
  rowMeans(abs(Z))
}
train_d <- manhattan_for(Xtr)
P90_chem <- unname(stats::quantile(train_d, probs=.90, type=7))
P95_chem <- unname(stats::quantile(train_d, probs=.95, type=7))
if (!all(is.finite(c(P90_chem,P95_chem))) || P90_chem >= P95_chem)
  stop('Invalid chemical Manhattan P90 / P95 thresholds.')
train_core <- sum(train_d <= P90_chem)
train_within <- sum(train_d <= P95_chem)
if (abs(train_within/n_train - .95) > .005)
  stop('Training P95 coverage sanity check failed.')

# Score a DISTINCT pollutant only once and reuse its value wherever the same
# DTXSID occurs in multiple pollutant categories (category denominators kept).
poll_unique <- poll[!duplicated(poll$DTXSID), c('DTXSID', DESCR), drop=FALSE]
Du <- manhattan_for(poll_unique[, DESCR, drop=FALSE])
lookup <- match(poll$DTXSID, poll_unique$DTXSID)
if (anyNA(lookup)) stop('Pollutant identifier matching failed.')
poll$Manhattan_Distance <- Du[lookup]
poll$Training_P90_Manhattan <- P90_chem
poll$Training_P95_Manhattan <- P95_chem
poll$AD_class <- ifelse(poll$Manhattan_Distance <= P90_chem, 'Core',
                        ifelse(poll$Manhattan_Distance <= P95_chem,
                               'Peripheral','Outside'))
poll$Training_overlap <- poll$DTXSID %in% train$DTXSID
AD_LEVELS <- c('Core', 'Peripheral', 'Outside')

# Each original pollutant category contains each DTXSID at most once.
category_levels <- unique(as.character(poll$Category))
tb <- table(
  factor(as.character(poll$Category), levels=category_levels),
  factor(as.character(poll$AD_class), levels=AD_LEVELS)
)
sumtab <- data.frame(Category=rownames(tb), stringsAsFactors=FALSE,
                     check.names=FALSE)
for (j in seq_along(AD_LEVELS))
  sumtab[[AD_LEVELS[j]]] <- as.integer(tb[,j])
sumtab$n <- rowSums(sumtab[,AD_LEVELS,drop=FALSE])
if (sum(sumtab$n) != nrow(poll) || any(sumtab$n <= 0L))
  stop('Category denominator or AD classification audit failed.')
for (level in AD_LEVELS)
  sumtab[[paste0(level,'_pct')]] <- 100*sumtab[[level]] / sumtab$n
sumtab$Within_chemical_AD_n <- sumtab$Core + sumtab$Peripheral
sumtab$Within_chemical_AD_pct <- 100*sumtab$Within_chemical_AD_n / sumtab$n
sumtab$Seen_in_training <- vapply(sumtab$Category,
  function(s) as.integer(sum(poll$Training_overlap[poll$Category==s])), integer(1L))
# Original v20 sorting: descending core-space coverage and then number of chemicals.
sumtab <- sumtab[order(-sumtab$Core_pct,-sumtab$n),,drop=FALSE]
row.names(sumtab) <- NULL
stopifnot(nrow(sumtab)==8L,
          all(abs(rowSums(sumtab[,paste0(AD_LEVELS,'_pct'),drop=FALSE])-100)<1e-7),
          all(sumtab$Within_chemical_AD_n+sumtab$Outside==sumtab$n))
write.csv(sumtab,
          file.path(OUT,'Figure4c_MANHATTAN_3GROUP_coverage_by_category.csv'),
          row.names=FALSE)
write.csv(poll[,c('Source_file','Category','DTXSID','Manhattan_Distance',
                  'Training_P90_Manhattan','Training_P95_Manhattan',
                  'AD_class','Training_overlap')],
          file.path(OUT,'Figure4c_MANHATTAN_3GROUP_pollutant_by_category.csv'),
          row.names=FALSE)
write.csv(data.frame(DTXSID=train$DTXSID,Manhattan_Distance=train_d),
          file.path(OUT,'Figure4c_MANHATTAN_training_2109_chemicals.csv'),
          row.names=FALSE)
write.csv(data.frame(
  Method='Equally weighted Manhattan distance to training centroid',
  Space='13 chemical descriptors',
  N_training_chemicals=n_train,
  Descriptor_weight=1/length(DESCR),
  Standardization='Z-score using means and SDs of unique training chemicals',
  Training_P90_Manhattan=P90_chem,
  Training_P95_Manhattan=P95_chem,
  Training_within_P90_N=train_core,
  Training_within_P95_N=train_within,
  Reference_file=normalizePath(train_file, winslash='/', mustWork=TRUE)),
  file.path(OUT,'Figure4c_MANHATTAN_3GROUP_thresholds.csv'),row.names=FALSE)
write.csv(data.frame(Descriptor=DESCR,Training_mean=as.numeric(ctr),
                     Training_sd=as.numeric(sc),Weight=rep(1/13,length(DESCR))),
          file.path(OUT,'Figure4c_MANHATTAN_scaling_and_weights.csv'),row.names=FALSE)
write.csv(data.frame(DTXSID=poll_unique$DTXSID,Manhattan_Distance=Du,
                     AD_class=ifelse(Du <= P90_chem,'Core',
                                     ifelse(Du <= P95_chem,'Peripheral','Outside'))),
          file.path(OUT,'Figure4c_MANHATTAN_unique_pollutants.csv'),row.names=FALSE)
message(sprintf('CHEMICAL MANHATTAN | n=%d | P90=%.6f | P95=%.6f | training within P95=%d',
                n_train,P90_chem,P95_chem,train_within))
print(sumtab[,c('Category','n','Core_pct','Peripheral_pct','Outside_pct',
               'Within_chemical_AD_pct')])

# PCA fitted ONLY to training chemistry; project external chemicals unchanged.
pca <- prcomp(train[,DESCR,drop=FALSE],center=TRUE,scale.=TRUE)
vshare <- 100*pca$sdev^2/sum(pca$sdev^2)
train_xy <- as.data.frame(pca$x[,1:2,drop=FALSE]);names(train_xy) <- c('PC1','PC2')
u <- poll[!duplicated(poll$DTXSID),,drop=FALSE]
poll_xy <- as.data.frame(predict(pca,u[,DESCR,drop=FALSE])[,1:2,drop=FALSE])
names(poll_xy) <- c('PC1','PC2')
poll_xy$AD_class <- factor(u$AD_class,levels=AD_LEVELS)
poll_xy$DTXSID <- u$DTXSID
write.csv(cbind(DTXSID=train$DTXSID,train_xy),
          file.path(OUT,'Figure4c_training_PCA_coordinates.csv'),row.names=FALSE)
write.csv(poll_xy,file.path(OUT,'Figure4c_pollutant_PCA_coordinates.csv'),row.names=FALSE)

# --------------------------------------------------------------
# Figure 4c | LARGE-TYPE / SENIOR-FRIENDLY TYPOGRAPHY
# --------------------------------------------------------------
# All plotted text scales from this base size. Core labels remain large while
# percentage labels in narrow bar segments are reduced proportionally to fit.
FONT <- 25.0
FONT_MM <- FONT / 2.845276
INK <- '#000000'
# Harmonized to Fig. 5 (Liao teal, Pearl orange, Danube rose).
# These are AD classes, not geographic regions; color reuse is visual only.
COL <- c('Core'='#37B8AE', 'Peripheral'='#EBA45D', 'Outside'='#E27D9F')
ELLIPSE_COL <- '#657580'

base_style <- theme_classic(base_size=FONT, base_family='sans') +
  theme(
    text=element_text(size=FONT, face='plain', colour=INK),
    axis.text=element_text(size=FONT, face='plain', colour=INK),
    axis.title=element_text(size=FONT, face='plain', colour=INK),
    plot.title=element_blank(),
    axis.line=element_line(linewidth=.45, colour=INK),
    axis.ticks=element_line(linewidth=.45, colour=INK),
    panel.grid=element_blank(),
    plot.margin=margin(8,6,3,6,unit='pt')
  )

# Main panel: centred view for readability. Inset: full PCA range so no pollutant
# points are hidden from the reader.
allx <- c(train_xy$PC1, poll_xy$PC1)
ally <- c(train_xy$PC2, poll_xy$PC2)
# Centred cropped main view requested by user; the inset retains all points
xr <- c(-8, 8)
yr <- c(-8, 8)
# Full-range inset (slight padding)
full_xr <- range(allx, na.rm=TRUE)
full_yr <- range(ally, na.rm=TRUE)
full_xpad <- 0.04 * diff(full_xr)
full_ypad <- 0.04 * diff(full_yr)
full_xr <- c(full_xr[1] - full_xpad, full_xr[2] + full_xpad)
full_yr <- c(full_yr[1] - full_ypad, full_yr[2] + full_ypad)

dens <- MASS::kde2d(train_xy$PC1, train_xy$PC2, n=220,
                    lims=c(xr,yr))
ix <- pmax(1L, pmin(length(dens$x), findInterval(train_xy$PC1,dens$x)))
iy <- pmax(1L, pmin(length(dens$y), findInterval(train_xy$PC2,dens$y)))
lev <- unique(as.numeric(quantile(
  dens$z[cbind(ix,iy)], c(.035,.14,.34,.55,.76,.91,.98))))
dgrid <- expand.grid(PC1=dens$x, PC2=dens$y)
dgrid$z <- as.vector(dens$z)
dgrid$band <- cut(dgrid$z, breaks=c(-Inf,lev,Inf),
                  include.lowest=TRUE, labels=FALSE)
dgrid$band[dgrid$band == 1L] <- NA_integer_
# seven visible grey bands, matching the clean background of the preview
graycols <- c('#F4F5F6','#ECEFF1','#E4E8EA','#D7DDE1',
              '#C8D0D5','#B7C2C8','#A8B4BB')
visible <- sort(unique(na.omit(dgrid$band)))
graymap <- setNames(rep(graycols,length.out=length(visible)), as.character(visible))

# Descriptive 95% ellipse in the 2-D PCA training-score space.
# Ellipse is ONLY a visual reference for the 2D PCA distribution.
# The 13D chemical Manhattan P95 boundary does not correspond to this ellipse.
train_pc <- as.matrix(train_xy[, c('PC1', 'PC2')])
pc_center <- colMeans(train_pc)
pc_eig <- eigen(stats::cov(train_pc), symmetric=TRUE)
angles <- seq(0, 2*pi, length.out=361L)
scale95 <- sqrt(pc_eig$values * stats::qchisq(.95, df=2))
ell_pts <- t(pc_eig$vectors %*% diag(scale95, 2L) %*%
             rbind(cos(angles), sin(angles)))
ell_pts <- sweep(ell_pts, 2L, pc_center, '+')
ellipse_p95 <- data.frame(PC1=ell_pts[,1], PC2=ell_pts[,2])


# Randomize plotting order so all three pollutant classes intermix visually
# rather than appearing as strict top/middle/bottom layers.
poll_xy_mix <- poll_xy[sample.int(nrow(poll_xy)), , drop=FALSE]
poll_xy_mix$AD_class <- factor(poll_xy_mix$AD_class, levels=AD_LEVELS)

# Equal point size for all three pollutant classes. In the MAIN panel the points
# are plotted in one shuffled layer for a more natural mixed cloud.
POINT_SIZE <- 0.22
p_pca_main <- ggplot() +
  geom_raster(data=dgrid[!is.na(dgrid$band),,drop=FALSE],
              aes(PC1,PC2,fill=factor(band)),alpha=.36) +
  scale_fill_manual(values=graymap,guide='none') +
  geom_point(data=poll_xy_mix,
             aes(PC1,PC2, colour=AD_class,
                 alpha=ifelse(AD_class=='Core', .44,
                              ifelse(AD_class=='Peripheral', .42, .30))),
             size=POINT_SIZE, show.legend=FALSE) +
  scale_colour_manual(values=COL, guide='none') +
  scale_alpha_identity() +
  geom_path(data=ellipse_p95, aes(PC1,PC2),
            linewidth=.77, linetype='22', colour=ELLIPSE_COL) +
  coord_fixed(ratio=1, xlim=xr, ylim=yr, expand=FALSE, clip='on') +
  labs(x=sprintf('PC1 (%.1f%%)',vshare[1]),
       y=sprintf('PC2 (%.1f%%)',vshare[2])) +
  base_style

# Bottom-right inset: full PCA range, same orientation as the main plot.
p_pca_inset <- ggplot() +
  geom_point(data=poll_xy[poll_xy$AD_class=='Outside',,drop=FALSE],
             aes(PC1,PC2),colour=COL['Outside'],size=.10,
             alpha=.30) +
  geom_point(data=poll_xy[poll_xy$AD_class=='Peripheral',,drop=FALSE],
             aes(PC1,PC2),colour=COL['Peripheral'],size=.10,
             alpha=.42) +
  geom_point(data=poll_xy[poll_xy$AD_class=='Core',,drop=FALSE],
             aes(PC1,PC2),colour=COL['Core'],size=.10,
             alpha=.44) +
  geom_path(data=ellipse_p95, aes(PC1,PC2),
            linewidth=.55, linetype='22', colour=ELLIPSE_COL) +
  coord_fixed(ratio=1, xlim=full_xr, ylim=full_yr, expand=FALSE) +
  annotate('text', x=full_xr[1] + 0.05*diff(full_xr),
           y=full_yr[2] - 0.07*diff(full_yr),
           label='Full range', hjust=0, vjust=1,
           size=FONT_MM*0.86, family='sans', colour='#3A3A3A') +
  theme_void(base_size=FONT, base_family='sans') +
  theme(panel.border=element_rect(colour='#7A7A7A', fill=NA, linewidth=.45),
        panel.background=element_rect(fill='white', colour=NA),
        plot.background=element_rect(fill='white', colour=NA),
        plot.margin=margin(0, 0, 0, 0, unit='pt'))

# Inset fractions of the PCA PANEL (not the complete exported page).
# V20 shifts the full-range inset right by 0.020 panel widths and
# down by 0.030 panel heights, without resizing it or changing its data.
# To move horizontally, adjust LEFT and RIGHT by the same amount;
# to move vertically, adjust BOTTOM and TOP by the same amount.
INSET_LEFT   <- 0.700
INSET_RIGHT  <- 0.995
INSET_BOTTOM <- 0.045
INSET_TOP    <- 0.290

p_pca <- p_pca_main +
  patchwork::inset_element(
    p_pca_inset,
    left=INSET_LEFT, right=INSET_RIGHT,
    bottom=INSET_BOTTOM, top=INSET_TOP,
    align_to='panel', clip=TRUE
  )

# Source-file categories remain separate, with one value per unique DTXSID
# WITHIN the corresponding category, and commas in n (e.g., n = 19,328).
N <- nrow(sumtab)
rects <- do.call(rbind,lapply(seq_len(N),function(i) {
  vals <- as.numeric(unlist(sumtab[i, paste0(AD_LEVELS,'_pct'), drop=FALSE], use.names=FALSE))
  stopifnot(length(vals) == length(AD_LEVELS), all(is.finite(vals)))
  data.frame(row=N-i+1L, AD=factor(AD_LEVELS,levels=AD_LEVELS),
             left=c(0,head(cumsum(vals),-1L)),right=cumsum(vals),pct=vals)
}))
rects$AD <- factor(rects$AD,levels=AD_LEVELS)
barlabels <- sprintf('%s  (n = %s)',sumtab$Category,
                     format(sumtab$n,big.mark=',',trim=TRUE))
label_df <- data.frame(y=N:1,lab=barlabels)

# Use a dedicated label column so category names never overlap the PCA panel.
p_names <- ggplot(label_df,aes(x=1,y=y,label=lab)) +
  geom_text(size=FONT_MM*0.93,hjust=1,colour='#222222',family='sans',fontface='plain', lineheight=0.97) +
  scale_x_continuous(limits=c(0,1.04),expand=c(0,0)) +
  scale_y_continuous(limits=c(.5,N+.5),expand=c(0,0)) +
  labs(title=NULL) +
  theme_void(base_size=FONT) +
  theme(text=element_text(size=FONT,face='plain',colour=INK),
        plot.title=element_text(size=FONT,colour=INK,margin=margin(b=13)),
        plot.margin=margin(8,0,3,0,unit='pt'))

# Keep large-segment labels inside; use a slightly smaller font for narrow segments
# so values such as 8.2% and 8.6% remain fully visible.
pct_labels <- subset(rects, pct>=7.0)
pct_labels$lab <- sprintf('%.1f%%',pct_labels$pct)
# All three lighter bar fills support dark text better than white text.
pct_labels$ink <- '#333333'
pct_labels$size_mm <- ifelse(pct_labels$pct < 10, FONT_MM*0.72,
                             ifelse(pct_labels$pct < 15, FONT_MM*0.78, FONT_MM*0.84))
p_bar <- ggplot(rects) +
  geom_rect(aes(xmin=left,xmax=right,ymin=row-.32,ymax=row+.32,fill=AD),
            colour='white',linewidth=.30) +
  geom_text(data=pct_labels,
            aes(x=(left+right)/2,y=row,label=lab,colour=ink,size=size_mm),
            family='sans',fontface='plain',show.legend=FALSE, lineheight=0.95) +
  scale_size_identity() +
  scale_colour_identity() +
  scale_fill_manual(values=COL,guide='none') +
  scale_y_continuous(limits=c(.5,N+.5),expand=c(0,0)) +
  scale_x_continuous(limits=c(0,102.5),breaks=seq(0,100,20),expand=c(0,0)) +
  labs(x='Proportion of chemicals (%)',y=NULL) +
  coord_cartesian(xlim=c(0,102.5), clip='off') +
  base_style +
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank(),
        axis.line.y=element_blank(),
        panel.grid.major.x=element_line(colour='#E9E9E9',linewidth=.27),
        plot.margin=margin(8,12,3,0,unit='pt'))


# BALANCED TWO-PANEL COMPOSITION ----------------------------------------------
# The PCA must stay square because PC1 and PC2 are on the same numeric scale.
# V16's bar panel fills all available height, making it much taller than PCA.
# Vertically center BOTH the right-hand labels and bars as one block, with
# moderate equal top/bottom breathing room. This fixes the visual imbalance
# WITHOUT distorting PCA geometry or changing data / percentage labels.
# Increasing RIGHT_PAD makes the right side shorter (and vice versa).
RIGHT_PAD <- 0.065  # slightly more breathing room to better align the right block with the square PCA panel
stopifnot(RIGHT_PAD >= 0, RIGHT_PAD < 0.25)
right_pair <- (p_names + p_bar) +
  plot_layout(nrow=1, widths=c(0.98, 1.34))
right_pair_balanced <- (plot_spacer() / right_pair / plot_spacer()) +
  plot_layout(ncol=1, heights=c(RIGHT_PAD, 1-2*RIGHT_PAD, RIGHT_PAD))
main_fig <- (p_pca + right_pair_balanced) +
  plot_layout(nrow=1, widths=c(1.62, 0.98+1.34))
# Final figure without the bottom legend strip for a cleaner, more compact layout.
fig <- main_fig

save_plot <- function(plot, stem, width=23.5, height=8.8) {
  if (capabilities('cairo')) {
    ggsave(file.path(OUT,paste0(stem,'.pdf')),plot,
           width=width,height=height,units='in',
           device=grDevices::cairo_pdf,bg='white',limitsize=FALSE)
  } else {
    ggsave(file.path(OUT,paste0(stem,'.pdf')),plot,
           width=width,height=height,units='in',
           device='pdf',useDingbats=FALSE,bg='white',limitsize=FALSE)
  }
  ggsave(file.path(OUT,paste0(stem,'.png')),plot,
         width=width,height=height,units='in',dpi=600,
         bg='white',limitsize=FALSE)
  ggsave(file.path(OUT,paste0(stem,'.tiff')),plot,
         width=width,height=height,units='in',dpi=600,
         compression='lzw',bg='white',limitsize=FALSE)
}
save_plot(fig,'Figure4c_chemical_space_MANHATTAN_3GROUP_NO_LEGEND_BALANCED')
print(fig)  # show completed figure in RStudio Plots pane

caption <- paste0(
  'Fig. 4c. Chemical-space coverage across eight pollutant categories. ',
  'Principal component analysis (PCA) was fitted to 13 standardized chemical ',
  'descriptors of ',nrow(train),' unique training chemicals, and ',nrow(u),
  ' distinct pollutants were projected. ',
  'Equally weighted Manhattan distances to the training chemical centroid ',
  '(13 descriptors, weight 1/13 each) were calculated after z-score ',
  'standardization using unique training chemicals. ',
  'Pollutants were classified as core (distance <= training P90), ',
  'peripheral (training P90 < distance <= training P95), ',
  'or outside (distance > training P95). ',
  'The training P90 and P95 are ',sprintf('%.5f',P90_chem),' and ',
  sprintf('%.5f',P95_chem),', respectively. ',
  'The right panel shows category-specific proportions; one DTXSID is counted ',
  'once per category, and pollutant categories may overlap. ',
  'The dashed training PCA ellipse is a descriptive 95% bivariate ellipse ',
  'and does not represent the 13-dimensional Manhattan AD boundary. ',
  'The inset shows the full PCA range.'
)
writeLines(caption,file.path(OUT,'Figure4c_MANHATTAN_3GROUP_caption.txt'),useBytes=TRUE)
message('DONE | Large-type Figure 4c using chemical-only equal-weight Manhattan (P90/P95): ',OUT)
print(sumtab[,c('Category','n','Core_pct','Peripheral_pct','Outside_pct')])
