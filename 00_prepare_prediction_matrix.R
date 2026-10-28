# Build the 2,109-chemical x 1,317-taxon prediction matrix from the supplied inputs.
suppressPackageStartupMessages(library(data.table))
ROOT <- normalizePath(".", winslash="/", mustWork=TRUE)
DATA <- file.path(ROOT,"data")
TRAIN <- fread(file.path(DATA,"Training_DATABASE_MOA_RECHECKED.csv"),
               na.strings=c("","NA","NaN"),encoding="UTF-8")
TAXON_FILE <- file.path(DATA,"Taxon_taxonomy_traits_prediction.csv")
ROSTER_FILE <- file.path(DATA,"SEVEN_REGION_TAXON_MEMBERSHIP_20261012.csv")
OUT <- file.path(DATA,"prediction_matrix.csv")

CHEM <- c("logS","logKow","logD7_4","MW","TPSA","log_Henry",
          "log_VaporPressure","MeltingPoint","HBA","HBD",
          "AromaticRingCount","RotatableBondCount","BertzCT","Model_MOA_final")
TAXON_COLS <- c("Taxon","Phylum","Class","Order","Family","Genus",
                "Log Body size","Feeding","Feeding_score","Respiration",
                "Respiration_score","Locomotion","Locomotion_score",
                "Body armor","Body_armor_score")
stopifnot(all(c("DTXSID","Taxon",CHEM) %in% names(TRAIN)))
taxa <- fread(TAXON_FILE,encoding="UTF-8")
roster <- fread(ROSTER_FILE,encoding="UTF-8")
if ("Species" %in% names(roster) && !"Taxon" %in% names(roster)) setnames(roster,"Species","Taxon")
stopifnot(all(TAXON_COLS %in% names(taxa)),all(c("Region","Taxon") %in% names(roster)))
taxa <- unique(taxa[,..TAXON_COLS])
roster_taxa <- unique(trimws(as.character(roster$Taxon)))
if (!setequal(roster_taxa,trimws(as.character(taxa$Taxon))))
  stop("Regional roster and trait/taxonomy table do not contain the same taxa.")

chem_cols <- c("DTXSID",CHEM)
consistency <- TRAIN[,lapply(.SD,function(x) uniqueN(x)),by="DTXSID",.SDcols=CHEM]
if (any(as.matrix(consistency[,-1,with=FALSE]) > 1L))
  stop("At least one chemical has inconsistent descriptors or broad MOA across training rows.")
chem <- unique(TRAIN[,..chem_cols])
stopifnot(nrow(chem)==uniqueN(TRAIN$DTXSID),uniqueN(chem$DTXSID)==2109L,
          nrow(taxa)==1317L,uniqueN(taxa$Taxon)==1317L)

grid <- CJ(DTXSID=chem$DTXSID,Taxon=taxa$Taxon,unique=TRUE,sorted=FALSE)
grid <- merge(grid,chem,by="DTXSID",all.x=TRUE,sort=FALSE)
grid <- merge(grid,taxa,by="Taxon",all.x=TRUE,sort=FALSE)
setcolorder(grid,c("DTXSID","Taxon",CHEM,setdiff(TAXON_COLS,"Taxon")))
stopifnot(nrow(grid)==uniqueN(chem$DTXSID)*uniqueN(taxa$Taxon),
          !anyDuplicated(grid,by=c("DTXSID","Taxon")))
fwrite(grid,OUT)
cat("Wrote ",nrow(grid)," chemical–taxon combinations to ",OUT,"\n",sep="")
