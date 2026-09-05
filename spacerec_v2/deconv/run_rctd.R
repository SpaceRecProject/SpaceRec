suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(spacexr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  flag <- paste0("--", name)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}
require_arg <- function(name) {
  value <- get_arg(name)
  if (is.null(value)) {
    stop(paste0("Missing required argument --", name), call. = FALSE)
  }
  value
}

rctd_input_dir <- require_arg("rctd-input-dir")
sc_ref_rds <- require_arg("sc-ref-rds")
output_dir <- require_arg("output-dir")
sample_id <- get_arg("sample-id", "BREAST")
annotation_column <- get_arg("annotation-column", "Annotation")
umi_min <- as.numeric(get_arg("umi-min", "100"))
max_cores <- as.numeric(get_arg("max-cores", "8"))
output_prefix <- get_arg("output-prefix", paste0("RCTD_", sample_id))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts <- readMM(file.path(rctd_input_dir, paste0(sample_id, "_counts.mtx")))
genes <- readLines(file.path(rctd_input_dir, paste0(sample_id, "_genes.tsv")))
spots <- readLines(file.path(rctd_input_dir, paste0(sample_id, "_spots.tsv")))
rownames(counts) <- genes
colnames(counts) <- spots
counts <- as(counts, "dgCMatrix")

coords <- read.csv(
  file.path(rctd_input_dir, paste0(sample_id, "_coords.csv")),
  row.names = 1,
  check.names = FALSE
)
coords <- coords[colnames(counts), c("x", "y")]

spatial <- SpatialRNA(
  coords = coords,
  counts = counts,
  nUMI = Matrix::colSums(counts)
)

ref_obj <- readRDS(sc_ref_rds)
ref_counts <- tryCatch(
  GetAssayData(ref_obj, assay = "RNA", slot = "counts"),
  error = function(e) GetAssayData(ref_obj, assay = "RNA", layer = "counts")
)
cell_types <- ref_obj[[annotation_column]][, 1]
names(cell_types) <- rownames(ref_obj[[annotation_column]])
keep <- !is.na(cell_types)
ref_counts <- ref_counts[, keep]
cell_types <- droplevels(factor(cell_types[keep]))
names(cell_types) <- colnames(ref_counts)

reference <- Reference(
  counts = ref_counts,
  cell_types = cell_types,
  nUMI = Matrix::colSums(ref_counts)
)

rctd <- create.RCTD(spatial, reference, max_cores = max_cores, UMI_min = umi_min)
rctd <- run.RCTD(rctd, doublet_mode = "full")

weights <- rctd@results$weights
weights <- as.matrix(weights)
weights <- weights / pmax(rowSums(weights), 1e-12)
write.csv(weights, file.path(output_dir, paste0(output_prefix, "_proportions.csv")), quote = FALSE)

write.csv(
  weights,
  file.path(output_dir, paste0(output_prefix, "_proportions_spacerec.csv")),
  quote = FALSE
)

saveRDS(rctd, file.path(output_dir, paste0(output_prefix, "_result.rds")))
