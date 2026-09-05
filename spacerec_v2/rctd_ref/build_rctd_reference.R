#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(hdf5r)
  library(jsonlite)
  library(Matrix)
  library(Seurat)
  library(spacexr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  flag <- paste0("--", name)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}
require_arg <- function(name) {
  value <- get_arg(name)
  if (is.null(value)) stop(paste0("Missing --", name), call. = FALSE)
  value
}

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (!length(file_arg)) stop("Run this script with Rscript so --file is available")
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE))

synthetic_h5 <- require_arg("synthetic-h5")
positions_csv <- require_arg("positions-csv")
reference_rds <- require_arg("reference-rds")
out_csv <- require_arg("out-csv")
out_summary <- require_arg("out-summary")
gene_list_output <- require_arg("gene-list-output")
cell_type_order_output <- require_arg("cell-type-order-output")
merge_json <- require_arg("merge-json")
annotation_column <- get_arg("annotation-column", "Annotation")
umi_min <- as.numeric(get_arg("umi-min", "100"))

merge_config <- fromJSON(merge_json, simplifyVector = FALSE)
merged_levels <- as.character(merge_config$cell_type_order)
merge_map <- c()
for (target in names(merge_config$merge)) {
  sources <- as.character(merge_config$merge[[target]])
  merge_map[sources] <- target
}
missing_targets <- setdiff(merged_levels, names(merge_config$merge))
if (length(missing_targets) > 0) {
  stop("Merge JSON is missing entries for: ", paste(missing_targets, collapse = ", "))
}

dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(gene_list_output), recursive = TRUE, showWarnings = FALSE)

h5 <- H5File$new(synthetic_h5, mode = "r")
data <- h5[["matrix/data"]]$read()
indices <- h5[["matrix/indices"]]$read()
indptr <- h5[["matrix/indptr"]]$read()
shape <- as.integer(h5[["matrix/shape"]]$read())
barcodes <- as.character(h5[["matrix/barcodes"]]$read())
genes <- as.character(h5[["matrix/features/name"]]$read())
h5$close_all()

counts <- sparseMatrix(
  i = as.integer(indices) + 1L,
  p = as.integer(indptr),
  x = as.numeric(data),
  dims = shape
)
rownames(counts) <- genes
colnames(counts) <- barcodes
counts <- as(counts, "dgCMatrix")

positions <- read.csv(positions_csv, stringsAsFactors = FALSE, check.names = FALSE)
required_position_cols <- c("barcode", "pxl_col_in_fullres", "pxl_row_in_fullres")
missing_position_cols <- setdiff(required_position_cols, colnames(positions))
if (length(missing_position_cols) > 0) {
  stop("Positions CSV is missing columns: ", paste(missing_position_cols, collapse = ", "))
}
coord_idx <- match(colnames(counts), as.character(positions$barcode))
if (any(is.na(coord_idx))) {
  missing <- colnames(counts)[is.na(coord_idx)]
  stop("Synthetic barcodes missing from positions CSV, first 20: ", paste(head(missing, 20), collapse = ", "))
}
coords <- as.data.frame(as.matrix(positions[coord_idx, c("pxl_col_in_fullres", "pxl_row_in_fullres")]))
colnames(coords) <- c("x", "y")
rownames(coords) <- colnames(counts)
puck <- SpatialRNA(coords = coords, counts = counts, nUMI = Matrix::colSums(counts))

ref_obj <- readRDS(reference_rds)
ref_counts <- tryCatch(
  GetAssayData(ref_obj, assay = "RNA", layer = "counts"),
  error = function(e) GetAssayData(ref_obj, assay = "RNA", slot = "counts")
)
raw_cell_types <- ref_obj[[annotation_column]][, 1]
names(raw_cell_types) <- rownames(ref_obj[[annotation_column]])
merged_cell_types <- unname(merge_map[as.character(raw_cell_types)])
keep <- !is.na(merged_cell_types)
ref_counts <- ref_counts[, keep]
cell_types <- factor(merged_cell_types[keep], levels = merged_levels)
names(cell_types) <- colnames(ref_counts)
rctd_ref <- Reference(counts = ref_counts, cell_types = cell_types, nUMI = Matrix::colSums(ref_counts))

source(file.path(script_dir, "rctd_reference.R"))
res <- build_rctd_reference(
  puck,
  reference = rctd_ref,
  UMI_min = umi_min,
  out_genes = "all",
  renormalize = TRUE,
  verbose = TRUE
)

ref <- as.matrix(res$reference)
aligned <- matrix(
  0,
  nrow = length(genes),
  ncol = length(merged_levels),
  dimnames = list(genes, merged_levels)
)
common_genes <- intersect(genes, rownames(ref))
common_types <- intersect(merged_levels, colnames(ref))
aligned[common_genes, common_types] <- ref[common_genes, common_types, drop = FALSE]

gene_mass <- rowSums(aligned)
nonzero <- gene_mass > 0
aligned <- aligned[nonzero, , drop = FALSE]
col_sums <- colSums(aligned)
if (any(col_sums <= 0)) {
  stop("Zero RCTD reference mass after synthetic-gene alignment for: ",
       paste(names(col_sums)[col_sums <= 0], collapse = ", "))
}
aligned <- sweep(aligned, 2, col_sums, "/")

write.csv(aligned, out_csv, quote = FALSE)
writeLines(rownames(aligned), gene_list_output)
writeLines(merged_levels, cell_type_order_output)

summary <- list(
  task = "build_rctd_reference",
  synthetic_h5 = normalizePath(synthetic_h5),
  positions_csv = normalizePath(positions_csv),
  reference_rds = normalizePath(reference_rds),
  merge_json = normalizePath(merge_json),
  annotation_column = annotation_column,
  source_reference_cells = ncol(ref_counts),
  source_reference_genes = nrow(ref_counts),
  synthetic_spots = ncol(counts),
  synthetic_genes = nrow(counts),
  output_genes = nrow(aligned),
  dropped_zero_genes = sum(!nonzero),
  synthetic_spots_umi_ge_min = sum(Matrix::colSums(counts) >= umi_min),
  umi_min = umi_min,
  rctd_output_genes_before_alignment = nrow(ref),
  common_genes = length(common_genes),
  cell_type_names = merged_levels,
  column_sum_min = min(colSums(aligned)),
  column_sum_max = max(colSums(aligned)),
  n_pixels_used = res$n_pixels_used,
  bulk_de_genes = length(res$gene_list_bulk),
  out_csv = normalizePath(out_csv, mustWork = FALSE),
  gene_list_output = normalizePath(gene_list_output, mustWork = FALSE),
  cell_type_order_output = normalizePath(cell_type_order_output, mustWork = FALSE)
)
write_json(summary, out_summary, pretty = TRUE, auto_unbox = TRUE)

cat("wrote", out_csv, "\n")
cat("summary", out_summary, "\n")
cat("genes", nrow(aligned), "cell_types", ncol(aligned), "common_genes", length(common_genes), "\n")
cat("column_sums", paste(signif(colSums(aligned), 8), collapse = ","), "\n")
