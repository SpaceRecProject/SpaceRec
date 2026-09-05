suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
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

method_input_dir <- require_arg("method-input-dir")
output_rds <- require_arg("output-rds")
counts <- readMM(file.path(method_input_dir, "reference_counts.mtx"))
genes <- readLines(file.path(method_input_dir, "reference_genes.tsv"))
cells <- readLines(file.path(method_input_dir, "reference_cells.tsv"))
metadata <- read.csv(file.path(method_input_dir, "reference_metadata.csv"), check.names = FALSE)
rownames(counts) <- genes
colnames(counts) <- cells
metadata <- metadata[match(cells, metadata$cell_id), , drop = FALSE]
rownames(metadata) <- metadata$cell_id
obj <- CreateSeuratObject(counts = as(counts, "dgCMatrix"), meta.data = metadata)
obj$Annotation <- obj$cell_type
saveRDS(obj, output_rds)
message("Wrote Seurat reference RDS: ", output_rds)
