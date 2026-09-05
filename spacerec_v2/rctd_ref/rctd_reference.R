## Extract RCTD's platform-normalized reference for use outside RCTD.
##
## Reproduces the reference-side half of create.RCTD + fitBulk:
##   1. per-cell sum-to-one normalization, then unweighted mean within cell type
##      (spacexr::get_cell_type_info)
##   2. DE gene selection against the spatial data, bulk thresholds
##      (spacexr::get_de_genes)
##   3. bulk deconvolution of the spatial pseudobulk -> global cell type proportions
##      (spacexr:::decompose_full, bulk_mode = TRUE; needs no Q_mat/sigma)
##   4. per-gene platform rescaling so the proportion-weighted average profile
##      reproduces the spatial data's observed gene fractions (spacexr::get_norm_ref)
##
## RCTD only ever keeps the DE genes, but the step-4 factor is well defined for any
## gene observed in both the reference and the spatial data, so out_genes = "all"
## applies it to the full shared gene set. The proportions in step 3 are always
## estimated on the DE subset, exactly as RCTD does.

library(spacexr)
library(Matrix)

build_rctd_reference <- function(puck,
                                 reference = NULL,
                                 cell_type_profiles = NULL,
                                 cell_type_names = NULL,
                                 CELL_MIN_INSTANCE = 25,
                                 gene_cutoff = 0.000125,
                                 fc_cutoff = 0.5,
                                 UMI_min = 100,
                                 UMI_max = 2e7,
                                 counts_MIN = 10,
                                 MIN_OBS = 3,
                                 out_genes = c("all", "bulk_de"),
                                 out_MIN_OBS = 3,
                                 factor_clip = NULL,
                                 MIN_CHANGE_BULK = 1e-4,
                                 renormalize = TRUE,
                                 verbose = TRUE) {

  out_genes <- match.arg(out_genes)
  if (is.null(reference) && is.null(cell_type_profiles))
    stop("build_rctd_reference: supply either `reference` or `cell_type_profiles`")
  say <- function(...) if (verbose) message(...)

  ## 1. cell type profiles (columns sum to 1) --------------------------------
  if (is.null(cell_type_profiles)) {
    if (is.null(cell_type_names)) cell_type_names <- levels(reference@cell_types)
    info <- spacexr:::process_cell_type_info(reference,
                                             cell_type_names = cell_type_names,
                                             CELL_MIN = CELL_MIN_INSTANCE)
  } else {
    cell_type_profiles <- as.data.frame(as.matrix(cell_type_profiles))
    cell_type_names <- colnames(cell_type_profiles)
    info <- list(cell_type_profiles, cell_type_names, length(cell_type_names))
  }
  profiles <- info[[1]]

  ## 2. pixel filtering and DE gene selection --------------------------------
  puck_all <- spacexr::restrict_counts(puck, rownames(puck@counts),
                                       UMI_thresh = UMI_min, UMI_max = UMI_max,
                                       counts_thresh = counts_MIN)
  say("build_rctd_reference: selecting platform-normalization DE genes")
  gene_list_bulk <- spacexr::get_de_genes(info, puck_all, fc_thresh = fc_cutoff,
                                          expr_thresh = gene_cutoff, MIN_OBS = MIN_OBS)
  if (length(gene_list_bulk) < 10)
    stop("build_rctd_reference: fewer than 10 bulk DE genes found")

  puck_de <- spacexr::restrict_counts(puck_all, gene_list_bulk,
                                      UMI_thresh = UMI_min, UMI_max = UMI_max,
                                      counts_thresh = counts_MIN)
  puck_de <- spacexr:::restrict_puck(puck_de, colnames(puck_de@counts))

  ## 3. bulk fit -> global cell type proportions -----------------------------
  say("build_rctd_reference: decomposing bulk")
  bulkData <- spacexr:::prepareBulkData(profiles, puck_de, gene_list_bulk)
  bulk_fit <- spacexr:::decompose_full(bulkData$X, sum(puck_de@nUMI), bulkData$b,
                                       verbose = FALSE, constrain = FALSE,
                                       MIN_CHANGE = MIN_CHANGE_BULK,
                                       n.iter = 100, bulk_mode = TRUE)
  proportions <- bulk_fit$weights

  ## 4. per-gene platform factor ---------------------------------------------
  ## nUMI is never recomputed by restrict_counts, so it stays the full-transcriptome
  ## total per pixel. target_means is therefore a genuine expected fraction and the
  ## factor is comparable across the two out_genes settings.
  nUMI_tot <- sum(puck_de@nUMI)
  if (out_genes == "bulk_de") {
    bulk_vec <- Matrix::rowSums(puck_de@counts)
    out_gene_list <- gene_list_bulk
  } else {
    puck_out <- spacexr:::restrict_puck(puck_all, colnames(puck_de@counts))
    bulk_vec <- Matrix::rowSums(puck_out@counts)
    out_gene_list <- intersect(rownames(profiles), names(bulk_vec))
    out_gene_list <- out_gene_list[bulk_vec[out_gene_list] >= out_MIN_OBS]
  }

  ref_sub      <- as.matrix(profiles[out_gene_list, , drop = FALSE])
  p            <- proportions / sum(proportions)
  weight_avg   <- as.vector(ref_sub %*% p[colnames(ref_sub)])
  target_means <- as.vector(bulk_vec[out_gene_list]) / nUMI_tot
  platform_factor <- weight_avg / target_means
  names(platform_factor) <- out_gene_list

  ## genes with no expression in any cell type give factor 0 and would divide to NaN
  keep <- is.finite(platform_factor) & platform_factor > 0
  if (sum(!keep) > 0)
    say("build_rctd_reference: dropping ", sum(!keep),
        " genes with zero reference expression or undefined platform factor")

  ## Outside the DE gene set the factor has a heavy tail: genes the reference and
  ## the spatial data disagree on by orders of magnitude get inflated or crushed.
  ## factor_clip = c(lo, hi) quantiles winsorizes it. NULL keeps RCTD's behaviour.
  if (!is.null(factor_clip)) {
    stopifnot(length(factor_clip) == 2)
    lim <- quantile(platform_factor[keep], factor_clip, na.rm = TRUE)
    n_clipped <- sum(platform_factor[keep] < lim[1] | platform_factor[keep] > lim[2])
    platform_factor <- pmin(pmax(platform_factor, lim[1]), lim[2])
    say("build_rctd_reference: winsorized platform factor to [",
        signif(lim[1], 3), ", ", signif(lim[2], 3), "], ", n_clipped, " genes clipped")
  }

  ref_norm <- sweep(ref_sub[keep, , drop = FALSE], 1, platform_factor[keep], "/")

  if (renormalize)
    ref_norm <- sweep(ref_norm, 2, colSums(ref_norm), "/")
  say("build_rctd_reference: final reference is ", nrow(ref_norm), " genes x ",
      ncol(ref_norm), " cell types")

  list(reference       = ref_norm,
       reference_raw   = ref_sub[keep, , drop = FALSE],
       proportions     = proportions,
       platform_factor = platform_factor[keep],
       gene_list_bulk  = gene_list_bulk,
       genes           = rownames(ref_norm),
       n_pixels_used   = ncol(puck_de@counts))
}

## Wrap a genes x cell-types matrix as an InstaPrism refPhi_cs. With no `map`,
## each column is its own cell type (one state per type).
as_refPhi_cs <- function(mat, map = NULL) {
  if (!requireNamespace("InstaPrism", quietly = TRUE))
    stop("as_refPhi_cs: InstaPrism is required to construct a refPhi_cs object")
  mat <- as.matrix(mat)
  if (is.null(map)) map <- setNames(as.list(colnames(mat)), colnames(mat))
  methods::new(methods::getClass("refPhi_cs", where = asNamespace("InstaPrism")),
               phi.cs = mat, map = map)
}

## The `gene_list_reg` RCTD actually regresses on, for restricting the output to
## RCTD's own working gene set.
rctd_reg_genes <- function(puck, reference = NULL, cell_type_profiles = NULL,
                           cell_type_names = NULL, CELL_MIN_INSTANCE = 25,
                           gene_cutoff_reg = 0.0002, fc_cutoff_reg = 0.75,
                           UMI_min = 100, UMI_max = 2e7, counts_MIN = 10,
                           MIN_OBS = 3) {
  if (is.null(cell_type_profiles)) {
    if (is.null(cell_type_names)) cell_type_names <- levels(reference@cell_types)
    info <- spacexr:::process_cell_type_info(reference,
                                             cell_type_names = cell_type_names,
                                             CELL_MIN = CELL_MIN_INSTANCE)
  } else {
    cell_type_profiles <- as.data.frame(as.matrix(cell_type_profiles))
    info <- list(cell_type_profiles, colnames(cell_type_profiles),
                 ncol(cell_type_profiles))
  }
  puck_all <- spacexr::restrict_counts(puck, rownames(puck@counts),
                                       UMI_thresh = UMI_min, UMI_max = UMI_max,
                                       counts_thresh = counts_MIN)
  spacexr::get_de_genes(info, puck_all, fc_thresh = fc_cutoff_reg,
                        expr_thresh = gene_cutoff_reg, MIN_OBS = MIN_OBS)
}
