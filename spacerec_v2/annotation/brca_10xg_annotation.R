#!/usr/bin/env Rscript

options(stringsAsFactors=FALSE)
set.seed(112358)

args <- commandArgs(trailingOnly=FALSE)
file_arg <- grep("^--file=", args, value=TRUE)
if (!length(file_arg)) stop("Run this script with Rscript so --file is available")
repo <- normalizePath(getwd(), mustWork=TRUE)

config_path <- Sys.getenv("SPACEREC_ANNO_CONFIG", "")
config <- list()
if (nzchar(config_path)) {
    if (!file.exists(config_path)) stop("SPACEREC_ANNO_CONFIG does not exist: ", config_path)
    config <- jsonlite::fromJSON(config_path, simplifyVector=TRUE)
}

pick_path <- function(name, default) {
    value <- config[[name]]
    if (is.null(value) || is.na(value) || !nzchar(as.character(value))) default else as.character(value)
}

paths <- list(
    sc_ref=pick_path("sc_ref_h5ad", file.path(repo, "resources/brca/deconv/reference/scRNA_adata_reannotated.h5ad")),
    xenium_outs=pick_path("xenium_outs", file.path(repo, "resources/brca/xen/outs")),
    out_dir=pick_path("output_dir", file.path(repo, "resources/brca/anno")),
    merge_json=pick_path("merge_json", file.path(repo, "spacerec_v2/deconv/brca_type_merge_17to11.json"))
)
paths$bp_dir <- file.path(paths$out_dir, "bpcells")
paths$reference_bp <- file.path(paths$bp_dir, "reference_counts")
paths$xenium_bp <- file.path(paths$bp_dir, "xenium_counts")
paths$annotation_csv <- file.path(paths$out_dir, "annotation_10xg.csv")
paths$cell_groups_csv <- file.path(paths$out_dir, "cell_groups.csv")
paths$expr_corr_csv <- file.path(paths$out_dir, "reference_xenium_gene_expression_correlation.csv")
paths$expr_corr_png <- file.path(paths$out_dir, "reference_xenium_gene_expression_correlation.png")
paths$cell_type_freq_csv <- file.path(paths$out_dir, "cell_type_frequencies.csv")
paths$prediction_score_csv <- file.path(paths$out_dir, "prediction_score_summary.csv")
paths$plot_dir <- file.path(paths$out_dir, "figures")
paths$prism_dir <- file.path(paths$out_dir, "prism")
paths$embedding_dir <- file.path(paths$out_dir, "embeddings")
paths$local_coords <- pick_path("cells_parquet", file.path(paths$xenium_outs, "cells.parquet"))

dir.create(paths$out_dir, recursive=TRUE, showWarnings=FALSE)
dir.create(paths$bp_dir, recursive=TRUE, showWarnings=FALSE)
dir.create(paths$plot_dir, recursive=TRUE, showWarnings=FALSE)
dir.create(paths$prism_dir, recursive=TRUE, showWarnings=FALSE)
dir.create(paths$embedding_dir, recursive=TRUE, showWarnings=FALSE)

required <- c("Seurat", "BPCells", "jsonlite", "ggplot2", "scales", "arrow", "Matrix", "rhdf5", "future")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) stop("Missing required R packages: ", paste(missing, collapse=", "))

suppressPackageStartupMessages({
    library(Seurat)
    library(BPCells)
    library(jsonlite)
    library(ggplot2)
    library(scales)
    library(arrow)
    library(Matrix)
    library(rhdf5)
    library(future)
})

cell_type_colors <- c(
    "B Cells"="#4C78A8",
    "CD4+ T Cells"="#1F77B4",
    "CD8+ T Cells"="#17BECF",
    "DCIS 1"="#F28E2B",
    "DCIS 2"="#FF9DA7",
    "Endothelial"="#76B7B2",
    "IRF7+ DCs"="#B07AA1",
    "Invasive Tumor"="#D62728",
    "LAMP3+ DCs"="#9467BD",
    "Macrophages 1"="#8CD17D",
    "Macrophages 2"="#59A14F",
    "Mast Cells"="#E377C2",
    "Myoepi ACTA2+"="#8C564B",
    "Myoepi KRT15+"="#C49C94",
    "Perivascular-Like"="#7F7F7F",
    "Prolif Invasive Tumor"="#B22222",
    "Stromal"="#EDC948",
    "Stromal & T Cell Hybrid"="#BAB0AC",
    "T Cell & Tumor Hybrid"="#AEC7E8",
    "T Cells"="#1F77B4",
    "DCIS"="#F28E2B",
    "DCs"="#9467BD",
    "Macrophages"="#59A14F",
    "Myoepi"="#8C564B",
    "Others"="#808080"
)

threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))
if (is.na(threads) || threads < 1L) threads <- 8L
options(future.globals.maxSize=200 * 1024^3)
future::plan("multicore", workers=threads)

message("hostname=", Sys.info()[["nodename"]])
message("repo=", repo)
message("threads=", threads)
message("xenium_outs=", paths$xenium_outs)
message("merge_json=", paths$merge_json)

if (!file.exists(paths$merge_json)) stop("Missing merge JSON: ", paths$merge_json)
merge_config <- jsonlite::fromJSON(paths$merge_json, simplifyVector=FALSE)
json_merged_cell_type_order <- as.character(merge_config$cell_type_order)
merged_cell_type_order <- unique(c(json_merged_cell_type_order, "Others"))
brca_cell_type_order <- unique(unlist(lapply(merge_config$merge, as.character), use.names=FALSE))
hybrid_cell_type_order <- c("Stromal & T Cell Hybrid", "T Cell & Tumor Hybrid")
merge_map <- c()
for (target in names(merge_config$merge)) {
    sources <- as.character(merge_config$merge[[target]])
    merge_map[sources] <- target
}
missing_targets <- setdiff(json_merged_cell_type_order, names(merge_config$merge))
if (length(missing_targets) > 0) {
    stop("Merge JSON is missing entries for: ", paste(missing_targets, collapse=", "))
}

merge_cell_types <- function(values) {
    values_chr <- as.character(values)
    merged <- unname(merge_map[values_chr])
    merged[is.na(merged) & !is.na(values_chr)] <- "Others"
    factor(merged, levels=merged_cell_type_order)
}

read_categorical <- function(path, group) {
    categories <- rhdf5::h5read(path, file.path(group, "categories"))
    codes <- rhdf5::h5read(path, file.path(group, "codes"))
    categories <- as.character(categories)
    out <- rep(NA_character_, length(codes))
    valid <- codes >= 0
    out[valid] <- categories[codes[valid] + 1L]
    out
}

read_h5ad_string_vector <- function(path, group, name) {
    target <- file.path(group, name)
    h5 <- rhdf5::H5Fopen(path)
    on.exit(rhdf5::H5Fclose(h5), add=TRUE)
    if (!rhdf5::H5Lexists(h5, target)) return(NULL)
    as.character(rhdf5::h5read(path, target))
}

read_optional_categorical <- function(path, group) {
    h5 <- rhdf5::H5Fopen(path)
    on.exit(rhdf5::H5Fclose(h5), add=TRUE)
    if (!rhdf5::H5Lexists(h5, file.path(group, "categories"))) return(NULL)
    read_categorical(path, group)
}

save_plot <- function(plot, filename, width=8, height=6) {
    path <- file.path(paths$plot_dir, filename)
    ggplot2::ggsave(path, plot=plot, width=width, height=height, dpi=300, limitsize=FALSE)
    message("Saved plot: ", path)
}

safe_dim_plot <- function(obj, filename, reduction=NULL, group.by, cols=NULL, width=8, height=6, label=TRUE) {
    if (!group.by %in% names(obj[[]])) {
        message("Skipping DimPlot; missing metadata column ", group.by)
        return(invisible(NULL))
    }
    if (!is.null(reduction) && !(reduction %in% SeuratObject::Reductions(obj))) {
        message("Skipping DimPlot; missing reduction ", reduction)
        return(invisible(NULL))
    }
    args <- list(object=obj, group.by=group.by, label=label, label.size=3, raster=TRUE)
    if (!is.null(reduction)) args$reduction <- reduction
    if (!is.null(cols)) args$cols <- cols
    p <- do.call(Seurat::DimPlot, args)
    save_plot(p, filename, width=width, height=height)
}

add_local_spatial_metadata <- function(obj) {
    if (!file.exists(paths$local_coords)) {
        message("Local centroid file not found; spatial plots will be skipped: ", paths$local_coords)
        obj$x_centroid <- NA_real_
        obj$y_centroid <- NA_real_
        return(obj)
    }
    if (grepl("\\.parquet$", paths$local_coords, ignore.case=TRUE)) {
        coords <- as.data.frame(arrow::read_parquet(paths$local_coords))
    } else {
        coords <- read.csv(paths$local_coords, check.names=FALSE)
    }
    if (all(c("cell_id", "x_centroid", "y_centroid") %in% names(coords))) {
        coords <- coords[, c("cell_id", "x_centroid", "y_centroid")]
        names(coords) <- c("xen_cell_id", "centroid_x", "centroid_y")
    }
    needed <- c("xen_cell_id", "centroid_x", "centroid_y")
    if (!all(needed %in% names(coords))) {
        message("Local centroid file lacks required columns; spatial plots will be skipped")
        obj$x_centroid <- NA_real_
        obj$y_centroid <- NA_real_
        return(obj)
    }
    coords$xen_cell_id <- as.character(coords$xen_cell_id)
    coords <- coords[!duplicated(coords$xen_cell_id), needed]
    obj_ids <- colnames(obj)
    obj_barcodes <- sub("_(Assay[^_]+)$", "", obj_ids)
    idx <- match(obj_ids, coords$xen_cell_id)
    missing_idx <- is.na(idx)
    idx[missing_idx] <- match(obj_barcodes[missing_idx], coords$xen_cell_id)
    obj$x_centroid <- as.numeric(coords$centroid_x[idx])
    obj$y_centroid <- as.numeric(coords$centroid_y[idx])
    message("Matched local centroids for ", sum(!is.na(idx)), " / ", ncol(obj), " Xenium cells")
    obj
}

plot_local_spatial <- function(obj, group.by, filename, cols=NULL, continuous=FALSE, width=7, height=7) {
    meta <- obj[[]]
    if (!all(c("x_centroid", "y_centroid", group.by) %in% names(meta))) {
        message("Skipping local spatial plot; missing metadata for ", group.by)
        return(invisible(NULL))
    }
    df <- meta[!is.na(meta$x_centroid) & !is.na(meta$y_centroid) & !is.na(meta[[group.by]]), , drop=FALSE]
    if (!nrow(df)) {
        message("Skipping local spatial plot; no matched local centroid rows for ", group.by)
        return(invisible(NULL))
    }
    p <- ggplot(df, aes(x=x_centroid, y=y_centroid, color=.data[[group.by]])) +
        geom_point(size=0.08, alpha=0.8) +
        coord_fixed() +
        scale_y_reverse() +
        theme_void()
    if (continuous) {
        p <- p + scale_color_viridis_c(option="A")
    } else if (!is.null(cols)) {
        p <- p + scale_color_manual(values=cols, na.value="#BDBDBD")
    }
    save_plot(p, filename, width=width, height=height)
}

export_reduction <- function(obj, reduction, filename) {
    if (!(reduction %in% SeuratObject::Reductions(obj))) {
        message("Skipping reduction export; missing reduction ", reduction)
        return(invisible(NULL))
    }
    emb <- as.data.frame(SeuratObject::Embeddings(obj, reduction=reduction))
    emb$cell_id <- rownames(emb)
    meta <- obj[[]]
    meta$cell_id <- rownames(meta)
    out <- merge(emb, meta, by="cell_id", all.x=TRUE)
    path <- file.path(paths$embedding_dir, filename)
    write.csv(out, path, row.names=FALSE)
    message("Saved embedding metadata: ", path)
}

read_brca_reference_counts <- function(path) {
    obs_names <- read_h5ad_string_vector(path, "obs", "_index")
    if (is.null(obs_names)) obs_names <- read_h5ad_string_vector(path, "obs", "Index")
    if (is.null(obs_names)) obs_names <- read_h5ad_string_vector(path, "obs", "cell_id")
    if (is.null(obs_names)) stop("Could not find obs index in h5ad reference")

    patients <- read_optional_categorical(path, "obs/Patient")
    samples <- read_optional_categorical(path, "obs/Sample")
    source_cell_type <- read_optional_categorical(path, "obs/Annotation")
    level1_label <- read_optional_categorical(path, "obs/Level1")
    if (is.null(level1_label)) stop("Missing obs/Level1 in BRCA reference")

    reference_cell_type_order <- unique(c(
        brca_cell_type_order,
        hybrid_cell_type_order,
        as.character(unique(level1_label[!is.na(level1_label)]))
    ))
    keep <- !is.na(level1_label)
    if (!any(keep)) stop("No reference cells with Level1 labels")
    message("Reference Level1 source classes=", paste(sort(unique(level1_label[keep])), collapse=", "))

    var_names <- read_h5ad_string_vector(path, "var", "_index")
    if (is.null(var_names)) var_names <- read_h5ad_string_vector(path, "var", "gene_symbols")
    if (is.null(var_names)) var_names <- read_h5ad_string_vector(path, "var", "gene_ids")
    if (is.null(var_names)) stop("Could not find var index/gene symbols in h5ad reference")

    indptr <- rhdf5::h5read(path, "layers/counts/indptr")
    n_obs <- length(obs_names)
    cell_map <- integer(n_obs)
    cell_map[which(keep)] <- seq_len(sum(keep))

    i_all <- list()
    j_all <- list()
    x_all <- list()
    if (length(indptr) == length(var_names) + 1L) {
        i_all <- vector("list", length(var_names))
        j_all <- vector("list", length(var_names))
        x_all <- vector("list", length(var_names))
        for (g in seq_along(var_names)) {
            start <- indptr[g] + 1L
            end <- indptr[g + 1L]
            if (end < start) {
                i_all[[g]] <- integer(0)
                j_all[[g]] <- integer(0)
                x_all[[g]] <- numeric(0)
                next
            }
            idx <- seq.int(start, end)
            rows0 <- rhdf5::h5read(path, "layers/counts/indices", index=list(idx))
            vals <- rhdf5::h5read(path, "layers/counts/data", index=list(idx))
            mapped <- cell_map[rows0 + 1L]
            retained <- mapped > 0L
            i_all[[g]] <- rep.int(g, sum(retained))
            j_all[[g]] <- mapped[retained]
            x_all[[g]] <- vals[retained]
        }
    } else if (length(indptr) == n_obs + 1L) {
        kept_cells <- which(keep)
        i_all <- vector("list", length(kept_cells))
        j_all <- vector("list", length(kept_cells))
        x_all <- vector("list", length(kept_cells))
        for (k in seq_along(kept_cells)) {
            cell <- kept_cells[k]
            start <- indptr[cell] + 1L
            end <- indptr[cell + 1L]
            if (end < start) {
                i_all[[k]] <- integer(0)
                j_all[[k]] <- integer(0)
                x_all[[k]] <- numeric(0)
                next
            }
            idx <- seq.int(start, end)
            cols0 <- rhdf5::h5read(path, "layers/counts/indices", index=list(idx))
            vals <- rhdf5::h5read(path, "layers/counts/data", index=list(idx))
            i_all[[k]] <- cols0 + 1L
            j_all[[k]] <- rep.int(k, length(cols0))
            x_all[[k]] <- vals
        }
    } else {
        stop("Unsupported layers/counts sparse layout: indptr length=", length(indptr),
             ", n_obs=", n_obs, ", n_var=", length(var_names))
    }

    counts <- Matrix::sparseMatrix(
        i=unlist(i_all, use.names=FALSE),
        j=unlist(j_all, use.names=FALSE),
        x=unlist(x_all, use.names=FALSE),
        dims=c(length(var_names), sum(keep)),
        dimnames=list(var_names, obs_names[keep])
    )
    meta <- data.frame(
        Patient=if (is.null(patients)) NA_character_ else patients[keep],
        Sample=if (is.null(samples)) NA_character_ else samples[keep],
        Source_Cell_type=if (is.null(source_cell_type)) NA_character_ else source_cell_type[keep],
        Level1=factor(level1_label[keep], levels=reference_cell_type_order),
        cell_type_19=factor(level1_label[keep], levels=reference_cell_type_order),
        cell_type=factor(level1_label[keep], levels=reference_cell_type_order),
        reference_mode="brca_Level1",
        row.names=obs_names[keep],
        check.names=FALSE
    )
    list(counts=counts, meta=meta)
}

resolve_xenium_dir <- function(path) {
    candidates <- unique(c(path, file.path(path, "outs")))
    for (candidate in candidates) {
        if (file.exists(file.path(candidate, "cell_feature_matrix.h5"))) {
            return(normalizePath(candidate, mustWork=TRUE))
        }
    }
    stop("Missing 10x Xenium outs directory containing cell_feature_matrix.h5: ", path)
}

write_bpcells_counts <- function(obj, assay_name, out_dir) {
    if (dir.exists(out_dir)) unlink(out_dir, recursive=TRUE)
    BPCells::write_matrix_dir(mat=obj[[assay_name]]$counts, dir=out_dir)
    obj[[assay_name]]$counts <- BPCells::open_matrix_dir(dir=out_dir)
    obj
}

average_counts <- function(obj, assay_name, genes) {
    Matrix::rowMeans(as(obj[[assay_name]]$counts[genes, , drop=FALSE], "dgCMatrix"))
}

write_frequency_outputs <- function(flex, xenium.obj) {
    freq_xenium <- as.data.frame(table(xenium.obj$cell_type), stringsAsFactors=FALSE)
    colnames(freq_xenium) <- c("cell_type", "count")
    freq_xenium$sample <- "Xenium"
    freq_xenium$relative_freq <- freq_xenium$count / sum(freq_xenium$count)

    freq_flex <- as.data.frame(table(flex$cell_type), stringsAsFactors=FALSE)
    colnames(freq_flex) <- c("cell_type", "count")
    freq_flex$sample <- "Flex"
    freq_flex$relative_freq <- freq_flex$count / sum(freq_flex$count)

    combined_freq <- rbind(freq_xenium, freq_flex)
    combined_freq$custom_hex <- cell_type_colors[combined_freq$cell_type]
    combined_freq$custom_hex[is.na(combined_freq$custom_hex)] <- "#808080"
    write.csv(combined_freq, file.path(paths$prism_dir, "step07_2_cell_type_frequency_long.csv"), row.names=FALSE)

    prism_freq <- reshape(
        combined_freq[, c("cell_type", "sample", "relative_freq")],
        idvar="cell_type",
        timevar="sample",
        direction="wide"
    )
    names(prism_freq) <- sub("^relative_freq\\.", "", names(prism_freq))
    write.csv(prism_freq, file.path(paths$prism_dir, "step07_2_cell_type_frequency_prism.csv"), row.names=FALSE)

    prism_counts <- reshape(
        combined_freq[, c("cell_type", "sample", "count")],
        idvar="cell_type",
        timevar="sample",
        direction="wide"
    )
    names(prism_counts) <- sub("^count\\.", "", names(prism_counts))
    write.csv(prism_counts, file.path(paths$prism_dir, "step07_2_cell_type_counts_prism.csv"), row.names=FALSE)

    p <- ggplot(subset(combined_freq, sample %in% c("Flex", "Xenium")),
                aes(x=sample, y=relative_freq, fill=cell_type)) +
        geom_bar(stat="identity", position="stack") +
        scale_fill_manual(values=cell_type_colors, na.value="#808080") +
        coord_flip() +
        labs(x="Dataset", y="Frequency") +
        theme_minimal()
    save_plot(p, "step07_2_cell_type_frequency.png", width=8, height=4)
}

write_prediction_score_outputs <- function(xenium.obj) {
    meta <- xenium.obj[[]]
    score_col <- if ("predicted.id_full.score" %in% names(meta)) {
        "predicted.id_full.score"
    } else if ("prediction.score.max" %in% names(meta)) {
        "prediction.score.max"
    } else {
        NA_character_
    }
    if (is.na(score_col)) {
        message("Skipping prediction-score plots; no prediction score column found")
        return(invisible(NULL))
    }

    score_df <- data.frame(
        cell_id=rownames(meta),
        cell_type=meta$cell_type,
        prediction_score=meta[[score_col]],
        stringsAsFactors=FALSE
    )
    write.csv(score_df, file.path(paths$prism_dir, "step07_3_prediction_scores_prism.csv"), row.names=FALSE)

    p <- ggplot(score_df, aes(x=cell_type, y=prediction_score, fill=cell_type)) +
        geom_violin(scale="width") +
        scale_fill_manual(values=cell_type_colors, na.value="#808080") +
        theme_minimal() +
        theme(legend.position="none", axis.text.x=element_text(angle=45, hjust=1)) +
        xlab("") +
        ylab("label prediction score") +
        ylim(0, 1)
    save_plot(p, "step07_3_prediction_score_violin.png", width=8, height=5)

    if ("full.umap" %in% SeuratObject::Reductions(xenium.obj)) {
        tryCatch({
            p2 <- FeaturePlot(xenium.obj, reduction="full.umap", features=score_col, raster=TRUE)
            save_plot(p2, "step07_3_prediction_score_full_umap.png", width=7, height=6)
        }, error=function(e) {
            message("Skipping prediction-score UMAP plot: ", conditionMessage(e))
        })
    }
}

write_cell_type_correlation_outputs <- function(flex, xenium.obj, common_genes) {
    cell_types <- sort(intersect(unique(as.character(flex$cell_type)), unique(as.character(xenium.obj$cell_type))))
    all_cor <- list()
    for (celltype in cell_types) {
        xen_cells <- colnames(xenium.obj)[xenium.obj$cell_type == celltype]
        flex_cells <- colnames(flex)[flex$cell_type == celltype]
        if (length(xen_cells) < 5 || length(flex_cells) < 5) next
        xen_means <- Matrix::rowMeans(as(xenium.obj[["Xenium"]]$counts[common_genes, xen_cells, drop=FALSE], "dgCMatrix"))
        flex_means <- Matrix::rowMeans(as(flex[["RNA"]]$counts[common_genes, flex_cells, drop=FALSE], "dgCMatrix"))
        df <- data.frame(
            cell_type=celltype,
            gene=common_genes,
            Flex=as.numeric(flex_means),
            Xenium=as.numeric(xen_means),
            stringsAsFactors=FALSE
        )
        df$pearson_r <- suppressWarnings(cor(df$Flex, df$Xenium, method="pearson"))
        all_cor[[celltype]] <- df
    }
    if (!length(all_cor)) {
        message("Skipping per-cell-type expression correlation; no shared cell types with enough cells")
        return(invisible(NULL))
    }

    out <- do.call(rbind, all_cor)
    write.csv(out, file.path(paths$prism_dir, "step07_1_per_cell_type_expression_correlation_prism.csv"), row.names=FALSE)
    summary <- unique(out[, c("cell_type", "pearson_r")])
    write.csv(summary, file.path(paths$prism_dir, "step07_1_per_cell_type_expression_correlation_summary.csv"), row.names=FALSE)

    p <- ggplot(out, aes(x=Flex, y=Xenium)) +
        geom_point(size=0.8, alpha=0.55) +
        geom_abline(intercept=0, slope=1, linetype=2) +
        scale_x_log10(labels=scales::label_log(digits=1)) +
        scale_y_log10(labels=scales::label_log(digits=1)) +
        facet_wrap(~cell_type, scales="free", ncol=3) +
        xlab("Flex") +
        ylab("Xenium") +
        theme_bw() +
        theme(axis.text=element_text(size=8, color="black"), axis.title=element_text(size=8))
    save_plot(p, "step07_1_per_cell_type_expression_correlation.png", width=9, height=8)
}

message("Step 1: load BRCA scRNA reference")
if (!file.exists(paths$sc_ref)) stop("Missing scRNA reference: ", paths$sc_ref)
ref <- read_brca_reference_counts(paths$sc_ref)
flex <- CreateSeuratObject(counts=ref$counts, assay="RNA", meta.data=ref$meta)
flex[["percent.mt"]] <- PercentageFeatureSet(flex, pattern="^MT-")
flex_qc <- flex[[]]
flex_qc$cell_id <- rownames(flex_qc)
write.csv(flex_qc[, c("cell_id", "Patient", "cell_type", "nFeature_RNA", "nCount_RNA", "percent.mt")],
          file.path(paths$prism_dir, "step02_1_flex_qc_prism.csv"), row.names=FALSE)
save_plot(
    VlnPlot(flex, features=c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol=3, group.by="Patient", pt.size=0),
    "step02_1_flex_qc_violin.png",
    width=11,
    height=4
)
flex <- subset(flex, subset=nCount_RNA > 200 & nCount_RNA < 10000 & percent.mt < 10)
message("Reference cells after QC=", ncol(flex), " genes=", nrow(flex))

message("Step 2: store reference counts with BPCells")
flex <- write_bpcells_counts(flex, "RNA", paths$reference_bp)

message("Step 3: load Xenium data")
xenium_dir <- resolve_xenium_dir(paths$xenium_outs)
xenium.obj <- LoadXenium(data.dir=xenium_dir, fov="fov", assay="Xenium", molecule.coordinates=FALSE)
DefaultAssay(xenium.obj) <- "Xenium"
xenium.obj <- add_local_spatial_metadata(xenium.obj)
xenium.obj$nCount_Xenium_log <- log1p(xenium.obj$nCount_Xenium)
xenium.obj$nFeature_Xenium_log <- log1p(xenium.obj$nFeature_Xenium)
xen_qc <- xenium.obj[[]]
xen_qc$cell_id <- rownames(xen_qc)
xen_qc_cols <- intersect(c("cell_id", "assay_id", "barcode", "orig.ident", "nFeature_Xenium", "nCount_Xenium", "nCount_Xenium_log", "nFeature_Xenium_log", "x_centroid", "y_centroid"), names(xen_qc))
write.csv(xen_qc[, xen_qc_cols, drop=FALSE],
          file.path(paths$prism_dir, "step03_2_xenium_qc_prism.csv"), row.names=FALSE)
xenium_group <- if ("assay_id" %in% names(xen_qc)) "assay_id" else "orig.ident"
save_plot(
    VlnPlot(xenium.obj, features=c("nCount_Xenium_log", "nFeature_Xenium_log"), ncol=2, pt.size=0, group.by=xenium_group),
    "step03_2_xenium_qc_violin.png",
    width=8,
    height=4
)
plot_local_spatial(xenium.obj, "nCount_Xenium_log", "step03_2_xenium_spatial_ncount_log_local_centroid.png", continuous=TRUE)
plot_local_spatial(xenium.obj, "nFeature_Xenium_log", "step03_2_xenium_spatial_nfeature_log_local_centroid.png", continuous=TRUE)
xenium.obj <- subset(xenium.obj, subset=nCount_Xenium > 0)
message("Xenium cells after QC=", ncol(xenium.obj), " genes=", nrow(xenium.obj))

message("Step 4: store Xenium counts with BPCells")
xenium.obj <- write_bpcells_counts(xenium.obj, "Xenium", paths$xenium_bp)

message("Step 5: write reference/Xenium gene expression correlation")
common_expr_genes <- intersect(rownames(flex), rownames(xenium.obj))
expr_corr <- data.frame(
    gene=common_expr_genes,
    reference_mean=average_counts(flex, "RNA", common_expr_genes),
    xenium_mean=average_counts(xenium.obj, "Xenium", common_expr_genes),
    row.names=NULL,
    check.names=FALSE
)
gene_panel_json <- file.path(xenium_dir, "gene_panel.json")
if (file.exists(gene_panel_json)) {
    gene_panel <- jsonlite::fromJSON(gene_panel_json)
    expr_corr$in_gene_panel <- expr_corr$gene %in% unique(gene_panel$payload$targets$gene_symbol)
} else {
    expr_corr$in_gene_panel <- NA
}
write.csv(expr_corr, paths$expr_corr_csv, row.names=FALSE)
write.csv(expr_corr, file.path(paths$prism_dir, "step03_3_expression_correlation_prism.csv"), row.names=FALSE)
corr_value <- suppressWarnings(cor(expr_corr$reference_mean, expr_corr$xenium_mean, method="pearson"))
write.csv(data.frame(metric="pearson_r", value=corr_value),
          file.path(paths$prism_dir, "step03_3_expression_correlation_summary.csv"), row.names=FALSE)
corr_label <- if (is.na(corr_value)) "Pearson r = NA" else sprintf("Pearson r = %.3f", corr_value)
png(paths$expr_corr_png, width=1600, height=1400, res=180)
print(
    ggplot(expr_corr, aes(x=reference_mean, y=xenium_mean)) +
        geom_point(alpha=0.45, size=1.4) +
        scale_x_continuous(trans="log1p") +
        scale_y_continuous(trans="log1p") +
        annotate("text", x=Inf, y=Inf, label=corr_label, hjust=1.1, vjust=1.4) +
        theme_bw() +
        labs(x="Reference mean counts", y="Xenium mean counts")
)
dev.off()
file.copy(paths$expr_corr_png, file.path(paths$plot_dir, "step03_3_expression_correlation.png"), overwrite=TRUE)

message("Step 6: prepare reference model")
DefaultAssay(flex) <- "RNA"
flex <- NormalizeData(flex)
flex <- FindVariableFeatures(flex)
flex <- ScaleData(flex)
flex <- RunPCA(flex)
flex <- RunUMAP(flex, dims=1:15)
flex <- FindNeighbors(flex, dims=1:15)
flex <- FindClusters(flex, resolution=0.5)
safe_dim_plot(flex, "step04_1_flex_umap_clusters.png", group.by="seurat_clusters", width=7, height=6)
safe_dim_plot(flex, "step04_1_flex_umap_cell_type.png", group.by="cell_type", cols=cell_type_colors, width=7, height=6)
export_reduction(flex, "umap", "step04_1_flex_umap_metadata.csv")

message("Step 7: sketch Xenium data")
DefaultAssay(xenium.obj) <- "Xenium"
sketch_ncells <- min(100000L, ncol(xenium.obj))
xenium.obj <- NormalizeData(xenium.obj)
xenium.obj <- FindVariableFeatures(xenium.obj)
xenium.obj <- SketchData(
    object=xenium.obj,
    ncells=sketch_ncells,
    method="LeverageScore",
    sketched.assay="sketch"
)
DefaultAssay(xenium.obj) <- "sketch"
xenium.obj <- FindVariableFeatures(xenium.obj)
xenium.obj <- ScaleData(xenium.obj)
xenium.obj <- RunPCA(xenium.obj, npcs=20)
xenium.obj <- RunUMAP(xenium.obj, dims=1:16, return.model=TRUE)
xenium.obj <- FindNeighbors(xenium.obj, reduction="pca", dims=1:16)
xenium.obj <- FindClusters(xenium.obj, resolution=0.6)
safe_dim_plot(xenium.obj, "step04_2_xenium_sketch_umap_clusters.png", group.by="sketch_snn_res.0.6", width=7, height=6)
plot_local_spatial(xenium.obj, "sketch_snn_res.0.6", "step04_2_xenium_sketch_spatial_clusters_local_centroid.png")
export_reduction(xenium.obj, "umap", "step04_2_xenium_sketch_umap_metadata.csv")

message("Step 8: transfer labels from BRCA reference to Xenium sketch")
common_genes <- intersect(rownames(flex), rownames(xenium.obj[["sketch"]]))
if (length(common_genes) < 50L) stop("Too few common genes for label transfer: ", length(common_genes))
message("Common genes for transfer=", length(common_genes))

flex_subset <- subset(flex, features=common_genes)
DefaultAssay(flex_subset) <- "RNA"
flex_subset <- NormalizeData(flex_subset)
flex_subset <- FindVariableFeatures(flex_subset)
flex_subset <- ScaleData(flex_subset)
flex_subset <- RunPCA(flex_subset)

anchors <- FindTransferAnchors(
    reference=flex_subset,
    query=xenium.obj,
    query.assay="sketch",
    features=common_genes,
    dims=1:20,
    reference.reduction="pca"
)
predictions <- TransferData(
    anchorset=anchors,
    refdata=flex_subset$cell_type,
    dims=1:20
)
xenium.obj <- AddMetaData(xenium.obj, metadata=predictions)
xenium.obj$predicted.id_19 <- xenium.obj$predicted.id
write.csv(predictions, file.path(paths$prism_dir, "step05_label_transfer_sketch_predictions_prism.csv"), row.names=TRUE)
safe_dim_plot(xenium.obj, "step05_xenium_sketch_umap_predicted_id.png", group.by="predicted.id", cols=cell_type_colors, width=7, height=6)
plot_local_spatial(xenium.obj, "predicted.id", "step05_xenium_sketch_spatial_predicted_id_local_centroid.png", cols=cell_type_colors)

message("Step 9: project sketch labels to full Xenium dataset")
xenium.obj <- ProjectData(
    object=xenium.obj,
    assay="Xenium",
    full.reduction="pca.full",
    sketched.assay="sketch",
    sketched.reduction="pca",
    umap.model="umap",
    dims=1:16,
    refdata=list(cluster_full="sketch_snn_res.0.6")
)
xenium.obj <- TransferSketchLabels(
    object=xenium.obj,
    sketched.assay="sketch",
    reduction="pca.full",
    dims=1:16,
    refdata=list(predicted.id_full="predicted.id"),
    k=50,
    reduction.model="umap",
    recompute.neighbors=FALSE,
    recompute.weights=FALSE
)
xenium.obj$predicted.id_full_19 <- xenium.obj$predicted.id_full
xenium.obj$cell_type_19 <- xenium.obj$predicted.id_full_19
xenium.obj$cell_type <- merge_cell_types(xenium.obj$cell_type_19)
flex$cell_type_19 <- flex$cell_type
flex$cell_type <- merge_cell_types(flex$cell_type_19)
safe_dim_plot(xenium.obj, "step06_xenium_full_umap_clusters.png", reduction="full.umap", group.by="cluster_full", width=7, height=6)
safe_dim_plot(xenium.obj, "step06_xenium_full_umap_cell_type.png", reduction="full.umap", group.by="cell_type", cols=cell_type_colors, width=7, height=6)
plot_local_spatial(xenium.obj, "cell_type", "step06_xenium_full_spatial_cell_type_local_centroid.png", cols=cell_type_colors)
export_reduction(xenium.obj, "full.umap", "step06_xenium_full_umap_metadata.csv")

message("Step 10: write annotation summaries")
write_cell_type_correlation_outputs(flex, xenium.obj, common_genes)
write_frequency_outputs(flex, xenium.obj)
write_prediction_score_outputs(xenium.obj)
cell_type_freq <- as.data.frame(table(xenium.obj$cell_type), stringsAsFactors=FALSE)
colnames(cell_type_freq) <- c("cell_type", "n")
cell_type_freq$fraction <- cell_type_freq$n / sum(cell_type_freq$n)
write.csv(cell_type_freq, paths$cell_type_freq_csv, row.names=FALSE)

meta <- xenium.obj[[]]
score_cols <- grep("^prediction.score", names(meta), value=TRUE)
score_summary <- data.frame(
    metric=score_cols,
    mean=vapply(score_cols, function(x) mean(meta[[x]], na.rm=TRUE), numeric(1)),
    median=vapply(score_cols, function(x) median(meta[[x]], na.rm=TRUE), numeric(1)),
    stringsAsFactors=FALSE
)
write.csv(score_summary, paths$prediction_score_csv, row.names=FALSE)

message("Step 11: export cell_groups.csv and annotation_10xg.csv")
cell_ids <- rownames(meta)
assay_id <- if ("assay_id" %in% names(meta)) meta$assay_id else sub("^.*_(Assay[^_]+)$", "\\1", cell_ids)
barcode <- if ("barcode" %in% names(meta)) meta$barcode else sub("_(Assay[^_]+)$", "", cell_ids)
predicted_id_19 <- if ("predicted.id_full_19" %in% names(meta)) meta$predicted.id_full_19 else meta$cell_type_19
predicted_id <- meta$cell_type
cluster_full <- if ("cluster_full" %in% names(meta)) meta$cluster_full else NA_character_

annotation <- data.frame(
    id=cell_ids,
    Count=meta$nCount_Xenium,
    assay_id=assay_id,
    Assay=assay_id,
	    Layer="",
	    Sample=assay_id,
	    TenXG_anno=meta$cell_type,
	    TenXG_anno_19=meta$cell_type_19,
	    barcode=barcode,
	    predicted_id=predicted_id,
	    predicted_id_19=predicted_id_19,
	    cluster_full=cluster_full,
    stringsAsFactors=FALSE,
    check.names=FALSE
)
if ("prediction.score.max" %in% names(meta)) {
    annotation$prediction_score_max <- meta$prediction.score.max
}
write.csv(annotation, paths$annotation_csv, row.names=FALSE)
write.csv(data.frame(cell_id=cell_ids, group=meta$cell_type), paths$cell_groups_csv, row.names=FALSE)

message("Finished annotation_10xg")
message("annotation_csv=", paths$annotation_csv)
message("cell_groups_csv=", paths$cell_groups_csv)
