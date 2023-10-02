library(dplyr)
library(EnhancedVolcano)
library(Seurat)

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("DESeq2")
#als_atlas <- readRDS('/data1/APA/Paul_ALS_Data/ALS_snRNA_final_with_scVI.RDS')
als_atlas <-readRDS('/data1/APA/Paul_ALS_Data/ALS_snRNA_final_with_scVI.RDS')
als_atlas <- subset(als_atlas, subset=chemistry=='V3')
rbps <- read.table('/home/aiden/codes/APA_stuff/post_qual/APA/For_ALS_atlas_paper/all_rbp_hits.txt')
colnames(rbps) <- c('rbps')

DefaultAssay(als_atlas) <- "RNA"
als_atlas[["RNA"]]@counts <- as.matrix(als_atlas[["RNA"]]@counts)+1
                          

length(rownames(als_atlas))
beforecorrection <- c('HNRPLL', 'HUR', 'HNRNPA1','BRUNOL6', 'Nab2p','BRUNOL4', 'BRUNOL5')
corrected_names <- c('HNRNPLL', 'ELAVL1', 'HNRNPA1','CELF6', 'NAB2', 'CELF4', 'CELF5')

features_to_check <- c(corrected_names, rbps$rbps[which(toupper(rbps$rbps) %in% rownames(als_atlas))])
features_to_check <- unique(features_to_check[which(features_to_check %in% rownames(als_atlas@assays$RNA))])

als_atlas <- SetIdent(als_atlas, value = 'simple_diagnosis')
rbp_DE <- FindMarkers(als_atlas, features = toupper(features_to_check),
                      ident.1 = 'C9ALS' , ident.2 = 'CTRL' ,
                      logfc.threshold = 0.05, test.use = "DESeq2")
rbp_DE$names <- rownames(rbp_DE)

plt <- EnhancedVolcano(rbp_DE,
                       x = 'avg_log2FC', y = 'p_val_adj',
                       FCcutoff = 0.25,  pointSize = 3.0,
                       title = paste0('C9ALS', ' vs CTRL'),
                       drawConnectors = TRUE, widthConnectors = 0.75,
                       lab = rownames(rbp_DE),
                       selectLab = rownames(rbp_DE),
                       xlab = bquote(~Log[2]~ 'fold change'),
                       labSize = 6.0,
                       legendLabels=c('Not sig.','Log (base 2) FC','p-value',
                                      'p-value & Log (base 2) FC'))
plot_name <- paste0('/data1/APA/Paul_ALS_Data/results/motif_analysis/rbps_de_volcanos_final_model_rbps_05_15/AllCTs_C9ALS_CTRL_modelrbp_DE_volcano_plot_th0.25.pdf')
pdf(file = plot_name)
print(plt)
dev.off()

### lets make the plot per celltype
saving_dir <- "/data1/APA/Paul_ALS_Data/results/motif_analysis/rbps_de_volcanos_final_model_rbps_05_15/"
als_atlas$als_groups = als_atlas$group 
levels(als_atlas$als_groups) <- c(levels(als_atlas$als_groups), "C9ALS")
als_atlas$als_groups[als_atlas$group == "C9ALSFTLD" | als_atlas$group == "C9ALSnoFTLD"] <- "C9ALS"

for (ct in names(table(als_atlas$celltype))){
  sub_so <- subset(als_atlas, subset=celltype==ct)
  sub_so <- SetIdent(sub_so, value = sub_so$als_groups)
  pathology = paste0('C9ALS_',ct)
  CTRL =  paste0('CTRL_',ct)
  features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(sub_so@assays$RNA))])
  rbp_DE <- FindMarkers(sub_so, features = features ,
                        ident.1 = 'C9ALS', ident.2 = 'control',
                        logfc.threshold = 0.05, test.use = "DESeq2")
  rbp_DE$names <- rownames(rbp_DE)
  plot_name <- paste0(saving_dir, ct, '_enriched_rbps_DE_C9ALS.pdf')
  pdf(plot_name,width=10, height = 10)
  plt <- EnhancedVolcano(rbp_DE,
                         x = 'avg_log2FC', y = 'p_val_adj',
                         FCcutoff = 0.25,  pointSize = 3.0,
                         title = paste0('C9ALS', ' vs ' , 'control'),
                         drawConnectors = TRUE, widthConnectors = 0.75,
                         lab = rownames(rbp_DE),
                         axisLabSize = 12,
                         titleLabSize = 12,
                         colAlpha = 0.6,
                         legendLabSize = 12, legendIconSize = 12,
                         selectLab = rownames(rbp_DE),
                         xlab = bquote(~Log[2]~ 'fold change'),
                         labSize = 6.0,
                         legendLabels=c('Not sig.','Log (base 2) FC','p-value',
                                        'p-value & Log (base 2) FC'))
  print(plt)
  dev.off()
  
}

###
# Assuming you have loaded required libraries and defined necessary variables

# List of cell types
celltypes <- names(table(als_atlas$celltype))

# Create empty lists to store the results
pairwise_results_C9ALS <- list()
pairwise_results_ctrl <- list()
pairwise_results_sALS <- list()
#
C9ALS_so <- subset(als_atlas, subset = als_groups == 'C9ALS')
C9ALS_so <- SetIdent(C9ALS_so, value = C9ALS_so$celltype)

sALS_so <- subset(als_atlas, subset = als_groups == 'sALSnoFTLD')
sALS_so <- SetIdent(sALS_so, value = sALS_so$celltype)

ctrl_so <- subset(als_atlas, subset = als_groups == 'control')
ctrl_so <- SetIdent(ctrl_so, value = ctrl_so$celltype)


# Loop through each cell type
for (i in 1:length(celltypes)) {
  ct1 <- celltypes[i]
  # Loop through other cell types for comparison
  for (j in 1:length(celltypes)) {
    if (i == j) {
      next  # Skip comparing the same cell type
    }
    ct2 <- celltypes[j]
    # Perform pairwise differential analysis for experimental condition
    features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(C9ALS_so@assays$RNA))])
    rbp_DE_C9ALS <- FindMarkers(C9ALS_so, features = features,
                          ident.1 = ct1, ident.2 = ct2,
                          logfc.threshold = 0.05, test.use = "DESeq2")
    
    features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(ctrl_so@assays$RNA))])
    rbp_DE_ctrl <- FindMarkers(ctrl_so, features = features,
                          ident.1 = ct1, ident.2 = ct2,
                          logfc.threshold = 0.05, test.use = "DESeq2")
    
    features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(sALS_so@assays$RNA))])
    rbp_DE_sALS <- FindMarkers(sALS_so, features = features,
                               ident.1 = ct1, ident.2 = ct2,
                               logfc.threshold = 0.05, test.use = "DESeq2")
    # Further processing, plotting, and saving for the control condition
    pairwise_results_C9ALS[[paste0(ct1, "_vs_", ct2)]] <- rbp_DE_C9ALS
    pairwise_results_ctrl[[paste0(ct1, "_vs_", ct2)]] <- rbp_DE_ctrl
    pairwise_results_sALS[[paste0(ct1, "_vs_", ct2)]] <- rbp_DE_sALS
    
  }
}

# Create an empty list to store the ordered and filtered avg_log2FC columns
avg_log2FC_list <- list()

for (comparison_name in names(pairwise_results_C9ALS)) {
  comparison_result <- pairwise_results_C9ALS[[comparison_name]]
  
  # Order the rows alphabetically based on row names
  ordered_result <- comparison_result[order(rownames(comparison_result)), ]
  print(ordered_result)
  
  # Apply filtering conditions and set avg_log2FC to zero
  ordered_result$avg_log2FC[(ordered_result$p_val_adj > 0.05) & (ordered_result$p_val > 0.05)] <- 0
  
  # Store the ordered and filtered avg_log2FC column along with rownames
  avg_log2FC_list[[comparison_name]] <- ordered_result$avg_log2FC
}

# Combine avg_log2FC columns into a single DataFrame
avg_log2FC_df <- as.data.frame(avg_log2FC_list)

# Restore rownames to the DataFrame
rownames(avg_log2FC_df) <- rownames(ordered_result)

# Print the resulting DataFrame
print(avg_log2FC_df)

csv_filename <- "/data1/APA/Paul_ALS_Data/results/avg_log2FC_results.csv"
write.csv(avg_log2FC_df, file = csv_filename, row.names = TRUE)
###
 # lets do celltypes vs others
all_results_C9ALS <- list()
for (ct in celltypes){
  features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(C9ALS_so@assays$RNA))])
  rbp_DE_C9ALS <- FindMarkers(C9ALS_so, features = features,
                              ident.1 = ct, logfc.threshold = 0.05, test.use = "DESeq2")
  all_results_C9ALS[[ct]] <- rbp_DE_C9ALS
}
# Create an empty list to store the ordered and filtered avg_log2FC columns
c9als_avg_log2FC_list <- list()

for (comparison_name in names(all_results_C9ALS)) {
  comparison_result <- all_results_C9ALS[[comparison_name]]
  
  # Order the rows alphabetically based on row names
  ordered_result <- comparison_result[order(rownames(comparison_result)), ]
  # Apply filtering conditions and set avg_log2FC to zero
  ordered_result$avg_log2FC[(ordered_result$p_val_adj > 0.05) & (ordered_result$p_val > 0.05)] <- 0
  # Store the ordered and filtered avg_log2FC column along with rownames
  c9als_avg_log2FC_list[[comparison_name]] <- ordered_result$avg_log2FC
}

# Combine avg_log2FC columns into a single DataFrame
c9als_avg_log2FC_df <- as.data.frame(c9als_avg_log2FC_list)

# Restore rownames to the DataFrame
rownames(c9als_avg_log2FC_df) <- rownames(ordered_result)
csv_filename <- "/data1/APA/Paul_ALS_Data/results/c9als_all_log2FC_results.csv"
write.csv(c9als_avg_log2FC_df, file = csv_filename, row.names = TRUE)


all_results_sALS <- list()
for (ct in celltypes){
  features <- unique(toupper(features_to_check)[which(toupper(features_to_check) %in% rownames(sALS_so@assays$RNA))])
  rbp_DE_sALS <- FindMarkers(sALS_so, features = features,
                              ident.1 = ct, logfc.threshold = 0.05, test.use = "DESeq2")
  all_results_sALS[[ct]] <- rbp_DE_sALS
}
# Create an empty list to store the ordered and filtered avg_log2FC columns
sals_avg_log2FC_list <- list()

for (comparison_name in names(all_results_sALS)) {
  comparison_result <- all_results_sALS[[comparison_name]]
  
  # Order the rows alphabetically based on row names
  ordered_result <- comparison_result[order(rownames(comparison_result)), ]
  # Apply filtering conditions and set avg_log2FC to zero
  ordered_result$avg_log2FC[(ordered_result$p_val_adj > 0.05) & (ordered_result$p_val > 0.05)] <- 0
  # Store the ordered and filtered avg_log2FC column along with rownames
  sals_avg_log2FC_list[[comparison_name]] <- ordered_result$avg_log2FC
}

# Combine avg_log2FC columns into a single DataFrame
sals_avg_log2FC_df <- as.data.frame(sals_avg_log2FC_list)

# Restore rownames to the DataFrame
rownames(sals_avg_log2FC_df) <- rownames(ordered_result)
csv_filename <- "/data1/APA/Paul_ALS_Data/results/sals_all_log2FC_results.csv"
write.csv(sals_avg_log2FC_df, file = csv_filename, row.names = TRUE)
