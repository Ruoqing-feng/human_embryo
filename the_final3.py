#!/usr/bin/python
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import seaborn as sb

from math import hypot
from matplotlib.collections import LineCollection
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg')
sb.set_style("white")

randomseed=123456

final = pd.read_csv('./finalcell_discardnonimmune.csv')
final = final['well_id'].to_list()
print(len(final))

raw = ad.read_h5ad('../new/final_all_qc_cells_last_run_new_new.h5ad')
print(len(list(set(raw.obs.index).intersection(set(final)))))
a = list(set(raw.obs.index).intersection(set(final)))
raw = raw[a,:]

mc.ut.set_name(raw, 'test')
print(raw.shape)

excluded_gene_names = ["MALAT1", "XIST", "XIST_intron"]
excluded_gene_patterns = ['MT-.*', 'RPS.*', 'RPL.*',"ERCC-.*","MTMR.*","MTND.*","MTRN.*","MTCO.*","MRPL.*","MRPS.*","MTATP.*"]

mc.pl.analyze_clean_genes(raw,
                          excluded_gene_names=excluded_gene_names,
                          excluded_gene_patterns=excluded_gene_patterns,
                          random_seed=randomseed)
mc.pl.pick_clean_genes(raw)

#raw.write('final_all_qc_cells_last_run.h5ad')

full = raw
print(full)
properly_sampled_min_cell_total = 400
properly_sampled_max_cell_total = 20000

total_umis_of_cells = mc.ut.get_o_numpy(full, name='__x__', sum=True)
plot = sb.distplot(total_umis_of_cells)
plot.set(xlabel='UMIs', ylabel='Density', yticks=[])
plot.axvline(x=properly_sampled_min_cell_total, color='darkgreen')
plot.axvline(x=properly_sampled_max_cell_total, color='crimson')
plt.savefig("./filename.png")

too_small_cells_count = sum(total_umis_of_cells < properly_sampled_min_cell_total)
too_large_cells_count = sum(total_umis_of_cells > properly_sampled_max_cell_total)

too_small_cells_percent = 100.0 * too_small_cells_count / len(total_umis_of_cells)
too_large_cells_percent = 100.0 * too_large_cells_count / len(total_umis_of_cells)

print(f"Will exclude %s (%.2f%%) cells with less than %s UMIs"
      % (too_small_cells_count,
         too_small_cells_percent,
         properly_sampled_min_cell_total))
print(f"Will exclude %s (%.2f%%) cells with more than %s UMIs"
      % (too_large_cells_count,
         too_large_cells_percent,
         properly_sampled_max_cell_total))

properly_sampled_max_excluded_genes_fraction = 0.5

excluded_genes_data = mc.tl.filter_data(full, var_masks=['~clean_gene'])[0]
excluded_umis_of_cells = mc.ut.get_o_numpy(excluded_genes_data, name='__x__', sum=True)
excluded_fraction_of_umis_of_cells = excluded_umis_of_cells / total_umis_of_cells

plot = sb.distplot(excluded_fraction_of_umis_of_cells)
plot.set(xlabel='Fraction of excluded gene UMIs', ylabel='Density', yticks=[])
plot.axvline(x=properly_sampled_max_excluded_genes_fraction, color='crimson')
plt.savefig("./filename2.png")

too_excluded_cells_count = sum(excluded_fraction_of_umis_of_cells > properly_sampled_max_excluded_genes_fraction)

too_excluded_cells_percent = 100.0 * too_excluded_cells_count / len(total_umis_of_cells)

print(f"Will exclude %s (%.2f%%) cells with more than %.2f%% excluded gene UMIs"
      % (too_excluded_cells_count,
         too_excluded_cells_percent,
         100.0 * properly_sampled_max_excluded_genes_fraction))

mc.pl.analyze_clean_cells(
    full,
    properly_sampled_min_cell_total=properly_sampled_min_cell_total,
    properly_sampled_max_cell_total=properly_sampled_max_cell_total,
    properly_sampled_max_excluded_genes_fraction=properly_sampled_max_excluded_genes_fraction)

mc.pl.pick_clean_cells(full)

clean = mc.pl.extract_clean_data(full)
#clean.write('./clean_before_anchor.h5ad')
#clean = ad.read_h5ad('./clean_before_anchor.h5ad')
print(clean)
suspect_gene_names = ['PCNA', 'MKI67',"TUBB", 'TOP2A', 'HIST1H1D','JUN', 'JUNB', 'HBA1','FOSB','ZFP36','FOS', 'JUN', 'HSP90AB1', 'HSPA1A','ISG15']
suspect_gene_patterns = ['5.*']
suspect_genes_mask = mc.tl.find_named_genes(clean, names=suspect_gene_names, patterns=suspect_gene_patterns)
print(suspect_genes_mask)
suspect_gene_names = sorted(clean.var_names[suspect_genes_mask])

mc.pl.relate_genes(clean,max_sampled_cells = 150000,random_seed=randomseed)

module_of_genes = clean.var['related_genes_module']
suspect_gene_modules = np.unique(module_of_genes[suspect_genes_mask])
suspect_gene_modules = suspect_gene_modules[suspect_gene_modules >= 0]
print(suspect_gene_modules)

similarity_of_genes = mc.ut.get_vv_frame(clean, 'related_genes_similarity')

for gene_module in suspect_gene_modules:
    module_genes_mask = module_of_genes == gene_module
    similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]
    similarity_of_module.index = \
    similarity_of_module.columns = [
        '(*) ' + name if name in suspect_gene_names else name
        for name in similarity_of_module.index
    ]
    similarity_of_module.to_csv("./afilename_"+str(gene_module)+"_2.csv")
    ax = plt.axes()
    sb.heatmap(similarity_of_module, vmin=0, vmax=1, ax=ax, cmap="YlGnBu")
    ax.set_title(f'Gene Module {gene_module}')
    plt.savefig("./afilename_"+str(gene_module)+"_2.png")


forbidden_genes_mask = suspect_genes_mask
for gene_module in suspect_gene_modules:
    module_genes_mask = module_of_genes == gene_module
    forbidden_genes_mask |= module_genes_mask
forbidden_gene_names = sorted(clean.var_names[forbidden_genes_mask])
fb_gene = pd.read_csv('./fb_gene.csv')
fb_gene = fb_gene['fb_gene'].to_list()

forbidden_gene_names = forbidden_gene_names+fb_gene
forbidden_gene_names = list(set(forbidden_gene_names))

print(len(forbidden_gene_names))
print(' '.join(forbidden_gene_names))

print(clean)
#clean.write('the_final_clean.h5ad')
clean.var['full_gene_index'].to_csv("./CLEANfullgeneindex.csv")
clean.var['related_genes_module'].to_csv("./CLEANrelatedgenemodule.csv")
clean.var['clean_gene'].to_csv('./CLEANclean_gene.csv')

max_parallel_piles = mc.pl.guess_max_parallel_piles(clean)
print(max_parallel_piles)
mc.pl.set_max_parallel_piles(max_parallel_piles)

mc.pl.divide_and_conquer_pipeline(clean,forbidden_gene_names=forbidden_gene_names, target_metacell_size=200000,random_seed=randomseed)

metacells = mc.pl.collect_metacells(clean, name='test_79')

mc.pl.compute_umap_by_features(metacells, max_top_feature_genes=2000,
                               min_dist=2.0, random_seed=randomseed)
umap_x = mc.ut.get_o_numpy(metacells, 'umap_x')
umap_y = mc.ut.get_o_numpy(metacells, 'umap_y')

#sb.scatterplot(x=umap_x, y=umap_y)
#plt.savefig("./umap.png",plot)

clean.write('the_clean_cells_20w_20wv2.h5ad')
metacells.write('the_clean_test_20w_20w_metacells_rightv2.h5ad')
#del metacells.uns['__name__']
#metacells.write('the_clean_test_for_seurat.h5ad')

name = metacells.uns['__name__']
del metacells.uns['__name__']
metacells.write('for_seurat.h5ad')
metacells.uns['__name__'] = name
outliers = mc.pl.compute_for_mcview(adata=clean, gdata=metacells, random_seed=123456, compute_var_var_similarity=dict(top=50, bottom=50))
outliers.write('outliers.h5ad')
clean.write('cells.h5ad')
metacells.write('metacells.h5ad')

