import pandas as pd
import numpy as np
import scipy.signal as sig
from typing import Tuple

def construct_antidiagonal_sum_vec(ld: np.ndarray) -> np.ndarray:
    J = ld.shape[0]
    v = np.zeros(2*J-1)
    for k in range(2*J-1):
        indices_1 = np.arange(k+1)
        indices_2 = indices_1[::-1]
        valid = (indices_1>=0) & (indices_1<J) & (indices_2>=0) & (indices_2<J)
        indices_1, indices_2 = indices_1[valid], indices_2[valid]
        v[k] = np.sum(ld[indices_1, indices_2])
    return v

def apply_filter(np_init_array: np.ndarray, width: int, n_pt=5, min_width=1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    a=sig.get_window('hann',2*width+1)
    ga = sig.convolve(np_init_array, a/a.sum(), mode='valid')
    ga = ga.astype(float)
    # find peaks
    peaks = sig.find_peaks(-ga, distance=min_width)
    minima_a = peaks[0]
    while len(minima_a)<(n_pt+2):
        # decrease the width
        width = width//2
        a=sig.get_window('hann',2*width+1)
        ga = sig.convolve(np_init_array, a/a.sum(), mode='valid')
        ga = ga.astype(float)
        peaks = sig.find_peaks(-ga, distance=min_width)
        minima_a = peaks[0]
        print(width, len(minima_a))
    minima_a_vals = [ga[i] for i in minima_a]
    minima_sorted_idx = np.argsort(minima_a_vals)

    cnt = 0
    new_minima_a = list()
    for idx in minima_sorted_idx:
        ii = minima_a[idx]
        if np.all([abs(ii-i)>min_width for i in new_minima_a]) and ii>min_width and ii<(len(np_init_array)-min_width):
            new_minima_a.append(ii)
            cnt +=1 
        if cnt==n_pt:
            break
    minima_a = np.sort(new_minima_a) 
    minima_a_vals = np.array([ga[i] for i in minima_a])
    
    return ga, minima_a, minima_a_vals, width

def local_search(ld: np.ndarray, minima_ind: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    J = ld.shape[0]
    
    # local search: percompute "outer sum"
    mid_pt = J//2 # expand from the middle
    mid_os = np.sum(ld[:mid_pt,(mid_pt+1):])
    os_pts = np.zeros(J)
    os_pts[mid_pt] = mid_os
    lower_os_running = mid_os
    for pt in np.arange(mid_pt-1, 0, -1):
        lower_os_running = lower_os_running-np.sum(ld[pt,(pt+2):])+np.sum(ld[:pt,pt+1])
        os_pts[pt] = lower_os_running
    upper_os_running = mid_os
    for pt in np.arange(mid_pt+1, J, 1):
        upper_os_running = upper_os_running-np.sum(ld[0:(pt-1),pt])+np.sum(ld[pt-1,(pt+1):])
        os_pts[pt] = upper_os_running
    
    # local search
    initial_brkpts = (minima_ind+1)//2
    initial_block_sizes = np.diff(initial_brkpts)
    initial_block_sizes = np.insert(initial_block_sizes,0,initial_brkpts[0])
    initial_block_sizes = np.insert(initial_block_sizes,len(initial_block_sizes),J-initial_brkpts[-1]+1)
    print("Before: number of breakpoints {}; min block size {}; max block size {}".format(len(initial_brkpts),initial_block_sizes.min(), initial_block_sizes.max()))
    
    # upper and lower bounds for the search space
    brkpt_idx = 0
    new_brkpts = list()
    for brkpt_idx in range(len(initial_brkpts)):
        brkpt = initial_brkpts[brkpt_idx]
        prv_pt = 0 if brkpt_idx==0 else initial_brkpts[brkpt_idx-1]
        nxt_pt = J-1 if brkpt_idx==(len(initial_brkpts)-1) else initial_brkpts[brkpt_idx+1]
        lb, ub = brkpt-(brkpt-prv_pt)//4, brkpt+(nxt_pt-brkpt)//4 # just 1/4
        # search within the local space
        outer_s = list()
        for pt in range(lb, ub):
            outer_s.append(os_pts[pt])
        min_s_idx = np.argmin(outer_s)
        opt_pt = np.arange(lb,ub)[min_s_idx]
        new_brkpts.append(opt_pt)
        
    new_brkpts = np.array(new_brkpts)
    new_block_sizes = np.diff(new_brkpts)
    new_block_sizes = np.insert(new_block_sizes,0,new_brkpts[0])
    new_block_sizes = np.insert(new_block_sizes,len(new_block_sizes),J-new_brkpts[-1]+1)
    print("After: number of new breakpoints {}; min block size {}; max block size {}".format(len(new_brkpts),new_block_sizes.min(), new_block_sizes.max()))

    return new_brkpts, new_block_sizes


def get_chr_genes(genes_chrom: pd.DataFrame, all_chr_pos: np.ndarray, buffer=0, pos_start_col='txStart', pos_end_col='txEnd') -> pd.DataFrame:
    genes_chrom['nsnps'] = 0
    for i_gene in range(len(genes_chrom)):
        gene_region = genes_chrom.iloc[i_gene][[pos_start_col,pos_end_col]].values
        snps_in_range = np.where((all_chr_pos>=(gene_region[0]-buffer)) & (all_chr_pos<=(gene_region[1]+buffer)))[0]
        idx = genes_chrom.iloc[i_gene].name
        if len(snps_in_range)>0:
            genes_chrom.loc[idx, 'nsnps'] = len(snps_in_range)
            genes_chrom.loc[idx,'start_idx_chr'] = snps_in_range[0]
            genes_chrom.loc[idx,'end_idx_chr'] = snps_in_range[-1]
    # only keep genes with >=1 SNPs
    genes_chrom = genes_chrom[genes_chrom['nsnps']>0].reset_index(drop=True)
    genes_chrom[['nsnps','start_idx_chr','end_idx_chr']] = genes_chrom[['nsnps','start_idx_chr','end_idx_chr']].astype(int)

    return genes_chrom

def get_chr_genes_from_rtable(gene_list: pd.DataFrame, bim, chrom: int, buffer: int = 0):
    # get non-overlapping genes
    upper, lower = buffer,buffer 
    gene_list = gene_list[gene_list['V1']==chrom].reset_index(drop=True)
    all_genes = gene_list['V4']
    gene_chr = gene_list['V1']
    gene_start = gene_list['V2']-lower
    gene_end = gene_list['V3']+upper
    
    genes_chr = pd.DataFrame(columns=['CHR','GENE','START','END','N_SNPS','start_idx_chr', 'end_idx_chr'])
    all_pos = bim["pos"].values
    for i_gene in range(len(gene_list)):
        snps_in_range = np.where((all_pos>=gene_start[i_gene]) & (all_pos<=gene_end[i_gene]))[0]
        nsnps = len(snps_in_range)
        if nsnps>1:
            start_idx_chr = snps_in_range[0]
            end_idx_chr = snps_in_range[-1]
            genes_chr.loc[len(genes_chr.index)] = [chrom, all_genes[i_gene], gene_start[i_gene], gene_end[i_gene], nsnps, start_idx_chr, end_idx_chr] 

    return genes_chr