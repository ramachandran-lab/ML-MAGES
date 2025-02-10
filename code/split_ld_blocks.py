import pandas as pd
import numpy as np
import os
import sys
import scipy.signal as sig
import argparse
import time

def apply_filter(orig_array, width, n_pt = 1, min_width = 1000):

    a=sig.get_window('hann',2*width+1)
    ga = sig.convolve(orig_array, a/a.sum(), mode='valid')

    peaks = sig.find_peaks(-ga, distance=min_width)
    minima_a = peaks[0]
    while len(minima_a)<(n_pt+1):
        # decrease the width
        width = width//2
        a=sig.get_window('hann',2*width+1)
        ga = sig.convolve(orig_array, a/a.sum(), mode='valid')
        peaks = sig.find_peaks(-ga, distance=min_width)
        minima_a = peaks[0]
    adjusted_width = width
    minima_a_vals = [ga[i] for i in minima_a]
    minima_sorted_idx = np.argsort(minima_a_vals)

    cnt = 0
    new_minima_a = list()
    for idx in minima_sorted_idx:
        ii = minima_a[idx]
        if np.all([abs(ii-i)>min_width for i in new_minima_a]) and ii>min_width and ii<(len(orig_array)-min_width):
            new_minima_a.append(ii)
            cnt +=1 
        if cnt==n_pt:
            break
    filtered_indices = np.sort(new_minima_a) 
    minima_a_vals = [ga[i] for i in minima_a]
    
    return filtered_indices, adjusted_width

def get_breakpoints(chr_ld, avg_block_size, filter_width=200):
    # initial filtering
    J = chr_ld.shape[0]
    v = np.zeros(2*J-1)
    for k in range(2*J-1):
        indices_1 = np.arange(k+1)
        indices_2 = indices_1[::-1]
        valid = (indices_1>=0) & (indices_1<J) & (indices_2>=0) & (indices_2<J)
        indices_1, indices_2 = indices_1[valid], indices_2[valid]
        # indices_1 = indices_1
        v[k] = np.sum(chr_ld[indices_1, indices_2])
    n_bpoints = int(np.floor(J/avg_block_size))+1
    filtered_indices, adjusted_width = apply_filter(v, width=filter_width, n_pt = n_bpoints, min_width = avg_block_size)

    brkpts = (filtered_indices+1)//2
    block_sizes = np.diff(brkpts)
    block_sizes = np.insert(block_sizes,0,brkpts[0])
    block_sizes = np.insert(block_sizes,len(block_sizes),J-brkpts[-1]+1)

    return brkpts, block_sizes

def adjust_breakpoints_local(chr_ld, brkpts):
    J = chr_ld.shape[0]
    # initialize local search 
    mid_pt = J//2 # expand from the middle
    mid_os = np.sum(chr_ld[:mid_pt,(mid_pt+1):])
    os_pts = np.zeros(J)
    os_pts[mid_pt] = mid_os
    lower_os_running = mid_os
    for pt in np.arange(mid_pt-1, 0, -1):
        lower_os_running = lower_os_running-np.sum(chr_ld[pt,(pt+2):])+np.sum(chr_ld[:pt,pt+1])
        os_pts[pt] = lower_os_running
    upper_os_running = mid_os
    for pt in np.arange(mid_pt+1, J, 1):
        upper_os_running = upper_os_running-np.sum(chr_ld[0:(pt-1),pt])+np.sum(chr_ld[pt-1,(pt+1):])
        os_pts[pt] = upper_os_running

    # conduct local search
    brkpt_idx = 0
    new_brkpts = list()
    print("Conducting local search...")
    for brkpt_idx in range(len(brkpts)):
        brkpt = brkpts[brkpt_idx]
        prv_pt = 0 if brkpt_idx==0 else brkpts[brkpt_idx-1]
        nxt_pt = J-1 if brkpt_idx==(len(brkpts)-1) else brkpts[brkpt_idx+1]
        lb, ub = brkpt-(brkpt-prv_pt)//4, brkpt+(nxt_pt-brkpt)//4 # just 1/4
        # search within the local space
        outer_s = list()
        for pt in range(lb, ub):
            outer_s.append(os_pts[pt])
        min_s_idx = np.argmin(outer_s)
        opt_pt = np.arange(lb,ub)[min_s_idx]
        print("Block {}. old break index {}, range[{},{}] -> new break index {}".format(brkpt_idx, brkpt, lb, ub, opt_pt))
        new_brkpts.append(opt_pt)
    
    # get indices of breakpoints, including both start and end
    new_brkpts = np.array(new_brkpts)
    new_block_sizes = np.diff(new_brkpts)
    new_block_sizes = np.insert(new_block_sizes,0,new_brkpts[0])
    new_block_sizes = np.insert(new_block_sizes,len(new_block_sizes),J-new_brkpts[-1]+1)

    return new_brkpts, new_block_sizes


def main(args):
    # take in command line arguments
    print("-----Required Arguments: ")
    full_ld_file = args.full_ld_file
    print("full_ld_file:", full_ld_file)
    output_file = args.output_file
    print("output_file:", output_file)

    print("-----Optional Arguments: ")
    avg_block_size = args.avg_block_size
    print("avg_block_size (default: 1000):", avg_block_size)

    print("=====//ML-MAGES// Helper Function: Split LD into Blocks=====")

    start_time = time.time()
    # load LD
    chr_ld = np.loadtxt(full_ld_file)
    print("LD size: {}x{}".format(chr_ld.shape[0],chr_ld.shape[1]))

    # split LD
    initial_brkpts, initial_block_sizes = get_breakpoints(chr_ld, avg_block_size, filter_width=200)
    print("{} breakpoints ({} blocks); min block size {}; max block size {}".format(len(initial_brkpts),len(initial_block_sizes),initial_block_sizes.min(), initial_block_sizes.max()))
    new_brkpts, new_block_sizes = adjust_breakpoints_local(chr_ld, initial_brkpts)
    print("Updated after local search:\n{} breakpoints ({} blocks); min block size {}; max block size {}".format(len(new_brkpts),len(new_block_sizes),new_block_sizes.min(), new_block_sizes.max()))

    # save breakpoints
    np.savetxt(output_file, new_brkpts, delimiter=',', fmt='%d') 

    end_time = time.time()
    print("Splitting LD takes {:.2f} seconds".format(end_time - start_time))

    print("============DONE============")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide the required arguments')

    # Required argument
    parser.add_argument('--full_ld_file', type=str, required=True, help='path to the full LD file (in a comma-separated matrix)')
    parser.add_argument('--output_file', type=str, required=True, help='path to the output file to store indices of the breakpoints to split the LD')

    # Optional arguments
    parser.add_argument('--avg_block_size', type=int, required=False, default=1000, help='approximated average LD block size after splitting')


    args = parser.parse_args()
    main(args)