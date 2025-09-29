import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D


def plot_effects(beta_list: list[np.ndarray], chr_is_odd=None, chr_switch_idx=None, 
                 sharey: bool=True, marker_s: int=1):
    assert len(beta_list)>0, "At least one set of effects is required!"
    nsnp = len(beta_list[0])
    if chr_is_odd is None:
        chr_is_odd = np.zeros(nsnp, dtype=bool)

    fig, axes = plt.subplots(len(beta_list),1,figsize=(10,1.5*len(beta_list)), sharex=True, sharey=sharey, dpi=200)
    for i, ax in enumerate(axes):
        
        if chr_switch_idx is None:
            chr_switch_idx = [0, nsnp]
        ax.scatter(np.arange(nsnp)[chr_is_odd], beta_list[i][chr_is_odd], 
                   s=marker_s, alpha=0.7, color='#999999',linewidth=0,)
        ax.scatter(np.arange(nsnp)[~chr_is_odd], beta_list[i][~chr_is_odd], 
                   s=marker_s, alpha=0.7, color='#404040',linewidth=0,)

    ax.set_xlim(0,nsnp)
    ax.set_xlabel("CHR")
    if chr_switch_idx is not None:
        for sw in chr_switch_idx:
            ax.axvline(x=sw, color='lightgray', ls='--', lw=0.5)
        # ax.set_xticks((chr_switch_idx[1:]+chr_switch_idx[:-1])/2)
        # ax.set_xticklabels(np.arange(1,len(chr_switch_idx)))
        
    fig.tight_layout()
    return fig


def plot_pvals(pvals: np.ndarray, names=[], chr_is_odd=None, chr_switch_idx=None, 
               sig_line: float=0.05, ylim=None, marker_s: int=1):
    nvals = len(pvals)
    if len(names)>0:
        assert len(names)==nvals, "Length of names ({}) does not match that of pvals ({})".format(len(names), nvals)
    if chr_is_odd is None:
        chr_is_odd = np.zeros(nvals, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.scatter(np.arange(nvals)[chr_is_odd], -np.log10(pvals[chr_is_odd]), 
               s=marker_s, alpha=0.7, color='#999999', linewidth=0)
    ax.scatter(np.arange(nvals)[~chr_is_odd], -np.log10(pvals[~chr_is_odd]), 
               s=marker_s, alpha=0.7, color='#404040', linewidth=0)
    ax.axhline(y=-np.log10(sig_line), color='red', linestyle='--', linewidth=0.5)
    ax.text(0, -np.log10(sig_line)+0.1, "p={:.1e}".format(sig_line), color='red', fontsize=8)
    if len(names)>0:
        if np.sum(pvals<sig_line)<10:
            for i, name in enumerate(names):
                if pvals[i]<sig_line:
                    ax.text(i, -np.log10(pvals[i]), name, color='black', fontsize=8)
        else:
            idx_pval = np.argsort(pvals)
            for rank in range(10):
                i = idx_pval[rank]
                ax.text(i, -np.log10(pvals[i]), names[i], color='black', fontsize=8)
    
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_ylabel("-log10(p)")
    ax.set_title("Negative log (base-10) of p values", loc='left', fontsize=10)
    ax.set_xlim(0,nvals)
    if chr_switch_idx is not None:
        for sw in chr_switch_idx:
            ax.axvline(x=sw, color='lightgray', ls='--', lw=0.5)

    fig.tight_layout()
    return fig


def plot_data_cls(df, x, hue, cls_perc, palette=None, ax=None, **kwargs):
    assert x in df.columns
    assert hue in df.columns

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    pred_K = len(cls_perc)
    if palette is None or len(palette)<pred_K:
        palette = sns.color_palette("colorblind", pred_K)

    sns.kdeplot(data=df, x=x, hue=hue, palette=palette, ax=ax, fill=True, legend=True, **kwargs) 
    leg = ax.legend_ #.legendHandles
    labels = [t.get_text() for t in ax.legend_.texts]
    new_labels = ["{}: {:.1%}".format(int(ii)+1,cls_perc.loc[int(ii)]) for ii in labels]
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    # leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.01, 1.01))             
    leg.set_loc("upper left")       
    leg._legend_handle_length = 1.0
    leg._legend_handle_textpad = 0.2
    ax.set_ylim(ymin=-0.1)
    ax.set_xlabel("$\\beta$ labeled by components")
    ax.set_title("$\\beta$ labeled by clusters")
    
    return ax

def plot_inf_cls(pred_K: int, Sigma: list[np.ndarray], pi: np.ndarray, x_extreme: float, palette: list=[], ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    if palette is None or len(palette)<pred_K:
        palette = sns.color_palette("colorblind", pred_K)
    x = np.linspace(-x_extreme,x_extreme,200)
    for k in range(pred_K):
        y = sp.stats.norm.pdf(x, 0, np.sqrt(float(Sigma[k][0][0])))
        ax.plot(x,y,lw=1,label="{} ({:.1%})".format(k+1,pi[k]), color=palette[k])
        ax.set_xlim(-x_extreme,x_extreme)
    leg = ax.legend(title="Cluster ($\\pi_k$)", 
              fontsize=10, title_fontsize=10,
              bbox_to_anchor=(1.01, 1.01), loc='upper left', handlelength=1.0, handletextpad=0.2)
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    ax.set_ylim(ymin=-0.1)
    ax.set_title("Gaussian mixtures")

    return ax

def plot_data_cls_bivar(df, phenos, hue, cls_perc, xylim, palette=None, ax=None, **kwargs):
    assert len(phenos)==2
    for pheno in phenos:
        assert pheno in df.columns
    assert hue in df.columns

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    pred_K = len(cls_perc)
    if palette is None or len(palette)<pred_K:
        palette = sns.color_palette("colorblind", pred_K)

    sns.scatterplot(data=df,x=phenos[0], y=phenos[1], hue=hue, ax=ax, 
                    palette=palette, **kwargs)
    ax.legend(bbox_to_anchor=(1.01, 1.01), numpoints=1, markerscale=2,
              title="Cluster", loc='upper left', handlelength=1.0, handletextpad=0.2)
    leg = ax.legend_ #.legendHandles
    labels = [t.get_text() for t in ax.legend_.texts]
    new_labels = ["{}: {:.1%}".format(int(ii)+1,cls_perc.loc[int(ii)]) for ii in labels]
    ax.set_xlim(-xylim,xylim)
    ax.set_ylim(-xylim,xylim)
    ax.set_aspect('equal', 'box')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    ax.set_title("$\\beta$ labeled by clusters")
    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])
    
    return ax


def plot_inf_cls_bivar(phenos, pred_K, Sigma, pi, xylim, nstd=np.sqrt(5.991), palette=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    if palette is None or len(palette)<pred_K:
        palette = sns.color_palette("colorblind", pred_K)

    for i, covar in enumerate(Sigma):
    
        # Calculate the eigenvectors and eigenvalues
        eigenval, eigenvec = np.linalg.eig(covar)
        s = np.sqrt(eigenval)
        largest_s = max(s)
        smallest_s = min(s)
    
        # Get the index of the largest eigenvector
        largest_eigenvec_ind_c = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:,largest_eigenvec_ind_c]
        # Calculate the angle between the x-axis and the largest eigenvector
        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        # Shift it such that the angle is between 0 and 2pi
        if (angle < 0):
            angle = angle + 2*np.pi
    
        # Get the covariance ellipse
        # np.sqrt(9.210) for 99% CI # np.sqrt(5.991) for 95% CI
        ell = Ellipse(xy=(0, 0), width=largest_s*nstd*2, height=smallest_s*nstd*2,
                      angle=np.rad2deg(angle), 
                      facecolor='none', edgecolor=palette[i], 
                      label="{} ({:.1%})".format(int(i)+1,pi[i]))
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.9)
        ax.add_patch(ell)
    
    ax.set_xlim(-xylim,xylim)
    ax.set_ylim(-xylim,xylim)
    ax.set_aspect('equal', 'box')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_title("Gaussian mixtures")
    ax.legend(bbox_to_anchor=(1.01, 1.01), title="Cluster ($\\pi_k$)", loc='upper left',
              fontsize=10, title_fontsize=10, handlelength=1.0, handletextpad=0.2)
    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])

    return ax


def plot_data_cls_bivar_merged(df, phenos, hue, cls_perc, pred_K, cls_cutoff, xylim, merged_palette, markers, ax=None, **kwargs):
    assert len(phenos)==2
    for pheno in phenos:
        assert pheno in df.columns
    assert hue in df.columns
    assert len(merged_palette)>=cls_cutoff
    assert len(markers)>=cls_cutoff

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        
    df = df.sort_values(by=hue)
    sns.scatterplot(data=df,x=phenos[0], y=phenos[1],hue=hue, ax=ax,
                    style=hue, style_order = np.arange(cls_cutoff+1), 
                    palette = merged_palette, markers = markers, s=10, 
                    **kwargs) #
    handles, labels = ax.get_legend_handles_labels()
    new_labels = list()
    for ii in range(cls_cutoff):
        new_labels.append("{}: {:.1%}".format(int(ii)+1,cls_perc.loc[int(ii)]))
    rest_perc = cls_perc.loc[np.arange(cls_cutoff,pred_K)].sum()
    new_labels.append("{}: {:.1%}".format("rest",rest_perc))
    new_handles = list()
    for lh in handles[1:]: #[1:5]: 
        lh.set_alpha(1)
        new_handles.append(lh)
    
    ax.legend(new_handles, new_labels, 
              bbox_to_anchor=(1.01, 1.01), numpoints=1, markerscale=1.5,
              title="Cluster", loc='upper left', handlelength=1.0, handletextpad=0.2)
    ax.set_xlim(-xylim, xylim)
    ax.set_ylim(-xylim, xylim)
    ax.set_aspect('equal', 'box')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    
    ax.set_title("$\\beta$ labeled by clusters")
    
    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])

    return ax


def plot_inf_cls_bivar_merged(phenos, pred_K, Sigma, pi, cls_cutoff, xylim, merged_palette, nstd=np.sqrt(5.991), ax=None):
    
    assert len(merged_palette)>=cls_cutoff
    
    for i, covar in enumerate(Sigma[:cls_cutoff]):
    
        # Calculate the eigenvectors and eigenvalues
        eigenval, eigenvec = np.linalg.eig(covar)
        s = np.sqrt(eigenval)
        largest_s = max(s)
        smallest_s = min(s)
    
        # Get the index of the largest eigenvector
        largest_eigenvec_ind_c = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:,largest_eigenvec_ind_c]
        # Calculate the angle between the x-axis and the largest eigenvector
        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        # Shift it such that the angle is between 0 and 2pi
        if (angle < 0):
            angle = angle + 2*np.pi
    
        # Get the covariance ellipse
        #np.sqrt(9.210) for 99% CI and np.sqrt(5.991) for 95% CI
        ell = Ellipse(xy=(0, 0), width=largest_s*nstd*2, height=smallest_s*nstd*2,
                      angle=np.rad2deg(angle), 
                      facecolor='none', edgecolor=merged_palette[i], 
                      label="{} ({:.1%})".format(int(i)+1,pi[i]))
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.9)
        ax.add_patch(ell)
    
    ax.set_xlim(-xylim,xylim)
    ax.set_ylim(-xylim,xylim)
    ax.set_aspect('equal', 'box')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_title("Gaussian mixtures")
    ax.legend(bbox_to_anchor=(1.01, 1.01), title="Cluster ($\\pi_k$)", loc='upper left',
              fontsize=10, title_fontsize=10, handlelength=1.0, handletextpad=0.2)

    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])

    return ax


def plot_data_cls_trivar(df, phenos, hue, cls_perc, xylim, palette=None, ax=None, view=(20, 30), **kwargs):
    """
    3D scatter plot of 3 traits, labeled by clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing columns for phenos and hue.
    phenos : list[str]
        Names of 3 trait columns in df.
    hue : str
        Column with cluster labels.
    cls_perc : pandas.Series
        Cluster proportions indexed by cluster id.
    xylim : float
        Axis limit for all three dimensions.
    palette : list, optional
        List of colors for clusters.
    ax : matplotlib 3D axes, optional
        If None, creates a new figure/axes.
    kwargs : dict
        Passed to scatter plot.

    Returns
    -------
    ax : matplotlib 3D axes
    """
    assert len(phenos) == 3, "Need exactly 3 traits for 3D plot"
    for pheno in phenos:
        assert pheno in df.columns
    assert hue in df.columns

    if ax is None:
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

    pred_K = len(cls_perc)
    if palette is None or len(palette) < pred_K:
        palette = sns.color_palette("colorblind", pred_K)

    # Map cluster labels -> colors
    unique_labels = sorted(df[hue].unique())
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    ax.scatter(
        df[phenos[0]], df[phenos[1]], df[phenos[2]],
        c=[color_map[l] for l in df[hue]],
        s=10, alpha=0.7, **kwargs
    )

    # Set axes
    ax.set_xlim(-xylim, xylim)
    ax.set_ylim(-xylim, xylim)
    ax.set_zlim(-xylim, xylim)
    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])
    ax.set_zlabel(phenos[2])
    ax.set_title("$\\beta$ labeled by clusters")

    # Apply 3D view angle
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    # Custom legend with percentages
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[lab], markersize=6)
               for lab in unique_labels]
    labels = ["{}: {:.1%}".format(int(lab)+1, cls_perc.loc[int(lab)]) for lab in unique_labels]
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
              title="Cluster", markerscale=2)

    return ax


def plot_inf_cls_trivar(phenos, pred_K, Sigma, pi, xylim, nstd=np.sqrt(5.991), view=(20, 30), palette=None, ax=None):
    """
    Plot 3D Gaussian mixture clusters as ellipsoids.

    Parameters
    ----------
    phenos : list[str]
        Names of the 3 traits.
    pred_K : int
        Number of clusters.
    Sigma : list[np.ndarray]
        List of 3x3 covariance matrices for each cluster.
    pi : list[float]
        Cluster weights (for legend labeling).
    xylim : float
        Axis limit.
    palette : list[str], optional
        Colors for clusters.
    ax : matplotlib 3D axes, optional
        If None, a new figure/axes is created.
    nstd : float
        Number of standard deviations for ellipsoid size (default 3~99% CI).
    """
    if ax is None:
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

    if palette is None or len(palette) < pred_K:
        palette = sns.color_palette("colorblind", pred_K)

    # Generate a sphere mesh
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    for i, covar in enumerate(Sigma):
        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(covar)
        radii = nstd * np.sqrt(eigenvals)

        # Transform sphere into ellipsoid
        ellipsoid = eigenvecs @ np.diag(radii)
        x = ellipsoid[0, 0]*x_sphere + ellipsoid[0, 1]*y_sphere + ellipsoid[0, 2]*z_sphere
        y = ellipsoid[1, 0]*x_sphere + ellipsoid[1, 1]*y_sphere + ellipsoid[1, 2]*z_sphere
        z = ellipsoid[2, 0]*x_sphere + ellipsoid[2, 1]*y_sphere + ellipsoid[2, 2]*z_sphere

        ax.plot_wireframe(x, y, z, color=palette[i], alpha=0.4, linewidth=1)

    ax.set_xlim(-xylim, xylim)
    ax.set_ylim(-xylim, xylim)
    ax.set_zlim(-xylim, xylim)
    ax.set_xlabel(phenos[0])
    ax.set_ylabel(phenos[1])
    ax.set_zlabel(phenos[2])
    ax.set_title("Gaussian mixture clusters")

    # Apply 3D view angle
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    # Legend
    handles = [plt.Line2D([0], [0], color=palette[i], lw=2) for i in range(pred_K)]
    labels = ["{}: {:.1%}".format(i+1, pi[i]) for i in range(pred_K)]
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
              title="Cluster ($\\pi_k$)", fontsize=10, title_fontsize=10)

    return ax