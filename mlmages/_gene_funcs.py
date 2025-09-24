import numpy as np
import scipy.stats as stats

CHISCORE_SUCCESS = True
try:
    from chiscore import liu_sf 
except:
    print("Unable to import chiscore. Please install chiscore package separately.")
    print("Using alternative method for linear-combination-of-chi-square test for gene-level test.")
    CHISCORE_SUCCESS = False
    pass

# from numpy import asarray, maximum, sqrt, sum
# from scipy.stats import ncx2

"""
Adapted from the original code of liu_sf() from the Python package chiscore (https://github.com/limix/chiscore).
The installation of the package may fail at some circumstances (most likely due to failure in installing the package chi2comb).
When this happens, the following function will be used alternatively for testing the linear combination of chi-squared variables.
"""

def test_chi2comb_alt(t, lambs, dofs=1, deltas=0):
    """
    Liu approximation [1] to linear combination of noncentral chi-squared variables.

    When ``kurtosis=True``, the approximation is done by matching the kurtosis, rather
    than the skewness, as derived in [2].

    Parameters
    ----------
    t : array_like
        Points at which the survival function will be applied, Pr(X>t).
    lambs : array_like
        Weights.
    dofs : array_like
        Degrees of freedom.
    deltas : array_like
        Noncentrality parameters.

    Returns
    -------
    q : float, ndarray
        Approximated survival function applied t: Pr(X>t).

    References
    ----------
    [1] Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation to the
        distribution of non-negative definite quadratic forms in non-central normal
        variables. Computational Statistics & Data Analysis, 53(4), 853-856.
    """
    t = np.asarray(t, float)
    lambs = np.asarray(lambs, float)
    dofs = np.asarray(dofs, float)
    deltas = np.asarray(deltas, float)

    lambs = {i: lambs ** i for i in range(1, 5)}

    c = {i: np.sum(lambs[i] * dofs) + i * np.sum(lambs[i] * deltas) for i in range(1, 5)}

    s1 = c[3] / np.sqrt(c[2]) ** 3
    s2 = c[4] / c[2] ** 2

    s12 = s1 ** 2
    if s12 > s2:
        a = 1 / (s1 - np.sqrt(s12 - s2))
        delta_x = s1 * a ** 3 - a ** 2
        dof_x = a ** 2 - 2 * delta_x
    else:
        delta_x = 0
        a = 1 / s1
        dof_x = 1 / s12

    mu_q = c[1]
    sigma_q = np.sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = np.sqrt(2 * (dof_x + 2 * delta_x))

    t_star = (t - mu_q) / sigma_q
    tfinal = t_star * sigma_x + mu_x

    q = stats.ncx2.sf(tfinal, dof_x, np.maximum(delta_x, 1e-9))

    return q


def test_gene(genes_start_idx_chr:np.ndarray, genes_nsnps:np.ndarray, epsilon_effect: float, betas_regularized:np.ndarray, ld:np.ndarray) -> dict:#, snp_cls=[]):

    # gene-level test    
    prior_weight = np.ones(len(betas_regularized))
    
    p_vals = list()
    test_stats = list()
    test_stat_vars = list()
    # snp_in_gene_cls = list()
    
    for i in range(len(genes_nsnps)):

        nsnps = genes_nsnps[i]
        gene_snps = np.arange(genes_start_idx_chr[i],genes_start_idx_chr[i]+nsnps)
        ld_g = ld[gene_snps,:][:,gene_snps]
        weight_matrix = np.diag(prior_weight[gene_snps])
        
        # compute eigenvalues
        mat = epsilon_effect * ld_g @ weight_matrix
        e_val, e_vec = np.linalg.eig(mat)
        
        #compute test statistics
        betas_g = betas_regularized[gene_snps]
        test_stat = betas_g.T @ weight_matrix @ betas_g
        #compute test statistics variance
        t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()
        
        if CHISCORE_SUCCESS:
            (p_val_g, _, _, _) = liu_sf(test_stat, e_val, 1, 0)
            # p_val_g = davies_pvalue(test_stat, mat)
        else:
            p_val_g = test_chi2comb_alt(test_stat, e_val)
        
        if p_val_g <= 0.0:
            p_val_g = 1e-20
        
        p_vals.append(p_val_g)
        test_stats.append(test_stat)
        test_stat_vars.append(t_var)
        # if len(snp_cls)>0:
        #     snp_in_gene_cls.append(snp_cls[gene_snps])
            

    return {'P':p_vals,'STAT':test_stats,'VAR':test_stat_vars}


def check_specific_or_shared_bivar(ang: float, eig: np.ndarray, angle_tol: float=30, axes_ratio_specific: float = 5, axes_ratio_shared: float = 1.5) -> tuple[list, bool]:
    is_specific = [False,False]
    is_shared = False
    angle_degrees = ang * (180 / np.pi)
    # Normalize the angle to be within the 0-360 degree range
    normalized_angle = angle_degrees % 360
    if eig[0]/eig[1]>axes_ratio_specific:
        # Check if close to the x-axis (0, 180, 360 degrees)
        if (normalized_angle < angle_tol) or (abs(normalized_angle - 180) < angle_tol) or (normalized_angle > 330):
            is_specific = [True,False]
        # Check if close to the y-axis (90, 270 degrees)
        elif (abs(normalized_angle - 90) < angle_tol) or (abs(normalized_angle - 270) < angle_tol):
            is_specific = [False,True]
        else:
            is_specific = [False,False]
            is_shared = True
    if eig[0]/eig[1]<axes_ratio_shared:
        is_shared = True

    return is_specific, is_shared


def check_specific_trivar(sigma_eiginfo, rad_thre=np.pi/12, eigval_times_thre=5):
    """
    Classify hyper-ellipses as specific to A, B, C, AB, AC, BC, or ABC
    based on eigenvalue ratios and angular alignment.

    Parameters
    ----------
    sigma_eiginfo : list of (eigvals, eigvecs, angs)
        Output from get_eiginfo, where angs[i][j] is the angle of eigenvector i to axis j
    rad_thre : float
        Angular threshold (in radians) for alignment with an axis
    eigval_times_thre : float
        Threshold ratio between largest eigenvalue and others

    Returns
    -------
    cls_specific : list of list[bool]
        For each Sigma entry, a vector:
        [is_A, is_B, is_C, is_AB, is_AC, is_BC, is_ABC]
    """
    cls_specific = []

    for eigvals, eigvecs, angs in sigma_eiginfo:
        # --- Single-axis check (A,B,C) ---
        is_specific = [False, False, False]

        if np.all([eigvals[0] / e > eigval_times_thre for e in eigvals[1:]]):
            for i_axis in range(3):  # x=A, y=B, z=C
                ang_to_axis = angs[0][i_axis]
                if ang_to_axis < rad_thre or ang_to_axis > (2 * np.pi - rad_thre):
                    is_specific[i_axis] = True

        # --- Project eigenvalues onto xyz axes ---
        proj_on_xyz = []
        for i_axis in range(3):
            proj_on_xyz.append([])
            for i_a in range(len(eigvals)):
                ang_to_axis = angs[i_a][i_axis]
                proj_on_xyz[i_axis].append(np.abs(eigvals[i_a] * np.cos(ang_to_axis)))
        proj_on_xyz = np.array(proj_on_xyz)  # shape: (3 axes, n eigvecs)

        # max projection along each axis
        max_on_xyz = np.max(proj_on_xyz, axis=1)

        # --- Pairwise checks (AB, AC, BC) ---
        is_AB = max_on_xyz[0]/max_on_xyz[2] > eigval_times_thre and max_on_xyz[1]/max_on_xyz[2] > eigval_times_thre
        is_AC = max_on_xyz[0]/max_on_xyz[1] > eigval_times_thre and max_on_xyz[2]/max_on_xyz[1] > eigval_times_thre
        is_BC = max_on_xyz[1]/max_on_xyz[0] > eigval_times_thre and max_on_xyz[2]/max_on_xyz[0] > eigval_times_thre

        # --- Triple check (ABC) ---
        is_ABC = True
        for i in range(3):
            for j in range(i+1, 3):
                ratio = max_on_xyz[i] / max_on_xyz[j]
                if ratio > eigval_times_thre or 1/ratio > eigval_times_thre:
                    is_ABC = False
        if np.any(is_specific):  # single-axis overrides "shared"
            is_ABC = False

        cls_specific.append(is_specific + [is_AB, is_AC, is_BC, is_ABC])

    return cls_specific