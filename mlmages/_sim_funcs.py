import numpy as np
import pandas as pd
import scipy as sp
import itertools

def simulate_effects(maf: np.ndarray, ld_scores: np.ndarray, nsnp: int,
                     p_causal: float, s: float, h2: float, w: float) -> np.ndarray:
    """
    Simulate causal effect sizes beta for each SNP.  
    """
    # simulate effects
    # Compute heterozygosity: Var(genotype) ≈ 2*p*(1-p) under HWE
    het = 2 * maf * (1 - maf) + 1e-8   # add small value to avoid zeros
    # Compute raw per-SNP variance (unnormalized) from MAF and LD:
    # Effect variance ~ (het^s) * (ld_score^w).  Using power s on heterozygosity.
    weights = (het ** s) * (ld_scores**w)
    # Select causal SNP indices
    n_causal = max([2,int(np.round(p_causal * nsnp))])
    causal_idx = np.random.choice(nsnp, size=n_causal, replace=False)
    # Draw raw betas for causals
    beta = np.zeros(nsnp)
    beta[causal_idx] = np.random.normal(loc=0, scale=np.sqrt(weights[causal_idx]), 
                                        size=n_causal)
    # Scale betas so that total genetic variance = h2
    # Compute current genetic variance: beta^2 * Var(genotype) = beta^2 * het
    current_var = np.sum((beta**2) * het)
    scale = np.sqrt(h2 / current_var)
    beta *= scale
    return beta

def simulate_phenotype(G: np.ndarray, beta: np.ndarray, h2: float) -> np.ndarray:
    """
    Given genotype matrix G (nind, nsnp) and effect sizes beta (length nsnp),
    simulate phenotype vector Y of length nind with SNP-heritability h2.
    """
    # Compute genetic values and their variance
    Gval = np.nansum(G * beta, axis=1)  # dot product (handling any NaNs)
    var_g = np.nanvar(Gval)
    # Generate noise with variance to achieve total variance=1
    # So that Var(Gval)/(Var(Gval)+Var(eps)) = h2.
    var_e = var_g * (1 - h2) / h2
    eps = np.random.normal(0, np.sqrt(var_e), size=Gval.shape)
    Y = Gval + eps
    return Y


def compute_gwas_summary(G: np.ndarray, y: np.ndarray):
    """
    Compute GWAS summary stats (beta, SE, p-value) for each SNP.
    
    Inputs:
      G: array of shape (n_individuals, n_SNPs), columns mean 0 var 1.
      y: array of shape (n_individuals,), continuous phenotype.
    
    Returns:
      DataFrame with columns ['beta_hat', 'SE', 'p_value'] of length n_SNPs.
    """
    n, m = G.shape #(nind, nsnp)
    
    # Center phenotype (remove mean) to match no-intercept model
    y_centered = y - np.mean(y)
    
    # Compute sum of squares for each SNP (denominator for beta)
    SSx = np.sum(G * G, axis=0)            # shape (m,)
    # Compute dot product of each SNP with y (numerator for beta)
    Sxy = G.T.dot(y_centered)             # shape (m,)
    
    # Effect sizes (OLS slope for each SNP)
    beta_hat = Sxy / SSx
    
    # Compute regression sum of squares (SSR) = beta^2 * SSx
    SSR = beta_hat**2 * SSx
    # Total sum of squares of y
    SSy = np.sum(y_centered * y_centered)
    # Residual sum of squares for each SNP
    SSE = SSy - SSR
    
    # Degrees of freedom for slope (n - 2)
    df = n - 2
    # Estimate of error variance for each SNP
    sigma2 = SSE / df
    
    # Standard error of beta for each SNP
    se = np.sqrt(sigma2 / SSx)
    
    # Compute t-statistics and two-sided p-values
    t_stats = beta_hat / se
    p_values = 2 * sp.stats.t.sf(np.abs(t_stats), df)
    
    # Return results in a DataFrame
    return pd.DataFrame({
        'beta_hat': beta_hat,
        'se': se,
        'p_value': p_values
    })


# simulate valid between-trait variance-covariance matrices
def simulate_covar(n_trait: int, sig_thre = 0.01, coeff_thre = 0.7, max_iter = 500) -> np.ndarray:
    i_iter = 0
    while i_iter<max_iter:
        C = np.random.rand(n_trait,n_trait)-0.5  
        Sigma = C @ C.T  # Symmetric, positive semi-definite
        X = np.random.multivariate_normal(np.zeros(n_trait), Sigma, size=500)
        is_correlated = True
        for pair in list(itertools.combinations(np.arange(n_trait),2)):
            pcoeff, pcsig = sp.stats.pearsonr(X[:,pair[0]],X[:,pair[1]])
            is_correlated &= (pcsig<sig_thre and np.abs(pcoeff)>coeff_thre)
        if is_correlated:
            d = np.sqrt(np.diag(Sigma))
            Sigma = Sigma / np.outer(d, d)
            return Sigma
        i_iter += 1
    print("Reaching maximum #iterations. Output identity as variance-covariance matrix.")
    return np.identity(n_trait)


def simulate_covar_base(n_trait: int, n_sim_cov: int=50, min_corr: float=0.6) -> list[np.ndarray]:
    """
    Simulate a variance-covariance matrix for n_trait traits with fixed correlation rho.
    """
    lower_limit_for_rho = -1/(n_trait-1)
    if -min_corr<lower_limit_for_rho:
        neg_upper_lim = min(0,lower_limit_for_rho*0.8)
    else:
        neg_upper_lim = -min_corr
    L1 = neg_upper_lim -lower_limit_for_rho 
    L2 = 1 - min_corr
    L = L1 + L2
    print("sampling from [{}, {}] and [{} ,{}] for correlation.".format(lower_limit_for_rho,neg_upper_lim,min_corr,1))
    sim_cov_base = list()
    for i in range(n_sim_cov):
        # sample correlation 
        choose_left = np.random.rand() < 0.5
        if choose_left: 
            rho = np.random.uniform(lower_limit_for_rho,neg_upper_lim,1)[0]
        else:
            rho = np.random.uniform(min_corr,1,1)[0]
        sigma_base = np.ones((n_trait,n_trait))*rho
        np.fill_diagonal(sigma_base, 1)
        sim_cov_base.append(sigma_base)
    return sim_cov_base


def scale_var_beta(beta: np.ndarray, af: np.ndarray, h2: float) -> np.ndarray:
    het = 2 * af * (1 - af) + 1e-8
    # Scale betas so that total genetic variance = h2
    # Compute current genetic variance: beta^2 * Var(genotype) = beta^2 * het
    current_var = np.sum((beta**2) * het)
    scale = np.sqrt(h2 / current_var)
    beta *= scale
    return beta

def get_topr_idx_from_ld(ld: np.ndarray, top_r: int):
    nsnp = ld.shape[0]
    ldsc = np.sum(ld**2,axis=0)-1
    idx_max_ldsc = np.argsort(-ld-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
    top_r_val = ld[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]
    
    return idx_max_ldsc, top_r_val, ldsc

def construct_features_from_sim_by_gene(sim_data_block: np.ndarray, ld: np.ndarray, top_r:int):

    X_sim_block = []
    y_sim_block = []
    nsnp = ld.shape[0]

    # compute LD score and get top r (within the block)
    idx_max_ldsc, top_r_val, ldsc = get_topr_idx_from_ld(ld, top_r)

    n_trait = sim_data_block.shape[1]//3

    for i_trait in range(n_trait):
        sim_data_block_bhat = sim_data_block[:,n_trait+i_trait]
        sim_data_block_se = sim_data_block[:,2*n_trait+i_trait]
        sim_data_block_btrue = sim_data_block[:,i_trait]
    
        top_r_beta = sim_data_block_bhat[idx_max_ldsc]
    
        X = np.concatenate([sim_data_block_bhat[:,None],sim_data_block_se[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
        y = sim_data_block_btrue

        X_sim_block.append(X)
        y_sim_block.append(y)

    return X_sim_block, y_sim_block