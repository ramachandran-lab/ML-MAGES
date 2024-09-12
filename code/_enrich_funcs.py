"""
Adapted from the original code of liu_sf() from the Python package chiscore (https://github.com/limix/chiscore).
The installation of the package may fail at some circumstances (most likely due to failure in installing the package chi2comb).
When this happens, the following function will be used alternatively for testing the linear combination of chi-squared variables.
"""

from numpy import asarray, maximum, sqrt, sum
from scipy.stats import ncx2


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
    t = asarray(t, float)
    lambs = asarray(lambs, float)
    dofs = asarray(dofs, float)
    deltas = asarray(deltas, float)

    lambs = {i: lambs ** i for i in range(1, 5)}

    c = {i: sum(lambs[i] * dofs) + i * sum(lambs[i] * deltas) for i in range(1, 5)}

    s1 = c[3] / sqrt(c[2]) ** 3
    s2 = c[4] / c[2] ** 2

    s12 = s1 ** 2
    if s12 > s2:
        a = 1 / (s1 - sqrt(s12 - s2))
        delta_x = s1 * a ** 3 - a ** 2
        dof_x = a ** 2 - 2 * delta_x
    else:
        delta_x = 0
        a = 1 / s1
        dof_x = 1 / s12

    mu_q = c[1]
    sigma_q = sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = sqrt(2 * (dof_x + 2 * delta_x))

    t_star = (t - mu_q) / sigma_q
    tfinal = t_star * sigma_x + mu_x

    q = ncx2.sf(tfinal, dof_x, maximum(delta_x, 1e-9))

    return q