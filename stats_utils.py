"""Statistical helper functions for fitting, testing, and resampling."""

import numpy as np
import math
from typing import Tuple, Dict, Callable

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

def erf(x: float) -> float:
    """Approximate the error function."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    
    y = 1.0 - (a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5) * math.exp(-x*x)
    return sign * y

def erfc(x: float) -> float:
    """Complementary error function."""
    return 1.0 - erf(x)

def beta_function(a: float, b: float) -> float:
    """Compute Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    def gamma_approx(z):
        """Approximate Gamma(z) with a Lanczos-style expansion."""
        if z < 0.5:
            return math.pi / (math.sin(math.pi * z) * gamma_approx(1 - z))
        
        z -= 1
        coeffs = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
                  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                  9.9843695780195716e-6, 1.5056327351493116e-7]
        
        base = z + 7 + 0.5
        result = math.sqrt(2 * math.pi) * (base ** (z + 0.5)) * math.exp(-base)
        
        x = 0.99999999999980993
        for i, coeff in enumerate(coeffs):
            x += coeff / (z + i + 1)
        
        return result * x
    
    return gamma_approx(a) * gamma_approx(b) / gamma_approx(a + b)

def incomplete_beta(x: float, a: float, b: float, max_iter: int = 100) -> float:
    """Compute incomplete beta function I_x(a,b) using continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - math.log(a) - math.log(beta_function(a, b)))
    
    f = 1.0
    c = 1.0
    d = 0.0
    
    for m in range(1, max_iter):
        m_val = float(m)
        numerator = m_val * (b - m_val) * x / ((a + 2*m_val - 1) * (a + 2*m_val))
        d = 1.0 + numerator * d
        d = 1.0 / d if abs(d) > 1e-10 else 1e10
        c = 1.0 + numerator / c
        f *= c * d
        
        if abs(c * d - 1) < 1e-10:
            break
    
    return front * f / a

def incomplete_gamma(x: float, a: float, max_iter: int = 100) -> float:
    """Compute lower incomplete gamma function P(a,x) using series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    
    term = 1.0 / a
    result = term
    
    for n in range(1, max_iter):
        term *= x / (a + n)
        result += term
        if abs(term) < 1e-12:
            break
    
    return result * math.exp(a * math.log(x) - x)

def chi2_cdf(x: float, k: int) -> float:
    """CDF of chi-squared distribution with k degrees of freedom."""
    if x <= 0:
        return 0.0
    
    return incomplete_gamma(x / 2.0, k / 2.0) / math.gamma(k / 2.0)

def kolmogorov_cdf(x: float) -> float:
    """CDF of Kolmogorov distribution K(x)."""
    if x <= 0:
        return 0.0
    if x >= 1.0:
        return 1.0
    
    result = 0.0
    for k in range(1, 100):
        term = (-1) ** (k - 1) * math.exp(-2 * k * k * x * x)
        result += 2 * term
        if abs(term) < 1e-12:
            break
    
    return result

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Probability density function of Gaussian distribution."""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Cumulative distribution function of Gaussian distribution."""
    return 0.5 * (1 + np.array([erf((xi - mu) / (sigma * math.sqrt(2))) for xi in np.atleast_1d(x)]))

def cauchy_pdf(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """Probability density function of Cauchy distribution."""
    return (gamma / math.pi) / ((x - x0) ** 2 + gamma ** 2)

def cauchy_cdf(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """Cumulative distribution function of Cauchy distribution."""
    return 0.5 + (1 / math.pi) * np.arctan((x - x0) / gamma)

def laplace_pdf(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    """Probability density function of Laplace (double exponential) distribution."""
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)

def laplace_cdf(x: np.ndarray, mu: float, b: float) -> np.ndarray:
    """Cumulative distribution function of Laplace distribution."""
    z = (x - mu) / b
    result = np.zeros_like(z, dtype=float)
    result[z >= 0] = 1 - 0.5 * np.exp(-z[z >= 0])
    result[z < 0] = 0.5 * np.exp(z[z < 0])
    return result

def student_t_pdf(x: np.ndarray, df: float, loc: float = 0, scale: float = 1) -> np.ndarray:
    """Probability density function of Student-t distribution."""
    if scipy_stats is not None:
        return scipy_stats.t.pdf(x, df, loc=loc, scale=scale)

    x_std = (x - loc) / scale
    
    coeff = math.gamma((df + 1) / 2) / (math.sqrt(math.pi * df) * math.gamma(df / 2))
    return (coeff / scale) * (1 + x_std ** 2 / df) ** (-(df + 1) / 2)

def student_t_cdf(x: np.ndarray, df: float, loc: float = 0, scale: float = 1) -> np.ndarray:
    """CDF of Student-t distribution using incomplete beta function."""
    if scipy_stats is not None:
        return scipy_stats.t.cdf(x, df, loc=loc, scale=scale)

    x_arr = np.atleast_1d(x)
    result = np.zeros_like(x_arr, dtype=float)
    
    x_std = (x_arr - loc) / scale
    
    for i, t in enumerate(x_std):
        if t == 0:
            result[i] = 0.5
        else:
            beta_arg = df / (df + t ** 2)
            if t > 0:
                result[i] = 1 - 0.5 * incomplete_beta(beta_arg, df / 2, 0.5)
            else:
                result[i] = 0.5 * incomplete_beta(beta_arg, df / 2, 0.5)
    
    if np.isscalar(x):
        return result[0]
    return result

def ks_test(data: np.ndarray, cdf_func: Callable, *args) -> Tuple[float, float]:
    """Return the two-sided KS statistic and p-value."""
    n = len(data)
    sorted_data = np.sort(data)
    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(0, n) / n
    theoretical_cdf = cdf_func(sorted_data, *args)
    
    d_plus = np.max(empirical_upper - theoretical_cdf)
    d_minus = np.max(theoretical_cdf - empirical_lower)
    d_stat = max(d_plus, d_minus)
    
    if scipy_stats is not None:
        p_value = scipy_stats.kstwo.sf(d_stat, n)
    else:
        ks_stat = d_stat * math.sqrt(n)
        p_value = 1 - kolmogorov_cdf(ks_stat)
    
    return d_stat, p_value

def chi2_goodness_of_fit(data: np.ndarray, pdf_func: Callable, args: Tuple,
                         n_bins: int = 10) -> Tuple[float, float, int]:
    """Return chi-squared statistic, p-value, and degrees of freedom."""
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    expected = pdf_func(bin_centers, *args) * bin_width * len(data)
    expected = np.maximum(expected, 1.0)
    
    chi2_stat = np.sum((hist - expected) ** 2 / expected)
    
    df = n_bins - 1 - len(args)
    
    p_value = 1 - chi2_cdf(chi2_stat, df)
    
    return chi2_stat, p_value, df

def anderson_darling_test(data: np.ndarray, cdf_func: Callable, *args) -> float:
    """
    Anderson-Darling test statistic.
    Higher values indicate worse fit.
    """
    n = len(data)
    sorted_data = np.sort(data)
    
    cdf_vals = cdf_func(sorted_data, *args)
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
    
    indices = np.arange(1, n + 1)
    term1 = (2 * indices - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
    
    a2 = -n - np.sum(term1) / n
    
    return a2

def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Shapiro-Wilk test for normality.
    Returns W statistic and approximate p-value.
    """
    n = len(data)
    if n < 3 or n > 5000:
        raise ValueError("Sample size must be between 3 and 5000")
    
    sorted_data = np.sort(data)
    mean_data = np.mean(data)
    
    W = 1 - (np.sum(np.abs(sorted_data - mean_data) ** 2)) / (np.sum(np.abs(sorted_data - mean_data)) ** 2 + 1e-10)
    W = max(0, min(1, W))
    
    p_value = min(1, max(0, 0.5 - abs(0.5 - W)))
    
    return W, p_value

def fit_gaussian(data: np.ndarray) -> Dict:
    """Fit Gaussian distribution via MLE."""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    return {'mu': mu, 'sigma': sigma}

def fit_laplace(data: np.ndarray) -> Dict:
    """Fit Laplace distribution via MLE."""
    mu = np.median(data)
    b = np.mean(np.abs(data - mu))
    
    return {'mu': mu, 'b': b}

def fit_cauchy(data: np.ndarray) -> Dict:
    """Fit Cauchy distribution via grid search + refinement."""
    x0_init = np.median(data)
    
    gammas = np.logspace(-1, 2, 50)
    best_gamma = gammas[0]
    best_ll = -np.inf
    
    for gamma in gammas:
        ll = np.sum(np.log(cauchy_pdf(data, x0_init, gamma) + 1e-300))
        if ll > best_ll and np.isfinite(ll):
            best_ll = ll
            best_gamma = gamma
    
    x0_range = np.linspace(x0_init - best_gamma, x0_init + best_gamma, 50)
    
    best_ll = -np.inf
    for x0_cand in x0_range:
        ll = np.sum(np.log(cauchy_pdf(data, x0_cand, best_gamma) + 1e-300))
        if ll > best_ll and np.isfinite(ll):
            best_ll = ll
            x0_init = x0_cand
    
    return {'x0': x0_init, 'gamma': best_gamma}

def fit_student_t(data: np.ndarray) -> Dict:
    """Fit Student-t distribution via MLE."""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    dfs = np.logspace(0, 2, 30)
    best_df = 1
    best_ll = -np.inf
    
    for df in dfs:
        try:
            pdf_vals = student_t_pdf(data, df, mu, sigma)
            pdf_vals = np.clip(pdf_vals, 1e-300, None)
            ll = np.sum(np.log(pdf_vals))
            
            if ll > best_ll and np.isfinite(ll):
                best_ll = ll
                best_df = df
        except:
            continue
    
    return {'df': best_df, 'loc': mu, 'scale': sigma}

def compute_aic(log_likelihood: float, n_params: int, n_data: int) -> float:
    """Compute the Akaike Information Criterion."""
    return 2 * n_params - 2 * log_likelihood

def compute_bic(log_likelihood: float, n_params: int, n_data: int) -> float:
    """Compute the Bayesian Information Criterion."""
    return n_params * math.log(n_data) - 2 * log_likelihood

def fit_all_distributions(data: np.ndarray) -> Dict:
    """Fit all candidate distributions and return model metrics."""
    results = {}
    n = len(data)
    
    gauss_params = fit_gaussian(data)
    gauss_pdf_vals = gaussian_pdf(data, gauss_params['mu'], gauss_params['sigma'])
    gauss_pdf_vals = np.clip(gauss_pdf_vals, 1e-300, None)
    gauss_ll = np.sum(np.log(gauss_pdf_vals))
    results['gaussian'] = {
        'params': gauss_params,
        'll': gauss_ll,
        'aic': compute_aic(gauss_ll, 2, n),
        'bic': compute_bic(gauss_ll, 2, n),
        'pdf': lambda x: gaussian_pdf(x, gauss_params['mu'], gauss_params['sigma']),
        'cdf': lambda x: gaussian_cdf(x, gauss_params['mu'], gauss_params['sigma'])
    }
    
    laplace_params = fit_laplace(data)
    laplace_pdf_vals = laplace_pdf(data, laplace_params['mu'], laplace_params['b'])
    laplace_pdf_vals = np.clip(laplace_pdf_vals, 1e-300, None)
    laplace_ll = np.sum(np.log(laplace_pdf_vals))
    results['laplace'] = {
        'params': laplace_params,
        'll': laplace_ll,
        'aic': compute_aic(laplace_ll, 2, n),
        'bic': compute_bic(laplace_ll, 2, n),
        'pdf': lambda x: laplace_pdf(x, laplace_params['mu'], laplace_params['b']),
        'cdf': lambda x: laplace_cdf(x, laplace_params['mu'], laplace_params['b'])
    }
    
    try:
        cauchy_params = fit_cauchy(data)
        cauchy_pdf_vals = cauchy_pdf(data, cauchy_params['x0'], cauchy_params['gamma'])
        cauchy_pdf_vals = np.clip(cauchy_pdf_vals, 1e-300, None)
        cauchy_ll = np.sum(np.log(cauchy_pdf_vals))
        results['cauchy'] = {
            'params': cauchy_params,
            'll': cauchy_ll,
            'aic': compute_aic(cauchy_ll, 2, n),
            'bic': compute_bic(cauchy_ll, 2, n),
            'pdf': lambda x: cauchy_pdf(x, cauchy_params['x0'], cauchy_params['gamma']),
            'cdf': lambda x: cauchy_cdf(x, cauchy_params['x0'], cauchy_params['gamma'])
        }
    except:
        results['cauchy'] = None
    
    try:
        t_params = fit_student_t(data)
        t_pdf_vals = student_t_pdf(data, t_params['df'], t_params['loc'], t_params['scale'])
        t_pdf_vals = np.clip(t_pdf_vals, 1e-300, None)
        t_ll = np.sum(np.log(t_pdf_vals))
        results['student_t'] = {
            'params': t_params,
            'll': t_ll,
            'aic': compute_aic(t_ll, 3, n),
            'bic': compute_bic(t_ll, 3, n),
            'pdf': lambda x: student_t_pdf(x, t_params['df'], t_params['loc'], t_params['scale']),
            'cdf': lambda x: student_t_cdf(x, t_params['df'], t_params['loc'], t_params['scale'])
        }
    except:
        results['student_t'] = None
    
    return results

def kl_divergence(data: np.ndarray, pdf_func: Callable, args: Tuple,
                  n_bins: int = 20) -> float:
    """Compute KL divergence between an empirical histogram and a PDF."""
    hist, bin_edges = np.histogram(data, bins=n_bins, density=False)
    hist = hist / np.sum(hist)
    hist = np.clip(hist, 1e-12, None)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    try:
        q = pdf_func(bin_centers, *args) * bin_width
        q = np.clip(q, 1e-12, None)
        
        kl = np.sum(hist * (np.log(hist) - np.log(q)))
        return kl
    except:
        return np.nan

def bootstrap_resample(data: np.ndarray, n_samples: int = 10000,
                       statistic: Callable = np.median) -> np.ndarray:
    """Bootstrap the sampling distribution of a statistic."""
    bootstrap_samples = np.zeros(n_samples)
    n = len(data)
    
    np.random.seed(42)
    for i in range(n_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_samples[i] = statistic(bootstrap_sample)
    
    return bootstrap_samples

def bootstrap_ci(data: np.ndarray, n_samples: int = 10000,
                 confidence: float = 0.68, statistic: Callable = np.median) -> Tuple:
    """Compute a percentile bootstrap confidence interval."""
    bootstrap_samples = bootstrap_resample(data, n_samples, statistic)
    point_estimate = statistic(data)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_samples, alpha * 100)
    upper = np.percentile(bootstrap_samples, (1 - alpha) * 100)
    
    return lower, point_estimate, upper

def gott_median_ci(n: int, confidence: float = 0.6827) -> Tuple[int, int]:
    """Find 1-indexed median confidence ranks using binomial weights."""
    from math import comb
    
    best_j = 1
    best_k = n
    best_prob = 0
    
    for j in range(1, n + 1):
        for k in range(j, n + 1):
            prob = 0
            for i in range(j, k + 1):
                if i <= n:
                    prob += comb(n, i) * (0.5 ** n)
            
            if prob >= confidence and (k - j) < (best_k - best_j):
                best_j = j
                best_k = k
                best_prob = prob
    
    return best_j, best_k, best_prob

def median_statistics_ci(data: np.ndarray, confidence: float = 0.6827) -> Tuple:
    """Compute a median and rank-based confidence interval."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    j, k, prob = gott_median_ci(n, confidence)
    
    ci_lower = sorted_data[j - 1]
    ci_upper = sorted_data[k - 1]
    median_val = np.median(data)
    
    return median_val, ci_lower, ci_upper, prob

if __name__ == '__main__':
    np.random.seed(42)
    test_data = np.random.normal(100, 10, 100)
    
    print("Testing Gaussian fit:")
    params = fit_gaussian(test_data)
    print(f"  mu={params['mu']:.2f}, sigma={params['sigma']:.2f}")
    
    print("\nTesting bootstrap CI:")
    low, est, high = bootstrap_ci(test_data, n_samples=1000)
    print(f"  CI: [{low:.2f}, {est:.2f}, {high:.2f}]")
    
    print("\nTesting Gott median CI:")
    med, low, high, prob = median_statistics_ci(test_data, confidence=0.6827)
    print(f"  Median: {med:.2f}, CI: [{low:.2f}, {high:.2f}], Prob: {prob:.4f}")

def erfcinv(x: float) -> float:
    """Inverse complementary error function using numerical inversion of erfc."""
    if x <= 0:
        return float('inf')
    if x >= 2:
        return float('-inf')
    
    y = math.sqrt(-math.log(x / 2.0))
    
    for _ in range(10):
        erfc_y = erfc(y)
        if abs(erfc_y - x) < 1e-14:
            break
        derivative = -2.0 / math.sqrt(math.pi) * math.exp(-y * y)
        if abs(derivative) < 1e-12:
            break
        y = y - (erfc_y - x) / derivative
    
    return y
