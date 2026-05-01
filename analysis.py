"""Core statistical analysis for the Θ₀ and H₀ project."""

import numpy as np
from typing import Dict, Tuple
import stats_utils as su
import data

def compute_median_statistics(values: np.ndarray, errors: np.ndarray) -> Dict:
    """Compute the median and rank-based confidence intervals."""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    
    n = len(sorted_values)
    median_idx = (n - 1) // 2
    median_value = sorted_values[median_idx]
    
    def central_rank_interval(n, sigma_level):
        """Return the central rank interval for the requested confidence level."""
        paper_rank_halfwidths = {
            29: {1: 3, 2: 5},
            19: {1: 3, 2: 5},
            11: {1: 2, 2: 4},
        }
        median_rank = (n + 1) // 2
        if n in paper_rank_halfwidths:
            halfwidth = paper_rank_halfwidths[n][sigma_level]
            lower_rank = max(1, median_rank - halfwidth)
            upper_rank = min(n, median_rank + halfwidth)
            return lower_rank - 1, upper_rank - 1, np.nan

        from math import comb
        target_prob = 0.6827 if sigma_level == 1 else 0.9545
        best = None
        center = (n - 1) / 2
        for lower in range(n):
            for upper in range(lower, n):
                prob = sum(comb(n, i) for i in range(lower, upper + 1)) / 2**n
                if prob < target_prob:
                    continue
                width = upper - lower
                offset = abs((lower + upper) / 2 - center)
                candidate = (width, offset, prob, lower, upper)
                if best is None or candidate < best:
                    best = candidate
        _, _, prob, lower, upper = best
        return lower, upper, prob

    j_1s, k_1s, prob_1s = central_rank_interval(n, 1)
    j_2s, k_2s, prob_2s = central_rank_interval(n, 2)
    
    ci_1sigma_lower = sorted_values[j_1s]
    ci_1sigma_upper = sorted_values[k_1s]
    ci_2sigma_lower = sorted_values[j_2s]
    ci_2sigma_upper = sorted_values[k_2s]
    
    sigma_median = (ci_1sigma_upper - ci_1sigma_lower) / 2.0
    
    return {
        'median': median_value,
        'sigma_median': sigma_median,
        'ci_1sigma_lower': ci_1sigma_lower,
        'ci_1sigma_upper': ci_1sigma_upper,
        'ci_1sigma_lower_error': median_value - ci_1sigma_lower,
        'ci_1sigma_upper_error': ci_1sigma_upper - median_value,
        'ci_1sigma_prob': prob_1s,
        'ci_2sigma_lower': ci_2sigma_lower,
        'ci_2sigma_upper': ci_2sigma_upper,
        'ci_2sigma_lower_error': median_value - ci_2sigma_lower,
        'ci_2sigma_upper_error': ci_2sigma_upper - median_value,
        'ci_2sigma_prob': prob_2s,
    }

def compute_weighted_mean(values: np.ndarray, errors: np.ndarray) -> Dict:
    """Compute the inverse-variance weighted mean and uncertainty."""
    weights = 1.0 / (errors ** 2)
    sum_weighted = np.sum(values * weights)
    sum_weights = np.sum(weights)
    
    weighted_mean = sum_weighted / sum_weights
    sigma_weighted_mean = 1.0 / np.sqrt(sum_weights)
    
    return {
        'weighted_mean': weighted_mean,
        'sigma_weighted_mean': sigma_weighted_mean,
    }

def compute_arithmetic_mean(values: np.ndarray, errors: np.ndarray) -> Dict:
    """Compute simple arithmetic mean."""
    mean = np.mean(values)
    sigma_mean = np.std(values, ddof=1) / np.sqrt(len(values))
    
    return {
        'mean': mean,
        'sigma_mean': sigma_mean,
    }

def compute_nsigma_deviations_median(values: np.ndarray, errors: np.ndarray,
                                     median_value: float, sigma_median: float) -> np.ndarray:
    """Compute normalized deviations from the median central estimate."""
    deviations = (values - median_value) / np.sqrt(errors**2 + sigma_median**2)
    return deviations

def compute_nsigma_deviations_weighted_mean(values: np.ndarray, errors: np.ndarray,
                                            weighted_mean: float, sigma_wm: float) -> np.ndarray:
    """Compute normalized deviations from the weighted mean estimate."""
    denominator = errors**2 - sigma_wm**2
    denominator = np.maximum(denominator, 0.1)
    
    deviations = (values - weighted_mean) / np.sqrt(denominator)
    return deviations

def fit_distribution_with_scale(data: np.ndarray, dist_name: str, 
                                 scale_range: Tuple = (0.001, 5.0),
                                 n_steps: int = 100) -> Dict:
    """Fit a distribution by choosing the scale with the best KS p-value."""
    
    if dist_name == 'gaussian':
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        scales = np.linspace(scale_range[0], scale_range[1], n_steps)
        best_p = -1
        best_s = scales[0]
        best_d = 1
        
        for s in scales:
            cdf_func = lambda x: su.gaussian_cdf(x, mu, s * sigma)
            d_stat, p_val = su.ks_test(data, cdf_func)
            
            if p_val > best_p:
                best_p = p_val
                best_s = s
                best_d = d_stat
        
        return {
            's': best_s,
            'p_value': best_p,
            'd_stat': best_d,
            'params': {'mu': mu, 'sigma': sigma},
        }
    
    elif dist_name == 'cauchy':
        x0 = np.median(data)
        
        scales = np.linspace(scale_range[0], scale_range[1], n_steps)
        best_p = -1
        best_s = scales[0]
        best_d = 1
        
        for s in scales:
            cdf_func = lambda x, s=s: su.cauchy_cdf(x, x0, s)
            try:
                d_stat, p_val = su.ks_test(data, cdf_func)
                if p_val > best_p:
                    best_p = p_val
                    best_s = s
                    best_d = d_stat
            except:
                pass
        
        return {
            's': best_s,
            'p_value': best_p,
            'd_stat': best_d,
            'params': {'x0': x0, 'gamma': best_s},
        }
    
    elif dist_name == 'laplace':
        mu = np.median(data)
        
        scales = np.linspace(scale_range[0], scale_range[1], n_steps)
        best_p = -1
        best_s = scales[0]
        best_d = 1
        
        for s in scales:
            cdf_func = lambda x, s=s: su.laplace_cdf(x, mu, s)
            try:
                d_stat, p_val = su.ks_test(data, cdf_func)
                if p_val > best_p:
                    best_p = p_val
                    best_s = s
                    best_d = d_stat
            except:
                pass
        
        return {
            's': best_s,
            'p_value': best_p,
            'd_stat': best_d,
            'params': {'mu': mu, 'b': best_s},
        }
    
    elif dist_name == 'student_t':
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        n_values = np.logspace(np.log10(2), np.log10(100), 20)
        scales = np.linspace(scale_range[0], scale_range[1], n_steps)
        
        best_p = -1
        best_s = scales[0]
        best_n = n_values[0]
        best_d = 1
        
        for n in n_values:
            for s in scales:
                cdf_func = lambda x, n=n, s=s: su.student_t_cdf(x, n, mu, s * sigma)
                try:
                    d_stat, p_val = su.ks_test(data, cdf_func)
                    if p_val > best_p:
                        best_p = p_val
                        best_s = s
                        best_n = n
                        best_d = d_stat
                except:
                    pass
        
        return {
            's': best_s,
            'n': best_n,
            'p_value': best_p,
            'd_stat': best_d,
            'params': {'df': best_n, 'loc': mu, 'scale': best_s * sigma},
        }
    
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

def analyze_subset(values: np.ndarray, errors: np.ndarray, 
                   subset_name: str = "All Tracers") -> Dict:
    """Run the full analysis for one measurement subset."""
    
    results = {
        'subset_name': subset_name,
        'n_measurements': len(values),
    }
    
    median_stats = compute_median_statistics(values, errors)
    results['median'] = median_stats['median']
    results['sigma_median'] = median_stats['sigma_median']
    results['median_ci_1sigma_lower'] = median_stats['ci_1sigma_lower_error']
    results['median_ci_1sigma_upper'] = median_stats['ci_1sigma_upper_error']
    results['median_ci_2sigma_lower'] = median_stats['ci_2sigma_lower_error']
    results['median_ci_2sigma_upper'] = median_stats['ci_2sigma_upper_error']
    
    wm_stats = compute_weighted_mean(values, errors)
    results['weighted_mean'] = wm_stats['weighted_mean']
    results['sigma_weighted_mean'] = wm_stats['sigma_weighted_mean']
    
    mean_stats = compute_arithmetic_mean(values, errors)
    results['arithmetic_mean'] = mean_stats['mean']
    results['sigma_arithmetic_mean'] = mean_stats['sigma_mean']
    
    nsigma_med = compute_nsigma_deviations_median(values, errors, 
                                                   median_stats['median'],
                                                   median_stats['sigma_median'])
    results['nsigma_deviations_median'] = nsigma_med
    results['nsigma_median_mean'] = np.mean(nsigma_med)
    results['nsigma_median_std'] = np.std(nsigma_med)
    
    nsigma_wm = compute_nsigma_deviations_weighted_mean(values, errors,
                                                        wm_stats['weighted_mean'],
                                                        wm_stats['sigma_weighted_mean'])
    results['nsigma_deviations_wm'] = nsigma_wm
    results['nsigma_wm_mean'] = np.mean(nsigma_wm)
    results['nsigma_wm_std'] = np.std(nsigma_wm)
    
    results['fits_median'] = {}
    
    for dist in ['gaussian', 'cauchy', 'student_t', 'laplace']:
        try:
            fit_result = fit_distribution_with_scale(nsigma_med, dist)
            results['fits_median'][dist] = fit_result
        except Exception as e:
            print(f"Warning: Failed to fit {dist} for median: {e}")
            results['fits_median'][dist] = None
    
    results['fits_wm'] = {}
    
    for dist in ['gaussian', 'cauchy', 'student_t', 'laplace']:
        try:
            fit_result = fit_distribution_with_scale(nsigma_wm, dist)
            results['fits_wm'][dist] = fit_result
        except Exception as e:
            print(f"Warning: Failed to fit {dist} for WM: {e}")
            results['fits_wm'][dist] = None
    
    results['kl_divergence'] = {}
    
    for dist in ['gaussian', 'cauchy', 'student_t', 'laplace']:
        fit = results['fits_median'][dist]
        if fit is not None:
            if dist == 'gaussian':
                args = (fit['params']['mu'], fit['s'] * fit['params']['sigma'])
            elif dist == 'cauchy':
                args = (fit['params']['x0'], fit['params']['gamma'])
            elif dist == 'student_t':
                args = (fit['params']['df'], fit['params']['loc'], fit['params']['scale'])
            elif dist == 'laplace':
                args = (fit['params']['mu'], fit['params']['b'])
            kl = su.kl_divergence(nsigma_med, _get_pdf_func(dist), args)
            results['kl_divergence'][f'{dist}_median'] = kl
    
    bootstrap_med = su.bootstrap_resample(values, n_samples=10000, 
                                          statistic=np.median)
    rng = np.random.default_rng(42)
    bootstrap_wm = np.zeros(10000)
    n_values = len(values)
    for i in range(len(bootstrap_wm)):
        sample_idx = rng.integers(0, n_values, size=n_values)
        bootstrap_wm[i] = compute_weighted_mean(
            values[sample_idx],
            errors[sample_idx]
        )['weighted_mean']
    
    results['bootstrap_median'] = bootstrap_med
    results['bootstrap_wm'] = bootstrap_wm
    results['bootstrap_median_std'] = np.std(bootstrap_med)
    results['bootstrap_wm_std'] = np.std(bootstrap_wm)
    
    return results

def _get_pdf_func(dist):
    """Return the PDF function for a fitted distribution name."""
    if dist == 'gaussian':
        return su.gaussian_pdf
    elif dist == 'cauchy':
        return su.cauchy_pdf
    elif dist == 'laplace':
        return su.laplace_pdf
    elif dist == 'student_t':
        return su.student_t_pdf
    else:
        raise ValueError(f"Unknown distribution: {dist}")

def print_analysis_results(results: Dict):
    """Pretty-print analysis results."""
    
    print("\n" + "="*80)
    print(f"ANALYSIS RESULTS: {results['subset_name']}")
    print(f"Number of measurements: {results['n_measurements']}")
    print("="*80)
    
    print("\n" + "-"*80)
    print("CENTRAL ESTIMATES")
    print("-"*80)
    print(f"Median:        {results['median']:.2f} "
          f"+{results['median_ci_1sigma_upper']:.2f} "
          f"-{results['median_ci_1sigma_lower']:.2f} (1σ)")
    print(f"               "
          f"+{results['median_ci_2sigma_upper']:.2f} "
          f"-{results['median_ci_2sigma_lower']:.2f} (2σ)")
    print(f"Weighted Mean: {results['weighted_mean']:.2f} ± {results['sigma_weighted_mean']:.2f}")
    print(f"Arithmetic Mean: {results['arithmetic_mean']:.2f} ± {results['sigma_arithmetic_mean']:.2f}")
    
    print("\n" + "-"*80)
    print("DISTRIBUTION FITS - MEDIAN CENTRAL ESTIMATE")
    print("-"*80)
    for dist in ['gaussian', 'cauchy', 'student_t', 'laplace']:
        if results['fits_median'][dist] is not None:
            fit = results['fits_median'][dist]
            if 'n' in fit:
                print(f"{dist:12s}: S={fit['s']:.3f}, n={fit['n']:.1f}, p={fit['p_value']:.4f}, D={fit['d_stat']:.4f}")
            else:
                print(f"{dist:12s}: S={fit['s']:.3f}, p={fit['p_value']:.4f}, D={fit['d_stat']:.4f}")

if __name__ == '__main__':
    theta0_vals, theta0_errs = data.get_theta0_values()
    
    print("TESTING ANALYSIS MODULE")
    print("="*80)
    
    results_all = analyze_subset(theta0_vals, theta0_errs, "All Tracers (N=29)")
    print_analysis_results(results_all)
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results_old = analyze_subset(old_vals, old_errs, "Old Tracers (N=19)")
    print_analysis_results(results_old)
    
    young_vals, young_errs, _, _ = data.get_theta0_subset("Young")
    results_young = analyze_subset(young_vals, young_errs, "Young Tracers (N=11)")
    print_analysis_results(results_young)
