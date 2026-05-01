"""Create the plots used in the Θ₀ and H₀ analysis."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import data
import analysis as ana
import stats_utils as su

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLOR_OLD = '#2E86AB'
COLOR_YOUNG = '#A23B72'
COLOR_BOTH = '#F18F01'
OUTPUT_SUFFIX = '_new'

PROJECT_DIR = Path(__file__).resolve().parent

def save_figure(filename):
    """Save a figure with the configured output suffix."""
    path = Path(filename)
    new_name = f'{path.stem}{OUTPUT_SUFFIX}{path.suffix}'
    plt.savefig(PROJECT_DIR / new_name, dpi=300, bbox_inches='tight')
    return new_name

def set_publication_style(fig_size=(12, 8)):
    """Apply shared plot styling."""
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['patch.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2

def get_tracer_color(tracer_type):
    """Return the display color for a tracer type."""
    if tracer_type == 'Old':
        return COLOR_OLD
    elif tracer_type == 'Young':
        return COLOR_YOUNG
    elif tracer_type == 'Both':
        return COLOR_BOTH
    else:
        return '#555555'

def fig01_theta0_overview():
    """Plot all Θ₀ measurements with central estimate reference lines."""
    set_publication_style((14, 8))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df = data.get_theta0_dataframe()
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    results = ana.analyze_subset(theta0_vals, theta0_errs, "All")
    
    sort_idx = np.argsort(theta0_vals)
    x_pos = np.arange(len(theta0_vals))
    
    for i, idx in enumerate(sort_idx):
        color = get_tracer_color(df.iloc[idx]['tracer_type'])
        ax.errorbar(i, theta0_vals[idx], yerr=theta0_errs[idx],
                   fmt='o', color=color, markersize=8, capsize=5, capthick=2,
                   alpha=0.7, elinewidth=1.5)
    
    ax.axhline(results['median'], color='red', linestyle='--', linewidth=2.5,
              label=f"Median = {results['median']:.2f} km/s")
    ax.axhline(results['weighted_mean'], color='green', linestyle='--', linewidth=2.5,
              label=f"Weighted Mean = {results['weighted_mean']:.2f} km/s")
    ax.axhline(results['arithmetic_mean'], color='blue', linestyle=':', linewidth=2.5,
              label=f"Arithmetic Mean = {results['arithmetic_mean']:.2f} km/s")
    
    ax.axhspan(results['median'] - results['median_ci_1sigma_lower'],
              results['median'] + results['median_ci_1sigma_upper'],
              alpha=0.1, color='red', label='Median 1σ CI')
    
    ax.set_xlabel('Measurement Index (sorted)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Θ₀ (km/s)', fontsize=13, fontweight='bold')
    ax.set_title('Galactic Rotational Velocity (Θ₀) Measurements\nAll 29 Measurements Rescaled to R₀ = 7.96 kpc',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos[::3])
    ax.grid(True, alpha=0.3)
    
    old_patch = mpatches.Patch(color=COLOR_OLD, label='Old Tracers')
    young_patch = mpatches.Patch(color=COLOR_YOUNG, label='Young Tracers')
    both_patch = mpatches.Patch(color=COLOR_BOTH, label='Both Types')
    line_red = plt.Line2D([0], [0], color='red', linewidth=2.5, linestyle='--')
    line_green = plt.Line2D([0], [0], color='green', linewidth=2.5, linestyle='--')
    line_blue = plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle=':')
    patch_red = mpatches.Patch(facecolor='red', alpha=0.1)
    
    ax.legend(handles=[old_patch, young_patch, both_patch, line_red, line_green, line_blue, patch_red],
             labels=['Old Tracers', 'Young Tracers', 'Both Types', 'Median', 'Weighted Mean', 'Arithmetic Mean', 'Median 1σ CI'],
             loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    saved_name = save_figure('Fig01_theta0_data_overview.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig02_theta0_deviations():
    """Plot Old-tracer deviations with fitted distributions."""
    set_publication_style((14, 10))
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results = ana.analyze_subset(old_vals, old_errs, "Old")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    dists = ['gaussian', 'cauchy', 'student_t', 'laplace']
    dist_names = ['Gaussian', 'Cauchy', "Student's t", 'Laplace']
    
    for idx, (dist, name) in enumerate(zip(dists, dist_names)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        ax.hist(results['nsigma_deviations_median'], bins=12, density=True,
               alpha=0.6, color='steelblue', edgecolor='black', linewidth=1.5,
               label='Empirical')
        
        if results['fits_median'][dist] is not None:
            fit = results['fits_median'][dist]
            x_range = np.linspace(-4, 4, 200)
            
            if dist == 'gaussian':
                params = fit['params']
                sigma = fit['s'] * params['sigma']
                y = su.gaussian_pdf(x_range, params['mu'], sigma)
                label = f"Gaussian (S={fit['s']:.2f})"
            elif dist == 'cauchy':
                params = fit['params']
                y = su.cauchy_pdf(x_range, params['x0'], params['gamma'])
                label = f"Cauchy (S={fit['s']:.2f})"
            elif dist == 'student_t':
                params = fit['params']
                y = su.student_t_pdf(x_range, params['df'], params['loc'], params['scale'])
                label = f"Student-t (n={fit['n']:.1f}, S={fit['s']:.2f})"
            elif dist == 'laplace':
                params = fit['params']
                y = su.laplace_pdf(x_range, params['mu'], params['b'])
                label = f"Laplace (S={fit['s']:.2f})"
            
            ax.plot(x_range, y, 'r-', linewidth=2.5, label=label)
            
            p_val = fit['p_value']
            d_stat = fit['d_stat']
            ax.text(0.98, 0.97, f"p-value = {p_val:.4f}\nD = {d_stat:.4f}",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Nσ Deviation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Fit (Old Tracers, N=19)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(-4, 4)
    
    fig.suptitle('Nσ Deviations: Old Tracers with Best-Fit Distributions\n(Median Central Estimate, Equation 6)',
                fontsize=14, fontweight='bold', y=0.995)
    
    saved_name = save_figure('Fig02_theta0_deviations.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig03_theta0_cdf():
    """Plot the empirical CDF against fitted theoretical CDFs."""
    set_publication_style((12, 8))
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results = ana.analyze_subset(old_vals, old_errs, "Old")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_dev = np.sort(results['nsigma_deviations_median'])
    empirical_cdf = np.arange(1, len(sorted_dev) + 1) / len(sorted_dev)
    ax.step(sorted_dev, empirical_cdf, where='mid', linewidth=2.5,
           label='Empirical CDF', color='black')
    
    x_range = np.linspace(-4, 4, 300)
    
    from stats_utils import gaussian_cdf, cauchy_cdf, laplace_cdf, student_t_cdf
    
    if results['fits_median']['gaussian'] is not None:
        fit = results['fits_median']['gaussian']
        params = fit['params']
        s = fit['s'] * params['sigma']
        y = gaussian_cdf(x_range, params['mu'], s)
        ax.plot(x_range, y, linewidth=2, linestyle='--', label=f"Gaussian (S={fit['s']:.2f})",
               color='#d62728')
    
    if results['fits_median']['cauchy'] is not None:
        fit = results['fits_median']['cauchy']
        params = fit['params']
        y = cauchy_cdf(x_range, params['x0'], params['gamma'])
        ax.plot(x_range, y, linewidth=2, linestyle='--', label=f"Cauchy (S={fit['s']:.2f})",
               color='#2ca02c')
    
    if results['fits_median']['laplace'] is not None:
        fit = results['fits_median']['laplace']
        params = fit['params']
        y = laplace_cdf(x_range, params['mu'], params['b'])
        ax.plot(x_range, y, linewidth=2, linestyle='--', label=f"Laplace (S={fit['s']:.2f})",
               color='#ff7f0e')
    
    if results['fits_median']['student_t'] is not None:
        fit = results['fits_median']['student_t']
        params = fit['params']
        y = student_t_cdf(x_range, params['df'], params['loc'], params['scale'])
        ax.plot(x_range, y, linewidth=2, linestyle='--',
               label=f"Student-t (n={fit['n']:.1f}, S={fit['s']:.2f})", color='#1f77b4')
    
    ax.set_xlabel('Nσ Deviation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title('Empirical vs Theoretical CDF: Old Tracers\n(Median Central Estimate)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    saved_name = save_figure('Fig03_theta0_cdf.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig04_theta0_qq():
    """Plot Old-tracer quantiles against Gaussian quantiles."""
    set_publication_style((10, 8))
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results = ana.analyze_subset(old_vals, old_errs, "Old")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sorted_dev = np.sort(results['nsigma_deviations_median'])
    n = len(sorted_dev)
    
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = np.array([np.sqrt(2) * su.erfcinv(2 * (1 - q))
                                      if q < 1 else 5 for q in quantiles])
    theoretical_quantiles[quantiles >= 1] = sorted_dev[-1]
    
    ax.scatter(theoretical_quantiles, sorted_dev, s=80, alpha=0.6, color='steelblue', edgecolor='black')
    
    lims = [min(theoretical_quantiles.min(), sorted_dev.min()),
            max(theoretical_quantiles.max(), sorted_dev.max())]
    ax.plot(lims, lims, 'r--', linewidth=2.5, label='Perfect Gaussian fit')
    
    ax.set_xlabel('Theoretical Quantiles (Gaussian)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    ax.set_title('Q-Q Plot: Old Tracers vs Gaussian Distribution\n(Median Central Estimate)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    saved_name = save_figure('Fig04_theta0_qq.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig05_theta0_bootstrap():
    """Plot bootstrap distributions for the median and weighted mean."""
    set_publication_style((14, 6))
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    results = ana.analyze_subset(theta0_vals, theta0_errs, "All")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.hist(results['bootstrap_median'], bins=50, density=True, alpha=0.7,
           color='steelblue', edgecolor='black', linewidth=1)
    ax.axvline(results['median'], color='red', linestyle='--', linewidth=2.5,
              label=f'Median = {results["median"]:.2f} km/s')
    ax.axvline(np.mean(results['bootstrap_median']), color='green', linestyle='--',
              linewidth=2.5, label=f'Bootstrap Mean = {np.mean(results["bootstrap_median"]):.2f} km/s')
    ax.set_xlabel('Θ₀ (km/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Bootstrap Distribution of Median\n(10,000 resamples, N=29)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    ax = axes[1]
    ax.hist(results['bootstrap_wm'], bins=50, density=True, alpha=0.7,
           color='coral', edgecolor='black', linewidth=1)
    ax.axvline(results['weighted_mean'], color='red', linestyle='--', linewidth=2.5,
              label=f'Weighted Mean = {results["weighted_mean"]:.2f} km/s')
    ax.axvline(np.mean(results['bootstrap_wm']), color='green', linestyle='--',
              linewidth=2.5, label=f'Bootstrap Mean = {np.mean(results["bootstrap_wm"]):.2f} km/s')
    ax.set_xlabel('Θ₀ (km/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Bootstrap Distribution of Weighted Mean\n(10,000 resamples, N=29)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    fig.suptitle('Bootstrap Analysis: All Tracers',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    saved_name = save_figure('Fig05_theta0_bootstrap.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig06_theta0_model_comparison():
    """Compare KS p-values and scale factors across tracer subsets."""
    set_publication_style((12, 8))
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    results_all = ana.analyze_subset(theta0_vals, theta0_errs, "All (N=29)")
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results_old = ana.analyze_subset(old_vals, old_errs, "Old (N=19)")
    
    young_vals, young_errs, _, _ = data.get_theta0_subset("Young")
    results_young = ana.analyze_subset(young_vals, young_errs, "Young (N=11)")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    dists = ['gaussian', 'cauchy', 'student_t', 'laplace']
    labels = ['Gaussian', 'Cauchy', "Student's t", 'Laplace']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for col, (results, title) in enumerate([(results_all, 'All Tracers'),
                                             (results_old, 'Old Tracers'),
                                             (results_young, 'Young Tracers')]):
        p_values = []
        for dist in dists:
            if results['fits_median'][dist] is not None:
                p_values.append(results['fits_median'][dist]['p_value'])
            else:
                p_values.append(0)
        
        ax = axes[0, col]
        bars = ax.bar(labels, p_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('KS Test p-value', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n(Median Central Estimate)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        scales = []
        for dist in dists:
            if results['fits_median'][dist] is not None:
                scales.append(results['fits_median'][dist]['s'])
            else:
                scales.append(0)
        
        ax = axes[1, col]
        bars = ax.bar(labels, scales, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Scale Factor S', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n(Best-fit Scale Factors)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, scales):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Distribution Goodness-of-Fit Comparison\n(KS Test p-values and Scale Factors)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    saved_name = save_figure('Fig06_theta0_model_comparison.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig07_theta0_kl():
    """Compare KL divergence across fitted distributions."""
    set_publication_style((12, 7))
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    results_all = ana.analyze_subset(theta0_vals, theta0_errs, "All (N=29)")
    
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    results_old = ana.analyze_subset(old_vals, old_errs, "Old (N=19)")
    
    young_vals, young_errs, _, _ = data.get_theta0_subset("Young")
    results_young = ana.analyze_subset(young_vals, young_errs, "Young (N=11)")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dists = ['gaussian', 'cauchy', 'student_t', 'laplace']
    labels = ['Gaussian', 'Cauchy', "Student's t", 'Laplace']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    x = np.arange(len(labels))
    width = 0.25
    
    results_list = [
        (results_all, 'All Tracers (N=29)'),
        (results_old, 'Old Tracers (N=19)'),
        (results_young, 'Young Tracers (N=11)')
    ]
    
    for i, (results, name) in enumerate(results_list):
        kl_values = []
        for dist in dists:
            kl_values.append(results['kl_divergence'].get(f'{dist}_median', np.nan))
        
        offset = (i - 1) * width
        ax.bar(x + offset, kl_values, width, label=name,
              color=plt.cm.Set3(i), edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Distribution Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL divergence\n[Lower is better]', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Goodness-of-Fit Comparison\n(Empirical vs Fitted PDF KL divergence)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    saved_name = save_figure('Fig07_theta0_kl.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig08_h0_comprehensive():
    """Plot the H₀ extension summary panels."""
    set_publication_style((14, 10))
    
    h0_vals, h0_errs = data.get_h0_values()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, :])
    sort_idx = np.argsort(h0_vals)
    x_pos = np.arange(len(h0_vals))
    colors_h0 = plt.cm.viridis(np.linspace(0.2, 0.8, len(h0_vals)))
    
    for i, idx in enumerate(sort_idx):
        ax.errorbar(i, h0_vals[idx], yerr=h0_errs[idx],
                   fmt='o', color=colors_h0[idx], markersize=8, capsize=5, capthick=2,
                   alpha=0.7, elinewidth=1.5)
    
    h0_median = np.median(h0_vals)
    h0_mean = np.mean(h0_vals)
    h0_weighted = np.average(h0_vals, weights=1/h0_errs**2)
    
    ax.axhline(h0_median, color='red', linestyle='--', linewidth=2, label=f'Median = {h0_median:.2f}')
    ax.axhline(h0_weighted, color='green', linestyle='--', linewidth=2, label=f'Weighted Mean = {h0_weighted:.2f}')
    ax.axhline(h0_mean, color='blue', linestyle=':', linewidth=2, label=f'Mean = {h0_mean:.2f}')
    
    ax.set_ylabel('H₀ (km/s/Mpc)', fontsize=12, fontweight='bold')
    ax.set_title('Hubble Constant (H₀) Measurements', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(h0_vals, bins=8, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.axvline(h0_median, color='red', linestyle='--', linewidth=2, label=f'Median')
    ax.axvline(h0_weighted, color='green', linestyle='--', linewidth=2, label=f'Weighted Mean')
    ax.set_xlabel('H₀ (km/s/Mpc)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('H₀ Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    h0_df = data.get_h0_dataframe()
    years = h0_df['year'].unique()
    years_sorted = sorted(years)
    h0_by_year = [h0_vals[h0_df['year'] == y] for y in years_sorted]
    h0_std_by_year = [np.std(h0_by_year[i]) if len(h0_by_year[i]) > 1 else 0 
                      for i in range(len(years_sorted))]
    
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(years_sorted, h0_std_by_year, s=100, alpha=0.6, color='coral', edgecolor='black')
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Std Dev (km/s/Mpc)', fontsize=11, fontweight='bold')
    ax.set_title('H₀ Measurement Scatter Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(h0_vals, h0_errs, s=100, alpha=0.6, color='orange', edgecolor='black')
    ax.set_xlabel('H₀ (km/s/Mpc)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Measurement Error', fontsize=11, fontweight='bold')
    ax.set_title('Measurement Precision vs Value', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 1])
    h0_normalized = (h0_vals - h0_weighted) / h0_errs
    colors_tension = ['red' if abs(x) > 2 else 'blue' for x in h0_normalized]
    ax.scatter(range(len(h0_normalized)), h0_normalized, s=100, c=colors_tension, 
              alpha=0.6, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='2σ')
    ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Measurement Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('(H₀ - Weighted Mean) / σ', fontsize=11, fontweight='bold')
    ax.set_title('H₀ Tension Analysis', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Hubble Constant (H₀) Comprehensive Analysis\n14 Measurements from Multiple Sources',
                fontsize=14, fontweight='bold', y=0.995)
    
    saved_name = save_figure('Fig08_h0_comprehensive.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig09_theta0_h0_comparison():
    """Compare the Θ₀ and H₀ data distributions."""
    set_publication_style((14, 10))
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    h0_vals, h0_errs = data.get_h0_values()
    
    results_theta0 = ana.analyze_subset(theta0_vals, theta0_errs, "All")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(theta0_vals, bins=15, density=True, alpha=0.6, color='steelblue', 
           edgecolor='black', linewidth=1)
    x_range = np.linspace(theta0_vals.min(), theta0_vals.max(), 100)
    mu, sigma = np.mean(theta0_vals), np.std(theta0_vals)
    from stats_utils import gaussian_pdf
    y = gaussian_pdf(x_range, mu, sigma)
    ax.plot(x_range, y, 'r-', linewidth=2.5, label='Gaussian fit')
    ax.set_xlabel('Θ₀ (km/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('Θ₀ Distribution (N=29)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(h0_vals, bins=10, density=True, alpha=0.6, color='coral', 
           edgecolor='black', linewidth=1)
    x_range = np.linspace(h0_vals.min(), h0_vals.max(), 100)
    mu, sigma = np.mean(h0_vals), np.std(h0_vals)
    y = gaussian_pdf(x_range, mu, sigma)
    ax.plot(x_range, y, 'r-', linewidth=2.5, label='Gaussian fit')
    ax.set_xlabel('H₀ (km/s/Mpc)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('H₀ Distribution (N=14)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 0])
    sorted_theta0 = np.sort(theta0_vals)
    n = len(sorted_theta0)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_q = np.sqrt(2) * np.array([su.erfcinv(2 * (1 - q)) if q < 1 else 5 for q in quantiles])
    sample_q = (sorted_theta0 - np.mean(theta0_vals)) / np.std(theta0_vals, ddof=1)
    ax.scatter(theoretical_q, sample_q, alpha=0.6, s=60, edgecolor='black')
    lims = [min(theoretical_q.min(), sample_q.min()),
            max(theoretical_q.max(), sample_q.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Gaussian')
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Standardized Sample Quantiles', fontsize=11, fontweight='bold')
    ax.set_title('Θ₀ Q-Q Plot', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 1])
    sorted_h0 = np.sort(h0_vals)
    n = len(sorted_h0)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_q = np.sqrt(2) * np.array([su.erfcinv(2 * (1 - q)) if q < 1 else 5 for q in quantiles])
    sample_q = (sorted_h0 - np.mean(h0_vals)) / np.std(h0_vals, ddof=1)
    ax.scatter(theoretical_q, sample_q, alpha=0.6, s=60, color='orange', edgecolor='black')
    lims = [min(theoretical_q.min(), sample_q.min()),
            max(theoretical_q.max(), sample_q.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Gaussian')
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Standardized Sample Quantiles', fontsize=11, fontweight='bold')
    ax.set_title('H₀ Q-Q Plot', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    table_data = [
        ['Statistic', 'Θ₀ (All 29)', 'H₀ (All 14)'],
        ['Mean', f'{np.mean(theta0_vals):.2f}', f'{np.mean(h0_vals):.2f}'],
        ['Median', f'{np.median(theta0_vals):.2f}', f'{np.median(h0_vals):.2f}'],
        ['Std Dev', f'{np.std(theta0_vals):.2f}', f'{np.std(h0_vals):.2f}'],
        ['Min', f'{np.min(theta0_vals):.2f}', f'{np.min(h0_vals):.2f}'],
        ['Max', f'{np.max(theta0_vals):.2f}', f'{np.max(h0_vals):.2f}'],
        ['Range', f'{np.max(theta0_vals) - np.min(theta0_vals):.2f}',
         f'{np.max(h0_vals) - np.min(h0_vals):.2f}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    fig.suptitle('Θ₀ vs H₀: Distribution Comparison and Non-Gaussianity Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    
    saved_name = save_figure('Fig09_theta0_h0_comparison.png')
    print(f"Saved {saved_name}")
    plt.close()

def fig10_results_table():
    """Save the central-estimate and KS result tables as a figure."""
    set_publication_style((14, 10))
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    old_vals, old_errs, _, _ = data.get_theta0_subset("Old")
    young_vals, young_errs, _, _ = data.get_theta0_subset("Young")
    
    results_all = ana.analyze_subset(theta0_vals, theta0_errs, "All")
    results_old = ana.analyze_subset(old_vals, old_errs, "Old")
    results_young = ana.analyze_subset(young_vals, young_errs, "Young")
    
    fig = plt.figure(figsize=(14, 10))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.axis('off')
    
    table2_data = [
        ['Tracers', 'N', 'Median (1σ CI)', 'Median (2σ CI)', 'Weighted Mean'],
        ['All', '29',
         f'{results_all["median"]:.2f} '
         f'+{results_all["median_ci_1sigma_upper"]:.2f}/'
         f'-{results_all["median_ci_1sigma_lower"]:.2f}',
         f'+{results_all["median_ci_2sigma_upper"]:.2f}/'
         f'-{results_all["median_ci_2sigma_lower"]:.2f}',
         f'{results_all["weighted_mean"]:.2f} ± {results_all["sigma_weighted_mean"]:.2f}'],
        ['Old', '19',
         f'{results_old["median"]:.2f} '
         f'+{results_old["median_ci_1sigma_upper"]:.2f}/'
         f'-{results_old["median_ci_1sigma_lower"]:.2f}',
         f'+{results_old["median_ci_2sigma_upper"]:.2f}/'
         f'-{results_old["median_ci_2sigma_lower"]:.2f}',
         f'{results_old["weighted_mean"]:.2f} ± {results_old["sigma_weighted_mean"]:.2f}'],
        ['Young', '11',
         f'{results_young["median"]:.2f} '
         f'+{results_young["median_ci_1sigma_upper"]:.2f}/'
         f'-{results_young["median_ci_1sigma_lower"]:.2f}',
         f'+{results_young["median_ci_2sigma_upper"]:.2f}/'
         f'-{results_young["median_ci_2sigma_lower"]:.2f}',
         f'{results_young["weighted_mean"]:.2f} ± {results_young["sigma_weighted_mean"]:.2f}'],
    ]
    
    table2 = ax1.table(cellText=table2_data, cellLoc='center', loc='center',
                      colWidths=[0.1, 0.05, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 3)
    
    for i in range(5):
        table2[(0, i)].set_facecolor('#40466e')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 4):
        for j in range(5):
            if i % 2 == 0:
                table2[(i, j)].set_facecolor('#f0f0f0')
            else:
                table2[(i, j)].set_facecolor('white')
    
    ax1.text(0.5, 1.15, 'Table 2: Central Estimates (Median Statistics)',
            transform=ax1.transAxes, fontsize=13, fontweight='bold',
            ha='center')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')
    
    table3_data = [
        ['Distribution', 'Scale Factor S', 'KS p-value', 'KS D-statistic'],
    ]
    
    for dist_name, dist in [('Gaussian', 'gaussian'), ('Cauchy', 'cauchy'),
                           ("Student's t", 'student_t'), ('Laplace', 'laplace')]:
        if results_old['fits_median'][dist] is not None:
            fit = results_old['fits_median'][dist]
            if 'n' in fit:
                s_str = f'{fit["s"]:.3f} (n={fit["n"]:.1f})'
            else:
                s_str = f'{fit["s"]:.3f}'
            table3_data.append([
                dist_name,
                s_str,
                f'{fit["p_value"]:.4f}',
                f'{fit["d_stat"]:.4f}'
            ])
        else:
            table3_data.append([dist_name, 'N/A', 'N/A', 'N/A'])
    
    table3 = ax2.table(cellText=table3_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.3, 0.25, 0.25])
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 3)
    
    for i in range(4):
        table3[(0, i)].set_facecolor('#40466e')
        table3[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table3_data)):
        for j in range(4):
            if i % 2 == 0:
                table3[(i, j)].set_facecolor('#f0f0f0')
            else:
                table3[(i, j)].set_facecolor('white')
    
    ax2.text(0.5, 1.15, 'Table 3: Distribution Fits - Old Tracers (N=19, Median Central Estimate)',
            transform=ax2.transAxes, fontsize=13, fontweight='bold',
            ha='center')
    
    plt.tight_layout()
    saved_name = save_figure('Fig10_results_table.png')
    print(f"Saved {saved_name}")
    plt.close()

def create_all_figures():
    """Create all project figures."""
    print("\nGenerating publication-quality figures with _new suffix...")
    print("="*80)
    
    fig01_theta0_overview()
    fig02_theta0_deviations()
    fig03_theta0_cdf()
    fig04_theta0_qq()
    fig05_theta0_bootstrap()
    fig06_theta0_model_comparison()
    fig07_theta0_kl()
    fig08_h0_comprehensive()
    fig09_theta0_h0_comparison()
    fig10_results_table()
    
    print("="*80)
    print("All figures generated successfully!")

if __name__ == '__main__':
    create_all_figures()
