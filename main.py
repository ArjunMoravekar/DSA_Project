#!/usr/bin/env python
"""Run the Θ₀ reproduction and H₀ comparison workflow."""

import numpy as np
import sys
import time
from datetime import datetime

import data
import analysis as ana
import plotting

def print_header():
    """Print a short run header."""
    print("\n" + "="*80)
    print("GALACTIC ROTATION VELOCITY AND HUBBLE CONSTANT ANALYSIS")
    print("="*80)
    print(f"Camarillo, Dredger & Ratra (2018) - arXiv:1805.01917")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

def main():
    """Run the full analysis pipeline."""
    
    print_header()
    start_time = time.time()
    
    print("STEP 1: LOADING DATA")
    print("-" * 80)
    
    theta0_vals, theta0_errs = data.get_theta0_values()
    print(f"Loaded {len(theta0_vals)} Θ₀ measurements (rescaled to R₀ = 7.96 kpc)")
    print(f"  Range: {np.min(theta0_vals):.2f} - {np.max(theta0_vals):.2f} km/s")
    print(f"  Errors: {np.min(theta0_errs):.2f} - {np.max(theta0_errs):.2f} km/s")
    
    h0_vals, h0_errs = data.get_h0_values()
    print(f"\nLoaded {len(h0_vals)} H₀ measurements")
    print(f"  Range: {np.min(h0_vals):.2f} - {np.max(h0_vals):.2f} km/s/Mpc")
    print(f"  Errors: {np.min(h0_errs):.2f} - {np.max(h0_errs):.2f} km/s/Mpc")
    
    print("\n" + "="*80)
    print("STEP 2: ANALYZING ALL TRACERS (N=29)")
    print("="*80)
    
    results_all = ana.analyze_subset(theta0_vals, theta0_errs, "All Tracers (N=29)")
    ana.print_analysis_results(results_all)
    
    print("\n" + "="*80)
    print("STEP 3: ANALYZING OLD TRACERS (N=19)")
    print("="*80)
    
    old_vals, old_errs, old_refs, old_types = data.get_theta0_subset("Old")
    results_old = ana.analyze_subset(old_vals, old_errs, "Old Tracers (N=19)")
    ana.print_analysis_results(results_old)
    
    print("\n" + "="*80)
    print("STEP 4: ANALYZING YOUNG TRACERS (N=11)")
    print("="*80)
    
    young_vals, young_errs, young_refs, young_types = data.get_theta0_subset("Young")
    results_young = ana.analyze_subset(young_vals, young_errs, "Young Tracers (N=11)")
    ana.print_analysis_results(results_young)
    
    print("\n" + "="*80)
    print("TABLE 2: CENTRAL ESTIMATES SUMMARY")
    print("="*80)
    print(f"\n{'Tracers':<20} {'N':<5} {'Median (1σ)':<30} {'Weighted Mean':<25}")
    print("-" * 80)
    
    for name, results in [("All Tracers", results_all),
                          ("Old Tracers", results_old),
                          ("Young Tracers", results_young)]:
        median_str = f"{results['median']:.2f} +{results['median_ci_1sigma_upper']:.2f}/-{results['median_ci_1sigma_lower']:.2f}"
        wm_str = f"{results['weighted_mean']:.2f} ± {results['sigma_weighted_mean']:.2f}"
        n_str = results['n_measurements']
        print(f"{name:<20} {n_str:<5} {median_str:<30} {wm_str:<25}")
    
    print("\n" + "="*80)
    print("TABLE 3: DISTRIBUTION GOODNESS-OF-FIT TESTS")
    print("Median Central Estimate - Old Tracers (N=19)")
    print("="*80)
    print(f"\n{'Distribution':<15} {'Scale S':<12} {'KS p-value':<12} {'KS D-stat':<12}")
    print("-" * 80)
    
    for dist_name, dist in [("Gaussian", "gaussian"), ("Cauchy", "cauchy"),
                           ("Student-t", "student_t"), ("Laplace", "laplace")]:
        if results_old['fits_median'][dist] is not None:
            fit = results_old['fits_median'][dist]
            s_str = f"{fit['s']:.3f}"
            if 'n' in fit:
                s_str += f" (n={fit['n']:.0f})"
            print(f"{dist_name:<15} {s_str:<12} {fit['p_value']:<12.4f} {fit['d_stat']:<12.4f}")
    
    print("\n" + "="*80)
    print("STEP 5: GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    
    try:
        plotting.create_all_figures()
        print("\nAll figures generated successfully!")
    except Exception as e:
        print(f"\nWarning: Error generating figures: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("COMPARISON WITH PAPER (Table 2 and Table 3)")
    print("="*80)
    
    print("\nAll Tracers (N=29) - Expected from paper:")
    print("  Median: 226.35 (+2.13/-2.89 1σ, +4.50/-8.44 2σ)")
    print("  Weighted Mean: 224.36 ± 1.67")
    print(f"  Our Results:")
    print(f"  Median: {results_all['median']:.2f} "
          f"(+{results_all['median_ci_1sigma_upper']:.2f}/"
          f"-{results_all['median_ci_1sigma_lower']:.2f} 1σ)")
    print(f"  Weighted Mean: {results_all['weighted_mean']:.2f} ± {results_all['sigma_weighted_mean']:.2f}")
    
    print("\nOld Tracers (N=19) - Expected from paper:")
    print("  Median: 219.70 (+6.67/-7.43 1σ)")
    print("  Weighted Mean: 222.01 ± 1.99")
    print(f"  Our Results:")
    print(f"  Median: {results_old['median']:.2f} "
          f"(+{results_old['median_ci_1sigma_upper']:.2f}/"
          f"-{results_old['median_ci_1sigma_lower']:.2f} 1σ)")
    print(f"  Weighted Mean: {results_old['weighted_mean']:.2f} ± {results_old['sigma_weighted_mean']:.2f}")
    
    print("\nYoung Tracers (N=11) - Expected from paper:")
    print("  Median: 228.85 (+3.98/-2.33 1σ)")
    print("  Weighted Mean: 230.05 ± 3.09")
    print(f"  Our Results:")
    print(f"  Median: {results_young['median']:.2f} "
          f"(+{results_young['median_ci_1sigma_upper']:.2f}/"
          f"-{results_young['median_ci_1sigma_lower']:.2f} 1σ)")
    print(f"  Weighted Mean: {results_young['weighted_mean']:.2f} ± {results_young['sigma_weighted_mean']:.2f}")
    
    print("\nOld Tracers Distribution Fits (paper Table 3):")
    print("  Expected (from paper):")
    print("    Gaussian: p=0.83, S=1.08")
    print("    Cauchy: p=0.71, S=0.70")
    print("    Student-t: p=0.83, S=1.07, n=36")
    print("    Laplace: p=0.76, S=1.04")
    print("  Our Results:")
    for dist_name, dist in [("Gaussian", "gaussian"), ("Cauchy", "cauchy"),
                           ("Student-t", "student_t"), ("Laplace", "laplace")]:
        if results_old['fits_median'][dist] is not None:
            fit = results_old['fits_median'][dist]
            if 'n' in fit:
                print(f"    {dist_name}: p={fit['p_value']:.2f}, S={fit['s']:.2f}, n={fit['n']:.0f}")
            else:
                print(f"    {dist_name}: p={fit['p_value']:.2f}, S={fit['s']:.2f}")
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("="*80 + "\n")
    
    return results_all, results_old, results_young

if __name__ == '__main__':
    try:
        results_all, results_old, results_young = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
