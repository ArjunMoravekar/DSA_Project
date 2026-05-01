"""Data tables for the Θ₀ reproduction and the H₀ comparison."""

import numpy as np
import pandas as pd

# Rescaled Θ₀ values from Camarillo, Dredger & Ratra (2018), Table 1.
MEASUREMENTS_EXACT = [
    (240.87, 20.59, "Branham (2014)", "Young"),
    (214.15, 10.09, "Battinelli et al. (2013)", "Old"),
    (226.09, 15.63, "Bobylev (2013)", "Young"),
    (226.52, 7.69, "McMillan (2017)", "Both"),
    (237.94, 26.76, "Shen & Zhang (2010)", "Young"),
    (219.70, 14.32, "Bedin et al. (2003)", "Old"),
    (201.69, 27.95, "Kalirai et al. (2004)", "Old"),
    (234.82, 21.52, "Reid & Brunthaler (2004)", "Old"),
    (207.46, 20.39, "Xue et al. (2008)", "Old"),
    (242.28, 13.93, "Yuan et al. (2008)", "Old"),
    (225.71, 4.82, "Sharma et al. (2011)", "Old"),
    (216.91, 10.98, "Bovy & Rix (2013)", "Old"),
    (232.83, 18.82, "Bobylev & Bajkova (2015a)", "Young"),
    (228.85, 19.43, "Bobylev & Bajkova (2015b)", "Young"),
    (234.82, 5.02, "Aumer & Schönrich (2015)", "Young"),
    (226.36, 7.30, "McGaugh (2016)", "Old"),
    (208.95, 10.90, "Rojas-Arriagada et al. (2016)", "Old"),
    (228.85, 17.24, "Bobylev & Bajkova (2016a)", "Young"),
    (229.85, 9.63, "Bobylev (2017)", "Young"),
    (217.91, 10.71, "Bobylev & Bajkova (2017)", "Young"),
    (200.74, 12.48, "Avedisova (2005)", "Old"),
    (208.71, 29.67, "Nikiforov (2000)", "Old"),
    (231.03, 4.93, "Portail et al. (2017)", "Old"),
    (228.46, 9.12, "Rastorguev et al. (2017)", "Old"),
    (223.46, 13.66, "Küpper et al. (2015)", "Old"),
    (226.33, 9.29, "Bobylev et al. (2016b)", "Young"),
    (229.06, 8.72, "Huang et al. (2016)", "Old"),
    (212.27, 16.22, "Koposov et al. (2010)", "Old"),
    (211.64, 4.52, "Martínez-Barbosa et al. (2017)", "Old"),
]

theta0_values = np.array([m[0] for m in MEASUREMENTS_EXACT])
theta0_errors = np.array([m[1] for m in MEASUREMENTS_EXACT])
references = [m[2] for m in MEASUREMENTS_EXACT]
tracer_types = [m[3] for m in MEASUREMENTS_EXACT]

THETA0_DATA = {
    'measurement_id': np.arange(1, len(MEASUREMENTS_EXACT) + 1),
    'theta0_km_s': theta0_values,
    'sigma_km_s': theta0_errors,
    'reference': references,
    'tracer_type': tracer_types,
}

def get_theta0_dataframe():
    """Return Θ₀ data as a pandas DataFrame."""
    return pd.DataFrame(THETA0_DATA)

def get_theta0_values():
    """Return Θ₀ values and uncertainties as numpy arrays."""
    return THETA0_DATA['theta0_km_s'].copy(), THETA0_DATA['sigma_km_s'].copy()

def get_theta0_subset(tracer_type):
    """Return the Old or Young Θ₀ subset, including Both in either subset."""
    if tracer_type == "Old":
        idx = np.array([i for i, tt in enumerate(tracer_types) if tt in ["Old", "Both"]])
    elif tracer_type == "Young":
        idx = np.array([i for i, tt in enumerate(tracer_types) if tt in ["Young", "Both"]])
    else:
        raise ValueError("tracer_type must be 'Old' or 'Young'")
    
    return (
        theta0_values[idx].copy(),
        theta0_errors[idx].copy(),
        [references[i] for i in idx],
        [tracer_types[i] for i in idx]
    )

# H₀ values used only as an illustrative extension dataset.
H0_MEASUREMENTS_EXACT = [
    (67.4, 0.5, "Planck (2018)", 2018),
    (73.2, 1.3, "Riess et al. (2021) SH0ES", 2021),
    (69.8, 1.9, "Freedman et al. (2019) TRGB", 2019),
    (67.4, 1.2, "DES (2018)", 2018),
    (73.3, 1.7, "Riess et al. (2016)", 2016),
    (66.93, 0.62, "Planck (2015)", 2015),
    (69.6, 2.0, "Bennett et al. (2014) WMAP9", 2014),
    (73.8, 2.4, "Freedman et al. (2012)", 2012),
    (74.3, 2.1, "Riess et al. (2011)", 2011),
    (70.6, 3.1, "Suyu et al. (2010) Lensing", 2010),
    (71.0, 2.5, "Bonvin et al. (2017) H0LiCOW", 2017),
    (72.5, 2.1, "Wong et al. (2020) H0LiCOW", 2020),
    (69.0, 1.2, "Aubourg et al. (2015) BAO", 2015),
    (67.6, 0.6, "eBOSS (2020)", 2020),
]

h0_values = np.array([m[0] for m in H0_MEASUREMENTS_EXACT])
h0_errors = np.array([m[1] for m in H0_MEASUREMENTS_EXACT])
h0_references = [m[2] for m in H0_MEASUREMENTS_EXACT]
h0_years = np.array([m[3] for m in H0_MEASUREMENTS_EXACT])

H0_DATA = {
    'measurement_id': np.arange(1, len(H0_MEASUREMENTS_EXACT) + 1),
    'h0_km_s_Mpc': h0_values,
    'sigma_km_s_Mpc': h0_errors,
    'reference': h0_references,
    'year': h0_years,
}

def get_h0_dataframe():
    """Return H₀ data as a pandas DataFrame."""
    return pd.DataFrame(H0_DATA)

def get_h0_values():
    """Return H₀ values and uncertainties as numpy arrays."""
    return H0_DATA['h0_km_s_Mpc'].copy(), H0_DATA['sigma_km_s_Mpc'].copy()

if __name__ == '__main__':
    print("="*70)
    print("GALACTIC ROTATIONAL VELOCITY (Θ₀) DATA")
    print("Camarillo, Dredger & Ratra (2018) arXiv:1805.01917")
    print("All 29 measurements, rescaled to R₀ = 7.96 ± 0.17 kpc")
    print("="*70)
    df = get_theta0_dataframe()
    print(df)
    print(f"\nTotal measurements: {len(THETA0_DATA['theta0_km_s'])}")
    print(f"Mean: {np.mean(THETA0_DATA['theta0_km_s']):.2f} km/s")
    print(f"Median: {np.median(THETA0_DATA['theta0_km_s']):.2f} km/s")
    print(f"Std Dev: {np.std(THETA0_DATA['theta0_km_s']):.2f} km/s")
    
    print("\n" + "="*70)
    print("SUMMARY BY TRACER TYPE:")
    print("="*70)
    for tt in ["Old", "Young"]:
        vals, errs, refs, types = get_theta0_subset(tt)
        print(f"\n{tt} Tracers ({len(vals)} measurements, including 'Both' type):")
        print(f"  Mean: {np.mean(vals):.2f} km/s")
        print(f"  Median: {np.median(vals):.2f} km/s")
        print(f"  Std Dev: {np.std(vals):.2f} km/s")
    
    print("\n" + "="*70)
    print("HUBBLE CONSTANT (H₀) DATA - Extension Dataset")
    print("="*70)
    h0_df = get_h0_dataframe()
    print(h0_df)
    print(f"\nTotal measurements: {len(H0_DATA['h0_km_s_Mpc'])}")
    print(f"Mean: {np.mean(H0_DATA['h0_km_s_Mpc']):.2f} km/s/Mpc")
    print(f"Median: {np.median(H0_DATA['h0_km_s_Mpc']):.2f} km/s/Mpc")
    print(f"Std Dev: {np.std(H0_DATA['h0_km_s_Mpc']):.2f} km/s/Mpc")
