#!/usr/bin/env python3
"""
This script processes LIGO LOSC HDF5 data to extract a noise exponent.
Steps performed:
  1. Loads the HDF5 file using h5py and extracts the strain data and the sampling interval (dt).
     If the attribute "Xspacing" is missing, a default dt (1/4096 sec) is used.
  2. Computes the power‐spectral density (PSD) of the strain using Welch’s method.
  3. Selects a frequency range (default 20–200 Hz) and fits a line in log–log space to obtain the noise exponent (slope) and intercept.
  4. Saves the extracted measurements (e.g. Target slope, intercept, mean PSD) into a CSV file for later comparison.
  5. Also saves a plot (psd_fit.png) showing the PSD and the fitted line.
  
Usage:
    python3 process_ligo_data.py <path_to_hdf5_file> --fmin 20 --fmax 200 --output observations.csv
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import logging
from scipy.signal import welch

logging.basicConfig(level=logging.INFO)

DEFAULT_DT = 1/4096.0  # Default sampling interval (in seconds) if Xspacing is not found.

def load_ligo_strain(filename):
    """
    Load the strain data and sampling interval from a LOSC HDF5 file.
    
    Parameters:
      filename: Path to the HDF5 file.
    
    Returns:
      strain: The strain time series as a NumPy array.
      dt: The sampling interval (in seconds).
    """
    with h5py.File(filename, 'r') as f:
        if 'strain' in f:
            strain = f['strain/Strain'][:]
            # Try to get 'Xspacing' from the attributes of the "strain" group.
            dt = f['strain'].attrs.get('Xspacing')
            if dt is None:
                logging.warning("Attribute 'Xspacing' not found in file. Using default dt = 1/4096 sec.")
                dt = DEFAULT_DT
            return strain, dt
        else:
            raise ValueError("No 'strain' group found in file.")

def compute_psd(strain, dt):
    """
    Compute the power spectral density (PSD) using Welch's method.
    
    Parameters:
      strain: The strain time series.
      dt: The sampling interval.
    
    Returns:
      frequencies: Frequency array.
      psd: PSD values.
    """
    fs = 1.0 / dt
    frequencies, psd = welch(strain, fs=fs, nperseg=1024)
    return frequencies, psd

def fit_noise_exponent(frequencies, psd, fmin=30, fmax=150):
    """
    Fit a line to the PSD in log–log space over the frequency range [fmin, fmax].
    
    Parameters:
      frequencies: Frequency array.
      psd: PSD values.
      fmin: Minimum frequency for fitting.
      fmax: Maximum frequency for fitting.
    
    Returns:
      slope: The noise exponent.
      intercept: Intercept of the fit.
    """
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    if np.sum(mask) < 2:
        raise ValueError("Not enough data points in the selected frequency range.")
    log_f = np.log10(frequencies[mask])
    log_psd = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_psd, 1)
    return slope, intercept

def export_csv(measurements, output_csv):
    """
    Export the measurements dictionary to a CSV file.
    
    Parameters:
      measurements: Dictionary of measurement names and values.
      output_csv: Filename for the CSV output.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Measurement", "Value"])
        for key, value in measurements.items():
            writer.writerow([key, value])
    logging.info(f"CSV exported to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Process LIGO LOSC HDF5 data and export comparison CSV.")
    parser.add_argument("filename", type=str, help="Path to the HDF5 file (e.g., L-L1_LOSC_16_V1-1126259446-32.hdf5)")
    parser.add_argument("--fmin", type=float, default=20, help="Minimum frequency for fitting (Hz)")
    parser.add_argument("--fmax", type=float, default=200, help="Maximum frequency for fitting (Hz)")
    parser.add_argument("--output", type=str, default="observations.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load strain data and dt
    strain, dt = load_ligo_strain(args.filename)
    logging.info(f"Loaded strain data with {len(strain)} samples, dt = {dt}")

    # Compute PSD using Welch's method
    frequencies, psd = compute_psd(strain, dt)
    logging.info(f"Computed PSD with {len(frequencies)} frequency bins.")

    # Fit the noise exponent (slope) in the specified frequency range
    slope, intercept = fit_noise_exponent(frequencies, psd, fmin=args.fmin, fmax=args.fmax)
    logging.info(f"Fitted noise exponent: slope = {slope:.3f}, intercept = {intercept:.3f}")

    # Plot PSD and fitted line
    plt.figure()
    plt.loglog(frequencies, psd, label="PSD")
    mask = (frequencies >= args.fmin) & (frequencies <= args.fmax)
    plt.loglog(frequencies[mask], 10**(intercept + slope*np.log10(frequencies[mask])), 'r--', label="Fit")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.title("PSD and Noise Exponent Fit")
    plt.savefig("psd_fit.png")
    plt.close()
    logging.info("PSD plot saved as psd_fit.png")

    # Prepare measurements for CSV export
    measurements = {
        "Target slope": slope,
        "Intercept": intercept,
        "Mean PSD": np.mean(psd)
    }
    export_csv(measurements, args.output)

if __name__ == "__main__":
    main()

