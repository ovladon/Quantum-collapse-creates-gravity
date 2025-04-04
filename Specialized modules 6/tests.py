#!/usr/bin/env python3
"""
Module: tests.py
Contains basic unit tests for key functions in the simulation package.
Run with: python3 tests.py
"""

import numpy as np
from simulation import (laplacian_3D, compute_power_spectrum_3D, estimate_noise_exponent,
                        compute_total_energy, gaussian_3d)
from simulation import SimulationParams, run_field_simulation_3D

def test_laplacian():
    # Test Laplacian on a simple quadratic function which has constant second derivative.
    N = 32
    L = 10.0
    x = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    field = X**2 + Y**2 + Z**2
    dx = L / N
    lap = laplacian_3D(field, dx, boundary_condition="dirichlet")
    # For f(x)=x^2, second derivative = 2; sum over 3 dims gives 6
    expected = 6 * np.ones_like(field)
    # Only test inner region to avoid boundary issues
    error = np.abs(lap[1:-1,1:-1,1:-1] - expected[1:-1,1:-1,1:-1]).mean()
    assert error < 1e-1, f"Laplacian error too high: {error}"
    print("test_laplacian passed.")

def test_power_spectrum():
    field = np.random.randn(32,32,32)
    psd = compute_power_spectrum_3D(field)
    assert psd.ndim == 1, "PSD should be a 1D array."
    print("test_power_spectrum passed.")

def test_noise_exponent():
    psd = np.linspace(1,100,50)
    slope, intercept, slope_std_error = estimate_noise_exponent(psd, fit_range=(1,40))
    assert slope is not None, "Slope should not be None."
    print("test_noise_exponent passed.")

def test_simulation_run():
    params = {
        "collapse_rate": 0.3,
        "collapse_sigma": 0.2,
        "collapse_amplitude": 1.0,
        "continuous_noise_amplitude": 0.01,
        "density_decay": 0.98,
        "relativistic_factor": 0.005,
        "G": 1.0,
        "L": 10.0,
        "N": 32,
        "steps_per_cycle": 10,
        "num_cycles": 1,
        "dt": 0.05,
        "m": 1.0,
        "boundary_condition": "periodic"
    }
    result = run_field_simulation_3D(params, snapshot_interval=10)
    assert "slope" in result, "Result should contain slope."
    print("test_simulation_run passed.")

if __name__ == "__main__":
    test_laplacian()
    test_power_spectrum()
    test_noise_exponent()
    test_simulation_run()
    print("All tests passed.")

