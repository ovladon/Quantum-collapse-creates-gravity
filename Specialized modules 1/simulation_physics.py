#!/usr/bin/env python3
"""
Module: simulation_physics.py
Provides the physics and numerical routines for the 3D field simulation with quantum collapse events.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
import time

def gaussian_3d(X, Y, Z, x0, y0, z0, sigma):
    """Compute a normalized 3D Gaussian."""
    norm = (2 * np.pi * sigma**2) ** (3/2)
    return np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2)) / norm

def laplacian_3D(field, dx):
    """Compute the 3D Laplacian of a field using finite differences."""
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / (dx**2)

def solve_poisson_3D(rho, dx, G, relativistic_factor=0.0):
    """
    Solve the 3D Poisson equation for gravitational potential using FFT.
    The potential is scaled by (1 + relativistic_factor) as a crude correction.
    """
    shape = rho.shape
    rho_k = fftn(rho)
    kx = 2 * np.pi * fftfreq(shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(shape[1], d=dx)
    kz = 2 * np.pi * fftfreq(shape[2], d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    # Avoid division by zero at k=0
    k_squared[0, 0, 0] = 1.0
    phi_k = -4 * np.pi * G * rho_k / k_squared
    phi_k[0, 0, 0] = 0.0
    phi = np.real(ifftn(phi_k))
    phi *= (1 + relativistic_factor)
    return phi

def compute_power_spectrum_3D(field):
    """Compute the radially averaged power spectral density of a 3D field."""
    F = fftn(field)
    F = np.fft.fftshift(F)
    psd = np.abs(F)**2
    shape = field.shape
    grid = np.indices(shape)
    grid = np.stack(grid, axis=-1)
    center = np.array([s//2 for s in shape])
    r = np.sqrt(np.sum((grid - center)**2, axis=-1))
    r_flat = r.flatten().astype(int)
    psd_flat = psd.flatten()
    radial_bins = np.bincount(r_flat, weights=psd_flat)
    counts = np.bincount(r_flat)
    radial_prof = radial_bins / (counts + 1e-8)
    return radial_prof

def estimate_noise_exponent(psd, fit_range=(1, 20)):
    """
    Estimate the noise exponent (slope) by fitting a line in log-log space
    over the given range of bins.
    """
    bins = np.arange(len(psd))
    idx = (bins >= fit_range[0]) & (bins < fit_range[1])
    x = np.log10(bins[idx] + 1e-12)
    y = np.log10(psd[idx] + 1e-12)
    if len(x) < 2:
        return None, None
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    return slope, intercept

def run_field_simulation_3D(params, snapshot_interval=50):
    """
    Run a 3D field simulation with collapse dynamics.

    params: dictionary containing keys:
       collapse_rate, collapse_sigma, collapse_amplitude, continuous_noise_amplitude,
       density_decay, relativistic_factor, G, L, N, steps_per_cycle, num_cycles, dt, m

    snapshot_interval: interval (in time steps) to save snapshots of the field.
    
    Returns a dictionary containing:
       - slope: estimated noise exponent from the gravitational potential.
       - intercept: intercept from the power spectrum fit.
       - final_field: final field configuration.
       - phi_grav: gravitational potential computed from energy density.
       - snapshots: list of field snapshots (for visualization).
       - runtime: total simulation time in seconds.
       - params: the simulation parameters used.
    """
    # Unpack simulation parameters
    collapse_rate = params["collapse_rate"]
    collapse_sigma = params["collapse_sigma"]
    collapse_amplitude = params["collapse_amplitude"]
    continuous_noise_amplitude = params["continuous_noise_amplitude"]
    density_decay = params["density_decay"]
    relativistic_factor = params["relativistic_factor"]
    G = params["G"]
    L = params["L"]
    N = params["N"]
    steps_per_cycle = params["steps_per_cycle"]
    num_cycles = params["num_cycles"]
    dt = params["dt"]
    m = params["m"]
    
    T = steps_per_cycle * num_cycles
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Initialize fields with small random fluctuations
    phi_prev = 0.01 * np.random.randn(N, N, N)
    phi = 0.01 * np.random.randn(N, N, N)
    
    snapshots = []
    start_time = time.time()
    
    for t in range(T):
        phi_avg = np.mean(phi)
        noise = continuous_noise_amplitude * np.random.randn(N, N, N)
        noise = gaussian_filter(noise, sigma=collapse_sigma/dx)
        laplacian_phi = laplacian_3D(phi, dx)
        
        # Add discrete collapse events as Gaussian deposits
        num_events = np.random.poisson(lam=collapse_rate)
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            phi += collapse_amplitude * gaussian_3d(X, Y, Z,
                                                    event_pos[0],
                                                    event_pos[1],
                                                    event_pos[2],
                                                    collapse_sigma)
        
        # Time-step using a finite-difference update
        phi_next = (2 * phi - phi_prev + dt**2 * (
            laplacian_phi - m**2 * phi - collapse_rate * (phi - phi_avg) + np.sqrt(collapse_rate) * noise
        ))
        phi_prev = phi
        phi = phi_next
        
        # Apply density decay to simulate energy loss
        phi *= density_decay
        
        # Save a snapshot at the specified interval
        if (t % snapshot_interval == 0):
            snapshots.append(phi.copy())
    
    runtime = time.time() - start_time
    
    # Compute effective energy density
    phi_t = (phi - phi_prev) / dt
    grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * dx)
    energy_density = 0.5 * (phi_t**2 + grad_x**2 + grad_y**2 + grad_z**2 + m**2 * phi**2)
    
    # Solve the Poisson equation for gravitational potential
    phi_grav = solve_poisson_3D(energy_density, dx, G, relativistic_factor)
    
    # Compute the power spectrum and extract the noise exponent (slope)
    psd = compute_power_spectrum_3D(phi_grav)
    slope, intercept = estimate_noise_exponent(psd, fit_range=(1, int(0.2 * N)))
    
    results = {
        "slope": slope,
        "intercept": intercept,
        "final_field": phi,
        "phi_grav": phi_grav,
        "snapshots": snapshots,
        "runtime": runtime,
        "params": params
    }
    
    return results

