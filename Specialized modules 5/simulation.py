#!/usr/bin/env python3
"""
Module: simulation.py
Provides the physics and numerical routines for the 3D field simulation with GRW collapse events.
Implements a specific collapse model based on GRW theory.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
import time
import logging
from dataclasses import dataclass, asdict

@dataclass
class SimulationParams:
    collapse_rate: float
    collapse_sigma: float
    collapse_amplitude: float
    continuous_noise_amplitude: float
    density_decay: float
    relativistic_factor: float
    G: float
    L: float
    N: int
    steps_per_cycle: int
    num_cycles: int
    dt: float
    m: float
    boundary_condition: str = "periodic"

def gaussian_3d(X, Y, Z, x0, y0, z0, sigma):
    """
    Compute a normalized 3D Gaussian function centered at (x0, y0, z0) with width sigma.
    """
    norm = (2 * np.pi * sigma**2) ** (3/2)
    return np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2)) / norm

def laplacian_3D(field, dx, boundary_condition="periodic"):
    """
    Compute the 3D Laplacian of a field using finite differences.
    """
    if boundary_condition.lower() == "periodic":
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        ) / (dx**2)
    elif boundary_condition.lower() == "dirichlet":
        lap = np.zeros_like(field)
        lap[1:-1,1:-1,1:-1] = (
            field[2:,1:-1,1:-1] + field[:-2,1:-1,1:-1] +
            field[1:-1,2:,1:-1] + field[1:-1,:-2,1:-1] +
            field[1:-1,1:-1,2:] + field[1:-1,1:-1,:-2] -
            6 * field[1:-1,1:-1,1:-1]
        ) / (dx**2)
        return lap
    else:
        raise ValueError("Unsupported boundary condition. Choose 'periodic' or 'dirichlet'.")

def solve_poisson_3D(rho, dx, G, relativistic_factor=0.0):
    """
    Solve the 3D Poisson equation for gravitational potential using FFT with periodic boundaries.
    """
    shape = rho.shape
    rho_k = fftn(rho)
    kx = 2 * np.pi * fftfreq(shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(shape[1], d=dx)
    kz = 2 * np.pi * fftfreq(shape[2], d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1.0  # Avoid division by zero
    phi_k = -4 * np.pi * G * rho_k / k_squared
    phi_k[0, 0, 0] = 0.0
    phi = np.real(ifftn(phi_k))
    phi *= (1 + relativistic_factor)
    return phi

def compute_power_spectrum_3D(field):
    """
    Compute the radially averaged power spectral density (PSD) of a 3D field.
    """
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
    Estimate the noise exponent (slope) by fitting a line in log-log space over the given range of bins.
    """
    bins = np.arange(len(psd))
    idx = (bins >= fit_range[0]) & (bins < fit_range[1])
    x = np.log10(bins[idx] + 1e-12)
    y = np.log10(psd[idx] + 1e-12)
    if len(x) < 2:
        return None, None, None
    A = np.vstack([x, np.ones(len(x))]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    slope = coeffs[0]
    intercept = coeffs[1]
    if len(x) > 2 and residuals.size > 0:
        residual_std = np.sqrt(residuals[0] / (len(x) - 2))
        sxx = np.sum((x - np.mean(x))**2)
        slope_std_error = residual_std / np.sqrt(sxx)
    else:
        slope_std_error = None
    return slope, intercept, slope_std_error

def compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition="periodic"):
    """
    Compute the total energy of the field (kinetic + gradient + potential energy).
    """
    phi_t = (phi - phi_prev) / dt
    kinetic = 0.5 * phi_t**2
    if boundary_condition.lower() == "periodic":
        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
        grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * dx)
    elif boundary_condition.lower() == "dirichlet":
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        grad_x[1:-1,:,:] = (phi[2:,:,:] - phi[:-2,:,:]) / (2 * dx)
        grad_y[:,1:-1,:] = (phi[:,2:,:] - phi[:,:-2,:]) / (2 * dx)
        grad_z[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,:-2]) / (2 * dx)
    else:
        raise ValueError("Unsupported boundary condition.")
    grad_energy = 0.5 * (grad_x**2 + grad_y**2 + grad_z**2)
    potential_energy = 0.5 * m**2 * phi**2
    total_energy = np.sum(kinetic + grad_energy + potential_energy) * (dx**3)
    return total_energy

def run_field_simulation_3D(params, snapshot_interval=50):
    """
    Run a 3D field simulation with GRW collapse dynamics and symplectic integration.
    Returns a dictionary with simulation results.
    """
    if not isinstance(params, dict):
        params = asdict(params)
    
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
    boundary_condition = params.get("boundary_condition", "periodic")
    
    T = steps_per_cycle * num_cycles
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    phi_prev = 0.01 * np.random.randn(N, N, N)
    phi = 0.01 * np.random.randn(N, N, N)
    
    energy_initial = compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition)
    
    snapshots = []
    start_time = time.time()
    
    for t in range(T):
        phi_avg = np.mean(phi)
        noise = continuous_noise_amplitude * np.random.randn(N, N, N)
        noise = gaussian_filter(noise, sigma=collapse_sigma/dx)
        lap_phi = laplacian_3D(phi, dx, boundary_condition)
        
        num_events = np.random.poisson(lam=collapse_rate)
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            phi += collapse_amplitude * gaussian_3d(X, Y, Z,
                                                    event_pos[0],
                                                    event_pos[1],
                                                    event_pos[2],
                                                    collapse_sigma)
        acceleration = lap_phi - m**2 * phi - collapse_rate * (phi - phi_avg) + np.sqrt(collapse_rate) * noise
        phi_next = 2 * phi - phi_prev + dt**2 * acceleration
        
        phi_prev = phi.copy()
        phi = phi_next.copy()
        phi *= density_decay
        
        if t % snapshot_interval == 0:
            snapshots.append(phi.copy())
    
    runtime = time.time() - start_time
    energy_final = compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition)
    energy_error = np.abs(energy_final - energy_initial) / (energy_initial + 1e-12)
    
    if boundary_condition.lower() == "periodic":
        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dx)
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)
        grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2*dx)
    elif boundary_condition.lower() == "dirichlet":
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        grad_x[1:-1,:,:] = (phi[2:,:,:] - phi[:-2,:,:]) / (2*dx)
        grad_y[:,1:-1,:] = (phi[:,2:,:] - phi[:,:-2,:]) / (2*dx)
        grad_z[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,:-2]) / (2*dx)
    
    energy_density = 0.5 * ( ((phi - phi_prev)/dt)**2 + grad_x**2 + grad_y**2 + grad_z**2 + m**2 * phi**2 )
    phi_grav = solve_poisson_3D(energy_density, dx, G, relativistic_factor)
    
    psd = compute_power_spectrum_3D(phi_grav)
    slope, intercept, slope_std_error = estimate_noise_exponent(psd, fit_range=(1, int(0.2 * N)))
    
    logging.info(f"Simulation completed in {runtime:.2f} seconds.")
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_std_error": slope_std_error,
        "energy_initial": energy_initial,
        "energy_final": energy_final,
        "energy_error": energy_error,
        "final_field": phi,
        "phi_grav": phi_grav,
        "snapshots": snapshots,
        "runtime": runtime,
        "params": params
    }

def convergence_test(base_params, resolutions, time_steps, ensemble_runs=3):
    """
    Perform convergence tests by running the simulation at different grid resolutions (N) and time steps (dt).
    """
    results = []
    for N in resolutions:
        for dt in time_steps:
            slopes = []
            energy_errors = []
            runtimes = []
            for _ in range(ensemble_runs):
                params = base_params.copy()
                params["N"] = N
                params["dt"] = dt
                sim_result = run_field_simulation_3D(params, snapshot_interval=params["steps_per_cycle"])
                if sim_result["slope"] is not None:
                    slopes.append(sim_result["slope"])
                    energy_errors.append(sim_result["energy_error"])
                    runtimes.append(sim_result["runtime"])
            if slopes:
                avg_slope = np.mean(slopes)
                std_slope = np.std(slopes)
                avg_energy_error = np.mean(energy_errors)
                std_energy_error = np.std(energy_errors)
                avg_runtime = np.mean(runtimes)
            else:
                avg_slope = std_slope = avg_energy_error = std_energy_error = avg_runtime = None
            results.append({
                "N": N,
                "dt": dt,
                "avg_slope": avg_slope,
                "std_slope": std_slope,
                "avg_energy_error": avg_energy_error,
                "std_energy_error": std_energy_error,
                "avg_runtime": avg_runtime
            })
            logging.info(f"Convergence test for N={N}, dt={dt}: avg_slope={avg_slope}, std_slope={std_slope}, avg_energy_error={avg_energy_error}")
    return results

