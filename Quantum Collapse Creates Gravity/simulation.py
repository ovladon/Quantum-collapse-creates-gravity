#!/usr/bin/env python3
"""
Module: simulation.py

This module implements a 3D field simulation with GRW collapse dynamics.
It provides two integration routines:
  1. run_field_simulation_3D_base(): the original (fixed time-step) integrator (fast).
  2. run_field_simulation_3D_adaptive(): an adaptive time-stepping version with ETA reporting.
  
Additional helper functions (Laplacian, energy computation, etc.) and routines for convergence
and sensitivity analysis are included.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
import time
import logging
import psutil
from dataclasses import dataclass, asdict
import itertools

try:
    from numba import njit, prange
except ImportError:
    def njit(func):
        return func
    def prange(x):
        return range(x)

# ---------------------------
# Data Class for Parameters
# ---------------------------
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

# ---------------------------
# Helper Functions
# ---------------------------
def gaussian_3d(X, Y, Z, x0, y0, z0, sigma):
    norm = (2 * np.pi * sigma**2) ** (3/2)
    return np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2)) / norm

def laplacian_3D(field, dx, boundary_condition="periodic"):
    if boundary_condition.lower() == "periodic":
        return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
                np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
                6 * field) / (dx**2)
    elif boundary_condition.lower() == "dirichlet":
        lap = np.zeros_like(field)
        lap[1:-1,1:-1,1:-1] = (field[2:,1:-1,1:-1] + field[:-2,1:-1,1:-1] +
                                field[1:-1,2:,1:-1] + field[1:-1,:-2,1:-1] +
                                field[1:-1,1:-1,2:] + field[1:-1,1:-1,:-2] -
                                6 * field[1:-1,1:-1,1:-1]) / (dx**2)
        return lap
    else:
        raise ValueError("Unsupported boundary condition.")

def spectral_laplacian_3D(field, L):
    N = field.shape[0]
    dx = L / N
    field_k = fftn(field)
    k = 2 * np.pi * fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    lap_k = -(KX**2 + KY**2 + KZ**2) * field_k
    lap = ifftn(lap_k).real
    return lap

@njit(parallel=True)
def laplacian_numba(field, dx):
    N0, N1, N2 = field.shape
    lap = np.empty_like(field)
    for i in prange(N0):
        for j in range(N1):
            for k in range(N2):
                ip = (i + 1) % N0
                im = (i - 1) % N0
                jp = (j + 1) % N1
                jm = (j - 1) % N1
                kp = (k + 1) % N2
                km = (k - 1) % N2
                lap[i, j, k] = (field[ip, j, k] + field[im, j, k] +
                                field[i, jp, k] + field[i, jm, k] +
                                field[i, j, kp] + field[i, j, km] - 6 * field[i, j, k]) / (dx * dx)
    return lap

def solve_poisson_3D(rho, dx, G, relativistic_factor=0.0):
    shape = rho.shape
    rho_k = fftn(rho)
    kx = 2 * np.pi * fftfreq(shape[0], d=dx)
    ky = 2 * np.pi * fftfreq(shape[1], d=dx)
    kz = 2 * np.pi * fftfreq(shape[2], d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1.0
    phi_k = -4 * np.pi * G * rho_k / k_squared
    phi_k[0, 0, 0] = 0.0
    phi = np.real(ifftn(phi_k))
    phi *= (1 + relativistic_factor)
    return phi

def compute_power_spectrum_3D(field):
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

def analytical_gravitational_potential(r, M, R, G):
    if r <= R:
        return -G * M * (3*R**2 - r**2) / (2 * R**3)
    else:
        return -G * M / r

def benchmark_comparison(sim_result, sim_params):
    L = sim_params["L"]
    N = sim_params["N"]
    dx = L / N
    M_est = np.sum(sim_result["phi_grav"]) * dx**3
    R = L / 2
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    phi_analytic = np.vectorize(analytical_gravitational_potential)(r, M_est, R, sim_params["G"])
    return phi_analytic

# ---------------------------
# Base Integration Routine (Fast)
# ---------------------------
def run_field_simulation_3D_base(params, snapshot_interval=50):
    if not isinstance(params, dict):
        params = asdict(params)
    T = params["steps_per_cycle"] * params["num_cycles"]
    N = params["N"]
    L = params["L"]
    dx = L / N
    t = 0.0
    # Initialize the fields with small random perturbations.
    phi_prev = 0.01 * np.random.randn(N, N, N)
    phi = 0.01 * np.random.randn(N, N, N)
    # Compute target energy from initial conditions.
    E_target = compute_total_energy(phi, phi_prev, params["dt"], dx, params["m"], params.get("boundary_condition", "periodic"))
    snapshots = []
    start_wall = time.time()
    
    for step in range(T):
        phi_avg = np.mean(phi)
        lap_phi = laplacian_3D(phi, dx, params.get("boundary_condition", "periodic"))
        noise = params["continuous_noise_amplitude"] * np.random.randn(N, N, N)
        noise = gaussian_filter(noise, sigma=params["collapse_sigma"]/dx)
        num_events = np.random.poisson(lam=params["collapse_rate"])
        X, Y, Z = np.meshgrid(np.linspace(-L/2, L/2, N, endpoint=False),
                               np.linspace(-L/2, L/2, N, endpoint=False),
                               np.linspace(-L/2, L/2, N, endpoint=False),
                               indexing='ij')
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            phi += params["collapse_amplitude"] * gaussian_3d(X, Y, Z,
                                                              event_pos[0],
                                                              event_pos[1],
                                                              event_pos[2],
                                                              params["collapse_sigma"])
        acceleration = (lap_phi - params["m"]**2 * phi -
                        params["collapse_rate"] * (phi - phi_avg) +
                        np.sqrt(params["collapse_rate"]) * noise)
        phi_next = 2 * phi - phi_prev + params["dt"]**2 * acceleration
        phi_prev = phi.copy()
        phi = phi_next.copy()
        t += params["dt"]
        if step % snapshot_interval == 0:
            snapshots.append(phi.copy())
    wall_time = time.time() - start_wall
    # Compute current energy and determine the relative energy error.
    E_current = compute_total_energy(phi, phi_prev, params["dt"], dx, params["m"], params.get("boundary_condition", "periodic"))
    energy_error = abs(E_current - E_target) / (E_target + 1e-12)
    
    if params.get("boundary_condition", "periodic").lower() == "periodic":
        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dx)
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)
        grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2*dx)
    else:
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        grad_x[1:-1,:,:] = (phi[2:,:,:] - phi[:-2,:,:]) / (2*dx)
        grad_y[:,1:-1,:] = (phi[:,2:,:] - phi[:,:-2,:]) / (2*dx)
        grad_z[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,:-2]) / (2*dx)
    energy_density = 0.5 * (((phi - phi_prev)/params["dt"])**2 +
                            grad_x**2 + grad_y**2 + grad_z**2 +
                            params["m"]**2 * phi**2)
    phi_grav = solve_poisson_3D(energy_density, dx, params["G"], params["relativistic_factor"])
    psd = compute_power_spectrum_3D(phi_grav)
    slope, intercept, slope_std_error = estimate_noise_exponent(psd, fit_range=(1, int(0.2 * N)))
    logging.info("Base simulation complete: slope=%.3f (std err: %s), energy_error=%.3e, wall_time=%.2f sec",
                 slope, slope_std_error, energy_error, wall_time)
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_std_error": slope_std_error,
        "energy_error": energy_error,
        "snapshots": snapshots,
        "simulation_time": t,
        "wall_time": wall_time
    }

# ---------------------------
# Adaptive Integration Routine (for Sensitivity Analysis)
# ---------------------------
def run_field_simulation_3D_adaptive(params, snapshot_interval=50, epsilon=1e-3,
                                       dt_initial=None, dt_min=1e-4, dt_max=0.1,
                                       use_spectral=False, use_numba_lap=False,
                                       energy_correction=True, progress_callback=None):
    if not isinstance(params, dict):
        params = asdict(params)
    dt_current = dt_initial if dt_initial is not None else params["dt"]
    T_total = params["steps_per_cycle"] * params["num_cycles"]
    t = 0.0
    N = params["N"]
    L = params["L"]
    dx = L / N
    X, Y, Z = np.meshgrid(np.linspace(-L/2, L/2, N, endpoint=False),
                           np.linspace(-L/2, L/2, N, endpoint=False),
                           np.linspace(-L/2, L/2, N, endpoint=False),
                           indexing='ij')
    phi_prev = 0.01 * np.random.randn(N, N, N)
    phi = 0.01 * np.random.randn(N, N, N)
    E_target = compute_total_energy(phi, phi_prev, dt_current, dx, params["m"], params.get("boundary_condition", "periodic"))
    E_current = E_target
    snapshots = []
    step_count = 0
    start_wall = time.time()
    
    while t < T_total:
        phi_old = phi.copy()
        phi_prev_old = phi_prev.copy()
        t_old = t
        
        phi_avg = np.mean(phi)
        if use_spectral:
            lap_phi = spectral_laplacian_3D(phi, L)
        else:
            if use_numba_lap:
                lap_phi = laplacian_numba(phi, dx)
            else:
                lap_phi = laplacian_3D(phi, dx, params.get("boundary_condition", "periodic"))
                
        noise = params["continuous_noise_amplitude"] * np.random.randn(N, N, N)
        noise = gaussian_filter(noise, sigma=params["collapse_sigma"] / dx)
        num_events = np.random.poisson(lam=params["collapse_rate"])
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            phi += params["collapse_amplitude"] * gaussian_3d(X, Y, Z,
                                                              event_pos[0],
                                                              event_pos[1],
                                                              event_pos[2],
                                                              params["collapse_sigma"])
        acceleration = (lap_phi - params["m"]**2 * phi -
                        params["collapse_rate"] * (phi - phi_avg) +
                        np.sqrt(params["collapse_rate"]) * noise)
        phi_next = 2 * phi - phi_prev + dt_current**2 * acceleration
        
        E_new = compute_total_energy(phi_next, phi, dt_current, dx, params["m"], params.get("boundary_condition", "periodic"))
        relative_error = abs(E_new - E_current) / (E_current + 1e-12)
        
        if relative_error > epsilon and dt_current > dt_min:
            dt_current = max(dt_current * 0.5, dt_min)
            phi = phi_old.copy()
            phi_prev = phi_prev_old.copy()
            t = t_old
            continue
        else:
            phi_prev = phi.copy()
            phi = phi_next.copy()
            t += dt_current
            E_current = E_new
            step_count += 1
            if energy_correction:
                correction_factor = np.sqrt(E_target / (E_current + 1e-12))
                phi *= correction_factor
                phi_prev *= correction_factor
                E_current = compute_total_energy(phi, phi_prev, dt_current, dx, params["m"], params.get("boundary_condition", "periodic"))
            if relative_error < epsilon / 10 and dt_current < dt_max:
                dt_current = min(dt_current * 1.1, dt_max)
        if step_count % snapshot_interval == 0:
            snapshots.append(phi.copy())
    
    wall_time = time.time() - start_wall
    if params.get("boundary_condition", "periodic").lower() == "periodic":
        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*dx)
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*dx)
        grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2*dx)
    else:
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        grad_x[1:-1,:,:] = (phi[2:,:,:] - phi[:-2,:,:]) / (2*dx)
        grad_y[:,1:-1,:] = (phi[:,2:,:] - phi[:,:-2,:]) / (2*dx)
        grad_z[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,:-2]) / (2*dx)
    energy_density = 0.5 * (((phi - phi_prev) / dt_current)**2 +
                            grad_x**2 + grad_y**2 + grad_z**2 +
                            params["m"]**2 * phi**2)
    phi_grav = solve_poisson_3D(energy_density, dx, params["G"], params["relativistic_factor"])
    psd = compute_power_spectrum_3D(phi_grav)
    slope, intercept, slope_std_error = estimate_noise_exponent(psd, fit_range=(1, int(0.2 * N)))
    logging.info("Adaptive simulation complete: slope=%.3f (std err: %s), final energy error=%.3e",
                 slope, slope_std_error, abs(E_current - E_target) / (E_target + 1e-12))
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_std_error": slope_std_error,
        "energy_initial": E_target,
        "energy_final": E_current,
        "energy_error": abs(E_current - E_target) / (E_target + 1e-12),
        "final_field": phi,
        "phi_grav": phi_grav,
        "snapshots": snapshots,
        "simulation_time": t,
        "wall_time": wall_time,
        "steps_taken": step_count,
        "params": params,
        "final_dt": dt_current
    }

# ---------------------------
# Convergence Test and Parameter Sensitivity Analysis
# ---------------------------
def convergence_test(base_params, resolutions, time_steps, ensemble_runs=3):
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
                sim_result = run_field_simulation_3D_base(params, snapshot_interval=params["steps_per_cycle"])
                if sim_result["slope"] is not None:
                    slopes.append(sim_result["slope"])
                    energy_errors.append(sim_result.get("energy_error", 0))
                    runtimes.append(sim_result["simulation_time"])
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
            logging.info("Convergence test for N=%d, dt=%.3f: avg_slope=%.3f, std_slope=%.3f, avg_energy_error=%.3e",
                         N, dt, avg_slope if avg_slope is not None else -1,
                         std_slope if std_slope is not None else -1,
                         avg_energy_error if avg_energy_error is not None else -1)
    return results

def parameter_sensitivity_analysis(base_params, param_grid, ensemble_runs=3):
    results = []
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        candidate = dict(zip(keys, values))
        sim_params = base_params.copy()
        sim_params.update(candidate)
        slopes = []
        energy_errors = []
        runtimes = []
        for _ in range(ensemble_runs):
            result = run_field_simulation_3D_base(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
            slopes.append(result["slope"])
            energy_errors.append(result.get("energy_error", 0))
            runtimes.append(result["simulation_time"])
        avg_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        avg_energy_error = np.mean(energy_errors)
        std_energy_error = np.std(energy_errors)
        avg_runtime = np.mean(runtimes)
        results.append({
            "candidate": candidate,
            "avg_slope": avg_slope,
            "std_slope": std_slope,
            "avg_energy_error": avg_energy_error,
            "std_energy_error": std_energy_error,
            "avg_runtime": avg_runtime
        })
        logging.info("Sensitivity analysis for candidate %s: avg_slope=%.3f, avg_energy_error=%.3e",
                     candidate, avg_slope, avg_energy_error)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3D Field Simulation with GRW Collapse Dynamics")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive time-stepping integration")
    parser.add_argument("--snapshot_interval", type=int, default=50, help="Interval for saving simulation snapshots")
    args = parser.parse_args()
    
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
    
    if args.adaptive:
        result = run_field_simulation_3D_adaptive(params, snapshot_interval=args.snapshot_interval)
    else:
        result = run_field_simulation_3D_base(params, snapshot_interval=args.snapshot_interval)
    
    print("Simulation complete.")
    print(f"Slope: {result['slope']}")
    print(f"Energy error: {result.get('energy_error', 'N/A')}")
    print(f"Wall time: {result['wall_time']} sec")
    print(f"Simulation time: {result['simulation_time']} sec")

