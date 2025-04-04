#!/usr/bin/env python3
"""
Module: simulation_physics.py
Provides the physics and numerical routines for the 3D field simulation with GRW collapse events.
This module implements a specific collapse model based on the Ghirardi–Rimini–Weber (GRW) theory.
References:
    - Ghirardi, G.C., Rimini, A. & Weber, T. (1986). "Unified dynamics for microscopic and macroscopic systems".
      Physical Review D, 34(2), 470–491.
      
The simulation uses a symplectic (leapfrog) integrator to update the field dynamics,
with options for periodic or Dirichlet boundary conditions.
Parameters such as collapse_rate, collapse_sigma, etc., are chosen based on estimates in the literature.
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
import time

def gaussian_3d(X, Y, Z, x0, y0, z0, sigma):
    """
    Compute a normalized 3D Gaussian function centered at (x0, y0, z0) with width sigma.
    """
    norm = (2 * np.pi * sigma**2) ** (3/2)
    return np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2)) / norm

def laplacian_3D(field, dx, boundary_condition="periodic"):
    """
    Compute the 3D Laplacian of a field using finite differences.
    
    Parameters:
        field: 3D numpy array
        dx: spatial resolution
        boundary_condition: "periodic" or "dirichlet"
    
    Returns:
        3D numpy array of the Laplacian.
    """
    if boundary_condition == "periodic":
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
            6 * field
        ) / (dx**2)
    elif boundary_condition.lower() == "dirichlet":
        lap = np.zeros_like(field)
        # Compute second derivatives for interior points only
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
    
    Parameters:
        rho: 3D density array.
        dx: spatial resolution.
        G: gravitational constant.
        relativistic_factor: crude correction factor.
        
    Returns:
        Gravitational potential phi.
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
    
    Returns:
        1D numpy array representing the radially averaged PSD.
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
    
    Also computes the standard error of the slope.
    
    Returns:
        slope, intercept, slope_std_error
    """
    bins = np.arange(len(psd))
    idx = (bins >= fit_range[0]) & (bins < fit_range[1])
    x = np.log10(bins[idx] + 1e-12)
    y = np.log10(psd[idx] + 1e-12)
    if len(x) < 2:
        return None, None, None
    # Linear regression using least squares
    A = np.vstack([x, np.ones(len(x))]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    slope = coeffs[0]
    intercept = coeffs[1]
    # Standard error estimation
    if len(x) > 2 and residuals.size > 0:
        residual_std = np.sqrt(residuals[0] / (len(x) - 2))
        sxx = np.sum((x - np.mean(x))**2)
        slope_std_error = residual_std / np.sqrt(sxx)
    else:
        slope_std_error = None
    return slope, intercept, slope_std_error

def compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition="periodic"):
    """
    Compute the total energy of the field.
    Energy = kinetic + gradient (spatial derivative) + potential energy.
    
    Returns:
        total energy (scalar)
    """
    # Kinetic energy from time derivative (using leapfrog estimate)
    phi_t = (phi - phi_prev) / dt
    kinetic = 0.5 * phi_t**2
    
    # Gradient energy using central differences
    if boundary_condition == "periodic":
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
    # Potential energy from the mass term
    potential_energy = 0.5 * m**2 * phi**2
    total_energy = np.sum(kinetic + grad_energy + potential_energy) * (dx**3)
    return total_energy

def run_field_simulation_3D(params, snapshot_interval=50):
    """
    Run a 3D field simulation with GRW collapse dynamics and symplectic integration.
    
    The collapse mechanism is tied to the GRW model. In the GRW model, collapse events occur 
    at random with a rate lambda (here collapse_rate) and have a localization width (collapse_sigma).
    For references, see Ghirardi, Rimini, Weber (1986).
    
    Parameters (in params dictionary):
       collapse_rate: event rate (Hz) – typically very low for individual particles; here rescaled for the field.
       collapse_sigma: localization width (m) – GRW suggests ~10^-7 m, but rescaled for simulation dimensions.
       collapse_amplitude: amplitude scaling for collapse events.
       continuous_noise_amplitude: amplitude of continuous stochastic noise.
       density_decay: decay factor per time step to simulate dissipation.
       relativistic_factor: correction factor for gravitational potential.
       G: gravitational constant.
       L: domain size.
       N: grid resolution.
       steps_per_cycle: number of time steps per cycle.
       num_cycles: number of cycles.
       dt: time step size.
       m: mass parameter for the field.
       boundary_condition: "periodic" or "dirichlet"
    
    Returns a dictionary containing:
       - slope, intercept, slope_std_error: estimated noise exponent and its error.
       - energy_initial: initial total energy.
       - energy_final: final total energy.
       - energy_error: relative energy change.
       - final_field: final field configuration.
       - phi_grav: gravitational potential computed from energy density.
       - snapshots: list of field snapshots (for visualization).
       - runtime: total simulation time in seconds.
       - params: simulation parameters used.
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
    boundary_condition = params.get("boundary_condition", "periodic")
    
    T = steps_per_cycle * num_cycles
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    dx = L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Initialize fields with small random fluctuations
    phi_prev = 0.01 * np.random.randn(N, N, N)
    phi = 0.01 * np.random.randn(N, N, N)
    
    # Record initial total energy for conservation check
    energy_initial = compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition)
    
    snapshots = []
    start_time = time.time()
    
    # Main simulation loop using leapfrog (symplectic) integration
    for t in range(T):
        phi_avg = np.mean(phi)
        noise = continuous_noise_amplitude * np.random.randn(N, N, N)
        noise = gaussian_filter(noise, sigma=collapse_sigma/dx)
        lap_phi = laplacian_3D(phi, dx, boundary_condition)
        
        # GRW collapse: add discrete collapse events as Gaussian deposits
        num_events = np.random.poisson(lam=collapse_rate)
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            phi += collapse_amplitude * gaussian_3d(X, Y, Z,
                                                    event_pos[0],
                                                    event_pos[1],
                                                    event_pos[2],
                                                    collapse_sigma)
        
        # Compute acceleration using a modified Klein–Gordon equation with collapse terms
        acceleration = lap_phi - m**2 * phi - collapse_rate * (phi - phi_avg) + np.sqrt(collapse_rate) * noise
        
        # Leapfrog integration: phi_next = 2*phi - phi_prev + dt^2 * acceleration
        phi_next = 2 * phi - phi_prev + dt**2 * acceleration
        
        # Update fields for next step
        phi_prev = phi.copy()
        phi = phi_next.copy()
        
        # Apply density decay to simulate dissipation
        phi *= density_decay
        
        # Save snapshots at specified intervals
        if (t % snapshot_interval == 0):
            snapshots.append(phi.copy())
    
    runtime = time.time() - start_time
    
    # Compute final energy and energy error (relative change)
    energy_final = compute_total_energy(phi, phi_prev, dt, dx, m, boundary_condition)
    energy_error = np.abs(energy_final - energy_initial) / (energy_initial + 1e-12)
    
    # Compute effective energy density for gravitational potential
    phi_t = (phi - phi_prev) / dt
    if boundary_condition == "periodic":
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
    
    energy_density = 0.5 * (phi_t**2 + grad_x**2 + grad_y**2 + grad_z**2 + m**2 * phi**2)
    
    # Solve Poisson equation to compute gravitational potential
    phi_grav = solve_poisson_3D(energy_density, dx, G, relativistic_factor)
    
    # Compute power spectrum and estimate the noise exponent with error estimation
    psd = compute_power_spectrum_3D(phi_grav)
    slope, intercept, slope_std_error = estimate_noise_exponent(psd, fit_range=(1, int(0.2 * N)))
    
    results = {
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
    
    return results

def convergence_test(base_params, resolutions, time_steps):
    """
    Perform convergence tests by running the simulation at different grid resolutions (N) and time steps (dt).
    
    Parameters:
        base_params: dictionary of base simulation parameters.
        resolutions: list of N values to test.
        time_steps: list of dt values to test.
        
    Returns:
        A list of dictionaries with keys: 'N', 'dt', 'slope', 'energy_error', etc.
    """
    results = []
    for N in resolutions:
        for dt in time_steps:
            params = base_params.copy()
            params["N"] = N
            params["dt"] = dt
            sim_result = run_field_simulation_3D(params, snapshot_interval=params["steps_per_cycle"])
            results.append({
                "N": N,
                "dt": dt,
                "slope": sim_result["slope"],
                "slope_std_error": sim_result["slope_std_error"],
                "energy_error": sim_result["energy_error"],
                "runtime": sim_result["runtime"]
            })
            print(f"Convergence test for N={N}, dt={dt}: slope={sim_result['slope']}, energy_error={sim_result['energy_error']}")
    return results

