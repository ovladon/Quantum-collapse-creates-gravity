#!/usr/bin/env python3
"""
Module: visualization.py
Creates static visualization charts from simulation snapshots.
This version simulates test particle trajectories (instead of matter points)
to show their response to the created gravitational field.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging

def get_grid(sim_params):
    L = sim_params["L"]
    N = sim_params["N"]
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    return x, y

def initialize_test_particles(num_particles, sim_params):
    """
    Initialize test particles in a small cluster near the center.
    """
    L = sim_params["L"]
    np.random.seed(42)
    positions = np.random.normal(loc=0.0, scale=L/20, size=(num_particles, 2))
    velocities = np.zeros((num_particles, 2))
    return positions, velocities

def update_particles(positions, velocities, grad_x, grad_y, sim_params, dt_effect):
    """
    Update test particle positions using Euler integration.
    Uses the nearest grid point (from the central slice) to compute acceleration.
    """
    L = sim_params["L"]
    N = sim_params["N"]
    dx = L / (N - 1)
    x_grid, y_grid = get_grid(sim_params)
    for i in range(positions.shape[0]):
        x_pos, y_pos = positions[i]
        # Find nearest grid indices
        idx_x = int(round((x_pos + L/2) / dx))
        idx_y = int(round((y_pos + L/2) / dx))
        idx_x = np.clip(idx_x, 0, N-1)
        idx_y = np.clip(idx_y, 0, N-1)
        # Acceleration is negative gradient at that grid point
        a_x = -grad_x[idx_y, idx_x]
        a_y = -grad_y[idx_y, idx_x]
        velocities[i, 0] += a_x * dt_effect
        velocities[i, 1] += a_y * dt_effect
        positions[i, 0] += velocities[i, 0] * dt_effect
        positions[i, 1] += velocities[i, 1] * dt_effect
    return positions, velocities

def create_visualization_with_test_particles(snapshots, sim_params, output_folder, dt_effect=0.1, num_particles=10):
    """
    For each simulation snapshot, update the test particle positions (using the local field gradient)
    and save a PNG image of the central slice with overlaid test particles.
    Also produces a summary collage from the first, middle, and last frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    x_grid, y_grid = get_grid(sim_params)
    N = sim_params["N"]
    # Initialize test particles
    positions, velocities = initialize_test_particles(num_particles, sim_params)
    
    saved_images = []
    for idx, frame in enumerate(snapshots):
        field_slice = frame[:, :, N//2]
        # Compute gradients of the field (np.gradient returns [grad_y, grad_x])
        grad_y, grad_x = np.gradient(field_slice, x_grid, y_grid)
        # Update test particles
        positions, velocities = update_particles(positions, velocities, grad_x, grad_y, sim_params, dt_effect)
        
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(field_slice, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                       origin='lower', cmap=cm.viridis)
        ax.scatter(positions[:,0], positions[:,1], c='red', s=50, label="Test Particles")
        ax.set_title(f"Frame {idx+1} / {len(snapshots)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        filename = os.path.join(output_folder, f"test_particle_frame_{idx+1}.png")
        fig.savefig(filename)
        plt.close(fig)
        saved_images.append(filename)
    
    # Create a summary collage if there are enough frames
    if len(snapshots) >= 3:
        indices = [0, len(snapshots)//2, len(snapshots)-1]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, idx in zip(axes, indices):
            frame = snapshots[idx]
            field_slice = frame[:, :, N//2]
            ax.imshow(field_slice, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                      origin='lower', cmap=cm.viridis)
            ax.set_title(f"Frame {idx+1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.suptitle("Summary Collage: Test Particle Trajectories")
        collage_path = os.path.join(output_folder, "test_particle_collage.png")
        fig.savefig(collage_path)
        plt.close(fig)
        saved_images.append(collage_path)
    
    logging.info(f"Static test particle visualization saved in {output_folder}")
    return saved_images

def create_results_chart(optimization_summary, output_folder):
    """
    Create a chart showing candidate results (slope vs energy error) from optimization.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_generations = optimization_summary.get("all_generations", [])
    slopes = []
    energy_errors = []
    for gen in all_generations:
        for res in gen:
            if res.get("slope") is not None and res.get("energy_error") is not None:
                slopes.append(res["slope"])
                energy_errors.append(res["energy_error"])
    
    best = optimization_summary.get("best_overall", {})
    best_slope = best.get("slope", None)
    best_energy_error = best.get("energy_error", None)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(slopes, energy_errors, c='blue', label='Candidates', alpha=0.6)
    if best_slope is not None and best_energy_error is not None:
        plt.scatter([best_slope], [best_energy_error], c='red', label='Best Candidate', s=100)
    
    target_slope = -5.0
    slope_tolerance = 0.5
    energy_threshold = 0.1
    plt.axvspan(target_slope - slope_tolerance, target_slope + slope_tolerance, color='green', alpha=0.2, label='Target Slope Range')
    plt.axhline(energy_threshold, color='orange', linestyle='--', label='Energy Error Threshold')
    
    plt.xlabel("Noise Exponent (Slope)")
    plt.ylabel("Relative Energy Error")
    plt.title("Optimization Candidate Results")
    plt.legend()
    chart_path = os.path.join(output_folder, "optimization_results_chart.png")
    plt.savefig(chart_path)
    plt.close()
    logging.info(f"Results chart saved to {chart_path}")
    return chart_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Static visualization with test particles.")
    parser.add_argument("snapshots", type=str, help="Path to snapshots.npy file")
    parser.add_argument("params", type=str, help="Path to simulation_params.json file")
    parser.add_argument("--output_folder", type=str, default="visualization_output", help="Folder to save visualization charts")
    parser.add_argument("--dt_effect", type=float, default=0.1, help="Effective time step for test particle update")
    parser.add_argument("--num_particles", type=int, default=10, help="Number of test particles")
    args = parser.parse_args()
    
    with open(args.params, "r") as f:
        sim_params = __import__("json").load(f)
    snapshots = np.load(args.snapshots, allow_pickle=True)
    
    create_visualization_with_test_particles(snapshots, sim_params, args.output_folder,
                                               dt_effect=args.dt_effect, num_particles=args.num_particles)

