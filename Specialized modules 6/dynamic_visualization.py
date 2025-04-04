#!/usr/bin/env python3
"""
Module: dynamic_visualization.py

An interactive visualization tool for the simulation using test particles.
It loads simulation snapshots and parameters, interpolates frames if needed,
and simulates test particle trajectories (updated via Euler integration) using
matplotlib’s FuncAnimation.

Advanced: A Numba-accelerated version of the particle update routine is provided for speed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import logging

# Attempt to import Numba for acceleration
try:
    from numba import njit
except ImportError:
    njit = lambda f: f

def load_simulation_data(folder):
    snapshots_path = os.path.join(folder, "snapshots.npy")
    params_path = os.path.join(folder, "simulation_params.json")
    if not os.path.exists(snapshots_path) or not os.path.exists(params_path):
        raise FileNotFoundError("snapshots.npy or simulation_params.json not found in the folder.")
    snapshots = np.load(snapshots_path, allow_pickle=True)
    with open(params_path, "r") as f:
        sim_params = json.load(f)
    return snapshots, sim_params

def interpolate_frames(snapshots, interp_factor=10):
    new_frames = []
    num = len(snapshots)
    for i in range(num - 1):
        frame1 = snapshots[i]
        frame2 = snapshots[i+1]
        new_frames.append(frame1)
        for j in range(1, interp_factor):
            alpha = j / interp_factor
            interp_frame = (1 - alpha) * frame1 + alpha * frame2
            new_frames.append(interp_frame)
    new_frames.append(snapshots[-1])
    return new_frames

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
    Update test particle positions using a pure-Python loop.
    """
    L = sim_params["L"]
    N = sim_params["N"]
    dx = L / (N - 1)
    x_grid, y_grid = get_grid(sim_params)
    for i in range(positions.shape[0]):
        x_pos, y_pos = positions[i]
        idx_x = int(round((x_pos + L/2) / dx))
        idx_y = int(round((y_pos + L/2) / dx))
        idx_x = min(max(idx_x, 0), N-1)
        idx_y = min(max(idx_y, 0), N-1)
        a_x = -grad_x[idx_y, idx_x]
        a_y = -grad_y[idx_y, idx_x]
        velocities[i, 0] += a_x * dt_effect
        velocities[i, 1] += a_y * dt_effect
        positions[i, 0] += velocities[i, 0] * dt_effect
        positions[i, 1] += velocities[i, 1] * dt_effect
    return positions, velocities

@njit
def update_particles_numba(positions, velocities, grad_x, grad_y, L, N, dt_effect):
    """
    Numba-accelerated version of test particle update.
    """
    dx = L / (N - 1)
    for i in range(positions.shape[0]):
        x_pos = positions[i, 0]
        y_pos = positions[i, 1]
        # Compute index using basic arithmetic
        idx_x = int(round((x_pos + L/2) / dx))
        idx_y = int(round((y_pos + L/2) / dx))
        if idx_x < 0:
            idx_x = 0
        elif idx_x >= N:
            idx_x = N - 1
        if idx_y < 0:
            idx_y = 0
        elif idx_y >= N:
            idx_y = N - 1
        a_x = -grad_x[idx_y, idx_x]
        a_y = -grad_y[idx_y, idx_x]
        velocities[i, 0] += a_x * dt_effect
        velocities[i, 1] += a_y * dt_effect
        positions[i, 0] += velocities[i, 0] * dt_effect
        positions[i, 1] += velocities[i, 1] * dt_effect
    return positions, velocities

def animate_test_particles(folder, interp_factor=10, dt_effect=0.1, num_particles=10, use_numba=True):
    """
    Launch an interactive animation showing simulation snapshots and test particle trajectories.
    If use_numba is True and Numba is available, the accelerated update function is used.
    """
    snapshots, sim_params = load_simulation_data(folder)
    if len(snapshots) < 10:
        frames = interpolate_frames(snapshots, interp_factor=interp_factor)
    else:
        frames = snapshots
    num_frames = len(frames)
    
    x_grid, y_grid = get_grid(sim_params)
    N = sim_params["N"]
    
    # Initialize test particles
    positions, velocities = initialize_test_particles(num_particles, sim_params)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(6,6))
    initial_field = frames[0][:, :, N//2]
    im = ax.imshow(initial_field, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                   origin='lower', cmap='viridis')
    scatter = ax.scatter(positions[:,0], positions[:,1], c='red', s=50, label="Test Particles")
    ax.set_title(f"Frame 1 / {num_frames}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    def update(frame_index):
        nonlocal positions, velocities
        current_frame = frames[frame_index]
        field_slice = current_frame[:, :, N//2]
        im.set_data(field_slice)
        
        # Compute gradients (np.gradient returns [grad_y, grad_x])
        grad_y, grad_x = np.gradient(field_slice, x_grid, y_grid)
        if use_numba:
            positions, velocities = update_particles_numba(positions, velocities, grad_x, grad_y,
                                                            sim_params["L"], N, dt_effect)
        else:
            positions, velocities = update_particles(positions, velocities, grad_x, grad_y, sim_params, dt_effect)
        scatter.set_offsets(positions)
        ax.set_title(f"Frame {frame_index+1} / {num_frames}")
        return im, scatter
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=dt_effect*1000, blit=False, repeat=True)
    logging.info("Launching interactive test particle animation.")
    plt.show()

# For backward compatibility with gui_app.py:
launch_interactive_visualization = animate_test_particles

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive test particle animation.")
    parser.add_argument("folder", type=str, help="Folder containing snapshots.npy and simulation_params.json")
    parser.add_argument("--interp_factor", type=int, default=10, help="Interpolation factor for frames")
    parser.add_argument("--dt_effect", type=float, default=0.1, help="Effective time step for particle update")
    parser.add_argument("--num_particles", type=int, default=10, help="Number of test particles")
    parser.add_argument("--use_numba", action="store_true", help="Use Numba acceleration if available")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    animate_test_particles(args.folder, interp_factor=args.interp_factor,
                           dt_effect=args.dt_effect, num_particles=args.num_particles,
                           use_numba=args.use_numba)

