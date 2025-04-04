#!/usr/bin/env python3
"""
Module: dynamic_visualization.py

A dynamic interactive visualization tool for the emergent gravity simulation.
This tool loads saved simulation snapshots (snapshots.npy) and simulation parameters
(simulation_params.json) from a specified folder. If the number of snapshots is low,
it interpolates additional frames for a smoother animation.
It then simulates matter point motion using simple Euler integration, where the acceleration
is approximated as the negative gradient of the central slice.
An interactive Matplotlib window with a slider allows frame-by-frame navigation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_simulation_data(folder):
    snapshots_path = os.path.join(folder, "snapshots.npy")
    params_path = os.path.join(folder, "simulation_params.json")
    if not os.path.exists(snapshots_path) or not os.path.exists(params_path):
        raise FileNotFoundError("snapshots.npy or simulation_params.json not found in the folder.")
    snapshots = np.load(snapshots_path, allow_pickle=True)
    import json
    with open(params_path, "r") as f:
        sim_params = json.load(f)
    return snapshots, sim_params

def interpolate_frames(snapshots, interp_factor=10):
    """
    Interpolate additional frames between each pair of snapshots.
    
    Parameters:
        snapshots: list/array of snapshots.
        interp_factor: number of frames to generate between original frames.
    
    Returns:
        A new list of frames (original + interpolated).
    """
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

def simulate_matter_points(frames, sim_params, num_points=50, dt_effect=0.1):
    """
    Simulate matter point trajectories using simple Euler integration.
    Acceleration is estimated as the negative gradient of the central slice.
    
    Parameters:
        frames: list of simulation frames (each a 3D array)
        sim_params: dictionary of simulation parameters (must include 'L' and 'N')
        num_points: number of matter points to simulate.
        dt_effect: effective time step for updating matter point dynamics.
    
    Returns:
        List of tuples (x_positions, y_positions) for each frame.
    """
    L = sim_params["L"]
    N = sim_params["N"]
    x_vals = np.linspace(-L/2, L/2, N)
    y_vals = np.linspace(-L/2, L/2, N)
    
    # Initialize matter point positions and velocities.
    mp_x = np.random.uniform(-L/2, L/2, num_points)
    mp_y = np.random.uniform(-L/2, L/2, num_points)
    mp_vx = np.zeros(num_points)
    mp_vy = np.zeros(num_points)
    
    positions = []
    for frame in frames:
        field_slice = frame[:, :, N//2]
        # np.gradient returns [grad_y, grad_x]
        grad_y, grad_x = np.gradient(field_slice, x_vals, y_vals)
        new_x = []
        new_y = []
        for i in range(num_points):
            idx_x = np.clip(np.searchsorted(x_vals, mp_x[i]), 0, N-1)
            idx_y = np.clip(np.searchsorted(y_vals, mp_y[i]), 0, N-1)
            a_x = -grad_x[idx_y, idx_x]
            a_y = -grad_y[idx_y, idx_x]
            mp_vx[i] += a_x * dt_effect
            mp_vy[i] += a_y * dt_effect
            mp_x[i] += mp_vx[i] * dt_effect
            mp_y[i] += mp_vy[i] * dt_effect
            new_x.append(mp_x[i])
            new_y.append(mp_y[i])
        positions.append((np.array(new_x), np.array(new_y)))
    return positions

def launch_interactive_visualization(folder, interp_factor=10, dt_effect=0.1, num_points=50):
    """
    Launch an interactive visualization window.
    
    Loads snapshots and parameters from the given folder.
    Interpolates frames if necessary, simulates matter point motion,
    and opens a Matplotlib window with a slider for frame navigation.
    """
    # Load data.
    snapshots, sim_params = load_simulation_data(folder)
    if len(snapshots) < 10:
        frames = interpolate_frames(snapshots, interp_factor=interp_factor)
    else:
        frames = snapshots
    matter_positions = simulate_matter_points(frames, sim_params, num_points=num_points, dt_effect=dt_effect)
    
    L = sim_params["L"]
    N = sim_params["N"]
    num_frames = len(frames)
    
    # Set up the Matplotlib figure and slider.
    fig, ax = plt.subplots(figsize=(6,6))
    plt.subplots_adjust(bottom=0.25)
    frame_idx = 0
    field_slice = frames[frame_idx][:, :, N//2]
    im = ax.imshow(field_slice, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='viridis')
    mp_pos = matter_positions[frame_idx]
    scat = ax.scatter(mp_pos[0], mp_pos[1], c='red', s=30, label="Matter Points")
    ax.set_title(f"Frame {frame_idx+1}/{num_frames}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        field_slice = frames[idx][:, :, N//2]
        im.set_data(field_slice)
        mp_pos = matter_positions[idx]
        scat.set_offsets(np.column_stack((mp_pos[0], mp_pos[1])))
        ax.set_title(f"Frame {idx+1}/{num_frames}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

