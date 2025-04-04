#!/usr/bin/env python3
"""
Module: visualization.py
Creates visualization charts and interactive animations from simulation snapshots and optimization results.
Instead of producing a video, this module now provides:
 - PNG charts (static snapshots and summary collage) via create_visualization_charts.
 - A results chart (slope vs energy error) via create_results_chart.
 - An interactive animation window via create_interactive_animation_window, which plays the simulation frame by frame,
   and simulates simple matter point motion under the influence of the gravitational potential (approximated by the central slice gradient).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def create_visualization_charts(snapshots, sim_params, output_folder, matter_points=100):
    """
    Generate PNG charts from simulation snapshots.
    
    For each selected snapshot (every n-th frame to avoid overload), the function:
      - Extracts a central slice (z = N//2) of the 3D field.
      - Overlays random matter points.
      - Saves the figure as a PNG.
    Additionally, a summary collage is generated combining the first, middle, and last snapshots.
    
    Parameters:
        snapshots: list of 3D numpy arrays (simulation snapshots)
        sim_params: dictionary of simulation parameters (must include 'L' and 'N')
        output_folder: directory where PNG images will be saved
        matter_points: number of random matter points to overlay
        
    Returns:
        List of saved image file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    L = sim_params["L"]
    N = sim_params["N"]
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Generate random matter points (for overlay)
    matter_x = np.random.uniform(-L/2, L/2, matter_points)
    matter_y = np.random.uniform(-L/2, L/2, matter_points)
    
    # Save individual snapshot images every n-th frame
    n_skip = max(1, len(snapshots)//10)  # save at most 10 images
    saved_images = []
    for idx in range(0, len(snapshots), n_skip):
        field = snapshots[idx]
        slice_field = field[:, :, N // 2]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(slice_field, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap=cm.viridis)
        ax.scatter(matter_x, matter_y, c='red', marker='o', s=10, label="Matter Points")
        ax.set_title(f"Snapshot {idx+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        filename = os.path.join(output_folder, f"snapshot_{idx+1}.png")
        fig.savefig(filename)
        plt.close(fig)
        saved_images.append(filename)
    
    # Create a summary collage from the first, middle, and last snapshots
    if len(snapshots) >= 3:
        indices = [0, len(snapshots)//2, len(snapshots)-1]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, idx in zip(axes, indices):
            field = snapshots[idx]
            slice_field = field[:, :, N // 2]
            ax.imshow(slice_field, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap=cm.viridis)
            ax.set_title(f"Snapshot {idx+1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.suptitle("Summary Collage: First, Middle, and Last Snapshots")
        collage_path = os.path.join(output_folder, "collage_summary.png")
        fig.savefig(collage_path)
        plt.close(fig)
        saved_images.append(collage_path)
    
    print(f"[INFO] Visualization charts saved in {output_folder}")
    return saved_images

def create_results_chart(optimization_summary, output_folder):
    """
    Create a chart showing the distribution of candidate results (slope vs energy error)
    and highlight the current best candidate relative to the target region.
    
    Parameters:
        optimization_summary: dictionary loaded from optimization_summary.json.
        output_folder: folder to save the chart image.
    
    Returns:
        The path to the saved chart image.
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
    energy_threshold = 0.1  # example threshold
    plt.axvspan(target_slope - slope_tolerance, target_slope + slope_tolerance, color='green', alpha=0.2, label='Target Slope Range')
    plt.axhline(energy_threshold, color='orange', linestyle='--', label='Energy Error Threshold')
    
    plt.xlabel("Noise Exponent (Slope)")
    plt.ylabel("Relative Energy Error")
    plt.title("Optimization Candidate Results")
    plt.legend()
    chart_path = os.path.join(output_folder, "optimization_results_chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"[INFO] Results chart saved to {chart_path}")
    return chart_path

def create_interactive_animation_window(snapshots, sim_params, matter_points_count=50):
    """
    Create an interactive animation window showing the evolution of the central slice and
    simulating the movement of matter points under the influence of the field's gradient.
    
    Matter point dynamics are approximated using simple Euler integration:
      v(t+dt) = v(t) - a * dt,  x(t+dt) = x(t) + v(t) * dt,
    where acceleration a is estimated as the negative gradient of the central slice.
    
    Parameters:
        snapshots: list of 3D numpy arrays (simulation snapshots)
        sim_params: dictionary with simulation parameters ('L', 'N', 'steps_per_cycle', 'num_cycles', 'dt')
        matter_points_count: number of matter points to simulate
    
    This function opens a new Tkinter Toplevel window with an embedded matplotlib figure and a slider.
    """
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    
    # Compute total simulation time and dt per snapshot.
    total_time = sim_params["steps_per_cycle"] * sim_params["num_cycles"] * sim_params["dt"]
    num_frames = len(snapshots)
    dt_snapshot = total_time / num_frames
    
    L = sim_params["L"]
    N = sim_params["N"]
    x_vals = np.linspace(-L/2, L/2, N)
    y_vals = np.linspace(-L/2, L/2, N)
    
    # Precompute central slices for each frame.
    central_slices = [snap[:, :, N//2] for snap in snapshots]
    
    # Initialize matter points: random positions and zero initial velocities.
    mp_x = np.random.uniform(-L/2, L/2, matter_points_count)
    mp_y = np.random.uniform(-L/2, L/2, matter_points_count)
    mp_vx = np.zeros(matter_points_count)
    mp_vy = np.zeros(matter_points_count)
    
    # Precompute matter points positions for each frame.
    matter_positions = []
    for frame in range(num_frames):
        field_slice = central_slices[frame]
        # Compute gradients along y and x (note: numpy.gradient returns [grad_y, grad_x])
        grad_y, grad_x = np.gradient(field_slice, x_vals, y_vals)
        # Assume acceleration = -gradient.
        new_x = []
        new_y = []
        for i in range(matter_points_count):
            # Find nearest grid indices for the current matter point.
            idx_x = np.clip(np.searchsorted(x_vals, mp_x[i]), 0, N-1)
            idx_y = np.clip(np.searchsorted(y_vals, mp_y[i]), 0, N-1)
            a_x = -grad_x[idx_y, idx_x]
            a_y = -grad_y[idx_y, idx_x]
            mp_vx[i] = mp_vx[i] + a_x * dt_snapshot
            mp_vy[i] = mp_vy[i] + a_y * dt_snapshot
            mp_x[i] = mp_x[i] + mp_vx[i] * dt_snapshot
            mp_y[i] = mp_y[i] + mp_vy[i] * dt_snapshot
            new_x.append(mp_x[i])
            new_y.append(mp_y[i])
        matter_positions.append((np.array(new_x), np.array(new_y)))
    
    # Create a new Toplevel window for interactive animation.
    win = tk.Toplevel()
    win.title("Interactive Animation")
    fig, ax = plt.subplots(figsize=(6,6))
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    slider = tk.Scale(win, from_=0, to=num_frames-1, orient=tk.HORIZONTAL, label="Frame")
    slider.pack(fill=tk.X)
    
    def update_frame(val):
        frame_idx = int(slider.get())
        ax.clear()
        field_slice = central_slices[frame_idx]
        ax.imshow(field_slice, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='viridis')
        mp_pos = matter_positions[frame_idx]
        ax.scatter(mp_pos[0], mp_pos[1], c='red', s=30, label="Matter Points")
        ax.set_title(f"Frame {frame_idx+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        canvas.draw()
    
    slider.configure(command=update_frame)
    update_frame(0)
    win.mainloop()

