#!/usr/bin/env python3
"""
Module: visualization.py
Creates visualizations from simulation snapshots and optimization results.
Includes a video visualization and a 'results chart' showing the candidate's performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation

def create_visualization(snapshots, params, output_folder, matter_points=100):
    """
    Generate a video visualization from simulation snapshots.
    
    Parameters:
        snapshots: list of 3D field snapshots.
        params: simulation parameters dictionary.
        output_folder: folder to save the video and frames.
        matter_points: number of random matter points to overlay.
    """
    os.makedirs(output_folder, exist_ok=True)
    L = params["L"]
    N = params["N"]
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Generate random matter points
    matter_x = np.random.uniform(-L/2, L/2, matter_points)
    matter_y = np.random.uniform(-L/2, L/2, matter_points)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def animate(i):
        ax.clear()
        # Extract a central slice (z = N//2)
        field = snapshots[i]
        slice_field = field[:, :, N // 2]
        im = ax.imshow(slice_field, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap=cm.viridis)
        ax.scatter(matter_x, matter_y, c='red', marker='o', s=10)
        ax.set_title(f"Snapshot {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return im,
    
    ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=200, blit=False)
    video_path = os.path.join(output_folder, "simulation_video.mp4")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='EmergentGravitySim'), bitrate=1800)
    ani.save(video_path, writer=writer)
    plt.close(fig)
    print(f"[INFO] Visualization video saved to {video_path}")

def create_results_chart(optimization_summary, output_folder):
    """
    Create a chart showing the distribution of candidate results (slope vs energy error)
    and highlight the current best candidate relative to the target region.
    
    Parameters:
        optimization_summary: dictionary loaded from optimization_summary.json.
        output_folder: folder to save the chart image.
    """
    all_generations = optimization_summary["all_generations"]
    slopes = []
    energy_errors = []
    for gen in all_generations:
        for res in gen:
            if res["slope"] is not None and res["energy_error"] is not None:
                slopes.append(res["slope"])
                energy_errors.append(res["energy_error"])
    
    best = optimization_summary["best_overall"]
    best_slope = best["slope"]
    best_energy_error = best["energy_error"]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(slopes, energy_errors, c='blue', label='Candidates', alpha=0.6)
    plt.scatter([best_slope], [best_energy_error], c='red', label='Best Candidate', s=100)
    
    # Draw target region for slope (TARGET_SLOPE +/- tolerance) and low energy error
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

