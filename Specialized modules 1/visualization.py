#!/usr/bin/env python3
"""
Module: visualization.py
Creates visualizations (including a video) from the simulation snapshots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation

def create_visualization(snapshots, params, output_folder, matter_points=100):
    """
    Generate a video visualization from simulation snapshots.
    
    snapshots: list of 3D fields (numpy arrays) captured during simulation.
    params: simulation parameters dictionary (must include 'L' and 'N').
    output_folder: directory to save the video and frame images.
    matter_points: number of random matter points to overlay.
    """
    os.makedirs(output_folder, exist_ok=True)
    L = params["L"]
    N = params["N"]
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Generate random matter points within the simulation domain
    matter_x = np.random.uniform(-L/2, L/2, matter_points)
    matter_y = np.random.uniform(-L/2, L/2, matter_points)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def animate(i):
        ax.clear()
        # Extract a central slice (z = N//2) of the current snapshot
        field = snapshots[i]
        slice_field = field[:, :, N // 2]
        im = ax.imshow(slice_field, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap=cm.viridis)
        ax.scatter(matter_x, matter_y, c='red', marker='o')
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

