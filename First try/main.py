import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.fft import fftn, ifftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
import csv


# ------------------------------
# 1. Helper Functions
# ------------------------------

def gaussian_3d(X, Y, Z, x0, y0, z0, sigma):
    """Returns a normalized 3D Gaussian evaluated on X, Y, Z with center (x0, y0, z0) and width sigma."""
    norm = (2 * np.pi * sigma**2) ** (3/2)
    return np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2)) / norm

def solve_poisson(rho, dx, G):
    """Solves ∇²Φ = 4πG ρ via FFT on a periodic grid."""
    N = rho.shape[0]
    rho_k = fftn(rho)
    k = 2 * np.pi * fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k_squared = KX**2 + KY**2 + KZ**2
    k_squared[0, 0, 0] = 1.0  # avoid division by zero
    phi_k = -4 * np.pi * G * rho_k / k_squared
    phi_k[0, 0, 0] = 0.0  # set the mean to zero
    phi = np.real(ifftn(phi_k))
    return phi

def interpolate_gradient(phi, grid, dx):
    """Computes the gradient of phi and returns interpolators for each component."""
    grad = np.gradient(phi, dx, edge_order=2)
    interp_funcs = []
    for comp in grad:
        interp = RegularGridInterpolator(grid, comp, bounds_error=False, fill_value=None)
        interp_funcs.append(interp)
    return interp_funcs

def apply_periodic_bc(pos, L):
    """Applies periodic boundary conditions for a position vector pos in a domain [-L/2, L/2]."""
    return ((pos + L/2) % L) - L/2

# ------------------------------
# 2. Main Simulation Function with Improved Visualization
# ------------------------------

def main():
    # Simulation parameters
    G = 1.0
    L = 10.0
    N = 64
    dx = L / N
    total_steps = 200
    dt = 0.05

    # Collapse event parameters
    collapse_rate = 0.5
    collapse_sigma = 0.2
    collapse_amplitude = 1.0
    density_decay = 0.99

    # Test particle parameters
    num_particles = 10
    particle_positions = np.random.uniform(-L/2, L/2, (num_particles, 3))
    particle_velocities = np.zeros((num_particles, 3))
    particle_traj = [particle_positions.copy()]

    # Create 3D grid
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    grid = (x, x, x)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    rho = np.zeros((N, N, N))

    # For recording summary data
    avg_rho_list = []
    avg_phi_list = []
    time_list = []

    # Set up figure with 4 panels (density, potential, time series, trajectories)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_rho = axs[0, 0]
    ax_phi = axs[0, 1]
    ax_time = axs[1, 0]
    ax_traj = axs[1, 1]

    # Fixed color limits for consistency
    dens_clim = (0, collapse_amplitude * 2)
    pot_clim = (-5, 5)

    slice_index = N // 2
    dens_slice = rho[:, :, slice_index]
    im_rho = ax_rho.imshow(dens_slice, extent=(-L/2, L/2, -L/2, L/2), origin='lower',
                           cmap='viridis', vmin=dens_clim[0], vmax=dens_clim[1])
    ax_rho.set_title("Density Slice (z=0)")
    ax_rho.set_xlabel("x")
    ax_rho.set_ylabel("y")
    fig.colorbar(im_rho, ax=ax_rho)

    phi_initial = np.zeros_like(rho)
    im_phi = ax_phi.imshow(phi_initial[:, :, slice_index], extent=(-L/2, L/2, -L/2, L/2), origin='lower',
                           cmap='inferno', vmin=pot_clim[0], vmax=pot_clim[1])
    ax_phi.set_title("Potential Slice (z=0)")
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")
    fig.colorbar(im_phi, ax=ax_phi)

    ax_time.set_title("Time Series: Avg. Density & Potential")
    ax_time.set_xlabel("Time step")
    ax_time.set_ylabel("Value")
    time_line, = ax_time.plot([], [], 'b-', label="Avg. Density")
    pot_line, = ax_time.plot([], [], 'r-', label="Avg. Potential")
    ax_time.legend()

    ax_traj.set_title("Particle Trajectories (x-y)")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    traj_lines = [ax_traj.plot([], [], marker='o', linestyle='-')[0] for _ in range(num_particles)]
    ax_traj.set_xlim(-L/2, L/2)
    ax_traj.set_ylim(-L/2, L/2)

    def update(step):
        nonlocal rho, particle_positions, particle_velocities

        # (a) Add collapse events
        num_events = np.random.poisson(lam=collapse_rate)
        for _ in range(num_events):
            event_pos = np.random.uniform(-L/2, L/2, size=3)
            rho += collapse_amplitude * gaussian_3d(X, Y, Z,
                                                    event_pos[0],
                                                    event_pos[1],
                                                    event_pos[2],
                                                    collapse_sigma)

        # (b) Decay density
        rho *= density_decay

        # (c) Solve for gravitational potential
        phi = solve_poisson(rho, dx, G)

        # (d) Compute gradient for dynamics
        interp_grad = interpolate_gradient(phi, grid, dx)

        # (e) Update test particles
        for i in range(num_particles):
            particle_positions[i] = apply_periodic_bc(particle_positions[i], L)
            acc = -np.array([interp_grad[dim](particle_positions[i].reshape(1, 3))[0]
                             for dim in range(3)])
            particle_velocities[i] += acc * dt
            particle_positions[i] += particle_velocities[i] * dt
            particle_positions[i] = apply_periodic_bc(particle_positions[i], L)
        particle_traj.append(particle_positions.copy())

        # (f) Update summary data
        dens_slice = rho[:, :, slice_index]
        phi_slice = phi[:, :, slice_index]
        avg_rho = np.mean(dens_slice)
        avg_phi = np.mean(phi_slice)
        avg_rho_list.append(avg_rho)
        avg_phi_list.append(avg_phi)
        time_list.append(step)

        im_rho.set_data(dens_slice)
        im_phi.set_data(phi_slice)
        time_line.set_data(time_list, avg_rho_list)
        pot_line.set_data(time_list, avg_phi_list)
        ax_time.relim()
        ax_time.autoscale_view()

        for i, line in enumerate(traj_lines):
            line.set_data(particle_traj[-1][i, 0], particle_traj[-1][i, 1])

        fig.suptitle(f"Time step {step+1}/{total_steps}")
        return im_rho, im_phi, time_line, pot_line, *traj_lines

    try:
        anim = FuncAnimation(fig, update, frames=total_steps, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Save summary data
        with open("simulation_summary.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time Step", "Avg Density", "Avg Potential"])
            for t, rho_val, phi_val in zip(time_list, avg_rho_list, avg_phi_list):
                writer.writerow([t, rho_val, phi_val])
        print("Time series data saved as simulation_summary.csv")

        # Save animation as video (requires ffmpeg)
        writer_ffmpeg = FFMpegWriter(fps=20, metadata=dict(artist='YourName'), bitrate=1800)
        anim.save("simulation.mp4", writer=writer_ffmpeg)
        print("Animation saved as simulation.mp4")

if __name__ == '__main__':
    main()

