#!/usr/bin/env python3
"""
Module: main.py
Ties together the simulation, optimization, and visualization modules for the emergent gravity simulation app.
Usage:
  --optimize    Run genetic algorithm parameter optimization.
  --simulate    Run a full simulation with best parameters from optimization.
  --visualize   Generate video visualization from simulation snapshots.
  --convergence Run convergence tests by varying resolution and time step.
"""

import argparse
import os
import json
from optimization import genetic_algorithm_optimization
from visualization import create_visualization, create_results_chart

def main():
    parser = argparse.ArgumentParser(description="Emergent Gravity Simulation Application")
    parser.add_argument("--optimize", action="store_true", help="Run genetic algorithm parameter optimization")
    parser.add_argument("--simulate", action="store_true", help="Run a full simulation with best parameters from optimization")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization from simulation snapshots")
    parser.add_argument("--convergence", action="store_true", help="Run convergence tests")
    args = parser.parse_args()
    
    if args.optimize:
        print("[INFO] Starting genetic algorithm optimization...")
        optimization_results = genetic_algorithm_optimization(generations=5, population_size=10)
        print(f"[INFO] Optimization complete. Results archived in {optimization_results['results_folder']}")
    
    if args.simulate:
        opt_folder = input("Enter folder path containing optimization_summary.json: ").strip()
        summary_path = os.path.join(opt_folder, "optimization_summary.json")
        with open(summary_path, "r") as f:
            optimization_results = json.load(f)
        best_candidate = optimization_results["best_overall"]["candidate"]
        base_params = optimization_results["base_params"]
        sim_params = base_params.copy()
        sim_params.update(best_candidate)
        
        from simulation_physics import run_field_simulation_3D
        print("[INFO] Running full simulation with best parameters from optimization...")
        simulation_result = run_field_simulation_3D(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
        
        sim_folder = os.path.join(opt_folder, "simulation_archive")
        os.makedirs(sim_folder, exist_ok=True)
        with open(os.path.join(sim_folder, "simulation_results.txt"), "w") as f:
            f.write(f"Final slope: {simulation_result['slope']}\n")
            f.write(f"Energy error: {simulation_result['energy_error']}\n")
            f.write(f"Runtime: {simulation_result['runtime']} seconds\n")
        import numpy as np
        np.save(os.path.join(sim_folder, "snapshots.npy"), simulation_result["snapshots"])
        with open(os.path.join(sim_folder, "simulation_params.json"), "w") as f:
            json.dump(sim_params, f, indent=4)
        print(f"[INFO] Full simulation complete. Results saved in {sim_folder}")
    
    if args.visualize:
        sim_folder = input("Enter simulation archive folder path (containing snapshots.npy and simulation_params.json): ").strip()
        snapshots_path = os.path.join(sim_folder, "snapshots.npy")
        import numpy as np
        snapshots = np.load(snapshots_path, allow_pickle=True)
        params_path = os.path.join(sim_folder, "simulation_params.json")
        with open(params_path, "r") as f:
            sim_params = json.load(f)
        output_folder = os.path.join(sim_folder, "visualization")
        create_visualization(snapshots, sim_params, output_folder)
        
        # Optionally, generate the results chart if an optimization summary is available.
        opt_folder = input("Enter folder path containing optimization_summary.json for results chart (or leave blank to skip): ").strip()
        if opt_folder:
            summary_path = os.path.join(opt_folder, "optimization_summary.json")
            with open(summary_path, "r") as f:
                optimization_summary = json.load(f)
            create_results_chart(optimization_summary, output_folder)
    
    if args.convergence:
        from simulation_physics import convergence_test
        base_params = {
            "collapse_rate": 0.3,
            "collapse_sigma": 0.2,
            "collapse_amplitude": 1.0,
            "continuous_noise_amplitude": 0.01,
            "density_decay": 0.98,
            "relativistic_factor": 0.005,
            "G": 1.0,
            "L": 10.0,
            "N": 64,
            "steps_per_cycle": 50,
            "num_cycles": 2,
            "dt": 0.05,
            "m": 1.0,
            "boundary_condition": "periodic"
        }
        resolutions = [32, 64, 128]
        time_steps = [0.05, 0.025]
        conv_results = convergence_test(base_params, resolutions, time_steps)
        conv_output = os.path.join("results", "convergence_results.json")
        os.makedirs("results", exist_ok=True)
        with open(conv_output, "w") as f:
            json.dump(conv_results, f, indent=4)
        print(f"[INFO] Convergence test results saved to {conv_output}")

if __name__ == "__main__":
    main()

