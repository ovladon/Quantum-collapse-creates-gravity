#!/usr/bin/env python3
"""
Module: main.py
Ties together the simulation, optimization, and visualization modules to run the emergent gravity app.
Usage:
  --optimize    Run the parameter optimization routine.
  --simulate    Run a simulation with the best parameters from optimization.
  --visualize   Generate a video from archived simulation snapshots.
"""

import argparse
import os
import json
from optimization import run_optimization
from visualization import create_visualization

def main():
    parser = argparse.ArgumentParser(description="Emergent Gravity Simulation Application")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--simulate", action="store_true", help="Run a simulation with best parameters from optimization")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization from simulation snapshots")
    args = parser.parse_args()
    
    if args.optimize:
        print("[INFO] Starting optimization...")
        optimization_results = run_optimization(num_iterations=5, samples_per_iteration=5)
        print(f"[INFO] Optimization complete. Results archived in {optimization_results['results_folder']}")
    
    if args.simulate:
        opt_folder = input("Enter folder path containing optimization_summary.json: ").strip()
        summary_path = os.path.join(opt_folder, "optimization_summary.json")
        with open(summary_path, "r") as f:
            optimization_results = json.load(f)
        # best_overall is stored as a tuple: [config, fitness, slope]
        best_config = optimization_results["best_overall"][0]
        base_params = optimization_results["base_params"]
        sim_params = base_params.copy()
        sim_params.update(best_config)
        
        # Run the simulation with the selected parameters
        from simulation_physics import run_field_simulation_3D
        print("[INFO] Running simulation with best parameters from optimization...")
        simulation_result = run_field_simulation_3D(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
        
        sim_folder = os.path.join(opt_folder, "simulation_archive")
        os.makedirs(sim_folder, exist_ok=True)
        # Save simulation results summary
        with open(os.path.join(sim_folder, "simulation_results.txt"), "w") as f:
            f.write(f"Final slope: {simulation_result['slope']}\n")
            f.write(f"Runtime: {simulation_result['runtime']} seconds\n")
        # Save snapshots (as a NumPy file) for visualization
        import numpy as np
        np.save(os.path.join(sim_folder, "snapshots.npy"), simulation_result["snapshots"])
        # Also save the simulation parameters used
        with open(os.path.join(sim_folder, "simulation_params.json"), "w") as f:
            json.dump(sim_params, f, indent=4)
        print(f"[INFO] Simulation complete. Results saved in {sim_folder}")
    
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
    
if __name__ == "__main__":
    main()

