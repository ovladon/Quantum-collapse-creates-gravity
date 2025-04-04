#!/usr/bin/env python3
"""
Module: optimization.py
Implements an automatic parameter optimization routine that runs multiple simulation tests,
archives results (CSV and JSON), and refines parameter ranges based on a target noise exponent.
"""

import os
import csv
import datetime
import time
import numpy as np
import json
from simulation_physics import run_field_simulation_3D
from resource_assessment import check_resources, estimate_simulation_time

def create_results_folder(base_folder="results"):
    """Create and return a timestamped folder for storing results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(base_folder, f"opt_results_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, timestamp

def sample_parameters(param_ranges):
    """Randomly sample parameter values within the given ranges."""
    return {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}

def fitness(slope, target=-5):
    """
    Define the fitness function as the negative absolute difference from the target slope.
    """
    if slope is None:
        return -float("inf")
    return -abs(slope - target)

def run_optimization(num_iterations=5, samples_per_iteration=10, initial_param_ranges=None, base_params=None):
    """
    Run evolutionary optimization over the parameter space.
    
    Returns a dictionary containing:
       - best_overall: the best configuration (parameters and slope) found.
       - all_iterations: list of all iteration results.
       - results_folder: folder where CSV files are stored.
       - timestamp: run timestamp.
       - base_params: the final simulation parameters.
       - final_param_ranges: the final parameter ranges.
    """
    if initial_param_ranges is None:
        initial_param_ranges = {
            "collapse_rate": (0.1, 0.5),
            "collapse_sigma": (0.1, 0.3),
            "collapse_amplitude": (0.5, 1.5),
            "continuous_noise_amplitude": (0.005, 0.02),
            "density_decay": (0.95, 0.99),
            "relativistic_factor": (0.0, 0.01)
        }
    if base_params is None:
        base_params = {
            "G": 1.0,
            "L": 10.0,
            "N": 64,
            "steps_per_cycle": 50,
            "num_cycles": 2,
            "dt": 0.05,
            "m": 1.0
        }
    results_folder, run_timestamp = create_results_folder()
    
    best_configurations = []
    all_iteration_results = []
    param_ranges = initial_param_ranges.copy()
    
    for iteration in range(num_iterations):
        avail_mem, cpu_load = check_resources()
        print(f"[INFO] Iteration {iteration+1}: Available Memory = {avail_mem:.2f}GB, CPU Load = {cpu_load:.1f}%")
        # Estimate simulation time for current settings
        sim_params = {**base_params, **{k: np.mean(param_ranges[k]) for k in param_ranges}}
        est_time = estimate_simulation_time(sim_params, test_steps=5)
        print(f"[INFO] Estimated simulation time: {est_time:.1f} seconds")
        
        iteration_results = []
        for sample in range(samples_per_iteration):
            config_sample = sample_parameters(param_ranges)
            sim_params_sample = {**base_params, **config_sample}
            
            # Warn if system resources are low
            avail_mem, cpu_load = check_resources()
            if avail_mem < 2.0 or cpu_load > 80.0:
                print("[WARNING] Limited system resources; simulation may take longer.")
            
            try:
                result = run_field_simulation_3D(sim_params_sample, snapshot_interval=sim_params_sample["steps_per_cycle"])
                slope = result["slope"]
            except Exception as e:
                print(f"[ERROR] Simulation failed for config {config_sample}: {e}")
                slope = None
            
            fit = fitness(slope)
            iteration_results.append((config_sample, fit, slope))
            print(f"[ITERATION {iteration+1}] Sample {sample+1}: Config: {config_sample} -> Slope: {slope}, Fitness: {fit}")
        
        # Archive iteration results to CSV
        csv_file = os.path.join(results_folder, f"iter_{iteration+1}_results_{run_timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["collapse_rate", "collapse_sigma", "collapse_amplitude",
                             "continuous_noise_amplitude", "density_decay",
                             "relativistic_factor", "slope"])
            for config, fit_val, slope_val in iteration_results:
                writer.writerow([config["collapse_rate"], config["collapse_sigma"],
                                 config["collapse_amplitude"], config["continuous_noise_amplitude"],
                                 config["density_decay"], config["relativistic_factor"],
                                 slope_val])
        
        # Select the best samples from this iteration
        iteration_results.sort(key=lambda x: x[1], reverse=True)
        top_n = max(1, int(0.2 * len(iteration_results)))
        best_configs = iteration_results[:top_n]
        best_configurations.extend(best_configs)
        all_iteration_results.append(iteration_results)
        
        # Refine parameter ranges based on the best configurations
        for key in param_ranges:
            values = [cfg[0][key] for cfg in best_configs if cfg[0][key] is not None]
            if values:
                mean_val = np.mean(values)
                current_min, current_max = param_ranges[key]
                width = current_max - current_min
                new_min = max(current_min, mean_val - 0.25 * width)
                new_max = min(current_max, mean_val + 0.25 * width)
                if new_max - new_min < 1e-4:
                    new_min, new_max = current_min, current_max
                param_ranges[key] = (new_min, new_max)
        print(f"[ITERATION {iteration+1}] Updated parameter ranges: {param_ranges}")
        
        # Increase resolution every 2 iterations if resources permit
        if (iteration + 1) % 2 == 0:
            avail_mem, _ = check_resources()
            if avail_mem > 4.0:
                old_N = base_params["N"]
                base_params["N"] = int(old_N * 1.5)
                base_params["steps_per_cycle"] = int(base_params["steps_per_cycle"] * 1.2)
                base_params["num_cycles"] = int(base_params["num_cycles"] * 1.2)
                print(f"[INFO] Increasing simulation resolution: N from {old_N} to {base_params['N']}")
            else:
                print("[INFO] Not enough extra memory to increase resolution.")
    
    best_overall = max(best_configurations, key=lambda x: x[1])
    print(f"[INFO] Best overall configuration: {best_overall[0]} with slope {best_overall[2]}")
    
    final_csv = os.path.join(results_folder, f"final_results_{run_timestamp}.csv")
    with open(final_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["collapse_rate", "collapse_sigma", "collapse_amplitude",
                         "continuous_noise_amplitude", "density_decay",
                         "relativistic_factor", "slope"])
        writer.writerow([best_overall[0]["collapse_rate"], best_overall[0]["collapse_sigma"],
                         best_overall[0]["collapse_amplitude"], best_overall[0]["continuous_noise_amplitude"],
                         best_overall[0]["density_decay"], best_overall[0]["relativistic_factor"],
                         best_overall[2]])
    
    optimization_results = {
        "best_overall": best_overall,
        "all_iterations": all_iteration_results,
        "results_folder": results_folder,
        "timestamp": run_timestamp,
        "base_params": base_params,
        "final_param_ranges": param_ranges
    }
    # Archive the optimization summary to a JSON file
    summary_path = os.path.join(results_folder, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(optimization_results, f, indent=4)
    
    return optimization_results

