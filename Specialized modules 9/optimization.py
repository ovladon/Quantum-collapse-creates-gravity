#!/usr/bin/env python3
"""
Module: optimization.py
Implements a genetic algorithm (GA) for parameter optimization in the emergent gravity simulation.
This module uses ensemble runs to average simulation outcomes for each candidate and records not only
the fitness but also the uncertainty in the noise exponent (slope_std_error). This is important for
assessing parameter sensitivity.
"""

import os
import csv
import datetime
import time
import numpy as np
import json
import logging
import multiprocessing as mp
from simulation import run_field_simulation_3D_base as run_field_simulation_3D, convergence_test
from resource_assessment import check_resources, estimate_simulation_time

TARGET_SLOPE = -5.0
ALPHA_ENERGY = 10.0  # Weight for energy conservation error in fitness
DEFAULT_ENSEMBLE_RUNS = 5
DEFAULT_MUTATION_RATE = 0.2

def default_json_converter(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def create_results_folder(base_folder="results"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(base_folder, f"opt_results_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, timestamp

def random_candidate(param_ranges):
    return {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}

def mutate(candidate, param_ranges, mutation_rate):
    new_candidate = candidate.copy()
    for key, (low, high) in param_ranges.items():
        if np.random.rand() < mutation_rate:
            range_width = high - low
            perturbation = np.random.uniform(-0.1 * range_width, 0.1 * range_width)
            new_candidate[key] = np.clip(new_candidate[key] + perturbation, low, high)
    return new_candidate

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if np.random.rand() < 0.5 else parent2[key]
    return child

def evaluate_candidate(candidate, base_params, num_cycles_test=1, ensemble_runs=DEFAULT_ENSEMBLE_RUNS, mutation_rate=DEFAULT_MUTATION_RATE):
    """
    Evaluate a candidate by running a short simulation multiple times.
    Returns the candidate along with average fitness, slope, and energy error.
    Fitness is computed as -abs(avg_slope - TARGET_SLOPE) - ALPHA_ENERGY * avg_energy_error.
    """
    slopes = []
    energy_errors = []
    slope_std_errors = []
    for _ in range(ensemble_runs):
        sim_params = base_params.copy()
        sim_params.update(candidate)
        sim_params["num_cycles"] = num_cycles_test
        try:
            result = run_field_simulation_3D(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
            if result["slope"] is None:
                slopes.append(np.nan)
                energy_errors.append(np.nan)
                slope_std_errors.append(np.nan)
            else:
                slopes.append(result["slope"])
                energy_errors.append(result["energy_error"])
                slope_std_errors.append(result["slope_std_error"] if result["slope_std_error"] is not None else 0)
        except Exception as e:
            logging.error("Simulation failed for candidate %s: %s", candidate, e)
            slopes.append(np.nan)
            energy_errors.append(np.nan)
            slope_std_errors.append(np.nan)
    avg_slope = np.nanmean(slopes)
    avg_energy_error = np.nanmean(energy_errors)
    avg_slope_std_error = np.nanmean(slope_std_errors)
    if np.isnan(avg_slope):
        fitness_val = -1e6
    else:
        fitness_val = -abs(avg_slope - TARGET_SLOPE) - ALPHA_ENERGY * avg_energy_error
    return (candidate, fitness_val, avg_slope, avg_energy_error, avg_slope_std_error)

def genetic_algorithm_optimization(generations=5, population_size=10, base_params=None, initial_param_ranges=None, ensemble_runs_param=None, mutation_rate_param=None):
    """
    Run the genetic algorithm optimization.
    Parameters:
      generations: Number of generations.
      population_size: Number of candidates per generation.
      base_params: Base simulation parameters (dictionary).
      initial_param_ranges: Dictionary of parameter ranges for random candidate generation.
      ensemble_runs_param: Number of ensemble runs per candidate evaluation.
      mutation_rate_param: Mutation rate for candidate mutation.
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
            "m": 1.0,
            "boundary_condition": "periodic"
        }
    if ensemble_runs_param is None:
        ensemble_runs_param = DEFAULT_ENSEMBLE_RUNS
    if mutation_rate_param is None:
        mutation_rate_param = DEFAULT_MUTATION_RATE

    results_folder, run_timestamp = create_results_folder()
    population = [random_candidate(initial_param_ranges) for _ in range(population_size)]
    
    pool = mp.Pool(processes=min(mp.cpu_count(), population_size))
    
    all_generations = []
    best_candidate_overall = None
    best_fitness_overall = -1e6
    
    for gen in range(generations):
        logging.info("Starting generation %d/%d", gen+1, generations)
        tasks = [(candidate, base_params, 1, ensemble_runs_param, mutation_rate_param) for candidate in population]
        results = pool.starmap(evaluate_candidate, tasks)
        
        gen_results = []
        for cand, fit, slope, energy_error, slope_std_error in results:
            gen_results.append({"candidate": cand, "fitness": fit, "slope": slope,
                                "energy_error": energy_error, "slope_std_error": slope_std_error})
            if fit > best_fitness_overall:
                best_fitness_overall = fit
                best_candidate_overall = cand
        all_generations.append(gen_results)
        
        csv_file = os.path.join(results_folder, f"gen_{gen+1}_results_{run_timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["collapse_rate", "collapse_sigma", "collapse_amplitude",
                             "continuous_noise_amplitude", "density_decay",
                             "relativistic_factor", "fitness", "slope", "slope_std_error", "energy_error"])
            for res in gen_results:
                cand = res["candidate"]
                writer.writerow([cand["collapse_rate"], cand["collapse_sigma"],
                                 cand["collapse_amplitude"], cand["continuous_noise_amplitude"],
                                 cand["density_decay"], cand["relativistic_factor"],
                                 res["fitness"], res["slope"], res["slope_std_error"], res["energy_error"]])
        logging.info("Generation %d results saved in %s", gen+1, csv_file)
        
        sorted_population = sorted(results, key=lambda x: x[1], reverse=True)
        survivors = [cand for cand, fit, _, _, _ in sorted_population[:max(2, population_size//2)]]
        
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parents = np.random.choice(survivors, 2, replace=False)
            child = crossover(parents[0], parents[1])
            child = mutate(child, initial_param_ranges, mutation_rate=mutation_rate_param)
            new_population.append(child)
        population = new_population
    
    pool.close()
    pool.join()
    
    final_sim_params = base_params.copy()
    final_sim_params.update(best_candidate_overall)
    final_result = run_field_simulation_3D(final_sim_params, snapshot_interval=final_sim_params["steps_per_cycle"])
    
    best_overall = {
        "candidate": best_candidate_overall,
        "fitness": best_fitness_overall,
        "slope": final_result["slope"],
        "slope_std_error": final_result.get("slope_std_error", None),
        "energy_error": final_result["energy_error"],
        "final_result": "Excluded from JSON summary"
    }
    
    optimization_results = {
        "best_overall": best_overall,
        "all_generations": all_generations,
        "results_folder": results_folder,
        "timestamp": run_timestamp,
        "base_params": base_params,
        "initial_param_ranges": initial_param_ranges,
        "ensemble_runs": ensemble_runs_param,
        "mutation_rate": mutation_rate_param
    }
    summary_path = os.path.join(results_folder, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(optimization_results, f, indent=4, default=default_json_converter)
    logging.info("Optimization summary saved in %s", summary_path)
    return optimization_results

if __name__ == "__main__":
    gen_results = genetic_algorithm_optimization(generations=10, population_size=20, ensemble_runs_param=10, mutation_rate_param=0.2)
    print("Optimization complete. Results archived in:", gen_results["results_folder"])

