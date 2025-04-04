#!/usr/bin/env python3
"""
Module: data_comparison.py
Provides functions to compare simulation output with real-world data.
This module loads simulation results and observational data from CSV files,
and generates comparative plots to assess how well the simulation matches real-life observations.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import logging

def load_simulation_results(sim_folder):
    """
    Load simulation parameters and metrics from the simulation archive.
    Assumes simulation_params.json and simulation_results.txt exist in sim_folder.
    """
    params_path = os.path.join(sim_folder, "simulation_params.json")
    with open(params_path, "r") as f:
        sim_params = json.load(f)
    results_path = os.path.join(sim_folder, "simulation_results.txt")
    results = {}
    with open(results_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    results[key.strip()] = float(value.strip())
                except ValueError:
                    results[key.strip()] = value.strip()
    return sim_params, results

def load_observational_data(csv_file):
    """
    Load observational data from a CSV file.
    The CSV file should have headers, e.g., "Measurement", "Value".
    Returns a dictionary of measurements.
    """
    obs_data = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurement = row.get("Measurement", "").strip()
            try:
                value = float(row.get("Value", "0"))
            except ValueError:
                value = row.get("Value", "").strip()
            obs_data[measurement] = value
    return obs_data

def compare_simulation_to_observations(sim_folder, obs_csv, output_folder="comparison_reports"):
    """
    Compare simulation results with observational data.
    Generates a bar plot comparing simulation metrics and observations.
    """
    sim_params, sim_results = load_simulation_results(sim_folder)
    obs_data = load_observational_data(obs_csv)
    
    # For demonstration, compare slope and energy error.
    sim_slope = sim_results.get("Final slope", None)
    sim_energy_error = sim_results.get("Energy error", None)
    
    obs_slope = obs_data.get("Target slope", None)
    obs_energy_error = obs_data.get("Energy error", None)
    
    labels = []
    sim_values = []
    obs_values = []
    
    if sim_slope is not None and obs_slope is not None:
        labels.append("Slope")
        sim_values.append(sim_slope)
        obs_values.append(obs_slope)
    if sim_energy_error is not None and obs_energy_error is not None:
        labels.append("Energy Error")
        sim_values.append(sim_energy_error)
        obs_values.append(obs_energy_error)
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, sim_values, width, label="Simulation")
    rects2 = ax.bar(x + width/2, obs_values, width, label="Observations")
    
    ax.set_ylabel("Values")
    ax.set_title("Simulation vs Observations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "comparison_plot.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Comparison plot saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare simulation results with observational data.")
    parser.add_argument("sim_folder", type=str, help="Folder containing simulation archive files (simulation_params.json and simulation_results.txt)")
    parser.add_argument("obs_csv", type=str, help="CSV file with observational data (columns: Measurement, Value)")
    parser.add_argument("--output_folder", type=str, default="comparison_reports", help="Folder to save comparison plot")
    args = parser.parse_args()
    
    compare_simulation_to_observations(args.sim_folder, args.obs_csv, output_folder=args.output_folder)

