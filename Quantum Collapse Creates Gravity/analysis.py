#!/usr/bin/env python3
"""
Module: analysis.py
Automates systematic error analysis and convergence studies.
"""

import os
import json
import logging
from simulation import convergence_test

def run_systematic_error_analysis(base_params, resolutions, time_steps, ensemble_runs=3, output_folder="analysis_reports"):
    """
    Run convergence tests over given resolutions and time steps and save a detailed report.
    """
    results = convergence_test(base_params, resolutions, time_steps, ensemble_runs)
    
    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, "systematic_error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    text_path = os.path.join(output_folder, "systematic_error_analysis.txt")
    with open(text_path, "w") as f:
        f.write("Systematic Error Analysis Report\n")
        f.write("===============================\n\n")
        for res in results:
            f.write(f"Grid Resolution N={res['N']}, Time Step dt={res['dt']}\n")
            f.write(f"  Average Slope: {res['avg_slope']:.3f} ± {res['std_slope']:.3f}\n")
            f.write(f"  Average Energy Error: {res['avg_energy_error']:.3f} ± {res['std_energy_error']:.3f}\n")
            f.write(f"  Average Runtime: {res['avg_runtime']:.3f} seconds\n\n")
    
    logging.info(f"Systematic error analysis reports saved to {output_folder}")
    return results

