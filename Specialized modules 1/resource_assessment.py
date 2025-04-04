#!/usr/bin/env python3
"""
Module: resource_assessment.py
Provides functions for checking system resources and estimating simulation runtime.
"""

import psutil
import time
import numpy as np
from simulation_physics import run_field_simulation_3D

def check_resources():
    """
    Returns available memory (in GB) and current CPU load (as a percentage).
    """
    mem = psutil.virtual_memory()
    avail_mem_gb = mem.available / (1024**3)
    cpu_load = psutil.cpu_percent(interval=1)
    return avail_mem_gb, cpu_load

def estimate_simulation_time(params, test_steps=5):
    """
    Estimates the simulation time by running a short test simulation (with test_steps)
    and scaling the result to the full simulation duration.
    """
    test_params = params.copy()
    original_steps = test_params["steps_per_cycle"]
    test_params["steps_per_cycle"] = test_steps
    test_params["num_cycles"] = 1
    start = time.time()
    _ = run_field_simulation_3D(test_params, snapshot_interval=test_steps)
    test_time = time.time() - start
    total_steps = params["steps_per_cycle"] * params["num_cycles"]
    estimated = (test_time / test_steps) * total_steps
    # Restore original steps (if needed elsewhere)
    test_params["steps_per_cycle"] = original_steps
    return estimated

