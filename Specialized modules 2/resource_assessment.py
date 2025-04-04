#!/usr/bin/env python3
"""
Module: resource_assessment.py
Provides functions for checking system resources and estimating simulation runtime.
Also includes helper functions to ensure that simulations do not overload system resources.
"""

import psutil
import time
import numpy as np
from simulation_physics import run_field_simulation_3D

def check_resources():
    """
    Check available system memory (in GB) and current CPU load (percentage).
    
    Returns:
        avail_mem_gb, cpu_load
    """
    mem = psutil.virtual_memory()
    avail_mem_gb = mem.available / (1024**3)
    cpu_load = psutil.cpu_percent(interval=1)
    return avail_mem_gb, cpu_load

def estimate_simulation_time(params, test_steps=5):
    """
    Estimate simulation time by running a short test simulation (with test_steps) and scaling.
    
    Parameters:
        params: simulation parameters dictionary.
        test_steps: number of steps to run for the test.
        
    Returns:
        Estimated total simulation time in seconds.
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
    test_params["steps_per_cycle"] = original_steps
    return estimated

