#!/usr/bin/env python3
"""
Module: gui_app.py
Provides a Tkinter-based GUI for the emergent gravity simulation project.
Includes tabs for optimization, simulation, convergence analysis, visualization, sensitivity analysis, and integrated run.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import json
import logging
import time

from optimization import genetic_algorithm_optimization
from simulation import run_field_simulation_3D_base, run_field_simulation_3D_adaptive, parameter_sensitivity_analysis, convergence_test
from visualization import create_visualization_with_test_particles, create_results_chart
from dynamic_visualization import launch_interactive_visualization

# Alias for modules that use run_field_simulation_3D
run_field_simulation_3D = run_field_simulation_3D_base

logging.basicConfig(level=logging.INFO)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emergent Gravity Simulation GUI")
        self.geometry("1000x800")
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both')

        self.optim_tab = ttk.Frame(notebook)
        notebook.add(self.optim_tab, text="Optimization")
        self.create_optim_tab()

        self.sim_tab = ttk.Frame(notebook)
        notebook.add(self.sim_tab, text="Simulation")
        self.create_sim_tab()

        self.conv_tab = ttk.Frame(notebook)
        notebook.add(self.conv_tab, text="Convergence Analysis")
        self.create_conv_tab()

        self.vis_tab = ttk.Frame(notebook)
        notebook.add(self.vis_tab, text="Visualization")
        self.create_vis_tab()

        self.sens_tab = ttk.Frame(notebook)
        notebook.add(self.sens_tab, text="Sensitivity Analysis")
        self.create_sens_tab()

        self.rep_tab = ttk.Frame(notebook)
        notebook.add(self.rep_tab, text="Reports")
        self.create_rep_tab()

        self.int_tab = ttk.Frame(notebook)
        notebook.add(self.int_tab, text="Integrated Run")
        self.create_int_tab()

    def log(self, widget, message):
        widget.insert(tk.END, message + "\n")
        widget.see(tk.END)

    # Optimization Tab
    def create_optim_tab(self):
        frame = self.optim_tab
        self.optim_output = scrolledtext.ScrolledText(frame, height=20)
        self.optim_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn = ttk.Button(frame, text="Run Optimization", command=self.run_optim_thread)
        btn.pack(pady=5)

    def run_optim_thread(self):
        thread = threading.Thread(target=self.run_optimization)
        thread.start()

    def run_optimization(self):
        self.log(self.optim_output, "[INFO] Starting genetic algorithm optimization...")
        try:
            opt_results = genetic_algorithm_optimization(generations=5, population_size=10)
            self.log(self.optim_output, f"[INFO] Optimization complete. Results saved in: {opt_results['results_folder']}")
        except Exception as e:
            self.log(self.optim_output, f"[ERROR] {str(e)}")

    # Simulation Tab
    def create_sim_tab(self):
        frame = self.sim_tab
        lbl = ttk.Label(frame, text="Select Optimization Summary File (optimization_summary.json):")
        lbl.pack(pady=5)
        self.sim_summary_path = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.sim_summary_path, width=50)
        entry.pack(pady=5)
        btn_browse = ttk.Button(frame, text="Browse", command=self.browse_sim_summary)
        btn_browse.pack(pady=5)
        btn_run = ttk.Button(frame, text="Run Full Simulation", command=self.run_sim_thread)
        btn_run.pack(pady=5)
        self.sim_output = scrolledtext.ScrolledText(frame, height=15)
        self.sim_output.pack(fill='both', expand=True, padx=10, pady=10)

    def browse_sim_summary(self):
        path = filedialog.askopenfilename(title="Select optimization_summary.json", filetypes=[("JSON Files", "*.json")])
        if path:
            self.sim_summary_path.set(path)

    def run_sim_thread(self):
        thread = threading.Thread(target=self.run_simulation)
        thread.start()

    def run_simulation(self):
        summary_file = self.sim_summary_path.get()
        if not summary_file:
            messagebox.showerror("Error", "Please select an optimization summary file.")
            return
        self.log(self.sim_output, f"[INFO] Loading optimization summary from: {summary_file}")
        try:
            with open(summary_file, "r") as f:
                opt_results = json.load(f)
            best_candidate = opt_results["best_overall"]["candidate"]
            base_params = opt_results["base_params"]
            sim_params = base_params.copy()
            sim_params.update(best_candidate)
            self.log(self.sim_output, "[INFO] Running full simulation with best candidate parameters...")
            sim_result = run_field_simulation_3D_base(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
            self.log(self.sim_output, f"[INFO] Simulation complete. Final slope: {sim_result['slope']}, Energy error: {sim_result.get('energy_error','N/A')}")
        except Exception as e:
            self.log(self.sim_output, f"[ERROR] {str(e)}")

    # Convergence Analysis Tab
    def create_conv_tab(self):
        frame = self.conv_tab
        lbl = ttk.Label(frame, text="Convergence Analysis: Automatic systematic error analysis")
        lbl.pack(pady=5)
        self.conv_output = scrolledtext.ScrolledText(frame, height=20)
        self.conv_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn = ttk.Button(frame, text="Run Convergence Analysis", command=self.run_conv_thread)
        btn.pack(pady=5)

    def run_conv_thread(self):
        thread = threading.Thread(target=self.run_convergence)
        thread.start()

    def run_convergence(self):
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
        self.log(self.conv_output, "[INFO] Running convergence analysis...")
        try:
            results = convergence_test(base_params, resolutions, time_steps, ensemble_runs=3)
            self.log(self.conv_output, "[INFO] Convergence analysis complete. See analysis_reports folder for detailed report.")
        except Exception as e:
            self.log(self.conv_output, f"[ERROR] {str(e)}")

    # Visualization Tab
    def create_vis_tab(self):
        frame = self.vis_tab
        explanation = (
            "Visualization Section:\n"
            "------------------------\n"
            "This section produces PNG charts from simulation snapshots with test particle trajectories\n"
            "and provides an interactive animation.\n"
            "Ensure that the simulation archive folder contains snapshots.npy and simulation_params.json."
        )
        lbl_exp = ttk.Label(frame, text=explanation, wraplength=800, justify="left")
        lbl_exp.pack(pady=5)
    
        lbl = ttk.Label(frame, text="Select Simulation Archive Folder:")
        lbl.pack(pady=5)
        self.vis_folder = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.vis_folder, width=50)
        entry.pack(pady=5)
        btn_browse = ttk.Button(frame, text="Browse", command=self.browse_vis_folder)
        btn_browse.pack(pady=5)
    
        btn_run_charts = ttk.Button(frame, text="Generate PNG Charts", command=self.run_vis_thread)
        btn_run_charts.pack(pady=5)
    
        btn_run_animation = ttk.Button(frame, text="Launch Interactive Animation", command=self.launch_interactive_animation)
        btn_run_animation.pack(pady=5)
    
        self.vis_output = scrolledtext.ScrolledText(frame, height=15)
        self.vis_output.pack(fill='both', expand=True, padx=10, pady=10)

    def browse_vis_folder(self):
        folder = filedialog.askdirectory(title="Select Simulation Archive Folder")
        if folder:
            self.vis_folder.set(folder)

    def run_vis_thread(self):
        thread = threading.Thread(target=self.run_visualization)
        thread.start()

    def run_visualization(self):
        folder = self.vis_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select a simulation archive folder.")
            return
        self.log(self.vis_output, f"[INFO] Running visualization for folder: {folder}")
        try:
            import numpy as np
            snapshots_path = os.path.join(folder, "snapshots.npy")
            with open(os.path.join(folder, "simulation_params.json"), "r") as f:
                sim_params = json.load(f)
            snapshots = np.load(snapshots_path, allow_pickle=True)
            output_folder = os.path.join(folder, "visualization_charts")
            saved_files = create_visualization_with_test_particles(snapshots, sim_params, output_folder,
                                                                   dt_effect=0.1, num_particles=10)
            self.log(self.vis_output, f"[INFO] Visualization charts created: {saved_files}")
        except Exception as e:
            self.log(self.vis_output, f"[ERROR] {str(e)}")

    def launch_interactive_animation(self):
        folder = self.vis_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select a simulation archive folder.")
            return
        self.log(self.vis_output, f"[INFO] Launching interactive animation for folder: {folder}")
        try:
            launch_interactive_visualization(folder, interp_factor=10, dt_effect=0.1, num_particles=10)
        except Exception as e:
            self.log(self.vis_output, f"[ERROR] {str(e)}")

    # Sensitivity Analysis Tab
    from simulation import run_field_simulation_3D_base as run_field_simulation_3D
    def create_sens_tab(self):
        frame = self.sens_tab
        lbl = ttk.Label(frame, text="Sensitivity Analysis:\nSystematically explore tolerance and integration parameters\n(on a smaller simulation).")
        lbl.pack(pady=5)
        self.sens_output = scrolledtext.ScrolledText(frame, height=20)
        self.sens_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_run = ttk.Button(frame, text="Run Sensitivity Analysis", command=self.run_sensitivity_thread)
        btn_run.pack(pady=5)
        self.sens_progress = ttk.Label(frame, text="Progress: Not started")
        self.sens_progress.pack(pady=5)

    def run_sensitivity_thread(self):
        thread = threading.Thread(target=self.run_sensitivity_analysis)
        thread.start()

    def sens_progress_callback(self, message):
        self.sens_progress.config(text=message)
        self.sens_output.insert(tk.END, message + "\n")
        self.sens_output.see(tk.END)

    def run_sensitivity_analysis(self):
        base_params = {
            "collapse_rate": 0.3,
            "collapse_sigma": 0.2,
            "collapse_amplitude": 1.0,
            "continuous_noise_amplitude": 0.01,
            "density_decay": 0.98,
            "relativistic_factor": 0.005,
            "G": 1.0,
            "L": 10.0,
            "N": 32,
            "steps_per_cycle": 20,
            "num_cycles": 1,
            "dt": 0.05,
            "m": 1.0,
            "boundary_condition": "periodic"
        }
        param_grid = {
            "dt": [0.05, 0.025, 0.1],
            "tolerance": [1e-3, 5e-3, 1e-2]
        }
        results_summary = []  # Ensure this is defined before the loop
        for dt_val in param_grid["dt"]:
            for tol in param_grid["tolerance"]:
                params = base_params.copy()
                params["dt"] = dt_val
                self.sens_progress_callback(f"Running simulation with dt={dt_val}, tolerance={tol}...")
                # Use the base integrator for sensitivity analysis
                from simulation import run_field_simulation_3D_base as run_field_simulation_3D
                sim_result = run_field_simulation_3D(params, snapshot_interval=5)
                result_entry = {
                    "dt": dt_val,
                    "tolerance": tol,
                    "slope": sim_result["slope"],
                    "energy_error": sim_result["energy_error"],
                    "simulation_time": sim_result["simulation_time"],
                    "wall_time": sim_result["wall_time"]
                }
                results_summary.append(result_entry)
                self.sens_progress_callback(
                    f"Completed dt={dt_val}, tol={tol}: slope={sim_result['slope']:.3f}, energy_error={sim_result['energy_error']:.3e}"
                )
        import os, json
        output_folder = "sensitivity_reports"
        os.makedirs(output_folder, exist_ok=True)
        json_path = os.path.join(output_folder, "sensitivity_analysis.json")
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=4)
        self.sens_progress_callback(f"Sensitivity analysis complete. Results saved to {json_path}")


    # Reports Tab
    def create_rep_tab(self):
        frame = self.rep_tab
        lbl = ttk.Label(frame, text="Reports and Findings")
        lbl.pack(pady=5)
        self.rep_output = scrolledtext.ScrolledText(frame, height=25)
        self.rep_output.pack(fill='both', expand=True, padx=10, pady=10)
        report_text = (
            "Integrated Emergent Gravity Simulation Report\n"
            "-------------------------------------------------\n"
            "This application automates simulation, optimization, convergence analysis, visualization,\n"
            "and sensitivity analysis of collapse dynamics for emergent gravity.\n"
            "Refer to the generated files in the output folders for detailed reports and plots.\n"
        )
        self.rep_output.insert(tk.END, report_text)

    # Integrated Run Tab
    def create_int_tab(self):
        frame = self.int_tab
        lbl = ttk.Label(frame, text="Integrated Run: Execute all modules sequentially")
        lbl.pack(pady=5)
        btn = ttk.Button(frame, text="Run All Modules", command=self.run_int_thread)
        btn.pack(pady=5)
        self.int_output = scrolledtext.ScrolledText(frame, height=20)
        self.int_output.pack(fill='both', expand=True, padx=10, pady=10)

    def run_int_thread(self):
        thread = threading.Thread(target=self.run_integrated)
        thread.start()

    def run_integrated(self):
        self.log(self.int_output, "[INFO] Starting integrated run...")
        try:
            self.log(self.int_output, "[INFO] Running optimization...")
            opt_results = genetic_algorithm_optimization(generations=3, population_size=5)
            self.log(self.int_output, f"[INFO] Optimization complete. Results folder: {opt_results['results_folder']}")
            best_candidate = opt_results["best_overall"]["candidate"]
            base_params = opt_results["base_params"]
            sim_params = base_params.copy()
            sim_params.update(best_candidate)
            self.log(self.int_output, "[INFO] Running full simulation with optimized parameters...")
            sim_result = run_field_simulation_3D_base(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
            self.log(self.int_output, f"[INFO] Simulation complete. Final slope: {sim_result['slope']}, Energy error: {sim_result.get('energy_error','N/A')}")
            self.log(self.int_output, "[INFO] Running convergence analysis...")
            resolutions = [32, 64, 128]
            time_steps = [0.05, 0.025]
            from simulation import convergence_test
            convergence_test(sim_params, resolutions, time_steps, ensemble_runs=3)
            self.log(self.int_output, "[INFO] Convergence analysis complete. See analysis_reports folder.")
            self.log(self.int_output, "[INFO] Running visualization...")
            sim_archive = os.path.join(opt_results["results_folder"], "simulation_archive")
            os.makedirs(sim_archive, exist_ok=True)
            import numpy as np
            np.save(os.path.join(sim_archive, "snapshots.npy"), sim_result["snapshots"])
            with open(os.path.join(sim_archive, "simulation_params.json"), "w") as f:
                json.dump(sim_params, f, indent=4)
            create_visualization_with_test_particles(sim_result["snapshots"], sim_params, os.path.join(sim_archive, "visualization_charts"),
                                                     dt_effect=0.1, num_particles=10)
            self.log(self.int_output, "[INFO] Visualization complete. Integrated run finished.")
        except Exception as e:
            self.log(self.int_output, f"[ERROR] {str(e)}")

if __name__ == "__main__":
    app = App()
    app.mainloop()

