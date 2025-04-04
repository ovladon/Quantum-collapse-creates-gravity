#!/usr/bin/env python3
"""
Re-designed GUI for the Emergent Gravity Simulation Project.
This GUI now contains detailed instructions on each tab so that users (even beginners)
understand the meaning of each tweakable parameter and the purpose of each analysis.

Tabs included:
  - Optimization: Configure and run the genetic algorithm.
  - Simulation: Adjust simulation duration (Steps per Cycle and Number of Cycles) and run a full simulation.
  - Convergence Analysis: Test how outputs vary with grid resolution and dt.
  - Visualization: Generate static charts and interactive animations.
  - Sensitivity Analysis: Explore the impact of varying dt and tolerance.
  - Data Comparison: Compare simulation results with observational data.
  - Integrated Run: Execute all modules sequentially.
  - Reports: View aggregated summary reports and logs.
  
All parameters and settings come with in-GUI help text to assist beginners.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import json
import logging
import time

from optimization import genetic_algorithm_optimization
from simulation import run_field_simulation_3D_base
from analysis import run_systematic_error_analysis
from visualization import create_visualization_with_test_particles, create_results_chart
from dynamic_visualization import launch_interactive_visualization
from data_comparison import compare_simulation_to_observations

logging.basicConfig(level=logging.INFO)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emergent Gravity Simulation GUI")
        self.geometry("1100x900")
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both')
        
        self.tabs = {}
        tab_info = {
            "Optimization": "In this tab, you can configure the genetic algorithm parameters:\n"
                            "  - Generations: Number of GA iterations (more generations may improve results but require more time).\n"
                            "  - Population Size: Number of candidate parameter sets per generation.\n"
                            "  - Ensemble Runs: How many simulation runs are averaged per candidate (higher values yield more robust results but are computationally expensive).\n"
                            "  - Mutation Rate: The probability of random perturbations to candidate parameters.\n"
                            "Click 'Run Optimization' to start the GA.",
            "Simulation": "This tab allows you to run a full simulation using the best candidate parameters from the optimization.\n"
                          "Adjust simulation duration:\n"
                          "  - Steps per Cycle: Number of integration steps in one cycle.\n"
                          "  - Number of Cycles: Total cycles (Total steps = Steps per Cycle × Number of Cycles).\n"
                          "The snapshot interval is automatically adjusted to yield at least 20 frames.",
            "Convergence Analysis": "Here you can run convergence analysis to see how simulation outputs (e.g., noise exponent, energy error) change\n"
                                    "with different grid resolutions and time steps. This helps determine the optimal simulation settings.\n"
                                    "Be aware that higher resolution and lower dt increase computation time.",
            "Visualization": "In this tab, generate visualizations from simulation snapshots.\n"
                             "Choose a simulation archive folder (containing snapshots.npy and simulation_params.json) and then:\n"
                             "  - 'Generate PNG Charts' produces static images with test particle trajectories.\n"
                             "  - 'Launch Interactive Animation' shows an animated visualization of the simulation.",
            "Sensitivity Analysis": "This tab explores how the simulation output changes when the integration time step (dt) and tolerance are varied.\n"
                                    "Lower dt values may improve accuracy at the cost of increased computation time.",
            "Data Comparison": "This tab lets you compare simulation results with observational data.\n"
                               "Select the simulation archive folder and a CSV file containing observational data\n"
                               "(with headers like 'Measurement' and 'Value'). A comparison plot will be generated.",
            "Integrated Run": "Execute the complete workflow sequentially: Optimization → Full Simulation → Convergence Analysis → Visualization.\n"
                              "This is useful for an end-to-end evaluation of the simulation pipeline.",
            "Reports": "View aggregated summary reports and log messages detailing simulation performance, optimization outcomes, and any errors.\n"
                       "This tab aggregates important messages for quick review."
        }
        
        for name, instr in tab_info.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=name)
            self.tabs[name] = frame
            label = ttk.Label(frame, text=instr, justify="left", wraplength=1050, foreground="blue")
            label.pack(pady=5, padx=10, anchor="w")

        self.create_optim_tab()
        self.create_sim_tab()
        self.create_conv_tab()
        self.create_vis_tab()
        self.create_sens_tab()
        self.create_data_comp_tab()
        self.create_int_tab()
        self.create_rep_tab()

    def log(self, widget, message):
        widget.insert(tk.END, message + "\n")
        widget.see(tk.END)

    # ---------------------
    # Optimization Tab
    # ---------------------
    def create_optim_tab(self):
        frame = self.tabs["Optimization"]
        param_frame = ttk.LabelFrame(frame, text="Optimization Parameters")
        param_frame.pack(pady=5, padx=10, fill='x')
        
        ttk.Label(param_frame, text="Generations:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_generations = ttk.Entry(param_frame, width=5)
        self.entry_generations.insert(0, "10")
        self.entry_generations.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Population Size:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        self.entry_pop_size = ttk.Entry(param_frame, width=5)
        self.entry_pop_size.insert(0, "20")
        self.entry_pop_size.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Ensemble Runs:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_ensemble = ttk.Entry(param_frame, width=5)
        self.entry_ensemble.insert(0, "5")
        self.entry_ensemble.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Mutation Rate:").grid(row=1, column=2, sticky="e", padx=5, pady=2)
        self.entry_mutation = ttk.Entry(param_frame, width=5)
        self.entry_mutation.insert(0, "0.2")
        self.entry_mutation.grid(row=1, column=3, padx=5, pady=2)
        
        self.optim_output = scrolledtext.ScrolledText(frame, height=15)
        self.optim_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_run_opt = ttk.Button(frame, text="Run Optimization", command=self.run_optim_thread)
        btn_run_opt.pack(pady=5)

    def run_optim_thread(self):
        thread = threading.Thread(target=self.run_optimization)
        thread.start()

    def run_optimization(self):
        try:
            generations = int(self.entry_generations.get())
            pop_size = int(self.entry_pop_size.get())
            ensemble_runs = int(self.entry_ensemble.get())
            mutation_rate = float(self.entry_mutation.get())
            self.log(self.optim_output, f"[INFO] Running GA with {generations} generations, population size {pop_size}, ensemble runs {ensemble_runs}, mutation rate {mutation_rate}")
            opt_results = genetic_algorithm_optimization(
                generations=generations, 
                population_size=pop_size, 
                ensemble_runs_param=ensemble_runs, 
                mutation_rate_param=mutation_rate
            )
            self.log(self.optim_output, f"[INFO] Optimization complete. Results saved in: {opt_results['results_folder']}")
        except Exception as e:
            self.log(self.optim_output, f"[ERROR] {str(e)}")

    # ---------------------
    # Simulation Tab
    # ---------------------
    def create_sim_tab(self):
        frame = self.tabs["Simulation"]
        sim_param_frame = ttk.LabelFrame(frame, text="Simulation Duration Parameters")
        sim_param_frame.pack(pady=5, padx=10, fill='x')
        
        ttk.Label(sim_param_frame, text="Steps per Cycle:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_steps = ttk.Entry(sim_param_frame, width=5)
        self.entry_steps.insert(0, "50")
        self.entry_steps.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(sim_param_frame, text="(Integration steps in one cycle)").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(sim_param_frame, text="Number of Cycles:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_cycles = ttk.Entry(sim_param_frame, width=5)
        self.entry_cycles.insert(0, "2")
        self.entry_cycles.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(sim_param_frame, text="(Total steps = Steps per Cycle × Number of Cycles)").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(frame, text="Select Optimization Summary File (optimization_summary.json):").pack(pady=5)
        self.sim_summary_path = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.sim_summary_path, width=60)
        entry.pack(pady=5)
        btn_browse = ttk.Button(frame, text="Browse", command=self.browse_sim_summary)
        btn_browse.pack(pady=5)
        
        self.sim_output = scrolledtext.ScrolledText(frame, height=15)
        self.sim_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_run_sim = ttk.Button(frame, text="Run Full Simulation", command=self.run_sim_thread)
        btn_run_sim.pack(pady=5)

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
            steps = int(self.entry_steps.get())
            cycles = int(self.entry_cycles.get())
            base_params["steps_per_cycle"] = steps
            base_params["num_cycles"] = cycles
            sim_params = base_params.copy()
            sim_params.update(best_candidate)
            self.log(self.sim_output, "[INFO] Running full simulation with best candidate parameters...")
            sim_result = run_field_simulation_3D_base(sim_params, snapshot_interval=None)
            self.log(self.sim_output, f"[INFO] Simulation complete. Final slope: {sim_result['slope']}, Energy error: {sim_result.get('energy_error','N/A')}")
            sim_archive = os.path.join(os.path.dirname(summary_file), "simulation_archive")
            os.makedirs(sim_archive, exist_ok=True)
            import numpy as np
            np.save(os.path.join(sim_archive, "snapshots.npy"), sim_result["snapshots"])
            with open(os.path.join(sim_archive, "simulation_params.json"), "w") as f:
                json.dump(sim_params, f, indent=4)
            with open(os.path.join(sim_archive, "simulation_results.txt"), "w") as f:
                f.write(f"Final slope: {sim_result['slope']}\n")
                f.write(f"Energy error: {sim_result['energy_error']}\n")
                f.write(f"Runtime: {sim_result['simulation_time']} seconds\n")
            self.log(self.sim_output, f"[INFO] Simulation archive saved in {sim_archive}")
        except Exception as e:
            self.log(self.sim_output, f"[ERROR] {str(e)}")

    # ---------------------
    # Convergence Analysis Tab
    # ---------------------
    def create_conv_tab(self):
        frame = self.tabs["Convergence Analysis"]
        lbl = ttk.Label(frame, text="Instructions:\nThis tab tests how the simulation output changes when varying grid resolution and dt.\nIt helps determine the optimal simulation settings (accuracy vs. computing time).\nResults are saved in the analysis_reports folder.", justify="left", wraplength=1050, foreground="purple")
        lbl.pack(pady=5, padx=10, anchor="w")
        self.conv_output = scrolledtext.ScrolledText(frame, height=15)
        self.conv_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_conv = ttk.Button(frame, text="Run Convergence Analysis", command=self.run_conv_thread)
        btn_conv.pack(pady=5)

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
            from analysis import run_systematic_error_analysis
            results = run_systematic_error_analysis(base_params, resolutions, time_steps, ensemble_runs=3)
            self.log(self.conv_output, "[INFO] Convergence analysis complete. See analysis_reports folder for detailed report.")
        except Exception as e:
            self.log(self.conv_output, f"[ERROR] {str(e)}")

    # ---------------------
    # Visualization Tab
    # ---------------------
    def create_vis_tab(self):
        frame = self.tabs["Visualization"]
        lbl = ttk.Label(frame, text="Instructions:\nSelect a simulation archive folder (with snapshots.npy and simulation_params.json).\nUse 'Generate PNG Charts' for static images or 'Launch Interactive Animation' for an animated view.", justify="left", wraplength=1050, foreground="darkgreen")
        lbl.pack(pady=5, padx=10, anchor="w")
        self.vis_folder = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.vis_folder, width=60)
        entry.pack(pady=5)
        btn_browse = ttk.Button(frame, text="Browse", command=self.browse_vis_folder)
        btn_browse.pack(pady=5)
        
        self.vis_output = scrolledtext.ScrolledText(frame, height=15)
        self.vis_output.pack(fill='both', expand=True, padx=10, pady=10)
        
        btn_static = ttk.Button(frame, text="Generate PNG Charts", command=self.run_vis_thread)
        btn_static.pack(pady=5)
        btn_dynamic = ttk.Button(frame, text="Launch Interactive Animation", command=self.launch_interactive_animation)
        btn_dynamic.pack(pady=5)

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
            saved_files = create_visualization_with_test_particles(snapshots, sim_params, output_folder, dt_effect=0.1, num_particles=10)
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

    # ---------------------
    # Sensitivity Analysis Tab
    # ---------------------
    def create_sens_tab(self):
        frame = self.tabs["Sensitivity Analysis"]
        lbl = ttk.Label(frame, text="Instructions:\nExplore how the simulation output varies with changes in dt and tolerance.\nLower dt values may improve accuracy but require more computation time.", justify="left", wraplength=1050, foreground="brown")
        lbl.pack(pady=5, padx=10, anchor="w")
        self.sens_output = scrolledtext.ScrolledText(frame, height=15)
        self.sens_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_sens = ttk.Button(frame, text="Run Sensitivity Analysis", command=self.run_sensitivity_thread)
        btn_sens.pack(pady=5)
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
        results_summary = []
        for dt_val in param_grid["dt"]:
            for tol in param_grid["tolerance"]:
                params = base_params.copy()
                params["dt"] = dt_val
                self.sens_progress_callback(f"Running simulation with dt={dt_val}, tolerance={tol}...")
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
                self.sens_progress_callback(f"Completed dt={dt_val}, tol={tol}: slope={sim_result['slope']:.3f}, energy_error={sim_result['energy_error']:.3e}")
        import os, json
        output_folder = "sensitivity_reports"
        os.makedirs(output_folder, exist_ok=True)
        json_path = os.path.join(output_folder, "sensitivity_analysis.json")
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=4)
        self.sens_progress_callback(f"Sensitivity analysis complete. Results saved to {json_path}")

    # ---------------------
    # Data Comparison Tab
    # ---------------------
    def create_data_comp_tab(self):
        frame = self.tabs["Data Comparison"]
        ttk.Label(frame, text="Simulation Archive Folder:").pack(pady=2)
        self.comp_sim_folder = tk.StringVar()
        entry_sim = ttk.Entry(frame, textvariable=self.comp_sim_folder, width=60)
        entry_sim.pack(pady=2)
        btn_sim_folder = ttk.Button(frame, text="Browse", command=self.browse_comp_sim_folder)
        btn_sim_folder.pack(pady=2)
        
        ttk.Label(frame, text="Observational Data CSV:").pack(pady=2)
        self.comp_obs_csv = tk.StringVar()
        entry_obs = ttk.Entry(frame, textvariable=self.comp_obs_csv, width=60)
        entry_obs.pack(pady=2)
        btn_obs = ttk.Button(frame, text="Browse", command=self.browse_obs_csv)
        btn_obs.pack(pady=2)
        
        self.comp_output = scrolledtext.ScrolledText(frame, height=15)
        self.comp_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_compare = ttk.Button(frame, text="Run Data Comparison", command=self.run_data_comparison)
        btn_compare.pack(pady=5)

    def browse_comp_sim_folder(self):
        folder = filedialog.askdirectory(title="Select Simulation Archive Folder")
        if folder:
            self.comp_sim_folder.set(folder)

    def browse_obs_csv(self):
        path = filedialog.askopenfilename(title="Select Observational Data CSV", filetypes=[("CSV Files", "*.csv")])
        if path:
            self.comp_obs_csv.set(path)

    def run_data_comparison(self):
        sim_folder = self.comp_sim_folder.get()
        obs_csv = self.comp_obs_csv.get()
        if not sim_folder or not obs_csv:
            messagebox.showerror("Error", "Please select both simulation archive folder and observational CSV file.")
            return
        self.log(self.comp_output, "[INFO] Running data comparison...")
        try:
            output_path = compare_simulation_to_observations(sim_folder, obs_csv)
            self.log(self.comp_output, f"[INFO] Data comparison complete. Plot saved to {output_path}")
        except Exception as e:
            self.log(self.comp_output, f"[ERROR] {str(e)}")

    # ---------------------
    # Integrated Run Tab
    # ---------------------
    def create_int_tab(self):
        frame = self.tabs["Integrated Run"]
        lbl = ttk.Label(frame, text="Instructions:\nThis tab runs the complete workflow sequentially:\n 1. Optimization\n 2. Full Simulation\n 3. Convergence Analysis\n 4. Visualization\nUseful for an end-to-end evaluation of the simulation pipeline.", justify="left", wraplength=1050, foreground="maroon")
        lbl.pack(pady=5, padx=10, anchor="w")
        self.int_output = scrolledtext.ScrolledText(frame, height=15)
        self.int_output.pack(fill='both', expand=True, padx=10, pady=10)
        btn_run_int = ttk.Button(frame, text="Run All Modules", command=self.run_int_thread)
        btn_run_int.pack(pady=5)

    def run_int_thread(self):
        thread = threading.Thread(target=self.run_integrated)
        thread.start()

    def run_integrated(self):
        self.log(self.int_output, "[INFO] Starting integrated run...")
        try:
            self.log(self.int_output, "[INFO] Running optimization...")
            opt_results = genetic_algorithm_optimization(generations=10, population_size=20, ensemble_runs_param=10, mutation_rate_param=0.2)
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
            from analysis import run_systematic_error_analysis
            run_systematic_error_analysis(sim_params, resolutions, time_steps, ensemble_runs=3)
            self.log(self.int_output, "[INFO] Convergence analysis complete. See analysis_reports folder.")
            self.log(self.int_output, "[INFO] Running visualization...")
            sim_archive = os.path.join(opt_results["results_folder"], "simulation_archive")
            os.makedirs(sim_archive, exist_ok=True)
            import numpy as np
            np.save(os.path.join(sim_archive, "snapshots.npy"), sim_result["snapshots"])
            with open(os.path.join(sim_archive, "simulation_params.json"), "w") as f:
                json.dump(sim_params, f, indent=4)
            with open(os.path.join(sim_archive, "simulation_results.txt"), "w") as f:
                f.write(f"Final slope: {sim_result['slope']}\n")
                f.write(f"Energy error: {sim_result['energy_error']}\n")
                f.write(f"Runtime: {sim_result['simulation_time']} seconds\n")
            self.log(self.int_output, f"[INFO] Full simulation complete. Archive saved in {sim_archive}")
        except Exception as e:
            self.log(self.int_output, f"[ERROR] {str(e)}")

    # ---------------------
    # Reports Tab
    # ---------------------
    def create_rep_tab(self):
        frame = self.tabs["Reports"]
        self.rep_output = scrolledtext.ScrolledText(frame, height=20)
        self.rep_output.pack(fill='both', expand=True, padx=10, pady=10)
        report_text = (
            "Integrated Emergent Gravity Simulation Report\n"
            "-------------------------------------------------\n"
            "This report aggregates outputs from all modules:\n"
            "- Optimization: Best candidate parameters and fitness scores.\n"
            "- Simulation: Final slope, energy error, and runtime.\n"
            "- Convergence Analysis: How simulation metrics vary with resolution and dt.\n"
            "- Visualization: Charts and animations of test particle trajectories.\n"
            "- Data Comparison: Comparison between simulation and observational data.\n"
            "\nReview the logs above for detailed performance information and error messages."
        )
        self.rep_output.insert(tk.END, report_text)

if __name__ == "__main__":
    app = App()
    app.mainloop()

