#!/usr/bin/env python3
"""
File: gui_app.py

This application provides an integrated GUI for the emergent gravity simulation project.
It lets you run:
 - Genetic algorithm optimization
 - Full simulation with best candidate
 - Convergence analysis with systematic error reporting
 - Visualization of simulation snapshots (producing PNG charts)
 - An integrated run that executes all modules sequentially

The GUI is built using Tkinter and includes multiple tabs for each module.
Each tab displays log output and explanations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import json

# Import our simulation modules with updated function names.
from optimization import genetic_algorithm_optimization
from simulation_physics import run_field_simulation_3D
from analysis import run_systematic_error_analysis
from visualization import create_visualization_charts, create_results_chart

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Emergent Gravity Simulation GUI")
        self.geometry("900x700")
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both')

        # Optimization Tab
        self.optim_tab = ttk.Frame(notebook)
        notebook.add(self.optim_tab, text="Optimization")
        self.create_optim_tab()

        # Simulation Tab
        self.sim_tab = ttk.Frame(notebook)
        notebook.add(self.sim_tab, text="Simulation")
        self.create_sim_tab()

        # Convergence Analysis Tab
        self.conv_tab = ttk.Frame(notebook)
        notebook.add(self.conv_tab, text="Convergence Analysis")
        self.create_conv_tab()

        # Visualization Tab
        self.vis_tab = ttk.Frame(notebook)
        notebook.add(self.vis_tab, text="Visualization")
        self.create_vis_tab()

        # Reports Tab
        self.rep_tab = ttk.Frame(notebook)
        notebook.add(self.rep_tab, text="Reports")
        self.create_rep_tab()

        # Integrated Run Tab
        self.int_tab = ttk.Frame(notebook)
        notebook.add(self.int_tab, text="Integrated Run")
        self.create_int_tab()

    def log(self, widget, message):
        widget.insert(tk.END, message + "\n")
        widget.see(tk.END)

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
            opt_results = genetic_algorithm_optimization(generations=3, population_size=5)
            self.log(self.optim_output, f"[INFO] Optimization complete. Results saved in: {opt_results['results_folder']}")
        except Exception as e:
            self.log(self.optim_output, f"[ERROR] {str(e)}")

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
        with open(summary_file, "r") as f:
            opt_results = json.load(f)
        best_candidate = opt_results["best_overall"]["candidate"]
        base_params = opt_results["base_params"]
        sim_params = base_params.copy()
        sim_params.update(best_candidate)
        self.log(self.sim_output, "[INFO] Running full simulation with best candidate parameters...")
        try:
            sim_result = run_field_simulation_3D(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
            self.log(self.sim_output, f"[INFO] Simulation complete. Final slope: {sim_result['slope']}, Energy error: {sim_result['energy_error']}")
        except Exception as e:
            self.log(self.sim_output, f"[ERROR] {str(e)}")

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
            results = run_systematic_error_analysis(base_params, resolutions, time_steps, ensemble_runs=3, output_folder="analysis_reports")
            self.log(self.conv_output, "[INFO] Convergence analysis complete. See analysis_reports folder for detailed report.")
        except Exception as e:
            self.log(self.conv_output, f"[ERROR] {str(e)}")

    def create_vis_tab(self):
        frame = self.vis_tab
        explanation = (
            "Visualization Section:\n"
            "------------------------\n"
            "This section produces PNG charts from simulation snapshots and provides an interactive animation.\n"
            "The interactive animation will play the simulation frame by frame, showing the central slice of the field\n"
            "and simulating the movement of matter points under the influence of the field's gradient.\n"
            "Please ensure that a valid simulation archive folder (containing snapshots.npy and simulation_params.json) is provided."
        )
        lbl_exp = ttk.Label(frame, text=explanation, wraplength=800, justify="left")
        lbl_exp.pack(pady=5)
        
        lbl = ttk.Label(frame, text="Select Simulation Archive Folder (must contain snapshots.npy and simulation_params.json):")
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
    
    def launch_interactive_animation(self):
        folder = self.vis_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select a simulation archive folder.")
            return
        self.log(self.vis_output, f"[INFO] Launching interactive animation for folder: {folder}")
        try:
            import numpy as np
            snapshots_path = os.path.join(folder, "snapshots.npy")
            with open(os.path.join(folder, "simulation_params.json"), "r") as f:
                sim_params = json.load(f)
            snapshots = np.load(snapshots_path, allow_pickle=True)
            # Launch the interactive animation window.
            from visualization import create_interactive_animation_window
            create_interactive_animation_window(snapshots, sim_params, matter_points_count=50)
        except Exception as e:
            self.log(self.vis_output, f"[ERROR] {str(e)}")


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
            saved_files = create_visualization_charts(snapshots, sim_params, output_folder)
            self.log(self.vis_output, f"[INFO] Visualization charts created: {saved_files}")
        except Exception as e:
            self.log(self.vis_output, f"[ERROR] {str(e)}")

    def create_rep_tab(self):
        frame = self.rep_tab
        lbl = ttk.Label(frame, text="Reports and Findings")
        lbl.pack(pady=5)
        self.rep_output = scrolledtext.ScrolledText(frame, height=25)
        self.rep_output.pack(fill='both', expand=True, padx=10, pady=10)
        report_text = (
            "Integrated Emergent Gravity Simulation Report\n"
            "-------------------------------------------------\n"
            "This application automates simulation, optimization, convergence analysis, and visualization.\n"
            "Please refer to the generated files in the output folders for detailed numerical reports and plots.\n"
            "For further information, see the technical preprint provided with the code.\n"
        )
        self.rep_output.insert(tk.END, report_text)

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
        # Run optimization
        self.log(self.int_output, "[INFO] Running optimization...")
        opt_results = genetic_algorithm_optimization(generations=3, population_size=5)
        self.log(self.int_output, f"[INFO] Optimization complete. Results folder: {opt_results['results_folder']}")
        # Run simulation with best candidate
        best_candidate = opt_results["best_overall"]["candidate"]
        base_params = opt_results["base_params"]
        sim_params = base_params.copy()
        sim_params.update(best_candidate)
        self.log(self.int_output, "[INFO] Running full simulation with optimized parameters...")
        sim_result = run_field_simulation_3D(sim_params, snapshot_interval=sim_params["steps_per_cycle"])
        self.log(self.int_output, f"[INFO] Simulation complete. Final slope: {sim_result['slope']}, Energy error: {sim_result['energy_error']}")
        # Run convergence analysis using full sim_params (including collapse parameters)
        self.log(self.int_output, "[INFO] Running convergence analysis...")
        resolutions = [32, 64, 128]
        time_steps = [0.05, 0.025]
        run_systematic_error_analysis(sim_params, resolutions, time_steps, ensemble_runs=3, output_folder="analysis_reports")
        self.log(self.int_output, "[INFO] Convergence analysis complete. See analysis_reports folder.")
        # Run visualization: create simulation archive if needed
        self.log(self.int_output, "[INFO] Running visualization...")
        import numpy as np
        sim_archive = os.path.join(opt_results["results_folder"], "simulation_archive")
        if not os.path.exists(sim_archive):
            os.makedirs(sim_archive)
        np.save(os.path.join(sim_archive, "snapshots.npy"), sim_result["snapshots"])
        with open(os.path.join(sim_archive, "simulation_params.json"), "w") as f:
            json.dump(sim_params, f, indent=4)
        create_visualization_charts(sim_result["snapshots"], sim_params, os.path.join(sim_archive, "visualization_charts"))
        self.log(self.int_output, "[INFO] Visualization complete. Integrated run finished.")

if __name__ == "__main__":
    app = App()
    app.mainloop()

