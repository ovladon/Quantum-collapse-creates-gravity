# Emergent Gravity Simulation Application

## Overview

This project implements a simulation to explore the hypothesis that gravity can emerge from quantum decoherence or collapse processes. The underlying idea is inspired by the GRW (Ghirardi–Rimini–Weber) collapse model, and the simulation studies how stochastic collapse events in a 3D quantum field may give rise to a gravitational potential. Test particles are then introduced into the simulation domain to probe how they are accelerated by the emergent gravitational field.

## Key Modules

- **simulation.py**  
  Implements the core simulation routines:
  - Uses a symplectic (leapfrog) integrator for evolving a 3D field.
  - Includes stochastic collapse events (based on a Poisson process) and continuous noise.
  - Computes the gravitational potential via FFT-based Poisson solving.
  - Provides an analytic benchmark (`analytical_gravitational_potential`) to compare against a uniform sphere.

- **optimization.py**  
  Implements a genetic algorithm for parameter optimization.
  - Evaluates candidate parameter sets by running short simulation ensembles.
  - Computes a fitness based on the deviation from a target noise exponent and energy conservation.
  - Records uncertainty (standard error) in the measured noise exponent.

- **visualization.py**  
  Generates static PNG charts from simulation snapshots.
  - Incorporates test particle trajectory tracking.
  - Overlays the evolving positions of test particles on the central slice of the simulation.

- **dynamic_visualization.py**  
  Provides an interactive animation using matplotlib’s `FuncAnimation`.
  - Uses a Numba-accelerated update routine (if available) for improved performance.
  - Allows smooth visualization of test particle trajectories over interpolated frames.

- **resource_assessment.py**  
  Checks system memory and CPU load, and provides an estimated simulation runtime.

- **tests.py**  
  Contains basic unit tests for important functions, helping ensure code robustness.

- **gui_app.py**  
  A Tkinter-based graphical user interface that ties together simulation, optimization, convergence analysis, and visualization.
  - Provides tabs for each module.
  - Uses threading to keep the interface responsive.
  
- **main.py**  
  A command-line interface that allows running each module separately.

## Scientific Motivation and Literature

The simulation is inspired by the idea that gravity might be an emergent phenomenon arising from quantum collapse or decoherence processes. For example, the GRW model proposes a stochastic collapse mechanism for the wavefunction. By simulating such collapse events in a 3D field, our code computes an emergent gravitational potential and then uses test particles to probe its effect.

Relevant literature:
- Ghirardi, G.C., Rimini, A., & Weber, T. (1986). "Unified dynamics for microscopic and macroscopic systems". *Physical Review D*, 34(2), 470–491.
- Further comparisons can be made with Verlinde’s emergent gravity proposals and other dynamical reduction models.

## How to Run

1. **Command-Line Interface:**  
   Run `python3 main.py --simulate` (or `--optimize`, `--visualize`, `--convergence`) to execute different parts of the simulation.

2. **GUI Interface:**  
   Run `python3 gui_app.py` to open the graphical user interface.

3. **Interactive Visualization:**  
   Run `python3 dynamic_visualization.py <folder> --use_numba` to launch the interactive animation.

4. **Unit Tests:**  
   Run `python3 tests.py` to run the unit tests.

## Parameter Sensitivity & Uncertainty

The optimization module now includes uncertainty quantification. Each candidate’s noise exponent is measured over an ensemble of simulation runs, and the standard error is recorded. This information can be used to assess the sensitivity of the model to parameter variations.

## Performance Indicators

- **Runtime & Energy Conservation:**  
  The simulation logs total runtime and energy error, along with available system memory, to help monitor performance.
  
- **Advanced Numerical Techniques:**  
  Parts of the dynamic visualization update loop are accelerated with Numba to allow real-time animation.

## Future Directions

- **Theoretical Refinement:**  
  Improve the analytical benchmark tests and compare the simulated gravitational potential with predictions from Newtonian or post-Newtonian gravity.
  
- **Experimental Comparison:**  
  Should relevant experimental or astrophysical data become available, the simulation outputs (e.g., gravitational potential profiles) can be compared to determine if the emergent gravity hypothesis is viable.

- **Extended Optimization:**  
  Further refine the genetic algorithm and add more comprehensive sensitivity analyses.

## Final Remarks

This application is intended as a research tool to explore whether quantum decoherence might be a source of emergent gravity. The code has been designed for modularity, reproducibility, and scientific rigor. Feedback and suggestions for improvement are welcome.

---

*Developed by Vld Belciug. For academic use and further collaboration, please contact vladbelciug80@gmail.com.*

