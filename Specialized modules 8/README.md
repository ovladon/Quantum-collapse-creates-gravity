# Emergent Gravity Simulation Project

## Overview
This project explores the hypothesis that gravity can emerge from quantum collapse dynamics. The simulation is inspired by GRW-type collapse models and examines whether collapse-induced noise in a quantum field yields a gravitational potential with a power spectrum scaling as \(k^{-5}\).

## Files Included
- **emergent_gravity.tex**: The main LaTeX preprint describing the research.
- **references.bib**: Bibliography file with relevant references.
- **README.md**: This file.
- **simulation.py**: Implements the 3D field simulation (both base and adaptive integrators).
- **optimization.py**: Genetic algorithm for parameter optimization.
- **visualization.py**: Generates static visualizations of the simulation.
- **dynamic_visualization.py**: Provides an interactive animation of the simulation.
- **analysis.py**: Automates convergence analysis.
- **gui_app.py**: A Tkinter-based GUI that integrates all modules.
- **tests.py**: Unit tests for key simulation functions.

## How to Run
1. **Preprint**: Compile `emergent_gravity.tex` with LaTeX to generate the PDF preprint.
2. **GUI Application**: Run `python3 gui_app.py` to open the graphical interface.
3. **Command-Line**: Individual modules can be run from the command line (e.g., `python3 simulation.py`, etc.).
4. **Sensitivity Analysis**: Use the Sensitivity Analysis tab in the GUI to explore parameter trade-offs.

## Requirements
- Python 3.10 or higher
- NumPy, SciPy, Matplotlib, Numba (optional, but recommended for acceleration)
- Tkinter (usually included with Python)

## Contact
For further inquiries or collaboration, please contact:
**Vlad Belciug**  
Email: [vladbelciug80@gmail.com](mailto:vladbelciug80@gmail.com)

## License
This project is released under the MIT License.

