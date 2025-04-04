import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace with the actual path to your CSV file
csv_filename = "/home/vlad/CSML/Doctorat/Quantum decoherence creates gravity/Sixth try/run_results_20250218_115750/param_sweep_results_20250218_115750.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_filename)
print("CSV loaded. Here are the first few rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Option 1: Create a pairplot to see relationships between all parameters and the slope.
sns.pairplot(df, diag_kind='kde')
plt.suptitle("Parameter Sweep Pairplot", y=1.02)
plt.show()

# Option 2: If you want to focus on a particular parameter (e.g., collapse_rate vs slope):
plt.figure(figsize=(8,6))
sns.scatterplot(x="collapse_rate", y="slope", data=df, hue="collapse_sigma", style="density_decay")
plt.title("Noise Exponent vs. Collapse Rate")
plt.xlabel("Collapse Rate")
plt.ylabel("Noise Exponent (Slope)")
plt.legend(title="Collapse Sigma / Density Decay")
plt.show()

