import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

# Load data from CSV files
acc_df = pd.read_csv("accuracy.csv", header=0)  # Replace with your file name
fix_train_output_df = pd.read_csv("fix_train_output.csv", header=None)  # Replace with your file name

# Extract error values from error.csv
errors = 100-acc_df["0"].values


# Extract cycle count from the second row of fix_train_output.csv
cycle_counts = -fix_train_output_df.iloc[:, 1].values 


# Combine data into a 2D array (error and cycle count)
data = np.column_stack((errors, cycle_counts))

# Define a reference point (choose values larger than the worst objectives)
#ref_point = np.array([max(data[:, 0]) + 1, max(data[:, 1]) + 1])  # Adjust based on your data
ref_point = np.array([100, 1e12])  # Adjust based on your data

# Compute hypervolume
hv = HV(ref_point=ref_point)

# Compute cumulative hypervolume every 10 rows
step = 5
cumulative_hypervolumes = []
row_labels = []

for i in range(step, len(data) + 1, step):  # Process in steps of 10
    subset = data[:i]  # Take rows from the 1st to the current row (1–10, 1–20, ...)
    hv_value = hv(subset)  # Compute hypervolume for the subset
    cumulative_hypervolumes.append(hv_value)
    row_labels.append(f"1-{i}")
    print(f"Rows 1 to {i}: Hypervolume = {hv_value}")

# Optional: Save results to a CSV
results_df = pd.DataFrame({
    "Rows": row_labels,
    "Cumulative Hypervolume": cumulative_hypervolumes
})
results_df.to_csv("cumulative_hypervolume.csv", index=False)


# Plot the cumulative hypervolume trend
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(row_labels, cumulative_hypervolumes, marker='o', linestyle='-', color='blue', label="Cumulative Hypervolume")
plt.title("Cumulative Hypervolume Every 10 Rows", fontsize=14)
plt.xlabel("Rows (1 to N)", fontsize=12)
plt.ylabel("Cumulative Hypervolume", fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()