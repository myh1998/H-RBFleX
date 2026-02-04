import numpy as np
import matplotlib.pyplot as plt

# Set font style and weight (IEEE-style)
plt.rcParams['font.family'] = 'Arial'  # Falls back to default if Arial is unavailable
plt.rcParams['font.weight'] = 'bold'

# Define custom colors using colormap (for red and blue tones)
color = plt.cm.coolwarm(np.linspace(0, 1, 5))
red = color[4]
blue = color[0]

# Define color mapping for each T value
colors = {30: 'black', 50: red, 100: blue}
T_values = [30, 50, 100]

# Compute log(T) for each T
log_T = {T: np.log(T) for T in T_values}

# Define the range of n values
n_range = np.arange(5, 11)
x = np.arange(len(n_range))  # Base x positions for bar groups

# Compute regret ratio for each T and each n
regret_ratios = {}
for T in T_values:
    regret_ratios[T] = (log_T[T]) ** ((n_range - 1) / 2)

# Define bar width and alignment offset
bar_width = 0.3

# Create the plot
plt.figure(figsize=(12, 5))
for i, T in enumerate(T_values):
    offset = (i - 1) * bar_width  # Offsets: -bar_width, 0, +bar_width
    values = regret_ratios[T]
    bars = plt.bar(x + offset, values, width=bar_width,
                   color=colors[T], label=f"T={T}")

    # Add text labels above each bar
    for xi, val in zip(x + offset, values):
        plt.text(xi, val + 0.1, f"{val:.0f}", ha='center', va='bottom', fontsize=20, fontweight='bold')

# Axis and legend formatting
plt.xlabel(r"$n$" + "\nDimensionality of Original DNN Design Space", fontsize=20, fontweight='bold')
plt.ylabel("The Ratio of \nCumulative Regret", fontsize=20, fontweight='bold')
plt.xticks(x, n_range, fontsize=20)
plt.ylim([0, 1100])
plt.tick_params(axis='y', labelsize=20)
plt.legend(fontsize=20)
plt.grid(True, axis='y')
plt.tight_layout()

# Save the figure as PDF
plt.savefig("[plot_regret]v1.pdf", format="pdf")
plt.show()