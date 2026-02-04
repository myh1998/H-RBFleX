import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.ticker as ticker



plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# Define a list of folder paths
folders = ['./NSGA_Acc_noband_NLP2/Results',
           "./RandomSearch_Large_Acc_noband_NLP/Results",
           "./MultiOpt_Large_Acc_noband_NLP/Results_original",
           "./MultiOpt_Large_noband_NLP/Results_2",
           './H-WOT_Large_noband_NLP/Results',
           "./H-RBFleX_Large_noband_NLP/Results"
           ]

labels = ["NSGA-II (Accuracy Metric)",
          "Random Search (Accuracy Metric)",
          'MOBO (Accuracy Metric)',
          'MOBO (RBFleX-NAS Score Metric)',
          'MOBO (NASWOT Ranking Table)',
          'H-RBFleX (Ours)',
          ]


color = plt.cm.coolwarm(np.linspace(0, 1, 5))
color_RBFp = color[4]
color_WOT = color[0]
color_RBF = color[3]
colors = ["#2ca02c",
          "#9467bd",
          "#8c564b",
          "#ff7f0e",
          color_WOT,
          color_RBFp
        ]

times = [3672820+79,
        4534927+79,
        5234897+178,
        459*2,
        1501,
        1441]

# List to store the data frames for hypervolume and time
data_frames = []
time_data_frames = []

hyperv = []

# Read CSV files from each folder
for folder in folders:
    print(folder)
    # Path to cumulative_hypervolume.csv
    hv_file_path = os.path.join(folder, 'cumulative_hypervolume_v3_norm.csv')
    
    # Read cumulative hypervolume data
    print(hv_file_path)
    hv_df = pd.read_csv(hv_file_path)
    hyperv.append(hv_df['Cumulative Hypervolume'].iloc[-1])


print(hyperv)



# Create the plot
plt.figure(figsize=(10, 4))
bars = plt.bar(range(len(hyperv)), hyperv, color=colors)

# Customize the plot
plt.ylabel('Cumulative \nHypervolume', fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)
plt.ylim(0, 1.5)
#plt.legend()
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
#plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0e}"))
plt.gca().yaxis.get_offset_text().set_fontsize(24)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.06,
        f"{round(height,2)}",
        ha='center',
        va='bottom',
        fontsize=24,
        bbox=dict(facecolor='white', edgecolor='none', pad=2.0)
    )

# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     if height > 6000:
#         T = np.round(times[i]/3600, 2)
#     else:
#         T = int(times[i]/3600)
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         height - 350,
#         f"{T}",
#         ha='center',
#         va='bottom',
#         fontsize=24,
#         bbox=dict(facecolor='none', edgecolor='none', pad=2.0)
#     )


# Display the plot
plt.savefig("[plot_hypervolume_with_traintime_nlp]ex_nlp_v4.pdf", format="pdf")
plt.show()