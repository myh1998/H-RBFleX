import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.ticker as ticker



plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# Define a list of folder paths
folders = ['./NSGA_Acc_noband_trans101/Results',
           "./MultiOpt_Large_Acc_noband_trans101/Results_random",
           "./MultiOpt_Large_Acc_noband_trans101/Results",
           #"./MultiOpt_Large_noband_trans101/Results",
           #"./H-RBFleX_Large_noband_trans101/Results_-1e10",
           #'./H-WOT_Large_noband_trans101/Results',
           ]

labels = ["NSGA-II (Accuracy Metric)",
          "Random Search (Accuracy Metric)",
          'MOBO (Accuracy Metric)',
          #'MOBO (RBFleX-NAS Score Metric)',
          #'H-RBFleX (Ours)',
          #'MOBO (NASWOT Ranking Table)',
          ]


color = plt.cm.coolwarm(np.linspace(0, 1, 5))
color_RBFp = color[4]
color_WOT = color[0]
color_RBF = color[3]
colors = ["#2ca02c",
          "#9467bd",
          "#8c564b",
          #"#ff7f0e",
          #color_RBFp,
          #color_WOT
        ]

# List to store the data frames for hypervolume and time
data_frames = []
time_data_frames = []

# Read CSV files from each folder
for folder in folders:
    print(folder)
    # Path to cumulative_hypervolume.csv
    hv_file_path = os.path.join(folder, 'cumulative_hypervolume_v2.csv')
    
    # Path to time.csv
    time_file_path = os.path.join(folder, 'time.csv')
    
    if os.path.exists(hv_file_path) and os.path.exists(time_file_path):
        # Read cumulative hypervolume data
        print(hv_file_path)
        hv_df = pd.read_csv(hv_file_path)
        if 'Cumulative Hypervolume' in hv_df.columns:
            # Drop the first 9 rows of cumulative hypervolume data (beacuse of random)
            hv_df = hv_df.drop(index=range(9))
            data_frames.append(hv_df['Cumulative Hypervolume'])
            
        else:
            print(f"'Cumulative Hypervolume' column not found in {hv_file_path}.")

        # Read time data (no removal of rows)
        time_df = pd.read_csv(time_file_path, header=None)
        if not 'H-RBFleX' in folder.split('/')[1] and not 'H-WOT' in folder.split('/')[1]:
            time_df = time_df*2
        if 'Acc' in folder.split('/')[1]:
            time_file_path = os.path.join(folder, 'time_traintime.csv')
            time_train_df = pd.read_csv(time_file_path, header=None)
            sums = [time_train_df[:100].sum().values[0]]
            for i in range(100, len(time_train_df), 10):
                sums.append(time_train_df[i:i+10].sum().values[0])
            sums_df = pd.DataFrame(sums, columns=['sum'])

        cumulative_time = time_df.cumsum().reset_index(drop=True)
        print(cumulative_time)
        
        if 'Acc' in folder.split('/')[1]:
            cumulative_time_train = sums_df['sum'].cumsum().reset_index(drop=True)
            print(type(cumulative_time_train))
            cumulative_time += cumulative_time_train
        # Append the entire time data
        time_data_frames.append(cumulative_time.reset_index(drop=True))  
    else:
        print(f"File {hv_file_path} or {time_file_path} not found.")
        exit()



# Create the plot
plt.figure(figsize=(10, 4))
for i, (hv_df, time_df) in enumerate(zip(data_frames, time_data_frames)):
    # Dynamically extract the label from the folder path
    label = labels[i]  # Extract 'H-RBFleX' from 'H-RBFleX/Results_N32780_qNargo'
    print(label)
    # Set the x-axis to time and y-axis to cumulative hypervolume
    plt.plot(time_df/3600, hv_df, color=colors[i], label=label, linewidth=2)
    plt.scatter(time_df/3600, hv_df, color=colors[i], s=100)

# Customize the plot
plt.xlabel('Time [hour]\n(b)', fontweight='bold',fontsize=24)
plt.ylabel('Cumulative \nHypervolume', fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)
plt.ylim(1e4, 7.5e5)
#plt.legend()
plt.grid(True)
#plt.xscale('log')
plt.tight_layout()
#plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0e}"))
plt.gca().yaxis.get_offset_text().set_fontsize(24)

# Display the plot
#plt.savefig("[plot_hypervolume_with_traintime_trans101]ex_segmentation_v4.pdf", format="pdf")
plt.show()