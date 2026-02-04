import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# Define a list of folder paths
"""
folders = ['./H-RBFleX/Results_N32780_qNargo',
           './MultiOpt/Result_qNargo_IM',
           './MultiOpt_Acc/Results_qNParEGO_IM']  # Replace with actual folder paths
"""
folders = ['./Results_NASWOT_ep30',
           './Results_TENAS_ep30',
           './Results_snip_ep30',
           './Results_synflow_ep30',
           './Results_zico_ep30',
           './Results_gradnorm_ep30',
           './Results_RBFleX_ep30']  # Replace with actual folder paths

labels = ["NASWOT", "TE-NAS", "snip", "synflow", "ZiCO", "grad_norm", "RBFleX-NAS"]

color = plt.cm.coolwarm(np.linspace(0, 1, 6))
colors = [color[0],"#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf",color[5]]

# List to store the data frames for hypervolume and time
data_frames = []
time_data_frames = []

# Read CSV files from each folder
for folder in folders:
    # Path to cumulative_hypervolume.csv
    hv_file_path = os.path.join(folder, 'cumulative_hypervolume_s2_v2.csv')
    
    # Path to time.csv
    time_file_path = os.path.join(folder, 'time.csv')
    
    if os.path.exists(hv_file_path) and os.path.exists(time_file_path):
        # Read cumulative hypervolume data
        hv_df = pd.read_csv(hv_file_path)
        if 'Cumulative Hypervolume' in hv_df.columns:
            # Drop the first 9 rows of cumulative hypervolume data (beacuse of random)
            #hv_df = hv_df.drop(index=range(9))
            data_frames.append(hv_df['Cumulative Hypervolume'])
            
        else:
            print(f"'Cumulative Hypervolume' column not found in {hv_file_path}.")

        # Read time data (no removal of rows)
        time_df = pd.read_csv(time_file_path, header=None)
        cumulative_time = time_df.cumsum().reset_index(drop=True)
        # Append the entire time data
        time_data_frames.append(cumulative_time.reset_index(drop=True))  
    else:
        print(f"File {hv_file_path} or {time_file_path} not found.")



# Create the plot
plt.figure(figsize=(10, 6))

for i, (hv_df, time_df) in enumerate(zip(data_frames, time_data_frames)):
    # Dynamically extract the label from the folder path
    label = folders[i].split('/')[1]  # Extract 'H-RBFleX' from 'H-RBFleX/Results_N32780_qNargo'
    
    hv_df = hv_df.drop(index=0).reset_index(drop=True)
    plt.plot(time_df/3600, hv_df, color=colors[i], label=labels[i], linewidth=2)
    plt.scatter(time_df/3600, hv_df, color=colors[i],s=150, edgecolors='k')

plt.axvline(
    x=0.3838,
    color='black',
    linestyle='dashed',
    linewidth=3
)

# Customize the plot
plt.xlabel('Time [hour]', fontweight='bold',fontsize=24)
plt.ylabel('Cumulative\n Hypervolume', fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)
#plt.xlim([0.3,0.8])
#plt.ylim([5.75e7,5.87e7])
#plt.title('Cumulative Hypervolume over Time')
#plt.grid(axis='y')
plt.grid(True)
plt.subplots_adjust(bottom=0)
plt.tight_layout()
#plt.yscale('log')
plt.gca().yaxis.get_offset_text().set_fontsize(24)


# Display the plot
plt.savefig("[plot_hypervolume]NAS_selection_hypervolume_v5.pdf", format="pdf")
plt.show()