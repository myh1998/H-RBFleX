import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
acc_code_path = "/Users/tomomasayamasaki/Library/CloudStorage/OneDrive-SingaporeUniversityofTechnologyandDesign/SUTD/Life_of_University/Lab/#5Research-FRCNSim/Program/RBFleX/imageNet_SSS"
sys.path.append(acc_code_path)
from Check_acc import get_acc

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

main_path = "./"
algo_path = ""  # Main folder having csv files
dataset = 'ImageNet16-120' # 'cifar10', 'cifar100', 'ImageNet16-120'

output_path = "fix_train_output.csv"  # input csv
input_path = "fix_train_input.csv"    # output csv
score_table = "../Score_table_250305_NAFBee_zico.csv"
file_output_path = output_path
file_input_path = input_path

df_output = pd.read_csv(file_output_path, header=None)
df_input = pd.read_csv(file_input_path, header=None)
df_table = pd.read_csv(score_table, header=0)
print(df_input)
print(df_table)



accuracy = list()
score_df = df_output.iloc[:, 0]
cycle_df = -df_output.iloc[:, 1]
for idx, row in df_input.iterrows():
    arch = int(row[0])
    accuracy.append(df_table['acc'].iloc[arch-1])
acc_df = pd.DataFrame(accuracy)
acc_df.to_csv('accuracy.csv', index=False)
all_df = pd.concat([cycle_df, acc_df], axis=1)
all_df.columns = ['cycle', 'acc']

#Distance (max-min norm)
if not "Hard" in algo_path:
    ref_point = np.array([0, 1])
    norm_all_data = all_df.apply(min_max_normalize)
    print(norm_all_data)
    distance_list = list()
    for idx, data in norm_all_data.iterrows():
        point = np.array([data[0], data[1]])
        distance = np.linalg.norm(point - ref_point)
        distance_list.append(distance)
    distance_df = pd.DataFrame(distance_list)

    all_df = pd.concat([all_df, distance_df], axis=1)
    all_df.columns = ['cycle', 'acc', 'distance']
    idx_mindis = all_df['distance'].idxmin()
    min_row = all_df.loc[idx_mindis]
    top5_row = all_df.nsmallest(5, 'distance')
    idxs = top5_row.index
    top5_config = df_input.loc[idxs]
    top5_config.columns = ['ch1', 'ArrayHeight', 'ArrayWeidth', 'IfmapSramSz', 'FilterSramSz', 'OfmapSramSz', 'bandwidth', 'map'] # ch: NATS-Bench-SSS channel size others: Scale-sim config
    print("Top-5 configuration")
    print("Result ")
    print(top5_row.to_string())
    print()
    print("Configuration")
    print(top5_config.to_string())
    pd.concat([top5_config, top5_row], axis=1).to_csv('top-5_maxmin.csv', index=True) #save
    if "Hard" in algo_path:
        print("Hardware solo opt.")
        top5_cycle = cycle_df.nsmallest(10)
        print(top5_cycle)
    if not "Hard" in algo_path:
        print()
        print("**Top-10 on Distance")
        top5_dis = all_df.nsmallest(20, 'distance')
        print(top5_dis)
        print("**Top-10 on acc")
        top5_acc = all_df.nlargest(10, "acc")
        print(top5_acc)
        top5_cycle = all_df.nsmallest(10, "cycle")
        print("**Top-10 on cycle")
        print(top5_cycle)
else:
    top5_row = all_df.nsmallest(5, 'cycle')
    print(top5_row)


plt.figure(figsize=(10, 6))
plt.scatter(all_df['acc'], all_df['cycle'], s=70, c='grey')
if not "Hard" in algo_path:
    plt.scatter(top5_row['acc'], top5_row['cycle'], s=70, c='steelblue')
    plt.scatter(min_row['acc'], min_row['cycle'], s=70, c='r')
plt.xlabel('ImageNet Accuracy (%)')
plt.ylabel('Cycle Count')
#plt.xlim(40, 48)
#plt.ylim(0, 5e7)
plt.grid(True)
plt.title("Accuracy vs Cycle Count")

plt.show()


