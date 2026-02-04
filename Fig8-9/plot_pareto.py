import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys
acc_code_path = "/Users/tomomasayamasaki/Library/CloudStorage/OneDrive-SingaporeUniversityofTechnologyandDesign/SUTD/Life_of_University/Lab/#5Research-FRCNSim/Program/RBFleX/imageNet_SSS"
sys.path.append(acc_code_path)
from Check_acc import get_acc
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

dataset = 'ImageNet16-120' # 'cifar10', 'cifar100', 'ImageNet16-120'
output_path = "fix_train_output.csv"  # input csv
acc_path = "accuracy.csv"

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

#########################################
#   READ FILES
#########################################

# Result 1
label_1_name = 'grad_norm'
main_path = './Results_gradnorm_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_1_output = pd.read_csv(re_output_path, header=None)
df_1_acc = pd.read_csv(re_acc_path, header=0)
all_1_df = pd.concat([df_1_output, df_1_acc], axis=1)
all_1_df.columns = ['score', 'cycle', 'acc']

# Result 2
label_2_name = 'NASWOT'
main_path = './Results_NASWOT_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_2_output = pd.read_csv(re_output_path, header=None)
df_2_acc = pd.read_csv(re_acc_path, header=0)
all_2_df = pd.concat([df_2_output, df_2_acc], axis=1)
all_2_df.columns = ['score', 'cycle', 'acc']

# Result 3
label_3_name = 'snip'
main_path = './Results_snip_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_3_output = pd.read_csv(re_output_path, header=None)
df_3_acc = pd.read_csv(re_acc_path, header=0)
all_3_df = pd.concat([df_3_output, df_3_acc], axis=1)
all_3_df.columns = ['score', 'cycle', 'acc']


# Result 4
label_4_name = 'synflow'
main_path = './Results_synflow_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_4_output = pd.read_csv(re_output_path, header=None)
df_4_acc = pd.read_csv(re_acc_path, header=0)
all_4_df = pd.concat([df_4_output, df_4_acc], axis=1)
all_4_df.columns = ['score', 'cycle', 'acc']

label_5_name = 'TE-NAS'
main_path = './Results_TENAS_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_5_output = pd.read_csv(re_output_path, header=None)
df_5_acc = pd.read_csv(re_acc_path, header=0)
all_5_df = pd.concat([df_5_output, df_5_acc], axis=1)
all_5_df.columns = ['score', 'cycle', 'acc']

label_6_name = 'ZiCO'
main_path = './Results_zico_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_6_output = pd.read_csv(re_output_path, header=None)
df_6_acc = pd.read_csv(re_acc_path, header=0)
all_6_df = pd.concat([df_6_output, df_6_acc], axis=1)
all_6_df.columns = ['score', 'cycle', 'acc']

label_7_name = 'RBFleX-NAS'
main_path = './Results_RBFleX_ep30'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_7_output = pd.read_csv(re_output_path, header=None)
df_7_acc = pd.read_csv(re_acc_path, header=0)
all_7_df = pd.concat([df_7_output, df_7_acc], axis=1)
all_7_df.columns = ['score', 'cycle', 'acc']


#########################################
#   PLOT
#########################################

# ACCURACY
color = plt.cm.coolwarm(np.linspace(0, 1, 6))
colors = [color[0],"#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf",color[5]]
plt.figure(figsize=(10, 6))
plt.scatter(100-all_2_df['acc'], -all_2_df['cycle'], color=color[0], label=label_2_name, s=150, edgecolors='k')
plt.scatter(100-all_3_df['acc'], -all_3_df['cycle'], color="#ff7f0e", label=label_3_name, s=150, edgecolors='k')
plt.scatter(100-all_1_df['acc'], -all_1_df['cycle'], color="#17becf", label=label_1_name, s=150, edgecolors='k')
plt.scatter(100-all_4_df['acc'], -all_4_df['cycle'], color="#9467bd", label=label_4_name, s=150, edgecolors='k')
plt.scatter(100-all_5_df['acc'], -all_5_df['cycle'], color="#2ca02c", label=label_5_name, s=150, edgecolors='k')
plt.scatter(100-all_6_df['acc'], -all_6_df['cycle'], color="#8c564b", label=label_6_name, s=150, edgecolors='k')
plt.scatter(100-all_7_df['acc'], -all_7_df['cycle'], color=color[5], label=label_7_name, s=150, edgecolors='k')
plt.scatter(100-91.06, 333901, color=color[5], marker="*", s=900, label="The most efficient config\n by RBFleX-NAS", edgecolors='k')

#plt.title("Full Data points (Accuracy)")
plt.xlabel("Error (%)", fontweight='bold',fontsize=24)
plt.ylabel("Cycle count", fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)
plt.ylim([3e5,5e5])
plt.xlim([8.8,15])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(0.4, -0.2),fontsize=20, ncol=3, loc='upper center', frameon=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.savefig("[plot_pareto]NAS_selection_v2.pdf", format="pdf")
plt.show()


