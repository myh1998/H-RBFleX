import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def pareto_front(df, color, label, s=20,alpha=1,edgecolors='none'):
    #finding the first front
    dominates = []
    fronts = [[]]

    #plt.scatter(df['acc'], df["edp"], color=color, alpha=0.3, s=s)
    for i in df.index.values:
        dominates.append([])
        isDominant = True
        for j in df.index.values:
            if i == j:
                continue
            # if i dominates j
            if df["edp"][i] < df["edp"][j] and df['acc'][i] < df['acc'][j]:
                dominates[i].append(j)
            # else if i is dominated by j
            elif df["edp"][j] < df["edp"][i] and df['acc'][j] < df['acc'][i]:
                df.loc[i, 'dom'] += 1
        if df.loc[i, 'dom'] == 0:
            fronts[0].append(i)

    for f in fronts:
        pfx = df.loc[f]['acc'].values
        pfy = df.loc[f]["edp"].values

        s_error = 100
        for acc in pfx:
            if acc < s_error:
                s_error = acc

        s_cycle = 1e10
        s_cycle_idx = 0
        for i, acc in enumerate(pfx):
            if s_error == acc:
                if s_cycle > pfy[i]:
                    s_cycle_idx = i
                    s_cycle = pfy[i]

        new_pfx = []
        new_pfy = []
        for i, acc in enumerate(pfx):
            if s_error == acc:
                if s_cycle_idx == i:
                    new_pfx.append(pfx[i])
                    new_pfy.append(pfy[i])
            else:
                new_pfx.append(pfx[i])
                new_pfy.append(pfy[i])

        
        new_pfx = np.array(new_pfx)
        new_pfy = np.array(new_pfy)
        if label == 'MOBO (Accuracy Metric)':
            print("new_pfx: ", new_pfx)
            print("new_pfy: ", new_pfy)
            plt.scatter(new_pfx[0:4], new_pfy[0:4], color=color, label=label, s=s,alpha=alpha,edgecolors=edgecolors)
        else:
            plt.scatter(new_pfx, new_pfy, color=color, label=label, s=s,alpha=alpha,edgecolors=edgecolors)
        print("label: ", label)
        print("num: ", len(new_pfy))
        
    
def pareto_front_line(df, color, label, s=20,alpha=1,edgecolors='none'):
    #finding the first front
    dominates = []
    fronts = [[]]

    #plt.scatter(df['acc'], df["edp"], color=color, alpha=0.3, s=s)
    for i in df.index.values:
        dominates.append([])
        isDominant = True
        for j in df.index.values:
            if i == j:
                continue
            # if i dominates j
            if df["edp"][i] < df["edp"][j] and df['acc'][i] < df['acc'][j]:
                dominates[i].append(j)
            # else if i is dominated by j
            elif df["edp"][j] < df["edp"][i] and df['acc'][j] < df['acc'][i]:
                df.loc[i, 'dom'] += 1
        if df.loc[i, 'dom'] == 0:
            fronts[0].append(i)

    for f in fronts:
        pfx = df.loc[f]['acc'].values
        pfy = df.loc[f]["edp"].values

        s_error = 200
        for acc in pfx:
            if acc < s_error:
                s_error = acc

        s_cycle = 100
        s_cycle_idx = 0
        for i, acc in enumerate(pfx):
            if s_error == acc:
                if s_cycle > pfy[i]:
                    s_cycle_idx = i
                    s_cycle = pfy[i]

        new_pfx = []
        new_pfy = []
        for i, acc in enumerate(pfx):
            if s_error == acc:
                if s_cycle_idx == i:
                    new_pfx.append(pfx[i])
                    new_pfy.append(pfy[i])
            else:
                new_pfx.append(pfx[i])
                new_pfy.append(pfy[i])

        
        new_pfx = np.array(new_pfx)
        new_pfy = np.array(new_pfy)
        if label == 'MOBO (Accuracy Metric)':
            new_pfx = new_pfx[0:4]
            new_pfy = new_pfy[0:4]
        mask = new_pfx <= 1.841 #180
        filtered_pfx = new_pfx[mask]
        filtered_pfy = new_pfy[mask]
        mask = filtered_pfy <= 2.822 #100
        filtered_pfx = filtered_pfx[mask]
        filtered_pfy = filtered_pfy[mask]
        sorted_indices = np.argsort(filtered_pfx)
        x_sorted = filtered_pfx[sorted_indices]
        y_sorted = filtered_pfy[sorted_indices]
        if "H-RBFleXa" in label or "NASWOTa" in label:
            print()
        else:
            if label == 'MOBO (Accuracy Metric)':
                x_sorted = x_sorted[0:3]
                y_sorted = y_sorted[0:3]
            plt.plot(
                x_sorted, y_sorted,
                linestyle='--',
                color=color,
                alpha=1,
                linewidth=2,
                label=label
            )
        print("label: ", label)



output_path = "fix_train_output.csv"  # input csv
acc_path = "accuracy.csv"

color = plt.cm.coolwarm(np.linspace(0, 1, 5))
color_RBFp = color[4]
color_WOT = color[0]
color_RBF = color[3]


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.figure(figsize=(10, 6))



# Result 1 Large
color1 = "#2ca02c"
label_1_name = 'NSGA-II (Accuracy Metric)'
main_path = './NSGA_Acc_noband_NLP2/Results'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_1_output = pd.read_csv(re_output_path, header=None)
df_1_acc = pd.read_csv(re_acc_path, header=0)
all_1_df = pd.concat([df_1_output, df_1_acc], axis=1)
all_1_df.columns = ['score', "edp", 'acc']
all_1_df["edp"] = all_1_df["edp"] * -1
all_1_df['dom'] = 0
#pareto_front(all_1_df, color, label_1_name)


# Result 2 Large
color2 = "#9467bd"
label_2_name = 'Random Search (Accuracy Metric)'
main_path = main_path = "./RandomSearch_Large_Acc_noband_NLP/Results"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_2_output = pd.read_csv(re_output_path, header=None)
df_2_acc = pd.read_csv(re_acc_path, header=0)
all_2_df = pd.concat([df_2_output, df_2_acc], axis=1)
all_2_df.columns = ['score', "edp", 'acc']
all_2_df["edp"] = all_2_df["edp"] * -1
all_2_df['dom'] = 0
#pareto_front(all_2_df, color, label_2_name)

# Result 3 Large
color3 = "#8c564b"
label_3_name = 'MOBO (Accuracy Metric)'
main_path = "./MultiOpt_Large_Acc_noband_NLP/Results_original"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_3_output = pd.read_csv(re_output_path, header=None)
df_3_acc = pd.read_csv(re_acc_path, header=0)
all_3_df = pd.concat([df_3_output, df_3_acc], axis=1)
all_3_df.columns = ['score', "edp", 'acc']
all_3_df["edp"] = all_3_df["edp"] * -1
all_3_df['dom'] = 0
#pareto_front(all_3_df, color, label_3_name)


# Result 4 Large
color4 = "#ff7f0e"
label_4_name = 'MOBO (RBFleX-NAS Score Metric)'
main_path = "./MultiOpt_Large_noband_NLP/Results_2"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_4_output = pd.read_csv(re_output_path, header=None)
df_4_acc = pd.read_csv(re_acc_path, header=0)
all_4_df = pd.concat([df_4_output, df_4_acc], axis=1)
all_4_df.columns = ['score', "edp", 'acc']
all_4_df["edp"] = all_4_df["edp"] * -1
all_4_df['dom'] = 0
#pareto_front(all_4_df, color, label_4_name)


# Result 5 Large
color5 = color_WOT
label_5_name = 'MOBO (NASWOT Ranking Table)'
main_path = './H-WOT_Large_noband_NLP/Results'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_5_output = pd.read_csv(re_output_path, header=None)
df_5_acc = pd.read_csv(re_acc_path, header=0)
all_5_df = pd.concat([df_5_output, df_5_acc], axis=1)
all_5_df.columns = ['score', "edp", 'acc']
all_5_df["edp"] = all_5_df["edp"] * -1
all_5_df['dom'] = 0
#pareto_front(all_5_df, color, label_5_name)

# Result 6 Large
color6 = color_RBFp #red
label_6_name = 'H-RBFleX (Ours)'
main_path = "./H-RBFleX_Large_noband_NLP/Results"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_6_output = pd.read_csv(re_output_path, header=None)
df_6_acc = pd.read_csv(re_acc_path, header=0)
all_6_df = pd.concat([df_6_output, df_6_acc], axis=1)
all_6_df.columns = ['score', "edp", 'acc']
all_6_df["edp"] = all_6_df["edp"] * -1
all_6_df['dom'] = 0
#pareto_front(all_6_df, color, label_6_name)

# Normlaized
base = all_6_df["edp"].min()
all_1_df["edp"] = all_1_df["edp"] / base
all_2_df["edp"] = all_2_df["edp"] / base
all_3_df["edp"] = all_3_df["edp"] / base
all_4_df["edp"] = all_4_df["edp"] / base
all_5_df["edp"] = all_5_df["edp"] / base
all_6_df["edp"] = all_6_df["edp"] / base

base = all_6_df["acc"].min()
all_1_df["acc"] = all_1_df["acc"] / base
all_2_df["acc"] = all_2_df["acc"] / base
all_3_df["acc"] = all_3_df["acc"] / base
all_4_df["acc"] = all_4_df["acc"] / base
all_5_df["acc"] = all_5_df["acc"] / base
all_6_df["acc"] = all_6_df["acc"] / base

s=40
a=0.2
#plt.scatter(all_1_df['acc'], all_1_df["edp"], color=color1, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_2_df['acc'], all_2_df["edp"], color=color2, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_3_df['acc'], all_3_df["edp"], color=color3, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_4_df['acc'], all_4_df["edp"], color=color4, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_5_df['acc'], all_5_df["edp"], color=color5, alpha=a, s=s,edgecolors='none')
drop_6_df = all_6_df.drop_duplicates(subset=["acc", "edp"])
plt.scatter(drop_6_df['acc'], drop_6_df["edp"], color=color6, alpha=a, s=s,edgecolors='none')

color5 = color_WOT
color6 = color_RBFp
a = 1

pareto_front_line(all_2_df, color2, label_2_name, s=170,alpha=1,edgecolors='k')
pareto_front_line(all_4_df, color4, label_4_name, s=170,alpha=1,edgecolors='k')
pareto_front_line(all_3_df, color3, label_3_name, s=170,alpha=1,edgecolors='k')
pareto_front_line(all_5_df, color5, label_5_name, s=170,alpha=1,edgecolors='k')
pareto_front_line(all_1_df, color1, label_1_name, s=170,alpha=1,edgecolors='k')
pareto_front_line(all_6_df, color6, label_6_name, s=170,alpha=1,edgecolors='k')


pareto_front(all_2_df, color2, label_2_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_4_df, color4, label_4_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_3_df, color3, label_3_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_1_df, color1, label_1_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_6_df, color6, label_6_name, s=370,alpha=1,edgecolors='k')
pareto_front(all_5_df, color5, label_5_name, s=170,alpha=1,edgecolors='k')




plt.xlabel("Normalized Perplexity", fontweight='bold',fontsize=24)
plt.ylabel("Normalized EDP", fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)

plt.ylim([0.8,2.822]) #a
plt.xlim([0.95,1.84]) #best

#plt.legend(fontsize=10)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.get_offset_text().set_fontsize(24)
plt.savefig("[plot_pareto_front_nlp]ex_nlp_v5.pdf", format="pdf")
#plt.show()



# スタイル設定（任意）
import matplotlib.cm as cm
plt.figure()

# 先頭100件を除いた300件を抽出
subset_df = drop_6_df.iloc[0:400].reset_index(drop=True)

# カラーマップを濃い色で15色生成（例: inferno, plasma, viridis などが濃い系）
cmap = cm.get_cmap('plasma', 15)  # inferno は濃い色調のグラデーション

"""
# プロット
fig, axes = plt.subplots(5, 6, figsize=(18, 12))
axes = axes.flatten()  # 2次元 → 1次元に変換して for ループで扱いやすく
for i in range(30):  # 300件 ÷ 10 = 30グループ
    ax = axes[i]
    group = subset_df.iloc[i*10:(i+1)*10]
    if i == 14:
        color = cmap(15-i % 15)  # 15色でループさせる
    else:
        color = 'gray'
    label = f'iter {i}'
    ax.scatter(group['acc'], group["edp"], color=color, alpha=1.0, s=30, edgecolors='none', label=label)
    ax.set_title(f'Group {i+1}\n({label})', fontsize=9)
    ax.set_ylim([0,0.4e6])
    ax.set_xlim([73,85])
"""

plt.figure(figsize=(10, 4))
for i in range(40): 
    if i < 24:
        group = subset_df.iloc[i*10:(i+1)*10]
        plt.scatter(group['acc'], group["edp"], color='gray', alpha=0.4, s=170, edgecolors='none')
    elif i > 24:
        group = subset_df.iloc[i*10:(i+1)*10]
        plt.scatter(group['acc'], group["edp"], color="#4CA1A3", alpha=0.4, s=170, edgecolors='none')

i = 24
group = subset_df.iloc[i*10:(i+1)*10]
plt.scatter(group['acc'], group["edp"], color='#D9856A', alpha=1.0, s=270, edgecolors='k')

plt.ylim([0,500]) #a
plt.xlim([0,500]) #best
plt.xlabel("Perplexity", fontweight='bold',fontsize=24)
plt.ylabel("EDP (second * nJ)", fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.get_offset_text().set_fontsize(24)
#plt.savefig("[plot_pareto_front_nlp]discussion_nlp_v2.pdf", format="pdf")
plt.show()