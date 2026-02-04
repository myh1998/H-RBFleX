import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def pareto_front(df, color, label, s=20,alpha=1,edgecolors='none'):
    #finding the first front
    dominates = []
    fronts = [[]]

    #plt.scatter(df['acc'], df['cycle'], color=color, alpha=0.3, s=s)
    for i in df.index.values:
        dominates.append([])
        isDominant = True
        for j in df.index.values:
            if i == j:
                continue
            # if i dominates j
            if df['cycle'][i] < df['cycle'][j] and df['acc'][i] < df['acc'][j]:
                dominates[i].append(j)
            # else if i is dominated by j
            elif df['cycle'][j] < df['cycle'][i] and df['acc'][j] < df['acc'][i]:
                df.loc[i, 'dom'] += 1
        if df.loc[i, 'dom'] == 0:
            fronts[0].append(i)

    for f in fronts:
        pfx = df.loc[f]['acc'].values
        pfy = df.loc[f]['cycle'].values

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
        plt.scatter(new_pfx, new_pfy, color=color, label=label, s=s,alpha=alpha,edgecolors=edgecolors)
    
def pareto_front_line(df, color, label, s=20,alpha=1,edgecolors='none'):
    #finding the first front
    dominates = []
    fronts = [[]]

    #plt.scatter(df['acc'], df['cycle'], color=color, alpha=0.3, s=s)
    for i in df.index.values:
        dominates.append([])
        isDominant = True
        for j in df.index.values:
            if i == j:
                continue
            # if i dominates j
            if df['cycle'][i] < df['cycle'][j] and df['acc'][i] < df['acc'][j]:
                dominates[i].append(j)
            # else if i is dominated by j
            elif df['cycle'][j] < df['cycle'][i] and df['acc'][j] < df['acc'][i]:
                df.loc[i, 'dom'] += 1
        if df.loc[i, 'dom'] == 0:
            fronts[0].append(i)

    for f in fronts:
        pfx = df.loc[f]['acc'].values
        pfy = df.loc[f]['cycle'].values

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
        mask = new_pfx <= 63
        filtered_pfx = new_pfx[mask]
        filtered_pfy = new_pfy[mask]
        sorted_indices = np.argsort(filtered_pfx)
        x_sorted = filtered_pfx[sorted_indices]
        y_sorted = filtered_pfy[sorted_indices]
        if "H-RBFleX" in label or "NASWOT" in label:
            print()
        else:
            plt.plot(
                x_sorted, y_sorted,
                linestyle='--',
                color=color,
                alpha=1,
                linewidth=2,
                label=label
            )


dataset = 'ImageNet16-120' # 'cifar10', 'cifar100', 'ImageNet16-120'
output_path = "fix_train_output.csv"  # input csv
acc_path = "accuracy.csv"

color = plt.cm.coolwarm(np.linspace(0, 1, 5))
color_RBFp = color[4]
color_WOT = color[0]
color_RBF = color[3]


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.figure(figsize=(8, 6))



# Result 1 Large
color1 = "#2ca02c"
label_1_name = 'NSGA-II (Accuracy Metric)'
main_path = './NSGA_Acc_noband/Results'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_1_output = pd.read_csv(re_output_path, header=None)
df_1_acc = pd.read_csv(re_acc_path, header=0)
all_1_df = pd.concat([df_1_output, df_1_acc], axis=1)
all_1_df.columns = ['score', 'cycle', 'acc']
all_1_df['cycle'] = all_1_df['cycle'] * -1
all_1_df['acc'] = 100-all_1_df['acc']
all_1_df['dom'] = 0
#pareto_front(all_1_df, color, label_1_name)


# Result 2 Large
color2 = "#9467bd"
label_2_name = 'Random Search (Accuracy Metric)'
main_path = "./MultiOpt_Large_Acc_noband/Results_randomsearch"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_2_output = pd.read_csv(re_output_path, header=None)
df_2_acc = pd.read_csv(re_acc_path, header=0)
all_2_df = pd.concat([df_2_output, df_2_acc], axis=1)
all_2_df.columns = ['score', 'cycle', 'acc']
all_2_df['cycle'] = all_2_df['cycle'] * -1
all_2_df['acc'] = 100-all_2_df['acc']
all_2_df['dom'] = 0
#pareto_front(all_2_df, color, label_2_name)

# Result 3 Large
color3 = "#8c564b"
label_3_name = 'MOBO (Accuracy Metric)'
main_path = "./MultiOpt_Large_Acc_noband/Results"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_3_output = pd.read_csv(re_output_path, header=None)
df_3_acc = pd.read_csv(re_acc_path, header=0)
all_3_df = pd.concat([df_3_output, df_3_acc], axis=1)
all_3_df.columns = ['score', 'cycle', 'acc']
all_3_df['cycle'] = all_3_df['cycle'] * -1
all_3_df['acc'] = 100-all_3_df['acc']
all_3_df['dom'] = 0
#pareto_front(all_3_df, color, label_3_name)


# Result 4 Large
color4 = "#ff7f0e"
label_4_name = 'MOBO (RBFleX-NAS Score Metric)'
main_path = "./MultiOpt_Large_noband/Result"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_4_output = pd.read_csv(re_output_path, header=None)
df_4_acc = pd.read_csv(re_acc_path, header=0)
all_4_df = pd.concat([df_4_output, df_4_acc], axis=1)
all_4_df.columns = ['score', 'cycle', 'acc']
all_4_df['cycle'] = all_4_df['cycle'] * -1
all_4_df['acc'] = 100-all_4_df['acc']
all_4_df['dom'] = 0
#pareto_front(all_4_df, color, label_4_name)


# Result 5 Large
color5 = color_WOT
label_5_name = 'MOBO (NASWOT Ranking Table)'
main_path = './H-WOT_Large_noband/Results'
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_5_output = pd.read_csv(re_output_path, header=None)
df_5_acc = pd.read_csv(re_acc_path, header=0)
all_5_df = pd.concat([df_5_output, df_5_acc], axis=1)
all_5_df.columns = ['score', 'cycle', 'acc']
all_5_df['cycle'] = all_5_df['cycle'] * -1
all_5_df['acc'] = 100-all_5_df['acc']
all_5_df['dom'] = 0
#pareto_front(all_5_df, color, label_5_name)

# Result 6 Large
color6 = color_RBFp #red
label_6_name = 'H-RBFleX (Ours)'
main_path = "./H-RBFleX_Large_noband/Results_init100"
re_output_path = main_path + '/'+ output_path
re_acc_path = main_path + '/'+ acc_path
df_6_output = pd.read_csv(re_output_path, header=None)
df_6_acc = pd.read_csv(re_acc_path, header=0)
all_6_df = pd.concat([df_6_output, df_6_acc], axis=1)
all_6_df.columns = ['score', 'cycle', 'acc']
all_6_df['cycle'] = all_6_df['cycle'] * -1
all_6_df['acc'] = 100-all_6_df['acc']
all_6_df['dom'] = 0

filtered_df = all_6_df[
    (all_6_df['cycle'] >= 9.5e4) & (all_6_df['cycle'] <= 1.05e5) &
    (all_6_df['acc'] >= 62) & (all_6_df['acc'] <= 63)
]
print(filtered_df)
exit()

s=40
a=0.2
#plt.scatter(all_1_df['acc'], all_1_df['cycle'], color=color1, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_2_df['acc'], all_2_df['cycle'], color=color2, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_3_df['acc'], all_3_df['cycle'], color=color3, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_4_df['acc'], all_4_df['cycle'], color=color4, alpha=a, s=s,edgecolors='none')
#plt.scatter(all_5_df['acc'], all_5_df['cycle'], color=color5, alpha=a, s=s,edgecolors='none')
drop_6_df = all_6_df.drop_duplicates(subset=["acc", "cycle"])
plt.scatter(drop_6_df['acc'], drop_6_df['cycle'], color=color6, alpha=a, s=s,edgecolors='none')
plt.show()
exit()

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
pareto_front(all_5_df, color5, label_5_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_1_df, color1, label_1_name, s=170,alpha=1,edgecolors='k')
pareto_front(all_6_df, color6, label_6_name, s=170,alpha=1,edgecolors='k')





plt.xlabel("Error (%)\n(a)", fontweight='bold',fontsize=24)
plt.ylabel("Cycle count", fontweight='bold',fontsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.tick_params(axis='x', labelsize=24)

plt.ylim([1e5*0.8,5e5]) #a
#plt.ylim([149000,152000]) #b

plt.xlim([52,63]) #best

#plt.legend(fontsize=10)
plt.tight_layout()
plt.grid(True)
plt.gca().yaxis.get_offset_text().set_fontsize(24)
#plt.savefig("[plot_pareto_front]ex_imageclass_v3a_extend.pdf", format="pdf")
#plt.show()




# スタイル設定（任意）
import matplotlib.cm as cm
plt.figure()
plt.style.use('ggplot')

# 先頭100件を除いた300件を抽出
subset_df = drop_6_df.iloc[100:400].reset_index(drop=True)

# カラーマップを濃い色で15色生成（例: inferno, plasma, viridis などが濃い系）
cmap = cm.get_cmap('inferno', 15)  # inferno は濃い色調のグラデーション

# プロット
plt.figure(figsize=(10, 6))
for i in range(30):  # 300件 ÷ 10 = 30グループ
    group = subset_df.iloc[i*10:(i+1)*10]
    color = cmap(15-i % 15)  # 15色でループさせる
    label = f'iter {i}'
    plt.scatter(group['acc'], group['cycle'], color=color, alpha=1.0, s=20, edgecolors='none', label=label)

plt.xlabel('Accuracy')
plt.ylabel('Cycle')
plt.title('Grouped Plot by Color')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # 凡例を右側に出す
plt.grid(True)
plt.show()