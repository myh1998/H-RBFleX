import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'


methods = ["RBFleX-NAS"]
all_dfs = []

i = 0
plt.figure(figsize=(10, 4))
for method in methods:

    # ファイル読み込み
    df = pd.read_csv("RBF_Score_table_241126_semantic.csv")
    df_score = df["score"]
    df_acc = df["acc"]

    # EProxy のときだけ tensor(...) の文字列を float に変換
    if method == "EProxy":
        df_score["score"] = df_score["score"].astype(str).str.extract(r"tensor\((.*?)\)").astype(float)

    # スコアと精度を1つのDataFrameに統合
    df = pd.concat([df_score, df_acc], axis=1)

    # 欠損や異常値のある行を削除
    df = df.dropna(subset=["score", "acc"]).copy()
    df = df[df["score"] != -100000000].copy()

    # スコアに基づくランク付け（高いスコアが上位）
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ヒストグラムの描画設定
    sns.set(style="whitegrid")
    
    
    # スコアのヒストグラム
    plt.subplot(1, 2, 2)
    sns.histplot(df["score"], bins=100, kde=True, color="steelblue", edgecolor="black")
    plt.title(f"{method}", fontweight='bold',fontsize=20)
    plt.xlabel("Score", fontweight='bold',fontsize=20)
    plt.ylabel("Count", fontweight='bold',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # Accのヒストグラム
    plt.subplot(1, 2, 1)
    sns.histplot(df["acc"], bins=100, kde=True, color="seagreen", edgecolor="black")
    plt.title(f"Distribution of DNN Accuracy", fontweight='bold',fontsize=20)
    plt.xlabel("Accuracy (%)", fontweight='bold',fontsize=20)
    plt.ylabel("Count", fontweight='bold',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    
    # Rank vs score
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x="rank", y="score", color="steelblue", linewidth=3)
    plt.fill_between(df["rank"], df["score"], df["score"].min() - 1, color="steelblue", alpha=0.2)
    plt.title("RBFleX-NAS Score vs Rank", fontweight='bold', fontsize=20)
    plt.xlabel("Rank", fontweight='bold',fontsize=20)
    plt.ylabel("Score", fontweight='bold',fontsize=20)
    plt.ylim([-180,-158])
    plt.xlim([1,4001])
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.figtext(0.98, 0.138, "1", 
            ha='center', fontsize=20, color='black', fontweight='bold')
    

    i += 1



plt.tight_layout()
plt.savefig("[modality_semantic]v2_rank.pdf", format="pdf")
plt.show()