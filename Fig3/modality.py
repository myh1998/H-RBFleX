import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'


#methods = ["RBFleX-NAS","RBFleX-NAS","NASWOT", "ZiCO"]
methods = ["RBFleX-NAS","RBFleX-NAS","Syn", "Snip"]
all_dfs = []

method = methods[1]
i = 0
plt.figure(figsize=(8, 8))
for method in methods:

    # ファイル読み込み
    df_score = pd.read_csv("./SegmentSemantic/{}_score.csv".format(method), header=0, names=["score"])
    df_acc = pd.read_csv("./SegmentSemantic/{}_accuracy.csv".format(method), header=0, names=["acc"])

    # EProxy のときだけ tensor(...) の文字列を float に変換
    if method == "EProxy":
        df_score["score"] = df_score["score"].astype(str).str.extract(r"tensor\((.*?)\)").astype(float)

    # スコアと精度を1つのDataFrameに統合
    df = pd.concat([df_score, df_acc], axis=1)

    # 欠損や異常値のある行を削除
    df = df.dropna(subset=["score", "acc"]).copy()
    df = df[df["score"] != -100000000].copy()

    if method == "Syn":
        df = df[df["score"] > -2e3].copy()

    if method == "TE":
        df = df[df["score"] < 1e7].copy()
        df = df[df["score"] > -1e8 ].copy()

    # スコアに基づくランク付け（高いスコアが上位）
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ヒストグラムの描画設定
    sns.set(style="whitegrid")
    

    # スコアのヒストグラム
    if i > 0:
        plt.subplot(2, 2, i+1)
        sns.histplot(df["score"], bins=100, kde=True, color="steelblue", edgecolor="black")
        plt.title(f"{method}", fontweight='bold')
        plt.xlabel("Score", fontweight='bold')
        plt.ylabel("Count", fontweight='bold')
    else:
        # Rankのヒストグラム
        plt.subplot(2, 2, i+1)
        sns.histplot(df["acc"], bins=100, kde=True, color="seagreen", edgecolor="black")
        plt.title(f"Distribution of DNN Accuracy", fontweight='bold')
        plt.xlabel("Accuracy (%)", fontweight='bold')
        plt.ylabel("Count", fontweight='bold')

    i += 1



plt.tight_layout()
#plt.savefig("[modality]v2.pdf", format="pdf")
plt.show()