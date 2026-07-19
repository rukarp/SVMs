"""
data_ccc.py

Default of Credit Card Clientsデータセット
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ファイルの読み込み
file_path = 'dccc(Default_of_Credit_Card_Clients).csv'
data = pd.read_csv(file_path)

# 不明データを削除する
data = data[~data['EDUCATION'].isin([0, 5, 6])]
data = data[~data['MARRIAGE'].isin([0])]

# ワンホットエンコーディング（カテゴリ列すべてに適用）--------------------------
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
data[categorical_cols] = data[categorical_cols].astype(str)
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 元の順番を維持して並べ替え
cols = []
for col in data.columns:
    if col in categorical_cols:
        # 該当カテゴリの展開後の列を追加
        cols.extend([c for c in data_encoded.columns if c.startswith(col + "_")])
    else:
        cols.append(col)

data_encoded = data_encoded[cols]
# -----------------------------------------------------------------------
# ID列の削除
data = data.drop(columns=['ID'])

#print(data_encoded.head())

# データとラベルに分割
X_data = data_encoded.drop(columns=['default payment next month'])
Y_data = data_encoded['default payment next month']

# Numpy形式に変更
X = X_data.astype(float).to_numpy()
Y = Y_data.astype(int).to_numpy()

X_0, X_1, Y_0, Y_1 = X[Y==0], X[Y==1], Y[Y==0], Y[Y==1]
X_0, _, Y_0, _ = train_test_split(X_0, Y_0, test_size=0.713, random_state=0, stratify=Y_0)
X, Y = np.concatenate((X_0, X_1)), np.concatenate((Y_0, Y_1))

# データ数を減らす
X, _, Y, _ = train_test_split(X, Y, test_size=0.6, random_state=44, stratify=Y)

# 正規化
X = MinMaxScaler().fit_transform(X)

# ラベルを -1 と 1 に変換
Y = np.where(Y == np.min(Y), -1, 1)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13, stratify=Y)

# ------ 各拠点に等しく分配* --------
X4_11_train, X_temp, Y4_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.75, stratify=Y_train, random_state=42)
X4_12_train, X_temp, Y4_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X4_13_train, X4_14_train, Y4_13_train, Y4_14_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X4_1_train = [X4_11_train, X4_12_train, X4_13_train, X4_14_train]
Y4_1_train = [Y4_11_train, Y4_12_train, Y4_13_train, Y4_14_train]

# X4_1m_train.shape -> (1330, 31), (1330, 31), (1331, 31), (1331, 31)
# ---------------------------------

# ----------------- K-means法1* -----------------
X_train_0, Y_train_0 = X_train[Y_train == -1], Y_train[Y_train == -1]
X_train_1, Y_train_1 = X_train[Y_train == 1], Y_train[Y_train == 1]

#kmeans_0 = KMeans(n_clusters=4, random_state=0).fit(X_train[:, [0, 1]][Y_train == -1])
kmeans_0 = KMeans(n_clusters=4, random_state=0).fit(X_train[Y_train == -1])
labels_0 = kmeans_0.labels_

#kmeans_1 = KMeans(n_clusters=4, random_state=0).fit(X_train[:, [0, 1]][Y_train == 1])
kmeans_1 = KMeans(n_clusters=4, random_state=0).fit(X_train[Y_train == 1])
labels_1 = kmeans_1.labels_

"""print(len(X_train_0[labels_0 == 0]))
print(len(X_train_0[labels_0 == 1]))
print(len(X_train_0[labels_0 == 2]))
print(len(X_train_0[labels_0 == 3]))
print()
print(len(X_train_1[labels_1 == 0]))
print(len(X_train_1[labels_1 == 1]))
print(len(X_train_1[labels_1 == 2]))
print(len(X_train_1[labels_1 == 3]))"""

X4_51_train, Y4_51_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 1]))
X4_52_train, Y4_52_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 2]))
X4_53_train, Y4_53_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 3]))
X4_54_train, Y4_54_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 0]))

X4_5_train = [X4_51_train, X4_52_train, X4_53_train, X4_54_train]
Y4_5_train = [Y4_51_train, Y4_52_train, Y4_53_train, Y4_54_train]

# X4_5m_train.shape -> (1141, 31), (1369, 31), (1378, 31), (1434, 31)
# --------------------------------


# ------ 各拠点に等しく分配* --------
X6_11_train, X_temp, Y6_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.8333, stratify=Y_train, random_state=42)
X6_12_train, X_temp, Y6_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.8, stratify=Y_temp, random_state=42)
X6_13_train, X_temp, Y6_13_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.75, stratify=Y_temp, random_state=42)
X6_14_train, X_temp, Y6_14_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X6_15_train, X6_16_train, Y6_15_train, Y6_16_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X6_1_train = [X6_11_train, X6_12_train, X6_13_train, X6_14_train, X6_15_train, X6_16_train]
Y6_1_train = [Y6_11_train, Y6_12_train, Y6_13_train, Y6_14_train, Y6_15_train, Y6_16_train]

# X6_1m_train.shape -> (887, 31), (887, 31), (887, 31), (887, 31), (887, 31), (887, 31)
# ---------------------------------

# ----------------- K-means法1* -----------------
X_train_0, Y_train_0 = X_train[Y_train == -1], Y_train[Y_train == -1]
X_train_1, Y_train_1 = X_train[Y_train == 1], Y_train[Y_train == 1]

#kmeans_0 = KMeans(n_clusters=4, random_state=0).fit(X_train[:, [0, 1]][Y_train == -1])
kmeans_0 = KMeans(n_clusters=6, random_state=0).fit(X_train[Y_train == -1])
labels_0 = kmeans_0.labels_

#kmeans_1 = KMeans(n_clusters=4, random_state=0).fit(X_train[:, [0, 1]][Y_train == 1])
kmeans_1 = KMeans(n_clusters=6, random_state=0).fit(X_train[Y_train == 1])
labels_1 = kmeans_1.labels_

"""print(len(X_train_0[labels_0 == 0]))
print(len(X_train_0[labels_0 == 1]))
print(len(X_train_0[labels_0 == 2]))
print(len(X_train_0[labels_0 == 3]))
print(len(X_train_0[labels_0 == 4]))
print(len(X_train_0[labels_0 == 5]))
print()
print(len(X_train_1[labels_1 == 0]))
print(len(X_train_1[labels_1 == 1]))
print(len(X_train_1[labels_1 == 2]))
print(len(X_train_1[labels_1 == 3]))
print(len(X_train_1[labels_1 == 4]))
print(len(X_train_1[labels_1 == 5]))"""

X6_51_train, Y6_51_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 0]))
X6_52_train, Y6_52_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 4])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 4]))
X6_53_train, Y6_53_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 2]))
X6_54_train, Y6_54_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 3]))
X6_55_train, Y6_55_train = np.concatenate((X_train_0[labels_0 == 5], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 5], Y_train_1[labels_1 == 1]))
X6_56_train, Y6_56_train = np.concatenate((X_train_0[labels_0 == 4], X_train_1[labels_1 == 5])), np.concatenate((Y_train_0[labels_0 == 4], Y_train_1[labels_1 == 5]))

X6_5_train = [X6_51_train, X6_52_train, X6_53_train, X6_54_train, X6_55_train, X6_56_train]
Y6_5_train = [Y6_51_train, Y6_52_train, Y6_53_train, Y6_54_train, Y6_55_train, Y6_56_train]

# X6_5m_train.shape -> (943, 31), (887, 31), (1080, 31), (847, 31), (925, 31), (640, 31)
# --------------------------------