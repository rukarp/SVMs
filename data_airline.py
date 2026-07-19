import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def preprocess_airline_data(file_path):
    # CSV読み込み
    df = pd.read_csv(file_path)
    
    # 不要列削除
    df = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')
    
    """# 欠損値補完
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)"""
    
    # 欠損値行削除
    df = df.dropna()
    
    # ワンホットエンコーディング
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    
    # データとラベルに分割
    X_df = df.drop(columns=['satisfaction_satisfied'])
    Y_df = df['satisfaction_satisfied']
    
    # Numpy配列へ変換
    X = X_df.astype(float).to_numpy()
    Y = Y_df.astype(int).to_numpy()
    
    return X, Y, df.columns.tolist()  # 列名も返す

# ----------------- train -----------------
file_path_train = 'Airline_Passenger_Satisfaction_train.csv'
X_train, Y_train, cols_train = preprocess_airline_data(file_path_train)

# ----------------- test ------------------
file_path_test = 'Airline_Passenger_Satisfaction_test.csv'
X_test, Y_test, cols_test = preprocess_airline_data(file_path_test)

# データ数の削減
X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.9, random_state=1, stratify=Y_train)
X_test, _, Y_test, _ = train_test_split(X_test, Y_test, test_size=0.9, random_state=1, stratify=Y_test)

# ----------------- 正規化 -----------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # train で基準を作る
X_test = scaler.transform(X_test)        # test も同じ基準でスケール

# ラベルを -1 と 1 に変換 ------------------
Y_min = np.min(Y_train)
Y_train = np.where(Y_train == Y_min, -1, 1)
Y_test = np.where(Y_test == Y_min, -1, 1)

# --------------------------------------------------------------------------------



# ------ 各拠点に等しく分配* --------
X4_11_train, X_temp, Y4_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.75, stratify=Y_train, random_state=42)
X4_12_train, X_temp, Y4_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X4_13_train, X4_14_train, Y4_13_train, Y4_14_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X4_1_train = [X4_11_train, X4_12_train, X4_13_train, X4_14_train]
Y4_1_train = [Y4_11_train, Y4_12_train, Y4_13_train, Y4_14_train]

#print(X4_11_train.shape, X4_12_train.shape, X4_13_train.shape, X4_14_train.shape)
#print(Y4_11_train.shape, Y4_12_train.shape, Y4_13_train.shape, Y4_14_train.shape)

# X4_1m_train.shape -> (2589, 23) (2590, 23) (2590, 23) (2590, 23)
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

X4_51_train, Y4_51_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 9]))
X4_52_train, Y4_52_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 1]))
X4_53_train, Y4_53_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 3]))
X4_54_train, Y4_54_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 2]))

X4_5_train = [X4_51_train, X4_52_train, X4_53_train, X4_54_train]
Y4_5_train = [Y4_51_train, Y4_52_train, Y4_53_train, Y4_54_train]

#print(X4_51_train.shape, X4_52_train.shape, X4_53_train.shape, X4_54_train.shape)
#print(Y4_51_train.shape, Y4_52_train.shape, Y4_53_train.shape, Y4_54_train.shape)

# X4_5m_train.shape -> (3191, 23) (2560, 23) (2528, 23) (2080, 23)
# --------------------------------


# ------ 各拠点に等しく分配* --------
X6_11_train, X_temp, Y6_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.8333, stratify=Y_train, random_state=42)
X6_12_train, X_temp, Y6_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.8, stratify=Y_temp, random_state=42)
X6_13_train, X_temp, Y6_13_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.75, stratify=Y_temp, random_state=42)
X6_14_train, X_temp, Y6_14_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X6_15_train, X6_16_train, Y6_15_train, Y6_16_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X6_1_train = [X6_11_train, X6_12_train, X6_13_train, X6_14_train, X6_15_train, X6_16_train]
Y6_1_train = [Y6_11_train, Y6_12_train, Y6_13_train, Y6_14_train, Y6_15_train, Y6_16_train]

#print(X6_11_train.shape, X6_12_train.shape, X6_13_train.shape, X6_14_train.shape, X6_15_train.shape, X6_16_train.shape)
#print(Y6_11_train.shape, Y6_12_train.shape, Y6_13_train.shape, Y6_14_train.shape, Y6_15_train.shape, Y6_16_train.shape)

# X6_1m_train.shape -> (1726, 23) (1726, 23) (1726, 23) (1727, 23) (1727, 23) (1727, 23)
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

X6_51_train, Y6_51_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 0]))
X6_52_train, Y6_52_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 5])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 5]))
X6_53_train, Y6_53_train = np.concatenate((X_train_0[labels_0 == 5], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 5], Y_train_1[labels_1 == 1]))
X6_54_train, Y6_54_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 2]))
X6_55_train, Y6_55_train = np.concatenate((X_train_0[labels_0 == 4], X_train_1[labels_1 == 4])), np.concatenate((Y_train_0[labels_0 == 4], Y_train_1[labels_1 == 4]))
X6_56_train, Y6_56_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 3]))

X6_5_train = [X6_51_train, X6_52_train, X6_53_train, X6_54_train, X6_55_train, X6_56_train]
Y6_5_train = [Y6_51_train, Y6_52_train, Y6_53_train, Y6_54_train, Y6_55_train, Y6_56_train]

#print(X6_51_train.shape, X6_52_train.shape, X6_53_train.shape, X6_54_train.shape, X6_55_train.shape, X6_56_train.shape)
#print(Y6_51_train.shape, Y6_52_train.shape, Y6_53_train.shape, Y6_54_train.shape, Y6_55_train.shape, Y6_56_train.shape)

# X6_5m_train.shape -> (1737, 23) (1964, 23) (2001, 23) (1507, 23) (1625, 23) (1525, 23)
# --------------------------------