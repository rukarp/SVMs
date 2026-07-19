import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[50:, 2:]
Y = iris.target[50:]

# 正規化
X = MinMaxScaler().fit_transform(X)

# ラベルを -1 と 1 に変換
Y = np.where(Y == np.min(Y), -1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
# X_train.shape, Y_train.shape, X_test.shape, Y_test.shape -> (455, 30), (455,), (114, 30), (114,)

# ------ 各拠点に等しく分配* --------
X4_11_train, X_temp, Y4_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.75, stratify=Y_train, random_state=42)
X4_12_train, X_temp, Y4_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X4_13_train, X4_14_train, Y4_13_train, Y4_14_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X4_1_train = [X4_11_train, X4_12_train, X4_13_train, X4_14_train]
Y4_1_train = [Y4_11_train, Y4_12_train, Y4_13_train, Y4_14_train]

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

X4_51_train, Y4_51_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 2]))
X4_52_train, Y4_52_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 1]))
X4_53_train, Y4_53_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 0]))
X4_54_train, Y4_54_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 3]))

X4_5_train = [X4_51_train, X4_52_train, X4_53_train, X4_54_train]
Y4_5_train = [Y4_51_train, Y4_52_train, Y4_53_train, Y4_54_train]

# --------------------------------

# ------ 各拠点に等しく分配* --------
X6_11_train, X_temp, Y6_11_train, Y_temp = train_test_split(X_train, Y_train, test_size=0.8333, stratify=Y_train, random_state=42)
X6_12_train, X_temp, Y6_12_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.8, stratify=Y_temp, random_state=42)
X6_13_train, X_temp, Y6_13_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.75, stratify=Y_temp, random_state=42)
X6_14_train, X_temp, Y6_14_train, Y_temp = train_test_split(X_temp, Y_temp, test_size=0.6666, stratify=Y_temp, random_state=42)
X6_15_train, X6_16_train, Y6_15_train, Y6_16_train = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42)

X6_1_train = [X6_11_train, X6_12_train, X6_13_train, X6_14_train, X6_15_train, X6_16_train]
Y6_1_train = [Y6_11_train, Y6_12_train, Y6_13_train, Y6_14_train, Y6_15_train, Y6_16_train]

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

X6_51_train, Y6_51_train = np.concatenate((X_train_0[labels_0 == 1], X_train_1[labels_1 == 2])), np.concatenate((Y_train_0[labels_0 == 1], Y_train_1[labels_1 == 2]))
X6_52_train, Y6_52_train = np.concatenate((X_train_0[labels_0 == 4], X_train_1[labels_1 == 1])), np.concatenate((Y_train_0[labels_0 == 4], Y_train_1[labels_1 == 1]))
X6_53_train, Y6_53_train = np.concatenate((X_train_0[labels_0 == 3], X_train_1[labels_1 == 4])), np.concatenate((Y_train_0[labels_0 == 3], Y_train_1[labels_1 == 4]))
X6_54_train, Y6_54_train = np.concatenate((X_train_0[labels_0 == 5], X_train_1[labels_1 == 5])), np.concatenate((Y_train_0[labels_0 == 5], Y_train_1[labels_1 == 5]))
X6_55_train, Y6_55_train = np.concatenate((X_train_0[labels_0 == 2], X_train_1[labels_1 == 0])), np.concatenate((Y_train_0[labels_0 == 2], Y_train_1[labels_1 == 0]))
X6_56_train, Y6_56_train = np.concatenate((X_train_0[labels_0 == 0], X_train_1[labels_1 == 3])), np.concatenate((Y_train_0[labels_0 == 0], Y_train_1[labels_1 == 3]))

X6_5_train = [X6_51_train, X6_52_train, X6_53_train, X6_54_train, X6_55_train, X6_56_train]
Y6_5_train = [Y6_51_train, Y6_52_train, Y6_53_train, Y6_54_train, Y6_55_train, Y6_56_train]

# --------------------------------
