import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

np.random.seed(2) 

# 内側（詰まった円形クラスタ）
n_inner = 50
r_inner = np.sqrt(np.random.uniform(0, 0.5**2, n_inner))  # 半径をsqrtで生成
theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
x_inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)], axis=1)

# 外側（リング状クラスタ）
n_outer = 50
r_outer = np.sqrt(np.random.uniform(0.6**2, 0.8**2, n_outer))
theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
x_outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)], axis=1)

# データ統合とラベル付け
X = np.vstack([x_inner, x_outer])
y = np.hstack([-np.ones(n_outer), np.ones(n_outer)])

# スケーリング（0〜1）
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ラベルを -1 と 1 に変換
y = np.where(y == np.min(y), -1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)

"""# プロット
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Filled Inner Circle - Synthetic Data (0-1 scaled)")
plt.grid(True)
plt.show()

# SVMで分類
clf = SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X, y)
print("Accuracy:", clf.score(X, y))"""
