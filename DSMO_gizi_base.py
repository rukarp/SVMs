# SVMの関数をまとめた親クラス
from DSMO_base import BaseDSMO

# Numpy
import numpy as np

# MPI
from mpi4py import MPI
import socket

# 時間計測用
import time

# グラフのプロット用
import matplotlib.pyplot as plt

# 立体凸包計算用
from scipy.spatial import ConvexHull

# 直交行列生成用
from scipy.linalg import orth

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()


# 疑似データ用のクラス
class BaseDSMO_for_Gizi(BaseDSMO):   
    
    
    
    def make_gradient_f(self, X, y, alphas, w, ind_sv, ind_inner):
        """
        SVM モデルのパラメータを使って決定関数 f(x) を生成する関数を作成する
        
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        alphas (array-like): ラグランジュ乗数
        w (array-like): SVM モデルの重み
        ind_sv (array-like): サポートベクターのインデックス
        ind_inner (array-like): 内部インデックス

        戻り値:
        function: 入力 x を受け取り 勾配∇f(x) を計算する関数
        """
        
        # 線形カーネルの勾配
        def linear_kernel_grad(x):
            grad_f = w
            return grad_f
        
        # RBFカーネルの勾配
        def rbf_kernel_grad(x):
            grad_f = np.zeros_like(x, dtype=float)#np.zeros(len(w), dtype=float)
            for i in np.concatenate((ind_sv, ind_inner)):
                diff = X[i] - x
                K = self._kernel(X[i], x)
                grad_f += y[i] * alphas[i] * diff * K
            grad_f *= 2 * self.gamma
            return grad_f
        
        if self.kernel == 'linear':
            return linear_kernel_grad
        elif self.kernel == 'rbf':
            return rbf_kernel_grad
        
        # 線形とRBF以外は一旦wを返す
        else:
            def default_grad(x):
                return w
            return default_grad
    
    def make_decision_func_squared(self, X, y, alphas, w, b, ind_sv, ind_inner):
        """
        SVM モデルのパラメータを使って決定関数 f(x) を生成する関数を作成する
        
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        alphas (array-like): ラグランジュ乗数
        w (array-like): SVM モデルの重み
        b (float): SVM モデルのバイアス項
        ind_sv (array-like): サポートベクターのインデックス
        ind_inner (array-like): 内部インデックス

        戻り値:
        function: 入力 x を受け取り f(x)^2 を計算する関数
        """
        def f(x):
            if self.kernel == 'linear':
                f_value = np.dot(w, x) + b
            else:
                f_value = 0
                for i in np.concatenate((ind_sv, ind_inner)):
                    f_value += alphas[i] * y[i] * self._kernel(X[i], x)
                f_value += b
        
            return f_value**2 + 0.1
        
        return f
    
    
    def make_gradient_f_squared(self, f, grad_f):
        """
        (f(x))^2 の勾配関数 ∇(f(x))^2 を返す
    
        パラメータ:
        f: 識別関数 f(x)
        grad_f: f(x) の勾配関数 ∇f(x)
    
        戻り値:
        function: 入力 x に対して ∇(f(x))^2 = 2 * f(x) * ∇f(x) を返す関数
        """
        def grad_f_squared(x):
            return 2 * f(x) * grad_f(x)
    
        return grad_f_squared

    
    def move_toward_decision_boundary(self, f, grad_f_squared, x_init, lr, bounds_eps, max_iter):
        """
        (f(x))^2 の勾配を使って、x を (f(x))^2=0 の場所（決定境界）に近づける
        終了判定はf(x)を使う
        """
        # 目的関数値に対して許容する範囲
        lower_bound, upper_bound = 1 - bounds_eps, 1 + bounds_eps
        #lower_bound, upper_bound = 0.5 - bounds_eps, 0.5 + bounds_eps
        
        x = x_init.copy()
        # 初期データのラベルを把握
        f_val_old = f(x)
        f_sign_init = np.sign(f_val_old)
        
        # 指定エリアより内にデータがある場合，上り向きの勾配
        if abs(f_val_old) < abs(lower_bound):
            sign = 1
        # 指定エリア内or指定エリアより外にデータがある場合，下り向きの勾配
        else:
            sign = -1

        for i in range(max_iter):
            
            # 収束判定(指定エリア内にデータが入っていて初期データと異なるか)
            if abs(lower_bound) <= abs(f_val_old) <= abs(upper_bound) and not np.array_equal(x, x_init):
                break
            
            # 最大回数ループが回った場合エラー
            if i == max_iter:
                print(f"error: i = {max_iter-1}", flush=True)
                
            # f(x)^2の勾配を計算し，xを更新
            grad = grad_f_squared(x)
            x_new = x + sign * lr * grad

            f_val_new = f(x_new)
            f_sign_new = np.sign(f_val_new)
            
            # 通り過ぎを検知しながらlrを更新
            if sign == -1:
                if abs(f_val_new) < abs(lower_bound):
                    lr /= 2
                    continue
                elif f_sign_new != f_sign_init:
                    lr /= 2
                    continue
            elif sign == 1:
                if abs(f_val_new) > abs(upper_bound):
                    lr /= 2
                    continue
            if abs(f_val_old - f_val_new) < upper_bound / 100:
                lr *= 2
                
            # 現在のxを更新
            x = x_new
            f_val_old = f_val_new


        if f_sign_init >= 0:
            label = self.max_val
        else:
            label = self.min_val
            
        return x, label
    
    
    
    
    def move_toward_decision_boundary_new(self, f, grad_f, x_init, lr, bounds_eps, max_iter):
        """
        (f(x) - sign(f(x))^2 の勾配を使って、x を (f(x))^2=+-1 の場所（決定境界）に近づける
        終了判定はf(x)を使う
        """
        
        x = x_init.copy()
        # 初期データのラベルを把握
        f_val_old = f(x)
        f_sign_init = np.sign(f_val_old)
        
        s = 1 if f_sign_init >= 0 else -1
        
        f_val = f_val_old
        t = (f_val - s)**2
        
        for i in range(max_iter):
                                    
            # 収束判定(指定エリア内にデータが入っていて初期データと異なるか)
            if 0 <= abs(f_val - s) <= bounds_eps and not np.array_equal(x, x_init):
                print(s, f_val)
                break
            
            # 最大回数ループが回った場合エラー
            if i == max_iter:
                print(f"error: i = {max_iter-1}", flush=True)
                break
                
            # (f(x) - sign(f(x))^2の勾配を計算し，xを更新
            grad = 2 * (f_val - s) * grad_f(x)
                    
            # 一旦更新候補を計算
            x_new = x - lr * grad
            f_val_new = f(x_new)
            t_new = (f_val_new - s)**2

            # lr の調整 
            # 底を通過した場合
            if t_new > t:
                lr *= 0.5
                # x は更新しない（戻る）
                continue

            # 変化量が小さすぎる場合
            if abs(t - t_new) < bounds_eps:
                lr *= 2.0

            # 更新を確定
            x = x_new
            f_val = f_val_new
            t = t_new
        
        if f_sign_init >= 0:
            label = self.max_val
        else:
            label = self.min_val
                    
        return x, label
    
    
    
    
    
    
    
    def projection_onto_1_all(self, w, b, X):
        """
        linearカーネル時に限り、wに直交する平面に X の各サンプルを射影する
        Xの全データについて行う 
        """
        # f(x) = Xw + b
        f_val = X @ w + b                     # shape: (n_samples,)
        f_sign = np.sign(f_val)               # shape: (n_samples,)
        #f_sign *= 0.05

        # 正規化項
        w_norm_sq = np.dot(w, w)              # scalar

        # 射影ベクトルの補正項
        correction = ((f_sign - f_val) / w_norm_sq)[:, np.newaxis] * w  # shape: (n_samples, n_features)

        # 射影後の全データ
        X_proj = X + correction

        # ラベルをmax_val/min_valで設定
        labels = np.where(f_sign >= 0, self.max_val, self.min_val)

        return X_proj, labels


    def select_independent_rows(self, X, y, min_dist=1e-2):
        """
        X: (n, d) ndarray
        
        return: 線形独立な行だけを抽出した ndarray
        """
        """selected_X = np.empty((0, X.shape[1]), dtype=float)
        selected_y = np.empty((0,), dtype=int)

        selected_X_rank = 0
        
        for i in range(X.shape[0]):
            candidate = X[i].reshape(1, -1)
            tmp = np.vstack([selected_X, candidate])
            tmp_rank = np.linalg.matrix_rank(tmp)
            if tmp_rank > selected_X_rank:
                # 距離チェック
                if selected_X.shape[0] == 0 or np.all(np.linalg.norm(selected_X - candidate, axis=1) >= min_dist):
                    selected_X = tmp
                    selected_y = np.append(selected_y, y[i])
                    selected_X_rank = tmp_rank"""

        X_0 = X[0]
        y_0 = y[0]
        
        X_diff = X - X_0
        X_diff_rank = np.linalg.matrix_rank(X_diff)
        
        selected_X = np.array([X_0])  # 基準点はまず追加
        selected_y = np.array([y_0])
        
        current_diff = np.empty((0, X.shape[1]))  # 現在の差分ベクトル集合
        current_diff_rank = 0
        
        for i in range(1, X.shape[0]):
            candidate = X_diff[i].reshape(1, -1)
            tmp = np.vstack([current_diff, candidate])

            #print(selected_X_rank, tmp_rank)
            
            # 距離チェック
            if np.all(np.linalg.norm(selected_X - candidate, axis=1) >= min_dist):
                # 線形独立性チェック
                tmp_rank = np.linalg.matrix_rank(tmp)
                if tmp_rank > current_diff_rank:
                    current_diff = tmp
                    selected_X = np.vstack([selected_X, X[i]])
                    selected_y = np.append(selected_y, y[i])

                    current_diff_rank = tmp_rank

                    #print(f"Added idx {i}, rank={selected_X_rank}")
            if len(selected_X) - 1 >= X_diff_rank:
                break  # 目標ランクに達したら終了

        return selected_X, selected_y


    def pick_affine_independent_points(self, X, y, min_dist=1e-2):
        """
        X: (n_samples, n_features) 数値データ（ワンホット後など）
        labels: ラベル（self.max_val / self.min_val）

        条件:
            - self.max_val, self.min_val の両クラスから1点以上
            - 合計でアフィン次元 + 1 個の独立点
        """
        
        # 独立なデータを選択
        selected_X, selected_y = self.select_independent_rows(X, y, min_dist=min_dist)
        
        # min_val が含まれていない場合のデータ追加
        if len(selected_X[selected_y == self.max_val]) == 0:
            for i in range(X.shape[0]):
                if y[i] == self.max_val and np.all(np.linalg.norm(selected_X - X[i], axis=1) >= min_dist):
                    if any(np.all(X[i] == x) for x in selected_X):
                        selected_X = np.vstack([selected_X, X[i].reshape(1, -1)])
                        selected_y = np.append(selected_y, y[i])
                        break  # 1点追加したら終了

        # min_val が含まれていない場合も同様
        
        elif len(selected_X[selected_y == self.min_val]) == 0:
            for i in range(X.shape[0]):
                if y[i] == self.min_val and np.all(np.linalg.norm(selected_X - X[i], axis=1) >= min_dist):
                    if any(np.all(X[i] == x) for x in selected_X):
                        selected_X = np.vstack([selected_X, X[i].reshape(1, -1)])
                        selected_y = np.append(selected_y, y[i])
                        break  # 1点追加したら終了
        
        # どちらもある場合のデータの追加
        else:
            for i in range(X.shape[0]):
                if any(np.all(X[i] == x) for x in selected_X):
                    selected_X = np.vstack([selected_X, X[i].reshape(1, -1)])
                    selected_y = np.append(selected_y, y[i])
                    break  # 1点追加したら終了

                    
        #print("selected_X shape:", selected_X.shape)           

        return selected_X, selected_y
    
    def pick_affine_independent_points1(self, X, y, min_dist=1e-2):
        """
        X: (n_samples, n_features) 数値データ（ワンホット後など）
        labels: ラベル（self.max_val / self.min_val）

        条件:
            - self.max_val, self.min_val の各クラスで
              基底を作り，結合する
        """

        X_up, y_up = X[y == self.max_val], y[y == self.max_val]
        X_down, y_down = X[y == self.min_val], y[y == self.min_val]

        selected_X_up, selected_y_up = self.select_independent_rows(X_up, y_up, min_dist=min_dist)
        selected_X_down, selected_y_down = self.select_independent_rows(X_down, y_down, min_dist=min_dist)

        selected_X = np.vstack([selected_X_up, selected_X_down])
        selected_y = np.concatenate([selected_y_up, selected_y_down])
        #print("selected_X shape:", selected_X.shape)           

        return selected_X, selected_y
    
    def get_convexHull_vertices(self, X, y):
        
        X_up = X[y == self.max_val]
        up_idx = np.where(y == self.max_val)[0]
        vertices_up = ConvexHull(X_up, qhull_options='QJ').vertices
        X_vertices_up = X[up_idx[vertices_up]]
        y_vertices_up = y[up_idx[vertices_up]]

        X_down = X[y == self.min_val]
        down_idx = np.where(y == self.min_val)[0]
        vertices_down = ConvexHull(X_down, qhull_options='QJ').vertices
        X_vertices_down = X[down_idx[vertices_down]]
        y_vertices_down = y[down_idx[vertices_down]]

        X_vertices = np.vstack([X_vertices_up, X_vertices_down])
        y_vertices = np.concatenate([y_vertices_up, y_vertices_down])
        
        return X_vertices, y_vertices

    def get_convexHull_vertices1(self, X, y, tol=1e-10):
        def hull_vertices(X_sub):
            """
            X_sub: 部分データ
            return: 元のインデックスに対応する頂点インデックス
            """
            n, d = X_sub.shape
            # 1. 中心化
            X_centered = X_sub - X_sub.mean(axis=0)

            # 2. SVDで有効次元を判定
            U, S, Vt = np.linalg.svd(X_centered)
            rank = np.sum(S > tol)

            if rank == 0:
                # 全て同一点
                return np.array([0])
            elif rank == 1:
                # 直線上 -> 両端の点を頂点
                X1D = X_centered @ Vt.T[:, :1]
                i_min = np.argmin(X1D[:, 0])
                i_max = np.argmax(X1D[:, 0])
                return np.array([i_min, i_max])
            else:
                # 低次元に射影
                X_reduced = X_centered @ Vt.T[:, :rank]
                try:
                    hull = ConvexHull(X_reduced)
                    return hull.vertices
                except:
                    # QhullErrorのときはQJで再試行
                    hull = ConvexHull(X_reduced, qhull_options='QJ')
                    return hull.vertices

        # -----------------------
        # max_val の凸包頂点
        X_up = X[y == self.max_val]
        up_idx = np.where(y == self.max_val)[0]
        vertices_up = hull_vertices(X_up)
        X_vertices_up = X_up[vertices_up]
        y_vertices_up = y[up_idx[vertices_up]]

        # min_val の凸包頂点
        X_down = X[y == self.min_val]
        down_idx = np.where(y == self.min_val)[0]
        vertices_down = hull_vertices(X_down)
        X_vertices_down = X_down[vertices_down]
        y_vertices_down = y[down_idx[vertices_down]]

        # 結合
        X_vertices = np.vstack([X_vertices_up, X_vertices_down])
        y_vertices = np.concatenate([y_vertices_up, y_vertices_down])

        return X_vertices, y_vertices
    
    def get_convexHull_vertices2(self, X, y, tol=1e-10, max_rank=10):
        """
        多次元データ向け高速版。
        max_rank: 射影先の次元数上限（高次元凸包は重いので制限）
        """

        # 1. 中心化して SVD（全データ一度だけ）
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        rank = np.sum(S > tol)
        rank_use = min(rank, max_rank)
        X_reduced = X_centered @ Vt.T[:, :rank_use]

        def hull_vertices(X_sub):
            n, d = X_sub.shape
            # rank1なら端点のみ
            if d == 1:
                i_min = np.argmin(X_sub[:, 0])
                i_max = np.argmax(X_sub[:, 0])
                return np.array([i_min, i_max])
            else:
                try:
                    hull = ConvexHull(X_sub)
                    return hull.vertices
                except:
                    hull = ConvexHull(X_sub, qhull_options='QJ')
                    return hull.vertices

        # max_val
        mask_up = y == self.max_val
        X_up = X_reduced[mask_up]
        idx_up = np.where(mask_up)[0]
        vertices_up = hull_vertices(X_up)
        X_vertices_up = X[idx_up[vertices_up]]
        y_vertices_up = y[idx_up[vertices_up]]

        # min_val
        mask_down = y == self.min_val
        X_down = X_reduced[mask_down]
        idx_down = np.where(mask_down)[0]
        vertices_down = hull_vertices(X_down)
        X_vertices_down = X[idx_down[vertices_down]]
        y_vertices_down = y[idx_down[vertices_down]]

        # 結合
        X_vertices = np.vstack([X_vertices_up, X_vertices_down])
        y_vertices = np.concatenate([y_vertices_up, y_vertices_down])

        return X_vertices, y_vertices

    
    def pick_far_and_independent_per_crass(self, X, y, tol=1e-10):
        """
        X: (n, d) ndarray
        y: (n,) ndarray または None
        tol: 線形独立判定用の閾値
        return: (selected_X, selected_y)
                selected_y は y が None の場合 None
        """
        n = X.shape[0]
        idx0 = np.random.randint(0, n)
        selected_X = np.array([X[idx0]])
        selected_y = np.array([y[idx0]]) if y is not None else None

        while True:
            # 各点と既存選択点の最小距離
            dist = np.min(np.linalg.norm(X[:, None, :] - selected_X[None, :, :], axis=2), axis=1)
            # 既に選んだ点は距離0なので除外
            for s in selected_X:
                mask = ~np.all(X == s, axis=1)
                X = X[mask]
                dist = dist[mask]
                if y is not None:
                    y = y[mask]
            if len(X) == 0:
                break

            # 最も遠い点を候補に
            next_idx = np.argmax(dist)
            next_point = X[next_idx]
            stacked = np.vstack([selected_X, next_point])
            rank = np.linalg.matrix_rank(stacked, tol=tol)

            # ランクが増える（＝独立）なら追加
            if rank > len(selected_X):
                selected_X = np.vstack([selected_X, next_point])
                if y is not None:
                    selected_y = np.append(selected_y, y[next_idx])
            else:
                # どの点を追加してもランクが増えないなら終了
                if np.allclose(rank, len(selected_X)):
                    break

        return selected_X, selected_y
    
    def pick_independent_and_far_points(self, X, y, tol=1e-10):
        
        X = np.array(X)
        n, d = X.shape
        rank_target = np.linalg.matrix_rank(X, tol=tol)

        # 1点目：原点または平均から最も遠い点を選ぶ
        mean = np.mean(X, axis=0)
        first_idx = np.argmax(np.linalg.norm(X - mean, axis=1))
        selected_X = np.array([X[first_idx]])
        selected_y = np.array([y[first_idx]])

        while len(selected_X) < rank_target:
            max_dist = -np.inf
            best_idx = None

            for i in range(n):
                # 既に選ばれていたらスキップ
                if np.any(np.all(X[i] == selected_X, axis=1)):
                    continue

                stacked = np.vstack([selected_X, X[i]])
                rank_before = np.linalg.matrix_rank(selected_X, tol=tol)
                rank_after = np.linalg.matrix_rank(stacked, tol=tol)

                # 線形独立か確認
                if rank_after > rank_before:
                    # 距離（既存点との平均距離）
                    dist = np.mean(np.linalg.norm(X[i] - selected_X, axis=1))
                    if dist > max_dist:
                        max_dist = dist
                        best_idx = i

            if best_idx is None:
                # これ以上独立な点がない
                break

            selected_X = np.vstack([selected_X, X[best_idx]])
            if y is not None:
                selected_y = np.append(selected_y, y[best_idx])

        return selected_X, selected_y
    
    def pick_far_and_independent(self, X, y, tol=1e-10):
        
        X_up = X[y == self.max_val]
        y_up = y[y == self.max_val]
        X_down = X[y == self.min_val]
        y_down = y[y == self.min_val]
        
        #selected_X_up, selected_y_up = self.pick_far_and_independent_per_crass(X_up, y_up, tol=tol)
        #selected_X_down, selected_y_down = self.pick_far_and_independent_per_crass(X_down, y_down, tol=tol)

        selected_X_up, selected_y_up = self.pick_independent_and_far_points(X_up, y_up, tol=tol)
        selected_X_down, selected_y_down = self.pick_independent_and_far_points(X_down, y_down, tol=tol)

        # 結合
        selected_X = np.vstack([selected_X_up, selected_X_down])
        selected_y = np.concatenate([selected_y_up, selected_y_down])

        return selected_X, selected_y
    
    def make_fake_data(self, X, ind_sv, lr, max_iter):
        """
        疑似データを生成する（マージン境界上への射影）
        Args:
            X (_type_): _description_
            lr (_type_): _description_
            max_iter (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # 線形カーネルの場合はwに直交する平面に射影する
        if self.kernel == 'linear':
            data, labels = self.projection_onto_1_all(self.w, self.b, X)

        # 非線形カーネルの場合は最小二乗法で求める
        else:
            bounds_eps = 10e-6
            data = np.empty((0, X.shape[1]))
            labels = np.empty(0)
        
            for i in range(X.shape[0]):
                x_init = X[i]

                x, label = self.move_toward_decision_boundary(self.f, self.grad_f_squared, x_init, lr, bounds_eps, max_iter)
                #x, label = self.move_toward_decision_boundary_new(self.f, self.grad_f, x_init, lr, bounds_eps, max_iter)
                
                data = np.vstack((data, x.reshape(1, -1)))
                labels = np.append(labels, label)

                if i % 100 == 0:
                   print(f"{i}")
                
        # 元データからサポートベクターを除去
        data = np.delete(data, ind_sv, axis=0)
        labels = np.delete(labels, ind_sv)

        return data, labels
    
    def make_fake_data_random(self, X, y, radius, max_retry):
        """
        ノイズを加えるだけの関数

        Args:
            X (_type_): _description_
            y (_type_): _description_
            radius (_type_): _description_
            max_retry (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        X_new = np.empty_like(X)
        X_delta = np.empty_like(X)
        dim = X.shape[1]
        flag = np.full(len(X), False, dtype=bool)
    
        while not np.all(flag):
            
            # False の点だけ再生成
            for i in range(X.shape[0]):
                for retry in range(max_retry):
                    X_delta[i] = self.generate_directional_noise(dim, radius, seed=i * 100000 + retry)
                    X_new[i] = X[i] + X_delta[i]
                    
                    cond_p_area = np.all((X_new[i] >= 0) & (X_new[i] <= 1))
                    
                    #if not cond_p_area:
                    #    continue
                    
                    flag[i] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[i]:
                    print(f"Retry failed for index {i}.")
            
            return X_new, y.copy()
    
    
    def make_fake_data_random_with_margin(self, X, y, alphas, radius, max_retry):
        """
        マージンの条件を満たしながらノイズを加える関数

        Args:
            X (_type_): _description_
            y (_type_): _description_
            alphas (_type_): _description_
            radius (_type_): _description_
            max_retry (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind_sv, ind_inner = self._get_SV_ind(alphas)
        ind_other = np.setdiff1d(np.arange(len(alphas)), np.concatenate([ind_sv, ind_inner]))
        
        X_new = np.empty_like(X)
        X_delta = np.empty_like(X)
        dim = X.shape[1]
        flag = np.full(len(X), False, dtype=bool)
    
        # マージンの条件ごとの各種設定
        configs = [
            {
                "ind": ind_other,
                "noise_func": self.generate_directional_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: y[i] * self.f(X_new[i]) > 1,
            },
            {
                "ind": ind_sv,
                "noise_func": self.generate_tangent_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: abs(self.f(X[i]) - self.f(X_new[i])) < self.ME,
            },
            {
                "ind": ind_inner,
                "noise_func": self.generate_directional_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: y[i] * self.f(X_new[i]) < 1,
            },
        ]
        
        while not np.all(flag):
            
            # False の点だけ再生成
            for cfg in configs:
               for i in cfg["ind"][~flag[cfg["ind"]]]:
                for retry in range(max_retry):
                    X_delta[i] = cfg["noise_func"](dim, radius, seed=i * 100000 + retry)
                    X_new[i] = X[i] + X_delta[i]
                    
                    cond_p_area = cfg["area_cond"](i)
                    cond_p_margin = cfg["margin_cond"](i)
                    
                    #if cond_p_area and cond_p_margin:
                    if cond_p_margin:
                        flag[i] = True
                        break
                
                # 失敗時のメッセージ
                if not flag[i]:
                    print(f"Retry failed for index {i}.")
            
            return X_new, y.copy()

    def make_fake_data_KKT(self, X, y, alphas, radius, max_retry):
        """
        KKT条件から動かす条件を決めてノイズを加える方法
        （linearカーネル限定）
        Args:
            X (_type_): _description_
            y (_type_): _description_
            alphas (_type_): _description_
            radius (_type_): _description_
            max_retry (_type_): _description_

        Returns:
            _type_: _description_
        """

        ind_sv, ind_inner = self._get_SV_ind(alphas)
        ind_other = np.setdiff1d(np.arange(len(alphas)), np.concatenate([ind_sv, ind_inner]))
        
        #print(f"Other Points: {ind_other}")
        #print(f"Support Vectors: {ind_sv}")
        #print(f"Inner Points: {ind_inner}")
        
        X_delta = np.zeros_like(X)
        X_new = np.zeros_like(X)
        
        # X_deltaの補正に用いるΣα_i^2の計算
        denom = np.dot(alphas, alphas)
        # X_deltaの補正に用いるΣy_iα_iの計算
        coef = y * alphas
        
        # Xの次元数を取得
        dim = X.shape[1]
        
        # すべての点のフラグを初期化
        flag = np.full(len(X), False, dtype=bool)

        # マージンの条件ごとの各種設定
        configs = [
            {
                "ind": ind_other,
                "noise_func": self.generate_directional_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: y[i] * self.f(X_new[i]) > 1,
            },
            {
                "ind": ind_sv,
                "noise_func": self.generate_tangent_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: abs(self.f(X[i]) - self.f(X_new[i])) < self.ME,
            },
            {
                "ind": ind_inner,
                "noise_func": self.generate_directional_noise,
                "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                "margin_cond": lambda i: y[i] * self.f(X_new[i]) < 1,
            },
        ]
            
        iter = 0
        
        while not np.all(flag):
            
            # False の点だけ再生成
            for cfg in configs:
               for i in cfg["ind"][~flag[cfg["ind"]]]:
                for retry in range(max_retry):
                    X_delta[i] = cfg["noise_func"](dim, radius, seed=i * 100000 + iter * 1000 + retry)
                    X_new[i] = X[i] + X_delta[i]
                    
                    cond_p_area = cfg["area_cond"](i)
                    cond_p_margin = cfg["margin_cond"](i)
                    
                    #if cond_p_area and cond_p_margin:
                    if cond_p_margin:
                        flag[i] = True
                        break
                
                # 失敗時のメッセージ
                if not flag[i]:
                    print(f"Retry failed for index {i}.")
            
            # 全体補正            
            w_delta = X_delta.T @ coef
            X_delta -= np.outer(coef, w_delta) / denom
            
            # 補正後に再チェック
            X_new = X + X_delta
            
            #cond_margin_other = np.array([y[i] * self.f(X_new[i]) > 1 for i in ind_other])
            cond_margin_sv = np.array([abs(self.f(X[i]) - self.f(X_new[i])) < self.ME for i in ind_sv])
            cond_margin_inner = np.array([y[i] * self.f(X_new[i]) < 1 for i in ind_inner])

            # マージンの条件をflagに反映
            #flag[ind_other] = cond_margin_other
            flag[ind_sv] = cond_margin_sv
            flag[ind_inner] = cond_margin_inner
            
            # エリアの違反があればFlaseにする．
            #cond_area = np.all((X_new >= 0) & (X_new <= 1), axis=1)
            #flag &= cond_area
        
            iter += 1
            
        return X_new, y.copy()
    



    
    
    
    
    
    
    
    
    def generate_directional_noise(self, dim, radius, seed=None):
        """
        指定された次元数と半径で、ランダムな方向のノイズを生成する関数
        """
        rng = np.random.default_rng(seed)

        # ランダムな方向の長さ1のベクトルを作る
        noise = rng.normal(size=dim)
        noise /= np.linalg.norm(noise)
        
        # 長さを [radius_min, radius_max] からランダムに決定
        r_min, r_max = radius
        length = rng.uniform(r_min, r_max)

        return length * noise
    
    def generate_tangent_noise(self, dim, radius, seed=None):
        
        rng = np.random.default_rng(seed)

        # ランダムベクトル
        v = rng.normal(size=dim)

        # w方向成分を除去して長さ1のベクトルを作る
        noise = v - (np.dot(v, self.w) / np.dot(self.w, self.w)) * self.w
        noise /= np.linalg.norm(noise)

        # 長さを [radius_min, radius_max] からランダムに決定
        r_min, r_max = radius
        length = rng.uniform(r_min, r_max)

        return length * noise