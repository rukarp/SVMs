# SVMの関数をまとめた親クラス
from svm_base import BaseSVM as MySVM

# Numpy
import numpy as np

# 時間計測用
import time

# グラフのプロット用
import matplotlib.pyplot as plt

# 立体凸包計算用
from scipy.spatial import ConvexHull

# 直交行列生成用
from scipy.linalg import orth

# 凸最適化用
import cvxpy as cp



# 疑似データ用のクラス
class BaseSVM_for_Gizi(MySVM):
    
    def fit(self, X, y):
        
        # 親クラスのfit関数を呼び出す
        super().fit(X, y)
        
        # -----識別関数の2乗の勾配(∇(f(x))^2)を求める -----
        self.grad_f = self.make_gradient_f(self.X, self.y, self.alphas, self.w, self.ind_sv, self.ind_inner)

        self.f_squared = self.make_decision_func_squared(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)
        self.grad_f_squared = self.make_gradient_f_squared(self.f, self.grad_f)
        # ----------------------------------------------
    
    
    
    
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

    
    """def move_toward_decision_boundary(self, f, grad_f_squared, x_init, lr, bounds_eps, max_iter):
        
        (f(x))^2 の勾配を使って、x を (f(x))^2=0 の場所（決定境界）に近づける
        終了判定はf(x)を使う
        
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
            
        return x, label"""
    
    
    
    
    def move_toward_decision_boundary(self, f, grad_f, x_init, f_val_proj, lr, bounds_eps, max_iter):
        """
        (f(x) - sign(f(x))^2 の勾配を使って、x を (f(x))^2=+-1 の場所（決定境界）に近づける
        終了判定はf(x)を使う
        f(x) = f_val_proj にするように近づける
        """
        
        # 初期データのラベルを把握
        x = x_init.copy()
        f_val_old = f(x)
                
        f_val = f_val_old
        t = (f_val - f_val_proj)**2
        
        for i in range(max_iter):
                                    
            # 収束判定(指定エリア内にデータが入っていて初期データと異なるか)
            if 0 <= abs(f_val - f_val_proj) <= bounds_eps and not np.array_equal(x, x_init):
                #print(s, f_val)
                break
            
            # 最大回数ループが回った場合エラー
            if i == max_iter:
                print(f"error: i = {max_iter-1}", flush=True)
                break
                
            # (f(x) - sign(f(x))^2の勾配を計算し，xを更新
            grad = 2 * (f_val - f_val_proj) * grad_f(x)
                    
            # 一旦更新候補を計算
            x_new = x - lr * grad
            f_val_new = f(x_new)
            t_new = (f_val_new - f_val_proj)**2

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
                    
        return x
    
    
    
    def projection_onto_1(self, w, b, x, f_val_proj):
        """
        linearカーネル時に限り，
        wに直交する平面へ1点xを射影する

        f(x) = f_val_proj となるように射影する
        """

        # 現在のf(x)
        f_val = x @ w + b

        # ||w||^2
        w_norm_sq = np.dot(w, w)

        # 射影補正
        correction = ((f_val_proj - f_val) / w_norm_sq) * w

        # 射影後
        x_proj = x + correction

        return x_proj
    
    
    
    def projection_onto_1_all(self, w, b, X, f_val_proj):
        """
        linearカーネル時に限り、wに直交する平面に X の各サンプルを射影する
        Xの全データについて行う 
        f(x) = f_val_proj にするように射影する
        """
        f_val = X @ w + b                     # shape: (n_samples,)

        # 正規化項
        w_norm_sq = np.dot(w, w)              # scalar

        # 射影ベクトルの補正項
        correction = ((f_val_proj - f_val) / w_norm_sq)[:, np.newaxis] * w  # shape: (n_samples, n_features)

        # 射影後の全データ
        X_proj = X + correction

        """# ラベルをmax_val/min_valで設定
        labels = np.where(f_val_proj >= 0, self.max_val, self.min_val)"""

        return X_proj



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
    


    def make_fake_data(self, X, ind_sv, lr, bounds_eps, max_iter):
        """
        疑似データを生成する（マージン境界上への射影）
        Args:
            X (_type_): _description_
            ind_sv (_type_): _description_
            lr (_type_): _description_
            bounds_eps (_type_): _description_
            max_iter (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # 線形カーネルの場合はwに直交する平面に射影する
        if self.kernel == 'linear':
            f_val = X @ self.w + self.b
            f_sign = np.where(f_val >= 0, 1, -1)            
            f_val_proj = f_sign
            
            data = self.projection_onto_1_all(self.w, self.b, X, f_val_proj)

        # 非線形カーネルの場合は最小二乗法で求める
        else:
            data = np.empty((0, X.shape[1]))
            
            f_val = np.array([self.f(x) for x in X])
            f_sign = np.where(f_val >= 0, 1, -1)
            f_val_proj = f_sign
        
            for i in range(X.shape[0]):                
                x = self.move_toward_decision_boundary(self.f, self.grad_f, X[i], f_val_proj[i], lr, bounds_eps, max_iter)                                
                data = np.vstack((data, x.reshape(1, -1)))

                if i % 100 == 0:
                   print(f"{i}")
                
        # 元データからサポートベクターを除去
        data = np.delete(data, ind_sv, axis=0)
        # サポートベクターを除去しながらラベルを設定
        labels = np.delete(np.where(f_sign >= 0, self.max_val, self.min_val), ind_sv)

        return data, labels
    

    def make_fake_data_shift(self, X, y, radius, max_retry, lr, bounds_eps, max_iter):
        """
        疑似データを生成する（ランダムな向きに移動後，元の等高線上に斜影）
        Args:
            X (_type_): _description_
            y (_type_): _description_
            radius (_type_): _description_
            max_retry (_type_): _description_
            lr (_type_): _description_
            bounds_eps (_type_): _description_
            max_iter (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        data = np.empty((0, X.shape[1]))
        
        dim = X.shape[1]
        
        # 線形カーネルの場合の斜影先の計算
        if self.kernel == 'linear':            
            f_val_proj = X @ self.w + self.b
            
        # 非線形カーネルの場合の斜影先の計算
        else:
            f_val_proj = np.array([self.f(x) for x in X])
            
        for i in range(X.shape[0]):   
                
            success = False  
                
            for retry in range(max_retry):
                
                # ランダム初期移動を毎回生成  
                x_delta = self.generate_directional_noise(dim, radius, seed=i * 100 + retry)  
                x_init = X[i] + x_delta
                
                # 目的関数値に向けて斜影 
                if self.kernel == 'linear':   
                    x = self.projection_onto_1(self.w, self.b, x_init, f_val_proj[i])
                else:
                    x = self.move_toward_decision_boundary(self.f, self.grad_f, x_init, f_val_proj[i], lr, bounds_eps, max_iter)
                    
                # [0,1] 判定
                if np.all((x >= 0) & (x <= 1)):
                    success = True
                    break  
                
            # 失敗時は元データを使う
            if not success:
                print(f"Retry failed for index {i}, using original data point.")
                #x = X[i].copy()      
                                          
            data = np.vstack((data, x.reshape(1, -1)))

            if i % 100 == 0:
                print(f"{i}")

        return data, y.copy()
    
    
    
    
    
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
                    
                    #if cond_p_area:
                    if True:
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
            
            X_new = np.zeros_like(X)
            X_delta = np.zeros_like(X)
            
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
                    "name": "other",
                    "ind": ind_other,
                    "noise_func": self.generate_directional_noise,
                    "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                    "margin_cond": lambda i: y[i] * self.f(X_new[i]) > 1,# + self.ME,
                },
                {
                    "name": "support_vector",
                    "ind": ind_sv,
                    "noise_func": self.generate_tangent_noise,
                    "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                    "margin_cond": lambda i: abs(self.f(X[i]) - self.f(X_new[i])) < self.ME,
                },
                {
                    "name": "inner",
                    "ind": ind_inner,
                    "noise_func": self.generate_directional_noise,
                    "area_cond": lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                    "margin_cond": lambda i: y[i] * self.f(X_new[i]) < 1,# - self.ME,
                },
            ]
                
            iter = 0
            
            while not np.all(flag):
                
                #print(f"Iteration {iter}: radius: ({radius[0]}, {radius[1]}), {np.sum(flag)} / {len(X)} points satisfied the conditions.", flush=True)
                #print(f"        False deta -> other: {np.sum(~flag[ind_other])}, sv: {np.sum(~flag[ind_sv])}, inner: {np.sum(~flag[ind_inner])}", flush=True)
                
                # (1) 全点について候補ノイズを生成
                for cfg in configs:
                    for i in cfg["ind"][~flag[cfg["ind"]]]:
                        for retry in range(max_retry):
                            X_delta[i] = cfg["noise_func"](dim, radius, seed=i * 100000 + iter * 1000 + retry)
                            X_new[i] = X[i] + X_delta[i]
                            
                            cond_i_area = cfg["area_cond"](i)
                            cond_i_margin = cfg["margin_cond"](i)
                            
                            #if cond_i_area and cond_i_margin:
                            if cond_i_margin:
                                flag[i] = True
                                break
                    
                        # 失敗時のメッセージ
                        if not flag[i]:
                            print(f"Retry failed for index {i} in '{cfg['name']}' category.", flush=True)
                
                # (2) 候補ノイズ全体に補正           
                w_delta = X_delta.T @ coef
                X_delta -= np.outer(coef, w_delta) / denom
                
                # 補正後に再チェック
                X_new = X + X_delta
                
                # (3) 補正後の条件をチェック
                cond_margin_other = np.array([y[i] * self.f(X_new[i]) > 1 for i in ind_other], dtype=bool)
                cond_margin_sv = np.array([abs(self.f(X[i]) - self.f(X_new[i])) < self.ME for i in ind_sv], dtype=bool)
                cond_margin_inner = np.array([y[i] * self.f(X_new[i]) < 1 for i in ind_inner], dtype=bool)
                
                flag[ind_other] = cond_margin_other
                flag[ind_sv] = cond_margin_sv
                flag[ind_inner] = cond_margin_inner
                
                # エリアの違反があればFlaseにする．
                #cond_area = np.all((X_new >= 0) & (X_new <= 1), axis=1)
                #flag &= cond_area
    
                iter += 1
                            
            # 元データからの平均移動距離
            d_move = np.mean(np.linalg.norm(X_new - X, axis=1))
                
            return X_new, y.copy(), d_move
    



    def make_fake_data_KKT_QP(self, X, y, alphas, radius, max_retry, seed = 42):
        """
        KKT条件から動かす条件を決めてノイズを加える方法
        （linearカーネル限定）
        wに平行な成分のノイズを決めてからwに垂直な成分のノイズを加える
        Args:
            X (_type_): _description_
            y (_type_): _description_
            alphas (_type_): _description_
            radius (_type_): _description_
            max_retry (_type_): _description_
            seed (_type_, optional): _description_. Defaults to 42.

        Returns:
            _type_: _description_
        """

        ind_sv, ind_inner = self._get_SV_ind(alphas)
        ind_other = np.setdiff1d(np.arange(len(alphas)), np.concatenate([ind_sv, ind_inner]))

        ind_corr = np.concatenate([ind_sv, ind_inner])

        #if len(ind_inner) > 0:
        #    print(f"innerのalpha平均: {np.mean(alphas[ind_inner]):.15f}")
        
        X_delta = np.zeros_like(X)
        X_delta_parallel = np.zeros_like(X)
        X_delta_perp = np.zeros_like(X)
    
        # Xの次元数を取得
        N, dim = X.shape
        
        # すべての点のフラグを初期化
        flag = np.full(N, False, dtype=bool)
        
        # シード値を固定して再現性を確保
        np.random.seed(seed)
        
        # randomな半径を生成
        r = np.random.uniform(radius[0], radius[1], size=N)
        
        """#補正後の条件をチェック
        X_new = X + X_delta + X_delta_parallel + X_delta_perp
        flag[ind_other] = np.array([y[i] * self.f(X_new[i]) > 1 for i in ind_other], dtype=bool)
        flag[ind_sv] = np.array([abs(self.f(X[i]) - self.f(X_new[i])) < self.ME for i in ind_sv], dtype=bool)
        flag[ind_inner] = np.array([y[i] * self.f(X_new[i]) < 1 for i in ind_inner], dtype=bool)
        
        # 一つでも条件を満たさない点があれば警告
        if not np.all(flag):
            print(f"ORIGINAL Warning: Not all points satisfy the conditions after noise addition.", flush=True)
            print(f"        False deta -> other: {np.sum(~flag[ind_other])}, sv: {np.sum(~flag[ind_sv])}, inner: {np.sum(~flag[ind_inner])}", flush=True)
            
            for i in ind_inner:
                f_val = self.f(X_new[i])
                if y[i] * f_val >= 1:
                    print(f"i={i}, f={f_val:.15f}, y*f={y[i] * f_val:.15f}, d={y[i] * (1.0 - y[i] * self.f(X_new[i])) / np.linalg.norm(self.w):.15f}")"""

        # otherのノイズを作成 ----------------------------------
        for i in ind_other:
            for retry in range(max_retry):
                current_seed = np.random.SeedSequence([seed, i, retry])
                X_delta[i] = self.generate_directional_noise(dim, (r[i], r[i]), seed=current_seed)
                                
                if y[i] * self.f(X[i] + X_delta[i]) > 1:
                    flag[i] = True
                    break
            
            # 失敗時のメッセージ
            if not flag[i]:
                print(f"Retry failed for index {i} in 'other' category.", flush=True)
        # -----------------------------------------------------------------------
        
        # w方向のノイズを作成 -----------------------------------------------------
        if len(ind_inner) > 0:
            s_inner = np.random.choice([-1, 1], size=len(X[ind_inner]))
            lambda_inner_0 = self.generate_lambda_sub_0(X[ind_inner], y[ind_inner], r[ind_inner], s_inner, ind_inner)
            lambda_inner = self.optimize_lambda(lambda_inner_0, X[ind_inner], y[ind_inner], alphas[ind_inner], r[ind_inner])
            
            # w方向のΔxを作成
            w_unit = self.w / np.linalg.norm(self.w)
            X_delta_parallel[ind_inner] = lambda_inner[:, np.newaxis] * w_unit
            
            """print(f"lambda min:", np.min(lambda_inner), flush=True)
            print(f"lambda max:", np.max(lambda_inner), flush=True)
            print(f"average abs lambda max:", np.mean(np.abs(lambda_inner)), flush=True)
            print(f"(radius max: {radius[1]})", flush=True)"""
        # -----------------------------------------------------------------------
        
        """# 補正後の条件をチェック
        X_new = X + X_delta + X_delta_parallel + X_delta_perp
        flag[ind_other] = np.array([y[i] * self.f(X_new[i]) > 1 for i in ind_other], dtype=bool)
        flag[ind_sv] = np.array([abs(self.f(X[i]) - self.f(X_new[i])) < self.ME for i in ind_sv], dtype=bool)
        flag[ind_inner] = np.array([y[i] * self.f(X_new[i]) < 1 for i in ind_inner], dtype=bool)
        
        # 一つでも条件を満たさない点があれば警告
        if not np.all(flag):
            print(f"INNER Warning: Not all points satisfy the conditions after noise addition.", flush=True)
            print(f"        False deta -> other: {np.sum(~flag[ind_other])}, sv: {np.sum(~flag[ind_sv])}, inner: {np.sum(~flag[ind_inner])}", flush=True)

        for i in ind_inner:
            f_val = self.f(X_new[i])
            if y[i] * f_val >= 1:
                print(f"i={i}, f={f_val:.15f}, y*f={y[i] * f_val:.15f}")"""
        
        # wに垂直なノイズの半径を計算
        r_perp = np.copy(r)
        if len(ind_inner) > 0:
            r_perp[ind_inner] = np.sqrt(np.maximum(r[ind_inner]**2 - lambda_inner**2, 0))
            
        # wと垂直なノイズを作成 -----------------------------------------------------
        for i in ind_corr:
            current_seed = np.random.SeedSequence([seed, i])
            X_delta_perp[i] = self.generate_tangent_noise(dim, (r_perp[i], r_perp[i]), seed=current_seed)
        
        # w_deltaを補正（w_deltaはwに垂直な成分しかもっていないため一括補正可能）
        coef = y[ind_corr] * alphas[ind_corr]
        denom = np.dot(coef, coef)
        w_delta = X_delta_perp[ind_corr].T @ coef
        X_delta_perp[ind_corr] -= np.outer(coef, w_delta) / denom
        # -----------------------------------------------------------------------
        
        
        
        #補正後の条件をチェック
        X_new = X + X_delta + X_delta_parallel + X_delta_perp
        flag[ind_other] = np.array([y[i] * self.f(X_new[i]) > 1 for i in ind_other], dtype=bool)
        flag[ind_sv] = np.array([abs(self.f(X[i]) - self.f(X_new[i])) < self.ME for i in ind_sv], dtype=bool)
        flag[ind_inner] = np.array([y[i] * self.f(X_new[i]) < 1 for i in ind_inner], dtype=bool)
        
        # 一つでも条件を満たさない点があれば警告
        if not np.all(flag):
            print(f"Warning: Not all points satisfy the conditions after noise addition.", flush=True)
            print(f"        False deta -> other: {np.sum(~flag[ind_other])}, sv: {np.sum(~flag[ind_sv])}, inner: {np.sum(~flag[ind_inner])}", flush=True)

        """w = X[ind_corr].T @ (y[ind_corr] * alphas[ind_corr])
        w_new = X_new[ind_corr].T @ (y[ind_corr] * alphas[ind_corr])
        w_error = w_new - w
        print("||w_new - w|| =", np.linalg.norm(w_error))
        print("relative error =", np.linalg.norm(w_error) / np.linalg.norm(w))"""
                        
        # 元データからの平均移動距離
        d_move = np.mean(np.linalg.norm(X_new - X, axis=1))
            
        return X_new, y.copy(), d_move
    
    def generate_lambda_sub_0(self, X_sub, y_sub, r_sub, s_sub, ind_sub):
        """
        内部点に対する初期のλを生成する関数
        """       
        f_sub = np.array([self.f(x) for x in X_sub])
        d_sub = y_sub * (1.0 - y_sub * f_sub) / np.linalg.norm(self.w)
            
        lambda_sub_0 = 0.5 * s_sub * np.minimum(d_sub, r_sub)
        lambda_sub_0 = 0.5 * s_sub * r_sub
        
        return lambda_sub_0
    
    
    def optimize_lambda(self, lambda_sub_0, X_sub, y_sub, alphas_sub, r_sub):

        # 制約の厳しさ
        epsilon = 1e-4
        
        # サイズ
        N_sub = len(lambda_sub_0)

        # QP変数
        x = cp.Variable(N_sub)

        # Q, c
        Q = np.eye(N_sub)
        c = -lambda_sub_0

        # A
        I = np.eye(N_sub)

        A = np.vstack([
            I,
            -I,
            (y_sub * alphas_sub).reshape(1, -1),
            -(y_sub * alphas_sub).reshape(1, -1),
            np.linalg.norm(self.w) * np.diag(y_sub)
        ])

        # b
        f_vec = np.array([self.f(X_sub[i]) for i in range(N_sub)])
        b = np.concatenate([
            r_sub,
            r_sub,
            np.array([0.0]),
            np.array([0.0]),
            1.0 - epsilon - y_sub * f_vec
        ])

        # 目的関数
        objective = cp.Minimize(
            0.5 * cp.quad_form(x, Q) + c @ x
        )

        # 制約
        constraints = [
            A @ x <= b
        ]        

        # QPを解く
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # 結果
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"QP failed: {problem.status}")

        lambda_sub = x.value

        return lambda_sub

    
    
    
    
    
    
    
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
    
    
    
    
    
    
