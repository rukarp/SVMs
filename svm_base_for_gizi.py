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
        6/1 KKT条件から動かす条件を決めてノイズを加える方法
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
        
        def plt_Data_1(x1, x2, y, x_range=(-0.05, 1.05), y_range=(-0.05, 1.05)):
        
            x_min, x_max = x_range
            y_min, y_max = y_range

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

            # 判別関数の出力を取得
            XY = np.vstack([xx.ravel(), yy.ravel()]).T
            Z = np.array([self.f(xy) for xy in XY]).reshape(xx.shape)

            # levelsを100段階の連続にする
            levels = np.linspace(Z.min(), Z.max(), 100)

            # フィールドをプロット
            cf = plt.contourf(xx, yy, Z, levels=levels, cmap='coolwarm', alpha=0.6)
            plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-')      # 決定境界を黒線で
            plt.contour(xx, yy, Z, levels=[-1, 1], colors='k', linestyles=':')  # f(x) = +-1を点線で
        
            # データプロット
            if y == -1:
                plt.scatter(x1[0], x1[1], c='blue', edgecolor='k', s=20, label='label = -1')
                plt.scatter(x2[0], x2[1], c='blue', edgecolor='k', s=20, label='label = -1')
            else:
                plt.scatter(x1[0], x1[1], c='red', edgecolor='k', s=20, label='label = +1')
                plt.scatter(x2[0], x2[1], c='red', edgecolor='k', s=20, label='label = +1')
            plt.scatter(x2[0], x2[1], s=100, facecolors='none', edgecolors='k') 

            plt.colorbar(cf)
            plt.xlabel('X1')
            plt.ylabel('X2')
            #plt.xlabel('petal length (cm)')
            #plt.ylabel('petal width (cm)')
            plt.legend#(fontsize=9)

            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.show()


        ind_sv, ind_inner = self._get_SV_ind(alphas)
        ind_other = np.setdiff1d(np.arange(len(alphas)), np.concatenate([ind_sv, ind_inner]))
        
        #print(f"Other Points: {ind_other}")
        #print(f"Support Vectors: {ind_sv}")
        #print(f"Inner Points: {ind_inner}")
        
        X_delta = np.zeros_like(X)
        X_new = np.zeros_like(X)
        alphas_new = np.zeros_like(alphas)
        
        flag = np.full(len(X), False, dtype=bool)
        
        dim = X.shape[1]
                
        # ind_otherのデータを動かす
        for p in ind_other: 
                
            for retry in range(max_retry):
                
                # ランダム初期移動を毎回生成  
                X_delta[p] = self.generate_directional_noise(dim, radius, seed=p * 100 + retry)
                X_new[p] = X[p] + X_delta[p]
                                    
                # [0,1] 且つマージンの外にあるか判定
                cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                cond_p_margin = y[p] * self.f(X_new[p]) > 1

                #if (not cond_p_area) or (not cond_p_margin):
                if not cond_p_margin:
                    continue
                    
                flag[p] = True
                break
                
            # 失敗時のメッセージ
            if not flag[p]:
                print(f"Retry failed for index {p}.")
                                          
            alphas_new[p] = 0
        
        """# ind_svのデータを動かす
        while not np.all(flag[ind_sv]):
            #print(flag[self.ind_sv])

            remain = ind_sv[~flag[ind_sv]]

            if len(remain) >= 2:
                p, q = remain[0], remain[1]

            elif len(remain) == 1:
                p = remain[0]
                
                q = None
                for ind in self.ind_sv:
                    if ind != p:
                        q = ind
                        break
                if q is None:
                    print(f"Error: No valid q found for p={p}.")
                    break

            else:
                break

            for retry in range(max_retry):
                
                #X[p]がwと直交する場合の処理を要検討
                #cond_p_w = abs(np.dot(self.w, X[p])) < self.ME
                
                # ランダム初期移動を毎回生成  
                x_p_delta = self.generate_tangent_noise(dim, radius, seed=p * 100 + retry)
                x_p = X[p] + x_p_delta
                                
                # x_pが[0,1] 且つサポートベクターであるか判定
                cond_p_area = np.all((x_p >= 0) & (x_p <= 1))
                cond_p_margin = abs(self.f(X[p]) - self.f(x_p)) < self.ME
                
                #if (not cond_p_area) or (not cond_p_margin):
                if not cond_p_margin:
                    print(f"abs(self.f(X[p]) - self.f(x_p)): {abs(self.f(X[p]) - self.f(x_p))}")
                    print(f"cond_p_area: {cond_p_area}, cond_p_margin: {cond_p_margin}")
                    continue
                
                # qはpに合わせて動かす
                x_q_delta = -1 * (y[p] * alphas[p]) / (y[q] * alphas[q]) * x_p_delta
                x_q = X[q] + x_q_delta

                # x_qが[0,1] 且つサポートベクターであるか判定
                cond_q_area = np.all((x_q >= 0) & (x_q <= 1))
                cond_q_margin = abs(self.f(X[q]) - self.f(x_q)) < self.ME

                #if (not cond_q_area) or (not cond_q_margin):
                if not cond_q_margin:
                    print(f"    abs(self.f(X[q]) - self.f(x_q)): {abs(self.f(X[q]) - self.f(x_q))}")
                    print(f"    cond_q_area: {cond_q_area}, cond_q_margin: {cond_q_margin}")
                    continue
                
                flag[p] = flag[q] = True
                break
                
            # 失敗時は元データを使う
            if not flag[p]:
                print(f"Retry failed for index {p}, {q}, using original data point.")
                x_p = X[p].copy()
                x_q = X[q].copy()
                
            X_new[p] = x_p
            alphas_new[p] = alphas[p]
            X_new[q] = x_q
            alphas_new[q] = alphas[q]"""
        
        # ind_svのデータを動かす
        t_start = time.time()
        iter = 0
        while not np.all(flag[ind_sv]):

            # False の点だけ再生成
            for p in ind_sv:

                if flag[p]:
                    continue

                for retry in range(max_retry):

                    X_delta[p] = self.generate_tangent_noise(dim, radius, seed=p * 100000 + iter * 1000 + retry)
                    X_new[p] = X[p] + X_delta[p]

                    cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                    cond_p_margin = abs(self.f(X[p]) - self.f(X_new[p])) < self.ME

                    #if not (cond_p_area and cond_p_margin):
                    if not cond_p_margin:
                        continue

                    flag[p] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[p]:
                    print(f"Retry failed for index {p}.")
                    
            # 全体補正
            r = np.sum(y[ind_sv, np.newaxis] * alphas[ind_sv, np.newaxis] * X_delta[ind_sv],axis=0)
            
            denom = np.sum(alphas[ind_sv] ** 2)
            X_delta[ind_sv] -= (y[ind_sv, np.newaxis] * alphas[ind_sv, np.newaxis] / denom) * r
            
            # 補正後に再チェック
            X_new[ind_sv] = X[ind_sv] + X_delta[ind_sv]

            cond_area = np.all((X_new[ind_sv] >= 0) & (X_new[ind_sv] <= 1),axis=1)
            cond_margin = np.array([abs(self.f(X[p]) - self.f(X_new[p])) < self.ME for p in ind_sv])

            #flag[ind_sv] = cond_area & cond_margin
            flag[ind_sv] = cond_margin
        
            iter += 1
        
        alphas_new[ind_sv] = alphas[ind_sv]
        
        t_end = time.time()
        print(f"Time taken for KKT-based fake data generation (inner): {t_end - t_start:.4f} seconds")
    
        """# ind_innerのデータを動かす
        t_start = time.time()
        while not np.all(flag[ind_inner]):
            #print(flag[self.ind_inner])

            remain = ind_inner[~flag[ind_inner]]

            if len(remain) >= 2:
                p, q = remain[0], remain[1]

            elif len(remain) == 1:
                p = remain[0]
                
                q = None
                for ind in self.ind_inner:
                    if ind != p:
                        q = ind
                        break
                if q is None:
                    print(f"Error: No valid q found for p={p}.")
                    break

            else:
                break

            for retry in range(max_retry):
                
                # ランダム初期移動を毎回生成  
                x_p_delta = self.generate_directional_noise(dim, radius, seed=p * 100 + retry)
                x_p = X[p] + x_p_delta
                
                # x_pが[0,1] 且つ違反サンプルであるか判定
                cond_p_area = np.all((x_p >= 0) & (x_p <= 1))
                cond_p_margin = y[p] * self.f(x_p) < 1
                
                #if (not cond_p_area) or (not cond_p_margin):
                if not cond_p_margin:
                    continue
                
                # qはpに合わせて動かす
                x_q_delta = -1 * y[p] * y[q] * x_p_delta
                x_q = X[q] + x_q_delta

                # x_qが[0,1] 且つ違反サンプルであるか判定
                cond_q_area = np.all((x_q >= 0) & (x_q <= 1))
                cond_q_margin = y[q] * self.f(x_q) < 1

                #if (not cond_q_area) or (not cond_q_margin):
                if not cond_q_margin:
                    continue
                
                flag[p] = flag[q] = True
                break
                
            # 失敗時は元データを使う
            if not flag[p]:
                print(f"Retry failed for index {p}, {q}, using original data point.")
                x_p = X[p].copy()
                x_q = X[q].copy()
                
            X_new[p] = x_p
            alphas_new[p] = self.C
            X_new[q] = x_q
            alphas_new[q] = self.C
        
        t_end = time.time()
        print(f"Time taken for KKT-based fake data generation (pairwise): {t_end - t_start:.4f} seconds")"""
                
        # ind_innerのデータを動かす
        t_start = time.time()
        iter = 0
        while not np.all(flag[ind_inner]):

            # False の点だけ再生成
            for p in ind_inner:

                if flag[p]:
                    continue

                for retry in range(max_retry):

                    X_delta[p] = self.generate_directional_noise(dim, radius, seed=p * 100000 + iter * 1000 + retry)
                    X_new[p] = X[p] + X_delta[p]

                    cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                    cond_p_margin = y[p] * self.f(X_new[p]) < 1

                    #if not (cond_p_area and cond_p_margin):
                    if not cond_p_margin:
                        continue

                    flag[p] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[p]:
                    print(f"Retry failed for index {p}.")
                    
            # 全体補正
            r = np.sum(y[ind_inner, np.newaxis] * X_delta[ind_inner], axis=0)

            num_inner = len(ind_inner)
            X_delta[ind_inner] -= (y[ind_inner, np.newaxis] / num_inner) * r

            # 補正後に再チェック
            X_new[ind_inner] = X[ind_inner] + X_delta[ind_inner]

            cond_area = np.all((X_new[ind_inner] >= 0) & (X_new[ind_inner] <= 1),axis=1)
            cond_margin = np.array([y[p] * self.f(X_new[p]) < 1 for p in ind_inner])

            #flag[ind_inner] = cond_area & cond_margin
            flag[ind_inner] = cond_margin
        
            iter += 1
        
        alphas_new[ind_inner] = self.C
        
        t_end = time.time()
        print(f"Time taken for KKT-based fake data generation (inner): {t_end - t_start:.4f} seconds")
        
        return X_new, y.copy()
    
    
    
    
    
    
    
    
    
    
    def make_fake_data_KKT(self, X, y, alphas, radius, max_retry):
        """
        6/1 KKT条件から動かす条件を決めてノイズを加える方法
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
        denom = np.sum(alphas ** 2)
        # X_deltaの補正に用いるΣy_iα_iの計算
        coef = (y * alphas)[:, None]
        
        # Xの次元数を取得
        dim = X.shape[1]
        
        # すべての点のフラグを初期化
        flag = np.full(len(X), False, dtype=bool)
        
        
        
        # X_deltaを作るための関数（マージンの条件ごとに違う関数でノイズを作るので関数に分けておく）
        def generate_noise(ind, X, X_delta, X_new, flag, noise_func, dim, radius, max_retry, area_cond, margin_cond):
            
            for i in ind[~flag[ind]]:
                for retry in range(max_retry):
                    X_delta[i] = noise_func(dim, radius, seed=i * 100000 + iter * 1000 + retry)
                    X_new[i] = X[i] + X_delta[i]
                    
                    cond_p_area = area_cond(i)
                    cond_p_margin = margin_cond(i)
                    
                    #if cond_p_area and cond_p_margin:
                    if cond_p_margin:
                        flag[i] = True
                        break
                
                # 失敗時のメッセージ
                if not flag[i]:
                    print(f"Retry failed for index {i}.")
        
            return X_delta



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
            
            """# ind_otherのFalseの点を生成
            X_delta = generate_noise(
                ind_other,
                X, X_delta, X_new, flag,
                self.generate_directional_noise,
                dim, radius, max_retry,
                lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                lambda i: y[i] * self.f(X_new[i]) > 1
            )
            
            # ind_svのFalseの点を生成
            X_delta = generate_noise(
                ind_sv,
                X, X_delta, X_new, flag,
                self.generate_tangent_noise,
                dim, radius, max_retry,
                lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                lambda i: abs(self.f(X[i]) - self.f(X_new[i])) < self.ME
            )
            
            # ind_innerのFalseの点を生成
            X_delta = generate_noise(
                ind_inner,
                X, X_delta, X_new, flag,
                self.generate_directional_noise,
                dim, radius, max_retry,
                lambda i: np.all((X_new[i] >= 0) & (X_new[i] <= 1)),
                lambda i: y[i] * self.f(X_new[i]) < 1
            )"""
            
            """# ind_otherのFalseの点を生成
            for p in ind_other[~flag[ind_other]]:

                for retry in range(max_retry):

                    X_delta[p] = self.generate_directional_noise(dim, radius, seed=p * 100000 + iter * 1000 + retry)
                    X_new[p] = X[p] + X_delta[p]

                    cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                    cond_p_margin = y[p] * self.f(X_new[p]) > 1

                    #if not (cond_p_area and cond_p_margin):
                    if not cond_p_margin:
                        continue

                    flag[p] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[p]:
                    print(f"Retry failed for index {p}.")
            
            # ind_svのFalseの点を生成
            for p in ind_sv[~flag[ind_sv]]:

                for retry in range(max_retry):

                    X_delta[p] = self.generate_tangent_noise(dim, radius, seed=p * 100000 + iter * 1000 + retry)
                    X_new[p] = X[p] + X_delta[p]

                    cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                    cond_p_margin = abs(self.f(X[p]) - self.f(X_new[p])) < self.ME

                    #if not (cond_p_area and cond_p_margin):
                    if not cond_p_margin:
                        continue

                    flag[p] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[p]:
                    print(f"Retry failed for index {p}.")
        
            # ind_innerのFalseの点を生成
            for p in ind_inner[~flag[ind_inner]]:

                for retry in range(max_retry):

                    X_delta[p] = self.generate_directional_noise(dim, radius, seed=p * 100000 + iter * 1000 + retry)
                    X_new[p] = X[p] + X_delta[p]

                    cond_p_area = np.all((X_new[p] >= 0) & (X_new[p] <= 1))
                    cond_p_margin = y[p] * self.f(X_new[p]) < 1

                    #if not (cond_p_area and cond_p_margin):
                    if not cond_p_margin:
                        continue

                    flag[p] = True
                    break
                
                # 失敗時のメッセージ
                if not flag[p]:
                    print(f"Retry failed for index {p}.")"""
        
            # 全体補正
            w_delta = np.sum(coef * X_delta, axis=0)            
            X_delta -= (coef / denom) * w_delta
            
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
    
    
    
    
    
    
