# Numpy
import numpy as np

# 時間計測用
import time

# グラフのプロット用
import matplotlib.pyplot as plt



# fit 関数以外をまとめた親クラス
class BaseSVM:
    def __init__(self, C=float("inf"), kernel='linear', degree=2, gamma=1.0, coef0=1.0, max_iterations= 1000000, tol= 1e-3, ME = 1e-10):
        """
        MySVMクラスの初期化関数
        パラメータ:
        C (float): ソフトマージンの重みパラメータ
        kernel (str): カーネルタイプ ('linear')
        degree (int): 多項式カーネルの次数
        gamma (float): カーネル係数
        coef0 (float): シグモイドカーネルおよび多項式カーネルの定数項

        max_iterations (int): イタレーション数 (パラメータを更新する最大回数)
        tol (float): 違反度を計算するときの許容誤差
        ME (float): Machine Epsilon
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.max_iterations = max_iterations
        self.tol = tol
        self.ME = ME

    def _scale_labels_to_pm1(self, labels):
        """
        ベクトルを-1と1にスケーリングする関数
        パラメータ:
        labels (array-like): スケーリングするラベル

        戻り値:
        replaced_labels (array-like): スケーリングされたラベル
        """

        # 置き換え
        replaced_labels = np.where(labels == self.min_val, -1, 1)

        return replaced_labels

    def _inverse_scale_labels(self, labels):
        """
        ラベルを-1と1から元の値に逆スケーリングする関数
        パラメータ:
        labels (array-like): 逆スケーリングするラベル

        戻り値:
        original_labels (array-like): 元のラベル
        """
        # 置き換え
        original_labels = np.where(labels >= 0, self.max_val, self.min_val)

        return original_labels

    def _get_SV_ind(self, alphas):
        """
        サポートベクターと内部サポートベクターのインデックスを取得する関数
        """
        ind_sv = np.where((self.ME < alphas) & (alphas < self.C - self.ME))[0]
        ind_inner = np.where(alphas >= self.C - self.ME)[0]
        
        #ind_sv = np.where((0 < alphas) & (alphas < self.C))[0]
        #ind_inner = np.where(alphas == self.C)[0]

        return ind_sv, ind_inner

    def _kernel(self, x1, x2):
        """
        カーネル関数を計算する関数
        パラメータ:
        x1 (array-like): 入力ベクトル1
        x2 (array-like): 入力ベクトル2

        戻り値:
        float: カーネル関数の値
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2.T) + self.coef0)
        else:
            raise ValueError("Unsupported kernel type: {self.kernel}")
    
    def _calculate_kernel_matrix(self, X):
        """
        カーネル行列を計算する関数
        
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        
        戻り値:
        array-like: カーネル行列。K(x_i, X)は各サンプルx_iと入力Xとの間のカーネル関数の出力を含む行列。
        """
        num_samples = X.shape[0]
        
        K = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                if i <= j:
                    K[i, j] = self._kernel(X[i], X[j])
                else:
                    K[i, j] = K[j, i]
        return K
    
    def _calculate_kernel_rows(self, p, q, X, K):
        """
        カーネル行列のp, q行を計算する関数
        パラメータ:
        p, q (int): アルファのインデックス
        X (array-like): トレーニングデータの特徴量
        K (array-like): カーネル行列

        戻り値:
        array-like: カーネル行列。K(x_i, X)は各サンプルx_iと入力Xとの間のカーネル関数の出力を含む行列        
        """
        
        #print(f"p:{p}", end='')
        n = X.shape[0]
        
        if np.isnan(K[p, 0]):
            #print("[ nan ]", end='')
            for i in range(X.shape[0]):
                K[p, i] = self._kernel(X[p], X[i])
        #print(f", q:{q}",end='')
        
        if np.isnan(K[q, 0]):
            #print("[ nan ]", end='')
            for i in range(X.shape[0]):
                K[q, i] = self._kernel(X[q], X[i])
        
        #print(", ",end='')
        
        return K    
    

    def _find_up_min_and_low_max(self, y, alphas, E):
        """
        違反ペア (p, q) を選択する関数
    
        パラメータ:
        y (array-like): トレーニングデータのラベル
        alphas (array-like): アルファ値の配列
        E (array-like): 誤差量を表す配列

        戻り値:
        up_min_ind (int): 違反条件を満たす最小誤差のインデックス (p)
        low_max_ind (int): 違反条件を満たす最大誤差のインデックス (q)
        up_min_val (float): p番目の誤差 E[p]
        low_max_val (float): q番目の誤差 E[q]
        """
        I_0 = np.where((alphas > self.ME) & (alphas < self.C - self.ME))[0]
        I_1 = np.where((alphas <= self.ME) & (y == 1))[0]
        I_2 = np.where((alphas >= self.C - self.ME) & (y == -1))[0]
        I_3 = np.where((alphas >= self.C - self.ME) & (y == 1))[0]
        I_4 = np.where((alphas <= self.ME) & (y == -1))[0]
            
        I_up = np.concatenate((I_0, I_1, I_2))
        I_low = np.concatenate((I_0, I_3, I_4))
            
        E_up, E_low = E[I_up], E[I_low]
        
        local_up_min_ind, local_low_max_ind = np.argmin(E_up), np.argmax(E_low)
        up_min_ind, low_max_ind = I_up[local_up_min_ind], I_low[local_low_max_ind]
        up_min_val, low_max_val = E_up[local_up_min_ind], E_low[local_low_max_ind]
            
        return up_min_ind, low_max_ind, up_min_val, low_max_val
        
    def _update_alphas(self, p, q, y, alphas, K, E):
        """
        アルファ値を更新する関数
        パラメータ:
        p, q (int): アルファのインデックス
        y (array-like): トレーニングデータのラベル
        alphas (array-like): アルファ値の配列
        K (array-like): カーネル行列
        E (array-like): 誤差量を表す配列

        戻り値:
        alphas (array-like): 更新されたアルファ値の配列
        E (array-like): 誤差量を表す配列
        d_obj (float): 目的関数の変化量
        """
        
        y_p, y_q = y[p], y[q]
        a_p, a_q = alphas[p], alphas[q]
        E_p, E_q = E[p], E[q]
        
        d_obj = 0
        
        # L, Hを計算        
        if y_p == y_q:
            L = max(0, a_p + a_q - self.C)
            H = min(self.C, a_p + a_q)
        else:
            L = max(0, a_p - a_q)
            H = min(self.C, self.C + a_p - a_q)

        if L == H:
            return alphas, E, d_obj
        
        # etaを計算
        eta = K[p, p] - 2 * K[p, q] + K[q, q]
        
        # eta > 0のとき，alphasを更新
        if eta > 0:
            a_p_new = a_p + y_p * (E_q - E_p) / eta
            a_p_new = min(H, max(L, a_p_new))
            
        # eta <= 0のとき，目的関数の値からalphasを更新
        else:
            L_obj = L * (eta * (0.5 * L - a_p) + y_p * (E_p - E_q))
            H_obj = H * (eta * (0.5 * H - a_p) + y_p * (E_p - E_q))
                    
            # LとHのどちらが目的関数を最小化するかでアルファを選択
            if H_obj < L_obj:
                a_p_new = H
            elif L_obj < H_obj:
                a_p_new = L
            else:
                a_p_new = a_p
        
        # alphas[q]を更新
        a_q_new = a_q + y_p * y_q * (a_p - a_p_new)

        # alphasの変化量を計算
        d_p, d_q = a_p_new - a_p, a_q_new - a_q
        
        # alphasに結果を格納
        alphas[p], alphas[q] = a_p_new, a_q_new

        #print(f"alpha[{p}]: {alphas[p]}, alpha[{p}]: {alphas[p]}, ", end='')
        
        # Eを更新
        """for i in range(len(E)):
            E[i] += K[p, i] * y_p * d_p + K[q, i] * y_q * d_q"""
        E += K[p, :] * y_p * d_p + K[q, :] * y_q * d_q
            
        # 目的関数の変化量を計算
        d_obj = d_p * (0.5 * d_p * eta + y_p * (E_p - E_q))

        return alphas, E, d_obj
            
    def _calculate_w(self, X, y, alphas, ind_sv, ind_inner):
        """
        重みベクトル w を計算する関数

        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        alphas (array-like): アルファ値の配列
        ind_sv (array-like): サポートベクターのインデックス
        ind_inner (array-like): サポートベクターの内積インデックス

        戻り値:
        array-like: w の値の配列
        """
        w = np.zeros(X.shape[1])
        for i in np.concatenate((ind_sv, ind_inner)):
            w += alphas[i] * y[i] * X[i]
        
        return w
    
    def _calculate_b(self, y, alphas, K, ind_sv, ind_inner):
        """
        バイアス項 b を計算する関数

        パラメータ:
        y (array-like): トレーニングデータのラベル
        alphas (array-like): アルファ値の配列
        K (array-like): カーネル行列
        num_samples (int): サンプルの数
        ind_sv (array-like): サポートベクターのインデックス
        ind_inner (array-like): サポートベクターの内積インデックス

        戻り値:
        float: b の値
        """
        if len(ind_sv) > 0:
            b = np.sum(y[ind_sv])
            for i in ind_sv:
                for j in np.concatenate((ind_sv, ind_inner)):
                    b -= alphas[j] * y[j] * K[j, i]
            b /= len(ind_sv)  
        
        else:
            b = 0  

        return b







    def make_decision_func(self, X, y, alphas, w, b, ind_sv, ind_inner):
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
        function: 入力 x を受け取り f(x) を計算する関数
        """
        def f(x):
            if self.kernel == 'linear':
                f_value = np.dot(w, x) + b
            else:
                f_value = 0
                for i in np.concatenate((ind_sv, ind_inner)):
                    f_value += alphas[i] * y[i] * self._kernel(X[i], x)
                f_value += b
        
            return f_value
        
        return f

    def predict(self, X):
        """
        新しいデータに対して予測を行う関数
        パラメータ:
        X (array-like): 予測するデータの特徴量

        戻り値:
        array-like: 予測ラベル
        """

        y_pred = np.array([self.f(x) for x in X])

        return np.where(y_pred >= 0, 1, -1)

    
    
    
    
    
    def fit(self, X, y):
        """
        SVMモデルをデータに適合させる関数
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        """
        
        # オリジナルデータのラベルの最小値と最大値を取得
        self.min_val = np.min(y)
        self.max_val = np.max(y)
        
        self.X = X
        self.y = y
        
        self.alphas = np.zeros(len(y), dtype=float)
        
        self.K = np.full((len(y), len(y)), np.nan, dtype=float)
        E = -self.y.astype(float)
                
        self.objective_value = np.array([0])

        #----------------------SMOアルゴリズムによるアルファの学習-----------------------#

        #t1 = time.time()
        
        #K = self._calculate_kernel_matrix(self.X)

        is_converged = False

        for count in range(self.max_iterations):

            #print(f"{count}: ", end='')

            # 違反ペアを得る
            #p, F_p = self._find_up_min(self.y, self.alphas, E)
            #q, F_q = self._find_low_max(self.y, self.alphas, E)
            p, q, F_p, F_q = self._find_up_min_and_low_max(self.y, self.alphas, E)
            
            #if count % 1000 == 0:
            #print(f'{count} -> p:{p}, q:{q}, objective_value:{self.objective_value[count]}')
            #print(f'    p:{p}, y_p:{self.y[p]}, a_p:{self.alphas[p]}, E_p:{E[p]}')
            #print(f'    q:{q}, y_q:{self.y[q]}, a_q:{self.alphas[q]}, E_q:{E[q]}')
            
            # 違反ペアが見つからない場合，学習を終了
            if (F_p > F_q - self.tol):
                is_converged = True
                
                self.iterations = count
                
                print(f"Successfully converged. Iterations: {count}")

                break
            
            # カーネル行列のp, q行を計算
            self.K = self._calculate_kernel_rows(p, q, self.X, self.K)
            
            # alphas を更新
            self.alphas, E, d_obj = self._update_alphas(p, q, self.y, self.alphas, self.K, E)
            
            # 目的関数の計算結果を配列に追加
            self.objective_value = np.append(self.objective_value, self.objective_value[count] + d_obj)



            #print(f'Obj: {self.objective_value[count+1]}')
            
            #print(f'{self.objective_value[count+1]}, ')
            
            
            
        if not is_converged:
            print(f"Reached maximum max_iterations. Iterations: {self.max_iterations}")
        
        #t2 = time.time()
        #print(f'fitting time: {t2-t1}')
        #-----------------------------------------------------------------------------#

        # サポートベクターのインデントを選択
        self.ind_sv, self.ind_inner = self._get_SV_ind(self.alphas)
        
        # 重みwを計算
        if self.kernel == 'linear':
            self.w = self._calculate_w(self.X, self.y, self.alphas, self.ind_sv, self.ind_inner)
        else:
            self.w = None

        # バイアスbを計算
        self.b = self._calculate_b(self.y, self.alphas, self.K, self.ind_sv, self.ind_inner)
        
        # 識別関数を生成
        self.f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)
    
    
    
    
    
    """def plt_Data_and_Boundary(self):
        
        #データポイントと決定境界をプロットする関数
        
        x_min, x_max = -0.05, 1.05
        y_min, y_max = -0.05, 1.05

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        plt.scatter(self.X[self.y == min(self.y), 0], self.X[self.y == min(self.y), 1], c='blue', edgecolor='k', s=20, label='label = -1')
        plt.scatter(self.X[self.y == max(self.y), 0], self.X[self.y == max(self.y), 1], c='red', edgecolor='k', s=20, label='label = +1')
        plt.scatter(self.X[self.ind_sv, 0], self.X[self.ind_sv, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

        XY = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = self.predict(XY).reshape(xx.shape)
        #Z = np.array([self.predict(np.array([xx.ravel()[i], yy.ravel()[i]]).reshape(1, -1)) for i in range(len(xx.ravel()))])
        levels = [self.min_val, (self.min_val + self.max_val) / 2, self.max_val]
        plt.contourf(xx, yy, Z, levels, alpha=0.3, colors=['lightblue', 'pink'])
        plt.contour(xx, yy, Z, levels= [(self.min_val + self.max_val) / 2], colors='k', linestyles='-')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()"""
    
    def plt_Data_and_Boundary(self, x_range=(-0.05, 1.05), y_range=(-0.05, 1.05)):
        
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
        plt.scatter(self.X[self.y == min(self.y), 0], self.X[self.y == min(self.y), 1], c='blue', edgecolor='k', s=20, label='label = -1')
        plt.scatter(self.X[self.y == max(self.y), 0], self.X[self.y == max(self.y), 1], c='red', edgecolor='k', s=20, label='label = +1')
        plt.scatter(self.X[self.ind_sv, 0], self.X[self.ind_sv, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')        
        plt.scatter(self.X[self.ind_inner, 0], self.X[self.ind_inner, 1], s=100, facecolors='none', edgecolors='k', marker='^', label='Margin-Violating Samples')
        
        plt.colorbar(cf)
        plt.xlabel('X1')
        plt.ylabel('X2')
        #plt.xlabel('petal length (cm)')
        #plt.ylabel('petal width (cm)')
        plt.legend#(fontsize=9)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.show()

        
    def plt_Objective_Values(self):
        
        # 目的関数の推移をプロットする関数
        
        plt.plot(self.objective_value, label='objective function\'s value')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.show()