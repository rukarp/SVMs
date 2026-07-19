"""
2024/12/06

作成者 : 山田 涼太 (B4)


徐さんのコードNSMO7.pyのアルゴリズムを
自分のコードに導入して実装
    
    
2024/11/29 更新分
・違反ペアはforループで探すのではなく
Numpy配列を使って計算するように変更

2024/12/06 更新分
・カーネル行列の計算方法を変更

2024/12/06 更新文
・Rjが代表して計算する場合
"""

# Numpy
import numpy as np

# MPI
from mpi4py import MPI
import socket

# リストのコピー用
import copy

# グラフのプロット用
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# 時間計測用
import time

# f1スコア計算用
from sklearn.metrics import f1_score

# 学習データ
import data_iris as ir
import data_cancer as ca
import data_adult as ad
import data_airline as ai



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

#buffer_size = 32768

class BaseDSMO:
    def __init__(self, C=float("inf"), kernel='linear', lamda =.005, degree=2, gamma=1.0, coef0=0.0, max_iterations= 100000, tol= 1e-3, ME = 1e-10):
        """
        BaseDSMOクラスの初期化関数
        パラメータ:
        C (float): ソフトマージンの重みパラメータ
        kernel (str): カーネルタイプ ('linear')
        lamda (int): 目的関数において，関数の合意を取る項につける重み
        degree (int): 多項式カーネルの次数
        gamma (float): カーネル係数
        coef0 (float): シグモイドカーネルおよび多項式カーネルの定数項

        max_iterations (int): イタレーション数 (パラメータを更新する最大回数)
        tol (float): 違反度を計算するときの許容誤差
        ME (float): Machine Epsilon
        """
        self.C = C
        self.kernel = kernel
        self.lamda = lamda
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
        # 最小値と最大値を取得
        self.min_val = np.min(labels)
        self.max_val = np.max(labels)

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
        original_labels = np.where(labels == -1, self.min_val, self.max_val)

        return original_labels

    def _get_SV_ind(self, alphas):
        """
        サポートベクターと内部サポートベクターのインデックスを取得する関数
        """
        ind_sv = np.where((self.ME < alphas) & (alphas < self.C - self.ME))[0]
        ind_inner = np.where(alphas >= self.C - self.ME)[0]

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
    
    def _kernel_row(self, x, X):
        """
        カーネル行を計算する関数（Xとxの間）

        パラメータ:
        x (ndarray): shape (d,)
        X (ndarray): shape (N, d)

        戻り値:
        ndarray: shape (N,) のカーネル値
        """

        if self.kernel == 'linear':
            return X @ x

        elif self.kernel == 'poly':
            return (X @ x.T + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            diff = X - x  # broadcasting (N, d)
            return np.exp(-self.gamma * np.sum(diff**2, axis=1))

        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * (X @ x.T) + self.coef0)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
    
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
        
        I_0 = np.where((alphas > self.ME) & (alphas < self.C - self.ME))[0]
        I_1 = np.where((alphas <= self.ME) & (y == 1))[0]
        I_2 = np.where((alphas >= self.C - self.ME) & (y == -1))[0]
        I_3 = np.where((alphas >= self.C - self.ME) & (y == 1))[0]
        I_4 = np.where((alphas <= self.ME) & (y == -1))[0]
            
        I_up = np.concatenate((I_0, I_1, I_2))
        I_low = np.concatenate((I_0, I_3, I_4))
        
        #print(f'rank {rank}, len(E_up) = {len(E_up)}, len(E_low) = {len(E_low)}',flush=True)
        
        if len(I_up) > 0:
            E_up = E[I_up]
            local_up_min_ind = np.argmin(E_up)
            up_min_ind, up_min_val = I_up[local_up_min_ind], E_up[local_up_min_ind]
        else:
            up_min_ind, up_min_val = 0, float("inf")
        
        if len(I_low) > 0:
            E_low = E[I_low]
            local_low_max_ind = np.argmax(E_low)
            low_max_ind, low_max_val = I_low[local_low_max_ind], E_low[local_low_max_ind]
        else:
            low_max_ind, low_max_val = 0, -float("inf")
        
        return up_min_ind, low_max_ind, up_min_val, low_max_val
            
    
    def _update_alphas(self, eta, y_p, y_q, a_p, a_q, E_p, E_q): 
        """
        アルファ値を更新する関数
        パラメータ:
        eta (float): カーネル行列の値から計算した目的関数の分母
        y_p, y_q (int): トレーニングデータのラベル
        a_p, a_q (float): アルファ値
        E_p, E_q (float): 誤差量

        戻り値:
        a_p_new, a_q_new (float): 更新されたアルファ値
        """
        
        # L, Hを計算        
        if y_p == y_q:
            L = max(0, a_p + a_q - self.C)
            H = min(self.C, a_p + a_q)
        else:
            L = max(0, a_p - a_q)
            H = min(self.C, self.C + a_p - a_q)

        if L == H:
            return 0, 0
        
        # etaを計算
        #eta = K_ii - 2 * K_ij + K_jj
        
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
                
        return a_p_new, a_q_new

            
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
        b = np.sum(y[ind_sv])
        for i in ind_sv:
            for j in np.concatenate((ind_sv, ind_inner)):
                b -= alphas[j] * y[j] * K[j, i]
        b /= len(ind_sv)

        return b
    
    def _calculate_b1(self, y, alphas, K, start_ind, end_ind, ind_sv, ind_inner):
        """
        バイアス項 b を計算する関数

        パラメータ:
        y (array-like): トレーニングデータのラベル
        alphas (array-like): アルファ値の配列
        K (array-like): カーネル行列
        start_ind (int): エージェントが初期状態でもつ1つ目のデータのインデックス
        end_ind (int): エージェントが初期状態でもつ最後のデータのインデックス
        ind_sv (array-like): サポートベクターのインデックス
        ind_inner (array-like): サポートベクターの内積インデックス

        戻り値:
        float: b の値
        """        
        # カーネル行列がエージェントごとに存在するので協力して計算
        local_active_sv = ind_sv[(start_ind <= ind_sv) & (ind_sv < end_ind)]
        local_active_inner = ind_inner[(start_ind <= ind_inner) & (ind_inner < end_ind)]
        local_active_sv_inner = np.concatenate((local_active_sv, local_active_inner))
        
        my_b = np.sum(y[local_active_sv])
        for i in ind_sv:
            for j in local_active_sv_inner:
                my_b -= alphas[j] * y[j] * K[i, j - start_ind]
                
        b = sum(comm.allgather(my_b)) / len(ind_sv)
        
        return b
    

    
    def fit_D(self, X, y): 
        """
        SVMモデルをデータに適合させる関数（分散学習）
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        """ 
                 
        # データの個数を共有し，箱を用意
        num_samples_ori_list = comm.allgather(len(X))
        start_ind = sum(num_samples_ori_list[:rank])
        end_ind = start_ind + num_samples_ori_list[rank]
        
        # エージェントが持っているデータのインデント
        self.my_ind = np.arange(start_ind, end_ind)
        
        self.num_samples_all = sum(num_samples_ori_list)
        
        # 各データの箱を初期化
        self.X = np.zeros((self.num_samples_all, X.shape[1]), dtype=float)
        self.y = np.zeros(self.num_samples_all, dtype=int)
        self.alphas = np.zeros(self.num_samples_all, dtype=float)
        
        self.X[self.my_ind] = X
        self.y[self.my_ind] = y
        
        # xの次元
        d = self.X.shape[1]
        
        # エラーキャッシュ(自分の初期状態で持つデータ数のみ)
        my_E = -self.y[self.my_ind].astype(float)
        
        # カーネル行列
        #K = np.full((self.num_samples_all, self.num_samples_all), np.nan, dtype=float)
        K = np.zeros((self.num_samples_all, len(self.my_ind)), dtype=float)
                
        self.objective_value = np.array([0])
        self.num_samples = np.array([])
        
        # 共有済みのデータに対応するインデックスをTrueとしていく配列
        is_shared = np.full(self.num_samples_all, False, dtype=bool)
        
        # MPI用のbufなど
        RANK, P, E_UP, Q, E_LOW = range(5)
        buf_E = np.empty((size, 5), dtype=np.float64)
        buf_X_y = np.empty(d + 1, dtype=np.float64)
        buf_a = np.empty(4, dtype=np.float64)
        # ----------------------------
        # 時間計測用
        t_E = t_E_1 = t_E_2 = t_E_3 = t_E_4 = t_K = t_K_1 = t_K_2 = t_a = 0
        
        is_converged = False

        for passes in range(self.max_iterations):
                                    
            self.num_samples = np.append(self.num_samples, np.count_nonzero(self.y))
            
            t_E_start = time.time()
            
            my_y = self.y[self.my_ind]
            my_alphas = self.alphas[self.my_ind]

            t_E_1_start = time.time()
            
            # 自分の拠点の違反ペアを得る
            my_p, my_q, my_E_up_min, my_E_low_max = self._find_up_min_and_low_max(my_y, my_alphas, my_E)
            
            # インデントをグローバルのインデント番号に書き換え
            my_p += start_ind
            my_q += start_ind
            
            t_E_1_end = time.time()
            t_E_1 += t_E_1_end - t_E_1_start
            
            # ---------------------------------------------------------------------------
            """t_E_2_start = time.time()
                        
            # 全エージェントの中で最小の E_up と、その値を持つエージェント(rank)を取得
            E_p, Rp = comm.allreduce((my_E_up_min, rank), op=MPI.MINLOC)
            # 全エージェントの中で最大の E_low と、その値を持つエージェント(rank)を取得
            E_q, Rq = comm.allreduce((my_E_low_max, rank), op=MPI.MAXLOC)

            # グローバルインデックスを共有
            p = comm.bcast(my_p, root=Rp)
            q = comm.bcast(my_q, root=Rq)
            
            t_E_2_end = time.time()
            t_E_2 += t_E_2_end - t_E_2_start"""

            # ---------------------------------------------------------------------------

            """t_E_3_start = time.time()
            
            # 各エージェントの結果を全共有し、最良のE_upとE_lowを探索
            my_val = (my_p, my_E_up_min, my_q, my_E_low_max, rank)
            all_vals = comm.allgather(my_val)
            # E_up最小, 同値ならrank最小
            p, E_p, _, _, Rp = min(all_vals, key=lambda x: x[1])
            # E_low最大, 同値ならrank最小
            _, _, q, E_q, Rq = max(all_vals, key=lambda x: x[3])
            
            t_E_3_end = time.time()
            t_E_3 += t_E_3_end - t_E_3_start"""

            # ---------------------------------------------------------------------------
            
            t_E_4_start = time.time()
            
            # 送信用バッファ（固定長5要素）
            buf = np.array([rank, my_p, my_E_up_min, my_q, my_E_low_max], dtype=np.float64)
            # 全エージェントで共有
            comm.Allgather(buf, buf_E)

            # E_up最小のE_pと, E_low最大のE_qを獲得, 同値ならrank最小
            Rp, p, E_p = buf_E[np.argmin(buf_E[:, E_UP]), [RANK, P, E_UP]]
            Rq, q, E_q = buf_E[np.argmax(buf_E[:, E_LOW]), [RANK, Q, E_LOW]]
            p, q, Rp, Rq = map(int, (p, q, Rp, Rq))
            
            t_E_4_end = time.time()
            t_E_4 += t_E_4_end - t_E_4_start
            
            # ---------------------------------------------------------------------------
           
            t_E_end = time.time()
            t_E += t_E_end - t_E_start
            
            # 違反ペアが見つからない場合，学習を終了
            if (E_p > E_q - self.tol):
                is_converged = True
                
                if rank == 0:
                    print(f'{passes} -> objective_value:{self.objective_value[passes]}, gap:{E_p - E_q}', flush=True)                
                break   
            
            t_K_start = time.time()
            
            # p, q 番目のデータが共有されていない場合の共有とカーネル行列の計算
            for Ri, i, my_i in ((Rp, p, my_p), (Rq, q, my_q)):
                if is_shared[i]:
                    continue               
                """t_K_1_start = time.time()
                self.X[i], self.y[i] = comm.bcast((self.X[my_i], self.y[my_i]), root=Ri)  
                t_K_1_end = time.time()
                t_K_1 += t_K_1_end - t_K_1_start"""
                
                t_K_2_start = time.time()
                if rank == Ri:
                    buf_X_y[:d] = self.X[my_i]
                    buf_X_y[d] = self.y[my_i]
                comm.Bcast(buf_X_y, root=Ri)
                self.X[i] = buf_X_y[:d]
                self.y[i] = int(buf_X_y[d])
                t_K_2_end = time.time()
                t_K_2 += t_K_2_end - t_K_2_start
                
                K[i, :] = [self._kernel(self.X[i], x) for x in self.X[self.my_ind]]
                #K[i, :] = self._kernel_row(self.X[i], self.X[self.my_ind])
                is_shared[i] = True
                        
            t_K_end = time.time()
            t_K += t_K_end - t_K_start
            
            """if rank == 0:
                print(f'{passes} -> p:{p}, q:{q}', flush=True)
                print(f'    E_p:{E_p}, E_q:{E_q}', flush=True)"""
            
            t_a_start = time.time()           
                        
            # カーネルを計算 --------------------------          
            K_pp = self._kernel(self.X[p], self.X[p])
            K_pq = self._kernel(self.X[p], self.X[q])
            K_qq = self._kernel(self.X[q], self.X[q])
            
            eta = K_pp - 2 * K_pq + K_qq
            # ----------------------------------------
            
            # 各エージェントがalphasの計算をする場合
            """y_p, y_q, alphas_p, alphas_q = self.y[p], self.y[q], self.alphas[p], self.alphas[q]
            alphas_p_new, alphas_q_new = self._update_alphas(eta, y_p, y_q, alphas_p, alphas_q, E_p, E_q)
            d_p = alphas_p_new - alphas_p
            d_q = alphas_q_new - alphas_q"""
            
            # Rp が代表してalphasの計算をする場合
            y_p, y_q, alphas_p, alphas_q = self.y[p], self.y[q], self.alphas[p], self.alphas[q]
            if rank == Rp:
                alphas_p_new, alphas_q_new = self._update_alphas(eta, y_p, y_q, alphas_p, alphas_q, E_p, E_q)
                d_p = alphas_p_new - alphas_p
                d_q = alphas_q_new - alphas_q
                buf_a[:] = alphas_p_new, alphas_q_new, d_p, d_q
            comm.Bcast(buf_a, root=Rp)
            alphas_p_new, alphas_q_new, d_p, d_q = buf_a
            
            # alphasに更新後の値を代入
            self.alphas[p], self.alphas[q] = alphas_p_new, alphas_q_new
            
            # 目的関数の変化量を計算し，値の推移を保存する配列に追加
            d_obj = d_p * (0.5 * d_p * eta + y_p * (E_p - E_q))
            self.objective_value = np.append(self.objective_value, self.objective_value[-1] + d_obj)
            
            # Eを更新
            #E[self.my_ind] += K[p, self.my_ind] * y_p * d_p + K[q, self.my_ind] * y_q * d_q
            my_E += K[p, :] * y_p * d_p + K[q, :] * y_q * d_q
            
            t_a_end = time.time()
            t_a += t_a_end - t_a_start
        
        if not is_converged:
            print(f"Reached maximum max_iterations. Iterations: {self.max_iterations}")
        
        #-----------------------------------------------------------------------------#
        
        if rank == 0:
            print("", flush=True)
            print(f"t_E = {t_E:.6f}", flush=True)
            print(f"(t_E_1) = ({t_E_1:.6f})", flush=True)
            print(f"(t_E_2) = ({t_E_2:.6f})", flush=True)
            print(f"(t_E_3) = ({t_E_3:.6f})", flush=True)
            print(f"(t_E_4) = ({t_E_4:.6f})", flush=True)
            print(f"t_K = {t_K:.6f}", flush=True)
            print(f"(t_K_1) = ({t_K_1:.6f})", flush=True)
            print(f"(t_K_2) = ({t_K_2:.6f})", flush=True)
            print(f"t_a = {t_a:.6f}", flush=True)
            print("", flush=True)
        

        # サポートベクターのインデントを選択
        self.ind_sv, self.ind_inner = self._get_SV_ind(self.alphas)
        
        # 必要ないインデックスをnanにした後で，Kを共有 (_calculate_bを使う場合)
        """ind_nan = np.where(self.alphas <= self.ME)[0]
        #ind_nan = np.setdiff1d(np.arange(len(self.alphas)), np.concatenate([self.ind_sv, self.ind_inner]))
        K[ind_nan, :] = np.nan
        K[:, ind_nan] = K[ind_nan, :].T
        all_K = comm.allgather(K[:, self.my_ind])
        K = np.hstack(all_K)
        #print(f"rank {rank}, K {K.shape}", flush=True)"""
        
        # Kをそのままスタック (_calculate_bを使う場合)
        #K = np.hstack(comm.allgather(K))
        # ----------------------------------------------


        

        # 重みwを計算
        if self.kernel == 'linear':
            self.w = self._calculate_w(self.X, self.y, self.alphas, self.ind_sv, self.ind_inner)
        else:
            self.w = None

        # バイアスbを計算
        #self.b = self._calculate_b(self.y, self.alphas, K, self.ind_sv, self.ind_inner)
        self.b = self._calculate_b1(self.y, self.alphas, K, start_ind, end_ind, self.ind_sv, self.ind_inner)

        self.f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)
                
                
            

    def fit_L(self, X, y):
        """
        SVMモデルをデータに適合させる関数（個別学習）
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        """
        
        #self.X = copy.deepcopy(X)
        #self.y = copy.deepcopy(y)
        
        # X, y のコピーを作成
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        
        N, d = self.X.shape
        
        print(rank, self.X.shape, self.y.shape, flush=True)
        
        self.alphas = np.zeros(N, dtype=float)
        
        K = np.full((N, N), np.nan, dtype=float)
        E = -y.astype(float)
        
        # Kの計算済みの行に対応するインデックスをTrueとしていく配列
        is_calculated = np.full(N, False, dtype=bool)
                
        self.objective_value = np.array([0])

        #----------------------SMOアルゴリズムによるアルファの学習-----------------------#

        is_converged = False

        for count in range(self.max_iterations):

            # 違反ペアを得る
            p, q, E_p, E_q = self._find_up_min_and_low_max(self.y, self.alphas, E)
            
            # 違反ペアが見つからない場合，学習を終了
            if (E_p > E_q - self.tol):
                is_converged = True
                print(f"    agent{rank} ({len(X)} datas) -> Successfully converged. Iterations: {count}", flush=True)

                break
            
            # カーネル行列のp, q行を計算
            for i in (p, q):
                if is_calculated[i]:
                    continue
                K[i, :] = [self._kernel(self.X[i], x) for x in self.X]
                #K[i] = self._kernel_row(self.X[i], self.X)
                is_calculated[i] = True
            
            # alphas を更新
            eta = K[p, p] - 2 * K[p, q] + K[q, q]
            y_p, y_q, alphas_p, alphas_q = self.y[p], self.y[q], self.alphas[p], self.alphas[q]
            a_p_new, a_q_new = self._update_alphas(eta, y_p, y_q, alphas_p, alphas_q, E_p, E_q)
            
            # alphasの変化量を計算
            d_p, d_q = a_p_new - alphas_p, a_q_new - alphas_q
            
            # alphasに結果を格納
            self.alphas[p], self.alphas[q] = a_p_new, a_q_new
            
            # Eを更新
            E += K[p, :] * y_p * d_p + K[q, :] * y_q * d_q   
            
            # 目的関数の変化量を配列に追加
            d_obj = d_p * (0.5 * d_p * eta + y_p * (E_p - E_q))
            self.objective_value = np.append(self.objective_value, self.objective_value[count] + d_obj)

        if not is_converged:
            print(f"    agent{rank} ({len(X)} datas) -> Reached maximum max_iterations. Iterations: {self.max_iterations}", flush=True)
        
        #-----------------------------------------------------------------------------#

        # サポートベクターのインデックスを獲得
        self.ind_sv, self.ind_inner = self._get_SV_ind(self.alphas)
        
        # 重みwを計算
        if self.kernel == 'linear':
            self.w = self._calculate_w(self.X, self.y, self.alphas, self.ind_sv, self.ind_inner)
        else :
            self.w = None

        # バイアスbを計算
        self.b = self._calculate_b(self.y, self.alphas, K, self.ind_sv, self.ind_inner)
                
        # 識別関数を計算
        self.f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)
            
           
            
    





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
                    f_value += alphas[i] * y[i] * self._kernel(x, X[i])
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
        f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)
        y_pred = np.array([f(x) for x in X])

        return np.where(y_pred >= 0, 1, -1)
    















 
    
    
    def plt_Objective_Values(self, filename):
        """
        目的関数の値をプロットする関数
        """
        plt.figure()
        colors='b'
        plt.plot(self.objective_value_list[rank], c = colors, label= f'Objective_Values')
        #plt.plot(np.sum(self.objective_value_list, axis=0), c = 'black', label= f'sum')

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()
        #plt.show()
        plt.savefig(f"fig/{filename}.pdf")
        
    
   
    
    def plt_Num_Samples(self, filename):
        """
        データ数の推移をプロット(データを削除しない場合の通算個数)
        """
        plt.figure()
        colors=['r', 'b', 'g', 'y', 'm', 'c']
        
        for i in range(size):
            plt.plot(self.num_samples_list[i], linestyle='-', c = colors[i], label=f'Agent {i}')
        # 目盛りを整数値に限定
        #max_iterations = len(self.num_samples_list[0])  # 横軸の最大値
        #plt.xticks(np.arange(0, max_iterations, step=1))  # 0から最大イテレーションまでの整数目盛り

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        plt.xlabel('Roop')
        plt.ylabel('Num of Samples')
        plt.grid(True)
        plt.legend()
        #plt.show()
        plt.savefig(f"fig/{filename}.pdf")
        
    


    def plt_Data_and_Boundary_D(self, filename, x_range=(-0.05, 1.05), y_range=(-0.05, 1.05)):
                
        #データポイントと決定境界をプロットする関数
        
        plt.figure()
        
        num_plots = size
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))  # 横並びにプロット
        
        x_min, x_max = x_range
        y_min, y_max = y_range

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        XY = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = np.array([self.f(xy) for xy in XY]).reshape(xx.shape)
        
        # データをエージェント0に集約する
        X_list = comm.gather(self.X, root=0)
        y_list = comm.gather(self.y, root=0)
        ind_sv_list = comm.gather(self.ind_sv, root=0)
        ind_inner_list = comm.gather(self.ind_inner, root=0)
        ori_ind_list = comm.gather(self.my_ind, root=0)
        
        xx_list = comm.gather(xx, root=0)
        yy_list = comm.gather(yy, root=0)
        Z_list = comm.gather(Z, root=0)
        
        if rank == 0:
            
            for i in range(size):
                ax = axes[i] if num_plots > 1 else axes  # サブプロットが1つの場合の対応
            
                ind_sv, ind_inner, ori_ind = ind_sv_list[i], ind_inner_list[i], ori_ind_list[i]
                X, y = X_list[i], y_list[i]
                X_ori, y_ori = X_list[i][ori_ind], y_list[i][ori_ind]
                X_sv, y_sv = X_list[i][ind_sv], y_list[i][ind_sv]
                X_inner, y_inner = X_list[i][ind_inner], y_list[i][ind_inner]
                xx, yy, Z = xx_list[i], yy_list[i], Z_list[i]
                
                # levelsを50段階の連続にする
                levels = np.linspace(Z.min(), Z.max(), 50)
                
                # フィールドをプロット
                cf = ax.contourf(xx, yy, Z, levels=levels, cmap='coolwarm', alpha=0.6)
                ax.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-')       # 決定境界を黒線で
                ax.contour(xx, yy, Z, levels=[-1, 1], colors='k', linestyles=':')   # f(x) = +-1を点線で

                # データプロット
                ax.scatter(X[y == -1, 0], X[y == -1, 1], c='lightsteelblue', edgecolor='k', s=20)
                ax.scatter(X[y == 1, 0], X[y == 1, 1], c='lightsalmon', edgecolor='k', s=20)
                ax.scatter(X_ori[y_ori == -1, 0], X_ori[y_ori == -1, 1], c='blue', edgecolor='k', s=20, label=f'label : -1')
                ax.scatter(X_ori[y_ori == 1, 0], X_ori[y_ori == 1, 1], c='red', edgecolor='k', s=20, label=f'label : 1')
                ax.scatter(X_sv[:, 0], X_sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
                ax.scatter(X_inner[:, 0], X_inner[:, 1], s=100, facecolors='none', edgecolors='k', marker='^', label='Inner Samples')
    
                #ax.set_xlabel('X1')
                #ax.set_ylabel('X2')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks([])
                ax.set_yticks([])

                ax.set_title(f'Agent {i+1}')
                ax.set_aspect('equal')
        
            #fig.suptitle(f'k = {loop}', fontsize=16)
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
    
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            plt.tight_layout()  
            #plt.tight_layout(rect=[0, 0, 1, 0.85])        
            #plt.show()
            plt.savefig(f"fig/{filename}.pdf")
    
    def plt_Data_and_Boundary_L(self, filename, x_range=(-0.05, 1.05), y_range=(-0.05, 1.05)):
        
        plt.figure()
        
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
        plt.scatter(self.X[self.ind_inner, 0], self.X[self.ind_inner, 1], s=100, facecolors='none', edgecolors='k', marker='^', label='Inner Samples')
        
        #plt.rcParams['pdf.fonttype'] = 42
        #plt.rcParams['ps.fonttype'] = 42

        plt.colorbar(cf)
        plt.xlabel('X1')
        plt.ylabel('X2')
        #plt.xlabel('petal length (cm)')
        #plt.ylabel('petal width (cm)')
        plt.legend()
        #plt.show()
        plt.savefig(f"fig/{filename}.pdf")