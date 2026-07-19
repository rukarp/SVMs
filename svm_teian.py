"""
提案手法用のコード
"""

# SVMのfit関数以外をまとめた親クラス
from svm_base_for_teian import BaseSVM_for_Teian as MySVM_t

# Numpy
import numpy as np

# MPI
from mpi4py import MPI

# リストのコピー用
import copy

# グラフのプロット用
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

buffer_size = 32768

class MySVM_T(MySVM_t):
    
    def __init__(self, max_roop=100, **kwargs):
        super().__init__(**kwargs)
        self.max_roop = max_roop

    
    def fit(self, X, y, A, L):
        """
        SVMモデルをデータに適合させる関数(提案手法)
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        A: 隣接行列(自分の拠点の列だけ)
        L: グラフ直径
        """
        # 隣接行列
        self.A = A

        # グラフ直径
        self.L = L

        # 目的関数の値の確認する回数
        self.history_check = size
        
        # オリジナルデータのラベルの最小値と最大値を取得
        self.min_val = np.min(y)
        self.max_val = np.max(y)
        
        # 初期状態のデータを保存
        self.X_ori = X
        self.y_ori = y
        
        # データの個数を保存する配列
        self.num_samples = np.empty((0, 2), dtype=int)
        
        # データの個数を保存する配列(データ消去をしない場合の通算個数)
        self.num_samples_total = np.empty(0, dtype=int)
        
        # 目的関数の末尾の値を保存する配列
        self.objective_last_value = np.zeros(1, dtype=float)

        self.is_converged = False
        
        
        #------------ alphaの学習(個別に学習してサポートベクターをもらう方法) -------------#
        
        # データの個数を共有し，箱を用意
        num_samples_ori_list = comm.allgather(len(self.X_ori))
        start_ind = sum(num_samples_ori_list[:rank])
        end_ind = start_ind + num_samples_ori_list[rank]
        
        self.my_ind = np.arange(start_ind, end_ind)
        
        self.num_samples_all = sum(num_samples_ori_list)
        
        #print(rank, start_ind, end_ind, num_samples_ori_list, flush=True)
        self.X = np.full((self.num_samples_all, self.X_ori.shape[1]), np.nan, dtype=float)
        self.y = np.zeros(self.num_samples_all, dtype=int)
        self.alphas = np.zeros(self.num_samples_all, dtype=float)
        
        self.X[self.my_ind] = self.X_ori
        self.y[self.my_ind] = self.y_ori
        #print(rank, self.X_all.shape, self.y_all.shape, self.y_all, flush=True)
        
        self.K = np.full((self.num_samples_all, self.num_samples_all), np.nan, dtype=float)
        
        for roop in range(self.max_roop): # ループ回数は仮置き
            
            comm.Barrier()
            if rank == 0:  
                print(f"roop {roop}", flush=True)
            comm.Barrier()

            # 学習データの通算個数を格納
            self.active_all_ind = np.where(~np.isnan(self.X[:, 0]))[0]
            self.num_samples_total = np.append(self.num_samples_total, len(self.active_all_ind))
            
            # 学習に使用するインデックスを獲得し，学習データを作成
            active_ind = np.where(self.y != 0)[0]
            active_X = self.X[active_ind]
            active_y = self.y[active_ind]
            self.num_samples = np.vstack((self.num_samples, np.array([roop, len(active_ind)]))) 

            # 個別に学習して alphas を得る
            self.alphas[:] = 0
            self.alphas[active_ind], self.K[np.ix_(active_ind, active_ind)], self.objective_value = self.fit_independent(active_X, active_y)
            self.ind_sv, self.ind_inner = self._get_SV_ind(self.alphas)

            # 目的関数の末尾の値を保存
            self.objective_last_value = np.append(self.objective_last_value, self.objective_value[-1])

            
            
            # 自分の拠点のデータでない且つalphasの値が0になったものを非アクティブ化する
            ind_zero_alphas = np.where(self.alphas <= self.ME)[0]                        
            self.y[ind_zero_alphas] = 0
            self.y[self.my_ind] = self.y_ori
                        
            # データ削除後の学習データの個数
            self.num_samples = np.vstack((self.num_samples, np.array([roop, len(np.where(self.y != 0)[0])])))             
            
            
            
            
            

            # 目的関数の値がsize回変化なかったか確認
            history_obj = self.objective_last_value[-self.L:]
            my_converged = all(obj == history_obj[0] for obj in history_obj)
            all_converged = comm.allgather(my_converged)
            self.is_converged = all(all_converged[i] == True for i in range(size))
            #print(rank, self.is_converged, history_obj, flush=True)


            comm.Barrier()  
            if rank == 0:  
                print("", flush=True)
            comm.Barrier()

            # 終了判定
            if (self.is_converged == True) or (roop == self.max_roop - 1):
                
                # 重みwを計算
                if self.kernel == 'linear':
                    self.w = self._calculate_w(self.X, self.y, self.alphas, self.ind_sv, self.ind_inner)
                else:
                    self.w = None

                # バイアスbを計算
                self.b = self._calculate_b(self.y, self.alphas, self.K, self.ind_sv, self.ind_inner)
                
                # 識別関数を計算
                self.f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)   
                
                self.roop = roop

                break  


            

            
            
            # alphasが0より大きくなるインデックスに対応するX, yを獲得
            ind_active_alphas = np.concatenate((self.ind_sv, self.ind_inner))
            X_sv_inner = self.X[ind_active_alphas]
            y_sv_inner = self.y[ind_active_alphas]
                            
                        
            # データ一組（特徴量1行 + ラベル1つ + インデント番号1つ）のバイトサイズを計算
            data_unit_size = (X_sv_inner.itemsize * X_sv_inner.shape[1]) + y_sv_inner.itemsize + ind_active_alphas.itemsize
            # チャンクサイズを計算（バッファサイズ内に収まるデータ一組の数）
            chunk_size = buffer_size // data_unit_size - 1
            # サポートベクターのチャンク数を計算
            num_chunks = (len(X_sv_inner) + chunk_size - 1) // chunk_size

            # データ送信・受信の実装
            for neighbor in range(size):
                if self.A[neighbor] == 1:
                            
                    # 送信するチャンク数を相手に送信
                    req_send_chunks = comm.isend(num_chunks, dest=neighbor)
                    req_recv_chunks = comm.irecv(source=neighbor)
                    recv_num_chunks = req_recv_chunks.wait()  # 受信するチャンク数
                    req_send_chunks.wait()

                    # 各チャンクを送信
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, len(X_sv_inner))
            
                        req_send = comm.isend((X_sv_inner[start:end], y_sv_inner[start:end], ind_active_alphas[start:end]), dest=neighbor)
                        req_send.wait()

                    # 各チャンクを受信し，データを追加
                    for i in range(recv_num_chunks):
                        req_recv = comm.irecv(source=neighbor)
                        recv_X, recv_y, recv_ind = req_recv.wait()

                        self.X[recv_ind] = recv_X
                        self.y[recv_ind] = recv_y

           
            




    

        
        
def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # ---------- iris ----------
    X_train, Y_train = ir.X4_5_train, ir.Y4_5_train
    X_test, Y_test = ir.X_test, ir.Y_test
    # --------------------------
    
    # --------- cancer ---------
    #X_train, Y_train = ca.X6_5_train, ca.Y6_5_train
    #X_test, Y_test = ca.X_test, ca.Y_test
    # --------------------------

    # ---------- adult ---------
    #X_train, Y_train = ad.X6_5_train, ad.Y6_5_train
    #X_test, Y_test = ad.X_test, ad.Y_test
    # --------------------------
    
    # -------- airline ---------
    #X_train, Y_train = ai.X6_5_train, ai.Y6_5_train
    #X_test, Y_test = ai.X_test, ai.Y_test
    # --------------------------

    # -------- 隣接行列 --------
    A = np.array([[0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0]])
    L = 2
    
    """A = np.array([[0, 1, 0, 0],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0]])
    L = 4"""
    
    """A = np.array([[0, 1, 1, 1, 1, 1],
                  [1, 0, 1, 1, 1, 1],
                  [1, 1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 1, 1],
                  [1, 1, 1, 1, 0, 1],
                  [1, 1, 1, 1, 1, 0]])
    L = 2"""
    
    """A = np.array([[0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0]])
    L = 6"""
    # -------------------------

    # --- カーネルを指定してインスタンスを生成 ---
    mysvm = MySVM_T(kernel = 'linear', C = 10)
    #mysvm = MySVM_T(kernel = 'poly', degree = 2, coef0 = 1.0, C = 1)
    #mysvm = MySVM_T(kernel = 'rbf', gamma = 1, C = 1)
    #mysvm = MySVM_T(kernel = 'sigmoid', gamma = 1.0, coef0 = 1.0, C = 1.0)
    # -----------------------------------------




    # --------------------------------------------------------
    
    #rankごとにデータを取得
    if rank < size:
        X_train = X_train[rank]
        Y_train = Y_train[rank]
        A = A[rank]
    else:
        X_train, Y_train, A = None, None, None
        
        
        
    if rank == 0:
        print("---------- Distributed Learning ----------\n", flush=True)
        t_start = time.time()
    comm.Barrier()
    
    # 学習(分散)
    mysvm.fit(X_train, Y_train, A, L)

    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        #print(f"fitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()

    # 目的関数の値を Agent0 に集約
    mysvm.objective_last_value_list = comm.gather(mysvm.objective_last_value, root=0)
    mysvm.num_samples_list = comm.gather(mysvm.num_samples, root=0)
    mysvm.num_samples_total_list = comm.gather(mysvm.num_samples_total, root=0)
    
    # テスト(分散)
    Y_pred = mysvm.predict(X_test)
    accuracy = np.mean(Y_pred == Y_test)
    f1 = f1_score(Y_pred, Y_test)

    # 目的関数値の値を再計算 -------------
    L1, L2 = 0, 0
    for i in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
        L2 -= mysvm.alphas[i]
        for j in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
            L1 += mysvm.alphas[i] * mysvm.alphas[j] * mysvm.y[i] * mysvm.y[j] * mysvm.K[i, j]
    Obj = 0.5 * L1 + L2

    Obj_list = comm.gather(Obj, root=0)
    # ----------------------------------
    
    accuracy_list = comm.gather(accuracy, root=0)
    f1_list = comm.gather(f1, root=0)
    
    for i in range(size):
        if rank == i:
            print(f'Agent {i}', flush=True)
            print(f'SV_len: {len(mysvm.ind_sv)}', flush=True)
            print(f'INNER_len: {len(mysvm.ind_inner)}', flush=True)
            print(f'w: {mysvm.w}', flush=True)
            print(f'b: {mysvm.b}', flush=True)
            #print(f'alpahs: {mysvm.alphas}', flush=True)
            #print(' ')
            #Y_pred = mysvm.predict(X_test)
            #print(f'Y_test: {Y_test}', flush=True)
            #print(f'Y_predict: {Y_pred}', flush=True)
            #accuracy = np.mean(Y_pred == Y_test)
            print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
            #f1 = f1_score(Y_pred, Y_test)
            print(f'F1: {f1 * 100:.2f}%\n', flush=True)
        
        comm.Barrier()
    
    # 受け取ったデータの中でalpha=0となるデータ(余計なデータを獲得)
    exclude_indices = np.concatenate((mysvm.ind_sv, mysvm.ind_inner, mysvm.my_ind))
    ind_ex = mysvm.active_all_ind[~np.isin(mysvm.active_all_ind, exclude_indices)]
    
    """# D_m 以外のデータでalpha=0で収束したデータ数 -------------------
    excluded_indices1 = np.setdiff1d(mysvm.active_all_ind, mysvm.my_ind)
    ind_ex_not_Dm = np.setdiff1d(excluded_indices1, np.concatenate((mysvm.ind_sv, mysvm.ind_inner)))"""
 
    
    # サポートベクター等の数を共有
    ind_sv_len_list = comm.gather(len(mysvm.ind_sv), root=0)
    ind_inner_len_list = comm.gather(len(mysvm.ind_inner), root=0)
    ind_ex_len_list = comm.gather(len(ind_ex), root=0)
    #ind_ex_not_Dm_len_list = comm.gather(len(ind_ex_not_Dm), root=0)
    
    # 各データについて確認
    if rank == 0:
        print(f"fitting time : {t_end - t_start}", end='', flush=True)
        print(f"\n\nroop : {mysvm.roop + 1}", end='', flush=True)
        
        print(f"\n\naccuracy(%)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{accuracy_list[i] * 100:.2f}, ", end='', flush=True)
        #print(f"\n\nf1(%)\n    ", end='', flush=True)
        #for i in range(size):
        #    print(f"{f1_list[i] * 100:.2f}, ", end='', flush=True)
        print(f"\n\nobjective_last_values\n    ", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.objective_last_value_list[i][-1]}, ", end='', flush=True)
        print(f"\n\nlen(ind_sv)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_sv_len_list[i]}, ", end='', flush=True)
        print(f"\n\nlen(ind_inner)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_inner_len_list[i]}, ", end='', flush=True)
        print(f"\n\nnum_samples_original\n    ", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_list[i][0, 1]}, ", end='', flush=True)
        print(f"\n\nnum_samples_end\n    ", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_list[i][-1, 1]}, ", end='', flush=True)
        print(f"\n\nnum_samples_total\n    ", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_total_list[i][-1]}, ", end='', flush=True)
        print(f"\n\nlen(ind_ex)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_ex_len_list[i]}, ", end='', flush=True)
        """print(f"\n\nlen(ind_ex_not_Dm)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_ex_not_Dm_len_list[i]}, ", end='', flush=True)"""
        print("\n\n", flush=True)
        
        
        
        
        # Excel貼り付け用 ---------------------------------------------------------
        print(f"print all with [,]", flush=True)
        print(f"{t_end - t_start},{mysvm.roop + 1},", end='', flush=True)        
        for i in range(size):
            print(f"{accuracy_list[i] * 100:.2f},", end='', flush=True)
        for i in range(size):
            #print(f"{mysvm.objective_last_value_list[i][-1]},", end='', flush=True)
            print(f"{Obj_list[i]},", end='', flush=True)
        for i in range(size):
            print(f"{ind_sv_len_list[i]},", end='', flush=True)
        for i in range(size):
            print(f"{ind_inner_len_list[i]},", end='', flush=True)
        for i in range(size):
            print(f"{ind_sv_len_list[i] + ind_inner_len_list[i]},", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_list[i][0, 1]},", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_list[i][-1, 1]},", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.num_samples_total_list[i][-1]},", end='', flush=True)
        for i in range(size):
            print(f"{ind_ex_len_list[i]},", end='', flush=True)
        """for i in range(size):
            print(f"{ind_ex_not_Dm_len_list[i]},", end='', flush=True)"""
        print("\n\n", flush=True)
        # -------------------------------------------------------------------------
        
        # 目的関数の末尾の値の推移，データの個数の推移をプロット
        mysvm.plt_Objective_Last_Values()
        mysvm.plt_Num_Samples()
        mysvm.plt_Num_Samples_Total()
    comm.Barrier()
    
    mysvm.plt_Data_and_Boundary()

    
    

if __name__ == "__main__":
    main()