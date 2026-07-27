"""
提案手法用のコード

疑似データを作成し，それを使って特別研究の提案手法を行う
"""

# SVMのfit関数以外をまとめた親クラス
from svm_base_for_teian import BaseSVM_for_Teian as MySVM_t
from svm_base_for_gizi import BaseSVM_for_Gizi as MySVM_g

# Numpy
import numpy as np

# MPI
from mpi4py import MPI
import socket

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
host = socket.gethostname()

buffer_size = 32768

class MySVM_TG(MySVM_t, MySVM_g):

    def calculate_SV_RMSE(self, Y_pred_ori_no_int, Y_pred_no_int):
        # 標準偏差を計算
        std_f = np.std(Y_pred_ori_no_int)
        if std_f < 1e-16:
            std_f += 1e-12
        
        Y_pred_ori = np.where(Y_pred_ori_no_int >= 0, 1, -1)
        Y_pred = np.where(Y_pred_no_int >= 0, 1, -1)

        SA = np.mean(Y_pred_ori == Y_pred)
        
        rmse = np.sqrt(np.mean((Y_pred_ori_no_int - Y_pred_no_int) ** 2))
        nrmse = rmse / std_f
        
        mae = np.mean(np.abs(Y_pred_ori_no_int - Y_pred_no_int))
        nmae = mae / std_f

        return SA, nrmse, nmae, np.max(Y_pred_no_int), np.min(Y_pred_no_int)    
    
    def calculate_w_cos(self, w_ori, w):
        
        # コサイン類似度
        cos_theta = np.dot(w_ori, w) / (np.linalg.norm(w_ori) * np.linalg.norm(w))
        # 丸め誤差対策
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        # 角度（ラジアン）
        theta = np.arccos(cos_theta)
        # 角度（度）
        theta_deg = np.degrees(theta)
        
        return cos_theta, theta, theta_deg
    



    

    
    
        
        
        
def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = socket.gethostname()

    # MPIのrankとhostを表示する
    for i in range(size):
        if rank == i:
            print(f"rank={rank}, host={host}")
        comm.Barrier()
    
    # ---------- iris ----------
    X_train, Y_train = ir.X6_5_train, ir.Y6_5_train
    X_test, Y_test = ir.X_test, ir.Y_test
    # --------------------------
    
    # --------- cancer ---------
    X_train, Y_train = ca.X6_5_train, ca.Y6_5_train
    X_test, Y_test = ca.X_test, ca.Y_test
    # --------------------------

    # ---------- adult ---------
    X_train, Y_train = ad.X6_5_train, ad.Y6_5_train
    X_test, Y_test = ad.X_test, ad.Y_test
    # --------------------------
    
    # -------- airline ---------
    #X_train, Y_train = ai.X6_5_train, ai.Y6_5_train
    #X_test, Y_test = ai.X_test, ai.Y_test
    # --------------------------

    # -------- 隣接行列 --------
    """A = np.array([[0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0]])
    L = 2"""
    
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
    
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0]])
    L = 6
    # -------------------------

    # -------- カーネルおよびパラメータ設定 ---------
    kernel = 'linear'
    #kernel = 'poly'
    #kernel = 'rbf'
    #kernel = 'sigmoid'
    
    C = 1
    gamma = .1
    
    degree = 2
    coef0 = 1
    
    plt = True
    plt = False
    # -----------------------------------------
    
    # ----- ノイズの半径の範囲を指定 (最小値, 最大値) -----
    radius = (0.1, 0.2)
    # --------------------------------------------------

    # --- カーネルを指定してインスタンスを生成 ---
    if kernel == 'linear':
        mysvm = MySVM_TG(kernel = 'linear', C = C)
    elif kernel == 'poly':
        mysvm = MySVM_TG(kernel = 'poly', degree = degree, coef0 = coef0, C = C)
    elif kernel == 'rbf':
        mysvm = MySVM_TG(kernel = 'rbf', gamma = gamma, C = C)
    elif kernel == 'sigmoid':
        mysvm = MySVM_TG(kernel = 'sigmoid', gamma = gamma, coef0 = coef0, C = C)
    # -----------------------------------------


    # --------------------------------------------------------
    
    #rankごとにデータを取得
    if rank < size:
        X_train = X_train[rank]
        Y_train = Y_train[rank]
        A = A[rank]
    else:
        X_train, Y_train, A = None, None, None
    
    # ----------------------
    mysvm.C = C
    # ----------------------

    # オリジナルデータのラベルの最小値と最大値を取得
    mysvm.min_val = np.min(Y_train)
    mysvm.max_val = np.max(Y_train)


    X_train_all = comm.gather(X_train, root=0)
    Y_train_all = comm.gather(Y_train, root=0)
    
    # 学習(集約，オリジナルデータ)
    if rank == 0:
        X_train_all = np.vstack(X_train_all)
        Y_train_all = np.concatenate(Y_train_all)

        print("---------- Centralized Learning ----------", flush=True) 
        
        X_low_len = np.count_nonzero(Y_train_all == mysvm.min_val)
        X_up_len = np.count_nonzero(Y_train_all == mysvm.max_val)
        print(f"low: {X_low_len} samples, up: {X_up_len} samples\n", flush=True)
        
        t_start = time.time()      
        
        mysvm.fit_each(X_train_all, Y_train_all)
        
        t_end = time.time()
        print(f"\nfitting time : {t_end - t_start}\n", flush=True)
        
        Y_pred_no_int = np.array([mysvm.f(x) for x in X_test])
        Y_pred = np.where(Y_pred_no_int >= 0, 1, -1) 
        accuracy = np.mean(Y_pred == Y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
        f1 = f1_score(Y_pred, Y_test)
        print(f'F1: {f1 * 100:.2f}%', flush=True)
                
        print(f'[MIN, MAX]: [{np.min(Y_pred_no_int):.6f}, {np.max(Y_pred_no_int):.6f}]\n', flush=True)


        if plt == True:
            mysvm.plt_Data_and_Boundary_L("plt_L_original")

    else:
        Y_pred_no_int = None
        mysvm.w = None
    comm.Barrier()

    Y_pred_ori_no_int = comm.bcast(Y_pred_no_int, root=0)
    w_ori = comm.bcast(mysvm.w, root=0)

    #print(f"{rank}: {Y_pred_no_int}", flush=True)
    #print(f"{rank}: {Y_pred_ori_no_int}", flush=True)



    

    if rank == 0:
        print("\n---------- Distributed Learning----------", flush=True)
    comm.Barrier()
    
    X_low_len = np.count_nonzero(Y_train == mysvm.min_val)
    X_up_len = np.count_nonzero(Y_train == mysvm.max_val)
    for i in range(size):
        if rank == i:
            print(f'Agent {i} ', end="", flush=True)
            print(f"low: {X_low_len} samples, up: {X_up_len}samples", flush=True)
        comm.Barrier()
    
    if rank == 0:
        print("", flush=True)
        t_start = time.time()
    comm.Barrier()
    
    # 学習(分散)
    mysvm.fit(X_train, Y_train, A, L)

    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        print(f"fitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()

    # 目的関数の値を Agent0 に集約
    mysvm.objective_last_value_list = comm.gather(mysvm.objective_last_value, root=0)
    mysvm.num_samples_list = comm.gather(mysvm.num_samples, root=0)
    mysvm.num_samples_total_list = comm.gather(mysvm.num_samples_total, root=0)
    
    # テスト(分散)
    Y_pred_no_int = np.array([mysvm.f(x) for x in X_test])
    Y_pred = np.where(Y_pred_no_int >= 0, 1, -1)  
    accuracy = np.mean(Y_pred == Y_test)
    f1 = f1_score(Y_pred, Y_test)
    SA, nrmse, nmae, max_f, min_f = mysvm.calculate_SV_RMSE(Y_pred_ori_no_int, Y_pred_no_int)
    if mysvm.kernel == "linear":
        cos_theta, theta, theta_deg = mysvm.calculate_w_cos(w_ori, mysvm.w)

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
            #print(f'w: {mysvm.w}', flush=True)
            #print(f'b: {mysvm.b}', flush=True)
            #print(f'alpahs: {mysvm.alphas}', flush=True)
            #print(' ')
            #Y_pred = mysvm.predict(X_test)
            #print(f'Y_test: {Y_test}', flush=True)
            #print(f'Y_predict: {Y_pred}', flush=True)
            #accuracy = np.mean(Y_pred == Y_test)
            print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
            #f1 = f1_score(Y_pred, Y_test)
            print(f'F1: {f1 * 100:.2f}%', flush=True)
            print(f'SA: {SA* 100:.2f}%', flush=True)
            print(f'NRMSE: {nrmse:.6f}', flush=True)
            print(f'NMAE: {nmae:.6f}', flush=True)
            print(f'[MIN, MAX]: [{min_f:.6f}, {max_f:.6f}]', flush=True)
            if mysvm.kernel == "linear":
                print(f'cos: {cos_theta:.6f}', flush=True)
                print(f'angle: {theta_deg:.6f} deg ({theta:.6f} rad)', flush=True)
            print('', flush=True)
        
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
        # 目的関数の末尾の値の推移，データの個数の推移をプロット
        mysvm.plt_Objective_Last_Values("plt_Objective_Last_Values_original")
        mysvm.plt_Num_Samples("plt_Num_Samples_original")
        mysvm.plt_Num_Samples_Total("plt_Num_Samples_Total_original")
    comm.Barrier()
    
    if plt == True:
        mysvm.plt_Data_and_Boundary_D("plt_D_original")
    comm.Barrier()



    # プロット用にmy_indを設定（X_train全部がmy_ind）
    mysvm.my_ind = list(range(len(X_train)))

    if rank == 0:
        print("\n-------- Original Learning ----------", flush=True)
        t_start = time.time()
    comm.Barrier()
    # 学習(分散，SV探し用)
    mysvm.fit_each(X_train, Y_train)
    
    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        print(f"\nfitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()
    
    # -----識別関数の2乗の勾配(∇(f(x))^2)を求める -----
    mysvm.grad_f = mysvm.make_gradient_f(mysvm.X, mysvm.y, mysvm.alphas, mysvm.w, mysvm.ind_sv, mysvm.ind_inner)

    mysvm.f_squared = mysvm.make_decision_func_squared(mysvm.X, mysvm.y, mysvm.alphas, mysvm.w, mysvm.b, mysvm.ind_sv, mysvm.ind_inner)
    mysvm.grad_f_squared = mysvm.make_gradient_f_squared(mysvm.f, mysvm.grad_f)
    # ----------------------------------------------
    
    if plt == True:
        mysvm.plt_Data_and_Boundary_D("plt_D_original_each")
    comm.Barrier()



    
    # 疑似データを作成
    comm.Barrier()
    if rank == 0:
        start = time.time()
        
        
        
    comm.Barrier()
    #X_train, Y_train = mysvm.make_fake_data(X_train, mysvm.ind_sv, lr = 0.01, bounds_eps=1e-6, max_iter=100)
    #X_train, Y_train = mysvm.make_fake_data_random(X_train, Y_train, radius = radius, max_retry = 10000)
    #X_train, Y_train = mysvm.make_fake_data_random_with_margin(X_train, Y_train, mysvm.alphas, radius = radius, max_retry = 10000)
    X_train, Y_train = mysvm.make_fake_data_KKT(X_train, Y_train, mysvm.alphas, radius = radius, max_retry = 10000)
    comm.Barrier()
    
    
    
    if rank == 0:
       end = time.time()
       print(f"make_fake_data time: {end - start}", flush=True)
    comm.Barrier()

    # ----------------------
    mysvm.C = C
    # ----------------------

    X_train_all = comm.gather(X_train, root=0)
    Y_train_all = comm.gather(Y_train, root=0)
    
    # 学習(集約，疑似データ)
    if rank == 0:
        X_train_all = np.vstack(X_train_all)
        Y_train_all = np.concatenate(Y_train_all)

        print("\n---------- Centralized Learning(Pseudo-Data) ----------", flush=True) 
        
        X_low_len = np.count_nonzero(Y_train_all == mysvm.min_val)
        X_up_len = np.count_nonzero(Y_train_all == mysvm.max_val)
        print(f"low: {X_low_len} samples, up: {X_up_len}samples\n", flush=True)
                       
        t_start = time.time()      
        
        mysvm.fit_each(X_train_all, Y_train_all)
        
        t_end = time.time()
        print(f"\nfitting time : {t_end - t_start}\n", flush=True)
        
        Y_pred_no_int = np.array([mysvm.f(x) for x in X_test])
        Y_pred = np.where(Y_pred_no_int >= 0, 1, -1) 
        
        accuracy = np.mean(Y_pred == Y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
        f1 = f1_score(Y_pred, Y_test)
        print(f'F1: {f1 * 100:.2f}%', flush=True)
        
        SA, nrmse, nmae, max_f, min_f = mysvm.calculate_SV_RMSE(Y_pred_ori_no_int, Y_pred_no_int)
        if mysvm.kernel == "linear":
            cos_theta, theta, theta_deg = mysvm.calculate_w_cos(w_ori, mysvm.w)

        print(f'SA: {SA* 100:.2f}%', flush=True)
        print(f'NRMSE: {nrmse:.6f}', flush=True)
        print(f'NMAE: {nmae:.6f}', flush=True)
        print(f'[MIN, MAX]: [{min_f:.6f}, {max_f:.6f}]', flush=True)
        if mysvm.kernel == "linear":
            print(f'cos: {cos_theta:.6f}', flush=True)
            print(f'angle: {theta_deg:.6f} deg ({theta:.6f} rad)', flush=True)
        print('', flush=True)
            
        if plt == True:
            mysvm.plt_Data_and_Boundary_L("plt_L_pseudo")
    comm.Barrier()



    
    if rank == 0:
        print("\n---------- Original Learning(Pseudo-Data) ----------", flush=True)
    comm.Barrier()
    
    X_low_len = np.count_nonzero(Y_train == mysvm.min_val)
    X_up_len = np.count_nonzero(Y_train == mysvm.max_val)
    for i in range(size):
        if rank == i:
            print(f'Agent {i} ', end="", flush=True)
            print(f"low: {X_low_len} samples, up: {X_up_len}samples", flush=True)
        comm.Barrier()
    
    if rank == 0:
        print("", flush=True)
        t_start = time.time()
    comm.Barrier()
    

    # 学習(個別，疑似データ)
    mysvm.my_ind = list(range(len(X_train)))
    
    mysvm.fit_each(X_train, Y_train)
    
    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        print(f"\nfitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()
        
    if plt == True:
        mysvm.plt_Data_and_Boundary_D("plt_D_pseudo_each")
    comm.Barrier()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    if rank == 0:
        print("\n---------- Distributed Learning(Pseudo-Data)----------", flush=True)
    comm.Barrier()
    
    X_low_len = np.count_nonzero(Y_train == mysvm.min_val)
    X_up_len = np.count_nonzero(Y_train == mysvm.max_val)
    for i in range(size):
        if rank == i:
            print(f'Agent {i} ', end="", flush=True)
            print(f"low: {X_low_len} samples, up: {X_up_len}samples", flush=True)
        comm.Barrier()
    
    if rank == 0:
        print("", flush=True)
        t_start = time.time()
    comm.Barrier()
    
    # 学習(分散)
    mysvm.fit(X_train, Y_train, A, L)

    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        print(f"fitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()

    # 目的関数の値を Agent0 に集約
    mysvm.objective_last_value_list = comm.gather(mysvm.objective_last_value, root=0)
    mysvm.num_samples_list = comm.gather(mysvm.num_samples, root=0)
    mysvm.num_samples_total_list = comm.gather(mysvm.num_samples_total, root=0)
    
    # テスト(分散)
    Y_pred_no_int = np.array([mysvm.f(x) for x in X_test])
    Y_pred = np.where(Y_pred_no_int >= 0, 1, -1)
    accuracy = np.mean(Y_pred == Y_test)
    f1 = f1_score(Y_pred, Y_test)
    SA, nrmse, nmae, max_f, min_f = mysvm.calculate_SV_RMSE(Y_pred_ori_no_int, Y_pred_no_int)
    if mysvm.kernel == "linear":
        cos_theta, theta, theta_deg = mysvm.calculate_w_cos(w_ori, mysvm.w)

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
            #print(f'w: {mysvm.w}', flush=True)
            #print(f'b: {mysvm.b}', flush=True)
            #print(f'alpahs: {mysvm.alphas}', flush=True)
            #print(' ')
            #Y_pred = mysvm.predict(X_test)
            #print(f'Y_test: {Y_test}', flush=True)
            #print(f'Y_predict: {Y_pred}', flush=True)
            #accuracy = np.mean(Y_pred == Y_test)
            print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
            #f1 = f1_score(Y_pred, Y_test)
            print(f'F1: {f1 * 100:.2f}%', flush=True)
            print(f'SA: {SA* 100:.2f}%', flush=True)
            print(f'NRMSE: {nrmse:.6f}', flush=True)
            print(f'NMAE: {nmae:.6f}', flush=True)
            print(f'[MIN, MAX]: [{min_f:.6f}, {max_f:.6f}]', flush=True)
            if mysvm.kernel == "linear":
                print(f'cos: {cos_theta:.6f}', flush=True)
                print(f'angle: {theta_deg:.6f} deg ({theta:.6f} rad)', flush=True)
            print('', flush=True)
        
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
        # 目的関数の末尾の値の推移，データの個数の推移をプロット
        mysvm.plt_Objective_Last_Values("plt_Objective_Last_Values_pseudo")
        mysvm.plt_Num_Samples("plt_Num_Samples_pseudo")
        mysvm.plt_Num_Samples_Total("plt_Num_Samples_Total_pseudo")
    comm.Barrier()
    
    if plt == True:
        mysvm.plt_Data_and_Boundary_D("plt_D_pseudo")

    
    

if __name__ == "__main__":
    main()