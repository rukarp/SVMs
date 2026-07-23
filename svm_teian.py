"""
提案手法用のコード
"""

# SVMのfit関数以外をまとめた親クラス
from svm_base_for_teian import BaseSVM_for_Teian as MySVM_T

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




    

        
        
def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # ---------- iris ----------
    X_train, Y_train = ir.X6_5_train, ir.Y6_5_train
    X_test, Y_test = ir.X_test, ir.Y_test
    # --------------------------
    
    # --------- cancer ---------
    X_train, Y_train = ca.X6_5_train, ca.Y6_5_train
    X_test, Y_test = ca.X_test, ca.Y_test
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
    
    A = np.array([[0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0]])
    L = 6
    # -------------------------

    # --- カーネルを指定してインスタンスを生成 ---
    mysvm = MySVM_T(kernel = 'linear', C = 1)
    #mysvm = MySVM_T(kernel = 'poly', degree = 2, coef0 = 1.0, C = 1)
    #mysvm = MySVM_T(kernel = 'rbf', gamma = 1, C = 1)
    #mysvm = MySVM_T(kernel = 'sigmoid', gamma = 1.0, coef0 = 1.0, C = 1.0)
    # -----------------------------------------


    plt = True
    plt = False

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
        mysvm.plt_Objective_Last_Values("plt_Objective_Last_Values")
        mysvm.plt_Num_Samples("plt_Num_Samples")
        mysvm.plt_Num_Samples_Total("plt_Num_Samples_Total")
    comm.Barrier()
    
    if plt == True:
        mysvm.plt_Data_and_Boundary_D("plt_D")

    
    

if __name__ == "__main__":
    main()