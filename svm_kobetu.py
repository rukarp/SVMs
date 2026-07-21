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

# 時間計測用
import time

# f1スコア計算用
from sklearn.metrics import f1_score

# 学習データ
import data_iris as ir
import data_cancer as ca
import data_adult as ad

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

buffer_size = 32768

SAVE_DIR = "fig/svm"

class MySVM_K(MySVM_t):
            
    def fit(self, X, y):
        """
        SVMモデルをデータに適合させる関数(個別学習)
        パラメータ:
        X (array-like): トレーニングデータの特徴量のリスト
        y (array-like): トレーニングデータのラベルのリスト
        """
        
        # オリジナルデータのラベルの最小値と最大値を取得
        self.min_val = np.min(y)
        self.max_val = np.max(y)
        
        # X, y のコピーを作成
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        
        # ------------------------ 学習 -------------------------
        
        # 個別に学習して alphas を得る
        self.alphas, self.K, self.objective_value = self.fit_independent(self.X, self.y)
        self.ind_sv, self.ind_inner = self._get_SV_ind(self.alphas)
        
        # 重みwを計算
        if self.kernel == 'linear':
            self.w = self._calculate_w(self.X, self.y, self.alphas, self.ind_sv, self.ind_inner)
        else :
            self.w = None

        # バイアスbを計算
        self.b = self._calculate_b(self.y, self.alphas, self.K, self.ind_sv, self.ind_inner)
                
        # 識別関数を計算
        self.f = self.make_decision_func(self.X, self.y, self.alphas, self.w, self.b, self.ind_sv, self.ind_inner)        
       
       
       
    
    
    def plt_Objective_Values(self, filename):
        """
        目的関数の値をプロットする関数
        """
        plt.figure()
        colors=['r', 'b', 'g', 'y', 'm', 'c']
        for i in range(len(self.objective_value_list)):
            plt.plot(self.objective_value_list[i], c = colors[i], label= f'Agent {i}')
            #plt.plot(np.sum(self.objective_value_list, axis=0), c = 'black', label= f'sum')

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()
        #plt.show()
        plt.savefig(f"{SAVE_DIR}/{filename}.pdf")

        
    
        
def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # ---------- iris ----------
    X_train, Y_train = ir.X6_5_train, ir.Y6_5_train
    X_test, Y_test = ir.X_test, ir.Y_test
    # --------------------------
    
    # --------- cancer ---------
    X_train, Y_train = ca.X6_1_train, ca.Y6_1_train
    X_test, Y_test = ca.X_test, ca.Y_test
    # --------------------------

    # ---------- adult ---------
    #X_train, Y_train = ad.X6_1_train, ad.Y6_1_train
    #X_test, Y_test = ad.X_test, ad.Y_test
    # --------------------------




    # -------------------------

    # --- カーネルを指定してインスタンスを生成 ---
    mysvm = MySVM_K(kernel = 'linear', C = 1)
    #mysvm = MySVM(kernel = 'poly', degree = 2, coef0 = 1.0, C = 1)
    #mysvm = MySVM(kernel = 'rbf', gamma = 1, C = 1)
    #mysvm = MySVM(kernel = 'sigmoid', gamma = 1.0, coef0 = 1.0, C = 1.0)
    # -----------------------------------------




    # --------------------------------------------------------
    
    #rankごとにデータを取得
    if rank < size:
        X_train = X_train[rank]
        Y_train = Y_train[rank]
    else:
        X_train, Y_train = None, None
        
    mysvm.my_ind = np.arange(len(Y_train))
    
    if rank == 0:
        print("---------- Individual Learning ----------\n", flush=True)
        t_start = time.time()
    comm.Barrier()
    
    # オリジナルデータのラベルの最小値と最大値を取得
    mysvm.min_val = np.min(Y_train)
    mysvm.max_val = np.max(Y_train)
    
    # プロット用にmy_indを設定
    mysvm.my_ind = np.arange(len(Y_train))
    
    # 学習(分散，SV探し用)
    mysvm.fit_each(X_train, Y_train)

    comm.Barrier()
    if rank == 0:
        t_end = time.time()
        #print(f"fitting time : {t_end - t_start}\n", flush=True)
    comm.Barrier()

    # 目的関数の値を Agent0 に集約
    mysvm.objective_value_list = comm.gather(mysvm.objective_value, root=0) 
    
    # テスト(個別)
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
            #print(' ')
            #Y_pred = mysvm.predict(X_test)
            #print(f'Y_test: {Y_test}', flush=True)
            #print(f'Y_predict: {Y_pred}', flush=True)
            #accuracy = np.mean(Y_pred == Y_test)
            print(f'Accuracy: {accuracy * 100:.2f}%', flush=True)
            #f1 = f1_score(Y_pred, Y_test)
            print(f'F1: {f1 * 100:.2f}%', flush=True)
            print(' ', flush=True)
        
        comm.Barrier()
    
    # サポートベクター等の数を共有
    ind_sv_len_list = comm.gather(len(mysvm.ind_sv), root=0)
    ind_inner_len_list = comm.gather(len(mysvm.ind_inner), root=0)
     
    # 各データについて確認
    if rank == 0:
        print(f"fitting time : {t_end - t_start}", end='', flush=True)
        
        print(f"\n\naccuracy(%)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{accuracy_list[i] * 100:.2f}, ", end='', flush=True)
        #print(f"\n\nf1(%)\n    ", end='', flush=True)
        #for i in range(size):
        #    print(f"{f1_list[i] * 100:.2f}, ", end='', flush=True)
        print(f"\n\nobjective_last_values\n    ", end='', flush=True)
        for i in range(size):
            print(f"{mysvm.objective_value_list[i][-1]}, ", end='', flush=True)
        print(f"\n\nlen(ind_sv)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_sv_len_list[i]}, ", end='', flush=True)
        print(f"\n\nlen(ind_inner)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_inner_len_list[i]}, ", end='', flush=True)
        print(f"\n\nlen(ind_inner)\n    ", end='', flush=True)
        for i in range(size):
            print(f"{ind_inner_len_list[i]}, ", end='', flush=True)
        print("\n\n", flush=True)
        
        
        
        # Excel貼り付け用 ---------------------------------------------------------
        print(f"print all with [,]", flush=True)
        print(f"{t_end - t_start},", end='', flush=True)        
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
        print("\n\n", flush=True)
        # -------------------------------------------------------------------------
    
        # 目的関数の値の推移をプロット
        mysvm.plt_Objective_Values("plt_Objective_Values_each")
    comm.Barrier()
    
    mysvm.plt_Data_and_Boundary_D("plt_D_each")
    

if __name__ == "__main__":
    main()