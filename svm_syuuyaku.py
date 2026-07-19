"""
個別学習用のコード
"""

# SVMのfit関数以外をまとめた親クラス
from svm_base import BaseSVM as MySVM

# Numpy
import numpy as np

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
#import data_dccc as dc
import data_airline as ai
import data_rbf as rd




    
    
    
    
def main():
    
    # ---------- iris ----------
    X_train, X_test, Y_train, Y_test = ir.X_train, ir.X_test, ir.Y_train, ir.Y_test
    # --------------------------
          
    # --------- cancer ---------
    #X_train, X_test, Y_train, Y_test = ca.X_train, ca.X_test, ca.Y_train, ca.Y_test
    # --------------------------
        
    # ---------- adult ---------
    #X_train, X_test, Y_train, Y_test = ad.X_train, ad.X_test, ad.Y_train, ad.Y_test
    #X_train, X_test, Y_train, Y_test = ad.X_train_91, ad.X_test_91, ad.Y_train_91, ad.Y_test_91
    #X_train, X_test, Y_train, Y_test = ad.X_train_73, ad.X_test_73, ad.Y_train_73, ad.Y_test_73
    #X_train, X_test, Y_train, Y_test = ad.X_train_55, ad.X_test_55, ad.Y_train_55, ad.Y_test_55
    # --------------------------
    
    # ---------- dccc ----------
    #X_train, X_test, Y_train, Y_test = dc.X_train, dc.X_test, dc.Y_train, dc.Y_test
    # --------------------------
    
    # -------- airline ---------
    #X_train, X_test, Y_train, Y_test = ai.X_train, ai.X_test, ai.Y_train, ai.Y_test
    # --------------------------
    
    # ------- rbfサンプル -------
    #X_train, X_test, Y_train, Y_test = rd.X_train, rd.X_test, rd.Y_train, rd.Y_test
    # --------------------------
    
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        
        
    # --- カーネルを指定してインスタンスを生成 ---
    mysvm = MySVM(kernel = 'linear', C = 10)#, tol = 0.1)
    #mysvm = MySVM(kernel = 'poly', degree = 2, coef0 = 1.0, C = 1)
    #mysvm = MySVM(kernel = 'rbf', gamma = 1, C = 1)#, tol = 0.1)
    #mysvm = MySVM(kernel = 'sigmoid', gamma = 1.0, coef0 = 1.0, C = 1.0)
    # -----------------------------------------
        
        
        
        
    # --------------------------------------------------------


    t_start = time.time()

    mysvm.fit(X_train, Y_train)

    t_end = time.time()
    print(f"fitting time : {t_end - t_start}")

    print(f'objective_value: {mysvm.objective_value[-1]}')

    print(f'SV_len: {len(mysvm.ind_sv)}')
    print(f'INNER_len: {len(mysvm.ind_inner)}')

    print(f'w: {mysvm.w}')
    print(f'b: {mysvm.b}')

    Y_pred = mysvm.predict(X_test)
    
    print(f'Y_test: {Y_test}')
    print(f'Y_pred: {Y_pred}')

    # 精度を計算
    accuracy = np.mean(Y_pred == Y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # F1スコアの計算
    f1 = f1_score(Y_pred, Y_test)
    print(f'F1: {f1 * 100:.2f}%')
        

    # 目的関数値の値を再計算 -------------
    L1, L2 = 0, 0
    for i in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
        L2 -= mysvm.alphas[i]
        for j in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
            L1 += mysvm.alphas[i] * mysvm.alphas[j] * mysvm.y[i] * mysvm.y[j] * mysvm.K[i, j]
    Obj = 0.5 * L1 + L2
    # ----------------------------------
    
    
    
    
    # Excel貼り付け用 ---------------------------------------------------------
    print("\n\nprint all with [,]")
    print(f"{t_end - t_start},{accuracy * 100:.2f},{Obj},{len(mysvm.ind_sv)},{len(mysvm.ind_inner)},{len(mysvm.ind_sv) + len(mysvm.ind_inner)}")
    # -------------------------------------------------------------------------

    """h = np.histogram(mysvm.alphas, 10)
    print(h[0])
    plt.plot(h[0])
    plt.show()"""



    mysvm.plt_Objective_Values()
    mysvm.plt_Data_and_Boundary()
    
    
    
if __name__ == "__main__":
    main()
        