"""
2025/7/27
疑似データを生成して同じ識別関数が得られるかを実験するコード
疑似データはf(x)^2に対して取る上限と下限の値を設定
"""

# SVMの関数をまとめた親クラス
from svm_base_for_gizi import BaseSVM_for_Gizi as MySVM_g

# Numpy
import numpy as np

# グラフのプロット用
import matplotlib.pyplot as plt

# 時間計測用
import time

# f1スコア計算用
from sklearn.metrics import f1_score

# 勾配降下法のライブラリ
from scipy.optimize import minimize

# 学習データ
import data_iris as ir
import data_cancer as ca
import data_adult as ad
import data_airline as ai
import data_rbf as rd
    
    




    
def main():
    
    # ---------- iris ----------
    X_train, X_test, Y_train, Y_test = ir.X_train, ir.X_test, ir.Y_train, ir.Y_test
    # --------------------------
          
    # --------- cancer ---------
    X_train, X_test, Y_train, Y_test = ca.X_train, ca.X_test, ca.Y_train, ca.Y_test
    # --------------------------
        
    # ---------- adult ---------
    #X_train, X_test, Y_train, Y_test = ad.X_train, ad.X_test, ad.Y_train, ad.Y_test
    # --------------------------
    
    # -------- airline ---------
    #X_train, X_test, Y_train, Y_test = ai.X_train, ai.X_test, ai.Y_train, ai.Y_test
    # --------------------------
    
    # ------- rbfサンプル -------
    #X_train, X_test, Y_train, Y_test = rd.X_train, rd.X_test, rd.Y_train, rd.Y_test
    # --------------------------
    
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        
        

        
        
    
    # --------------------------------------------------------
    
    OLL_roop = 2 #4
    Y_pred = np.empty(OLL_roop, dtype=object)
    Y_pred_no_int = np.empty(OLL_roop, dtype=object)
    w_list = np.empty(OLL_roop, dtype=object)
    
    # --- カーネルを指定してインスタンスを生成 ---
    mysvm = MySVM_g(kernel = 'linear', C = 1)#, tol = 1e-10)
    #mysvm = MySVM_g(kernel = 'poly', degree = 2, coef0 = 1.0, C = 1)
    #mysvm = MySVM_g(kernel = 'rbf', gamma = 1, C = 1)#, tol = 0.1)
    #mysvm = MySVM_g(kernel = 'sigmoid', gamma = 1.0, coef0 = 1.0, C = 1.0)
    
    plt = True
    plt = False
    # -----------------------------------------
    
    # ----- ノイズの半径の範囲を指定 (最小値, 最大値) -----
    radius = (0.1, 0.2)
    # --------------------------------------------------

    for i in range(OLL_roop):
                
        # 疑似データを生成
        if i == 1:
            start = time.time()
            #X_train, Y_train = mysvm.make_fake_data(X_train, mysvm.ind_sv, lr = 1e-3, bounds_eps=1e-6, max_iter=10000)
            #X_train, Y_train = mysvm.make_fake_data_shift(X_train, Y_train, radius = radius, max_retry = 1000, lr = 1e-3, bounds_eps=1e-6, max_iter=10000)
            #X_train, Y_train = mysvm.make_fake_data_random(X_train, Y_train, radius = radius, max_retry = 1000)
            #X_train, Y_train = mysvm.make_fake_data_random_with_margin(X_train, Y_train, mysvm.alphas, radius = radius, max_retry = 1000)
            X_train, Y_train = mysvm.make_fake_data_KKT(X_train, Y_train, mysvm.alphas, radius = radius, max_retry = 10000)
            end = time.time()
            print(f"make_fake_data time: {end - start:.6f} sec\n")
            
            # ハードマージンに変更
            #mysvm.C = float("inf")
            #mysvm.tol= 1e-5
        
        # サポートベクターとなる疑似データのみを使う．
        if i == 2:
            X_train, Y_train = X_train[mysvm.ind_sv], Y_train[mysvm.ind_sv]
            
        # ごく一部のデータのみを抽出する
        if i == 3:
            X_up = X_train[Y_train == mysvm.max_val]
            X_down = X_train[Y_train == mysvm.min_val]
            
            rank = np.linalg.matrix_rank(X_up.T)
            print("rank(up):", rank)

            rank = np.linalg.matrix_rank(X_down.T)
            print("rank(down):", rank)

            rank = np.linalg.matrix_rank(X_train.T)
            print("rank(all):", rank)
            
            min_dist=0.1*np.sqrt(X_train.shape[1])
            print(f"min_dist: {min_dist}")
            X_train, Y_train = mysvm.pick_affine_independent_points1(X_train, Y_train, min_dist=min_dist)

            #X_train, Y_train = mysvm.get_convexHull_vertices2(X_train, Y_train)
            
            #X_train, Y_train = mysvm.pick_far_and_independent(X_train, Y_train)
            
    
        print(f"Roop {i}: {X_train.shape}, {Y_train.shape}\n")
        values, counts = np.unique(Y_train, return_counts=True)
        for val, count in zip(values, counts):
            print(f"value {val}: {count} samples")
            
        t_start = time.time()
        mysvm.fit(X_train, Y_train)
        t_end = time.time()
        print(f"fitting time : {t_end - t_start}")

        print(f'objective_value: {mysvm.objective_value[-1]}')

        print(f'SV_len: {len(mysvm.ind_sv)}')
        print(f'INNER_len: {len(mysvm.ind_inner)}')

        print(f'w: {mysvm.w}')
        print(f'b: {mysvm.b}')

        #Y_pred[i] = mysvm.predict(X_test)
        Y_pred_no_int[i] = np.array([mysvm.f(x) for x in X_test])
        Y_pred[i] = np.where(Y_pred_no_int[i] >= 0, 1, -1)
        
        print(f'Y_test: {Y_test}')
        print(f'Y_pred: {Y_pred[i]}')

        # 精度を計算
        accuracy = np.mean(Y_pred[i] == Y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # F1スコアの計算
        f1 = f1_score(Y_pred[i], Y_test)
        print(f'F1: {f1 * 100:.2f}%\n')
        
        #wを保存
        w_list[i] = mysvm.w
    

        """# 目的関数値の値を再計算 -------------
        L1, L2 = 0, 0
        for i in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
            L2 -= mysvm.alphas[i]
            for j in np.concatenate((mysvm.ind_sv, mysvm.ind_inner)):
                L1 += mysvm.alphas[i] * mysvm.alphas[j] * mysvm.y[i] * mysvm.y[j] * mysvm.K[i, j]
        Obj = 0.5 * L1 + L2
        # ----------------------------------"""

        if i >= 0:
            mysvm.plt_Objective_Values()
            if plt == True:
                mysvm.plt_Data_and_Boundary(x_range=(-0.05, 1.05), y_range=(-0.05, 1.05))
    
    # Y_predがオリジナルデータとどの程度一致するか
    
    # 標準偏差を計算
    std_f = np.std(Y_pred_no_int[0])
    if std_f < 1e-16:
        std_f += 1e-12

    for i in range(1, OLL_roop):
        Y_pred_accuracy = np.mean(Y_pred[0] == Y_pred[i])
        print(f'Y_pred_same({i}): {Y_pred_accuracy * 100:.2f}%')
        
        rmse = np.sqrt(np.mean((Y_pred_no_int[0] - Y_pred_no_int[i]) ** 2))
        print(f'RMSE({i}): {rmse:.6f}')
        
        mae = np.mean(np.abs(Y_pred_no_int[0] - Y_pred_no_int[i]))
        print(f'MAE({i}): {mae:.6f}')
        
        #rms_f = np.sqrt(np.mean(Y_pred_no_int[0]**2))
        nrmse = rmse / std_f
        print(f'NRMSE_std({i}): {nrmse:.6f}')
        
        nmae = mae / std_f
        print(f'NMAE_std({i}): {nmae:.6f}')
        
        print(f'pred [Max, Min]: [{np.max(Y_pred_no_int[i]):.6f}, {np.min(Y_pred_no_int[i]):.6f}]')
        
        # wのコサイン類似度を計算
        if mysvm.kernel == "linear":
            # コサイン類似度
            cos_theta = np.dot(w_list[0], w_list[i]) / (np.linalg.norm(w_list[0]) * np.linalg.norm(w_list[i]))
            # 丸め誤差対策
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            # 角度（ラジアン）
            theta = np.arccos(cos_theta)
            # 角度（度）
            theta_deg = np.degrees(theta)

            print(f"cos({i}): {cos_theta:.6f}")
            print(f"angle({i}): {theta_deg:.6f} deg ({theta:.6f} rad)")

        

    
    
    
if __name__ == "__main__":
    main()
        