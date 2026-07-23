"""
提案手法のplt関数等をまとめた親クラス
"""

# SVMの関数をまとめた親クラス
from svm_base import BaseSVM as MySVM

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

buffer_size = 32768

SAVE_DIR = "fig/svm"


# 提案手法用のクラス
class BaseSVM_for_Teian(MySVM):   

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
    
    
    def fit_independent(self, X, y):
        """
        SVMモデルをデータに適合させる関数
        パラメータ:
        X (array-like): トレーニングデータの特徴量
        y (array-like): トレーニングデータのラベル
        """
        
        alphas = np.zeros(len(y), dtype=float)
        
        K = np.full((len(y), len(y)), np.nan, dtype=float)
        E = -y.astype(float)
                
        objective_value = np.array([0])

        #----------------------SMOアルゴリズムによるアルファの学習-----------------------#

        is_converged = False

        for count in range(self.max_iterations):

            # 違反ペアを得る
            #p, F_p = self._find_up_min(y, alphas, E)
            #q, F_q = self._find_low_max(y, alphas, E)
            p, q, F_p, F_q = self._find_up_min_and_low_max(y, alphas, E)
            
            #print(f'p:{p}, q:{q}, objective_value:{objective_value[count]}')
            
            # 違反ペアが見つからない場合，学習を終了
            if (F_p > F_q - self.tol):
                is_converged = True
                print(f"    agent{rank} ({len(X)} datas) -> Successfully converged. Iterations: {count}", flush=True)

                break
            
            # カーネル行列のp, q行を計算
            K = self._calculate_kernel_rows(p, q, X, K)
            
            # alphas を更新
            alphas, E, d_obj = self._update_alphas(p, q, y, alphas, K, E)
            
            # 目的関数の計算結果を配列に追加
            objective_value = np.append(objective_value, objective_value[count] + d_obj)

        if not is_converged:
            print(f"    agent{rank} ({len(X)} datas) -> Reached maximum max_iterations. Iterations: {self.max_iterations}", flush=True)
        
        #-----------------------------------------------------------------------------#

        return alphas, K, objective_value
    
    def fit_each(self, X, y):
        """
        SVMモデルをデータに適合させる関数(個別学習)
        パラメータ:
        X (array-like): トレーニングデータの特徴量のリスト
        y (array-like): トレーニングデータのラベルのリスト
        """
        
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
            plt.savefig(f"{SAVE_DIR}/{filename}.pdf")

    
    def plt_Objective_Last_Values(self, filename):
        """
        ステップごとの目的関数の値をプロットする関数
        """
        plt.figure()
        colors=['r', 'b', 'g', 'y', 'm', 'c']
        for i in range(len(self.objective_last_value_list)):
            data = self.objective_last_value_list[i][:-1]
            plt.plot(data, c = colors[i], label= f'Agent {i+1}')
        #plt.plot(np.sum(self.objective_value_list, axis=0), c = 'black', label= f'sum')
        
        # 目盛りを整数値に限定
        #max_iterations = len(self.objective_last_value_list[0])  # 横軸の最大値
        #plt.xticks(np.arange(0, max_iterations, step=1))  # 0から最大イテレーションまでの整数目盛り

        #plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1))
        #plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)

        plt.xlabel('k')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.legend()

        # 横軸目盛りを「ちょうどいい」間隔の整数にする
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        #plt.show()
        plt.savefig(f"{SAVE_DIR}/{filename}.pdf")
        
    
    def plt_Num_Samples(self, filename):
        """
        データ数の推移をプロット
        """
        plt.figure()
        colors=['r', 'b', 'g', 'y', 'm', 'c']
        
        for i in range(size):
            data = self.num_samples_list[i]
            plt.plot(data[:, 0], data[:, 1], linestyle='-', c = colors[i], label=f'Agent {i+1}')
        # 目盛りを整数値に限定
        #max_iterations = len(self.num_samples_list[0][0])  # 横軸の最大値
        #plt.xticks(np.arange(0, max_iterations, step=1))  # 0から最大イテレーションまでの整数目盛り

        #plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1))
        #plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)

        plt.xlabel('k')
        plt.ylabel('Num of Samples')
        plt.grid(True)
        plt.legend()
        
        # 横軸目盛りを「ちょうどいい」間隔の整数にする
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        #plt.show()
        plt.savefig(f"{SAVE_DIR}/{filename}.pdf")

    def plt_Num_Samples_Total(self, filename):
        """
        データ数の推移をプロット(データを削除しない場合の通算個数)
        """
        plt.figure()
        colors=['r', 'b', 'g', 'y', 'm', 'c']
        
        for i in range(size):
            plt.plot(self.num_samples_total_list[i], linestyle='-', c = colors[i], label=f'Agent {i+1}')
        # 目盛りを整数値に限定
        #max_iterations = len(self.num_samples_list[0][0])  # 横軸の最大値
        #plt.xticks(np.arange(0, max_iterations, step=1))  # 0から最大イテレーションまでの整数目盛り

        #plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1))
        #plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)

        plt.xlabel('k')
        plt.ylabel('Num of Samples (Total)')
        plt.grid(True)
        plt.legend()
        
        # 横軸目盛りを「ちょうどいい」間隔の整数にする
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        #plt.show()
        plt.savefig(f"{SAVE_DIR}/{filename}.pdf")
    