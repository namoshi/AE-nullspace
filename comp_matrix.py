import numpy as np

def calculate_projection_to_null_space(A):
    """
    行列Aの特異値分解（SVD）を用いて、Aの零空間への射影行列Bを計算する。

    Args:
        A (np.ndarray): 入力行列

    Returns:
        np.ndarray: Aの零空間への射影行列B
    """
    # 行列Aを特異値分解する
    # U: 左特異ベクトル, s: 特異値, Vh: 右特異ベクトルの転置（V^T）
    U, s, Vh = np.linalg.svd(A)

    # 浮動小数点誤差を考慮し、非常に小さい特異値を0とみなす閾値（しきいち）を設定
    # 一般的に、最大の特異値にマシンの浮動小数点精度を掛けたものが使われる
    tolerance = np.finfo(s.dtype).eps * max(A.shape) * np.max(s)

    # 閾値より小さい特異値の数を数えることで、零空間の次元を決定
    # または、閾値より大きい特異値の数（ランクr）を数える
    rank = np.sum(s > tolerance)

    # Aの零空間の正規直交基底を抽出する
    # Vh (V^T) の (rank)行目以降が零空間の基底ベクトルとなる
    null_space_basis_T = Vh[rank:, :]

    # 射影行列Bを計算する
    # B = V_n @ V_n.T (ここで V_n は零空間の基底ベクトルを列に持つ行列)
    # V_n = null_space_basis_T.T なので、
    # B = (null_space_basis_T.T) @ (null_space_basis_T.T).T
    # B = null_space_basis_T.T @ null_space_basis_T
    B = null_space_basis_T.T @ null_space_basis_T

    return B

# --- メインの実行部分 ---
if __name__ == '__main__':
    # 例：ランクが2である3x3行列を作成
    # 3行目は1行目と2行目の和なので、行は線形従属
    A = np.array([
        [1, 2, 3],
#        [4, 5, 6],
        [5, 7, 9]
    ])

    print("入力行列 A:\n", A)
    
    print("\n---")
    print("rank(A)=", np.linalg.matrix_rank(A))
    print("---")

    # 行列Bを計算
    B = calculate_projection_to_null_space(A)

    print("\n---")
    print("計算された行列 B (Aの零空間への射影行列):\n", np.round(B, 4))
    print("---")

    print("\n---")
    print("rank(B)=", np.linalg.matrix_rank(B))
    print("---")

    # --- 検証 ---
    print("\n[検証1] Bが射影行列であることの確認 (B^2 = B)")
    is_idempotent = np.allclose(B @ B, B)
    print(f"B @ B = B は正しいか？ -> {is_idempotent}\n")

    # Aの零空間のベクトルを求める (Ax = 0 を満たすx)
    # 上記の例では、[1, -2, 1] が零空間のベクトル
    x_null = np.array([1, -2, 1])
    print(f"[検証2] Aの零空間のベクトル x_null = {x_null}")
    print("A @ x_null の結果 (ほぼゼロになるはず):\n", A @ x_null)
    print("B @ x_null の結果 (x_null自身になるはず):\n", np.round(B @ x_null, 4))
    
    # Aの行空間のベクトルを用意 (例：Aの1行目)
    x_row = A[0, :]
    print(f"\n[検証3] Aの行空間のベクトル x_row = {x_row}")
    print("B @ x_row の結果 (ゼロベクトルになるはず):\n", np.round(B @ x_row, 4))