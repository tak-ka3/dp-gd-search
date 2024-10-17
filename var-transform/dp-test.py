import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from search import search_all, search_by_threshold
from noise_alg import laplace_func
from transform_alg import transform_linear_sum, transform_logsumexp, trasform_arg_max
from transform import transform

def dp_test(input_data1: np.ndarray, input_data2: np.ndarray) -> np.float64:
    x, y1, y2 = transform(input_data1, input_data2, transform_linear_sum, laplace_func)
    plt.scatter(x, y1, color="green", s=0.2, label="x1")
    plt.scatter(x, y2, color="orange", s=0.2, label="x2")
    plt.legend()
    plt.show()
    
    eps = search_by_threshold(x, y1, y2)
    return eps

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    x_data1 = np.array([1.0, 3.0, 5.0, 7.0])
    x_data2 = np.array([2.0, 4.0, 6.0, 8.0])
    eps = dp_test(x_data1, x_data2)

    print("estimated eps: ", eps)
