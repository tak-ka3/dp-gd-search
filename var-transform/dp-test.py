import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from search import search_all, search_by_threshold
from noise_alg import laplace_func
from transform_alg import transform_linear_sum

"""
laplace分布を前提とした実装
x_val: 変換前の確率密度関数の範囲を表す配列
pdf_vals: 変換前の確率密度関数の値の二次元配列
pdf_func: 変換前の確率変数が従う確率密度関数
beta: logexpsumのスケールパラメタ
"""
# def transform_logexpsum(X_vals, X_pdf_vals, beta=10):
#     Y_vals = []
#     Y_pdf_vals = []
#     for X_val, X_pdf_val in zip(X_vals, X_pdf_vals):
#         Y_val, Y_pdf_val = trasform_var(X_val, X_pdf_val, lambda x: np.exp(x*beta))
#         Y_vals.append(Y_val)
#         Y_pdf_vals.append(Y_pdf_val)
    
#     # 合計の確率密度関数を求めるX軸、つまりY_valsの範囲が異なる恐れがある

#         Y_pdf_val = np.exp(Y_pdf_val)
#         Y_pdf_val = Y_pdf_val / np.sum(Y_pdf_val)
#         plt.scatter(Y_val, Y_pdf_val, s=0.5)
#     return (1 / beta) * np.log(np.sum(np.exp(beta * arr)))

"""
ノイズを付与後に適用する関数
"""
def test_func(x: np.array):
    # return x**5
    # return np.exp(x)
    return x

# 元の確率変数の確率密度関数であるoriginal_fを仮定した変数変換
def var_trasform(y, x_diff, original_f, loc=0):
    if type(loc) == np.ndarray:
        return [original_f(y, loc=val) * np.abs(x_diff) for val in loc]
    else:
        return original_f(y, loc=loc) * np.abs(x_diff)

"""
data1: 隣接入力データ
data2: 隣接入力データ
noise_func: ノイズを加える関数 (Laplaceメカニズムなど)
func: ノイズを加えた値に対して施す関数
    - 出力はスカラーorベクター(funcに依存する)も必要
"""
def transform(input_data1: np.ndarray, input_data2: np.ndarray, transform_func, noise_func) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: ここのxの範囲を元の確率密度関数が大きい値を取る範囲に限定するのが良さそう。現状確率が0となるような不要な部分も多い
    # 例えばLaplace分布だと確率が0.9以上が入っているのがどの範囲であるかなどを元にしてxの値を決めるとか
    # つまりこのxの値は元の確率変数の確率分布の概形（中心、分散など）に依存する
    # リダクションの場合は要素ごとに入力値が確率分布が変わるので、全ての入力で試す必要がある
    if noise_func == laplace_func:
        sample_num = 1000
        y_sample_data1 = [np.random.laplace(data)  for data in input_data1 for _ in range(sample_num)]
        y_sample_data2 = [np.random.laplace(data)  for data in input_data2 for _ in range(sample_num)]
        range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2))*10, max(max(y_sample_data1), max(y_sample_data2))*10, 1000)

        x, pdf1, pdf2 = transform_func(range_x, input_data1, input_data2)
        return x, pdf1, pdf2
    else:
        raise NotImplementedError

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    x_data1 = np.array([1.0, 3.0, 5.0, 7.0])
    x_data2 = np.array([2.0, 4.0, 6.0, 8.0])
    x, y1, y2 = transform(x_data1, x_data2, transform_linear_sum, laplace_func)
    plt.scatter(x, y1, color="green", s=0.2, label="x1")
    plt.scatter(x, y2, color="orange", s=0.2, label="x2")
    plt.legend()
    plt.show()
    
    eps = search_all(x, y1, y2)

    print("estimated eps: ", eps)
