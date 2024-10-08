import numpy as np
from scipy import interpolate
from utils import spline1, inv_calc, diff_calc
from noise_alg import laplace_func

"""
変数変換の関数
X_val: 変換前の確率密度関数の範囲を表す配列
X_pdf_val: 変換前の確率密度関数の値の配列
transform_func: 変換のための関数
"""
def trasform_var(X_val: np.ndarray, X_pdf_val: np.ndarray, transform_func) -> tuple[np.ndarray, np.ndarray]:
    assert X_val.size == X_pdf_val.size
    Y_val = transform_func(X_val) # Y = F(X)
    # 逆関数を数値的に計算する
    x_inv, y_inv = inv_calc(X_val, Y_val)
    # スプライン補間の際のデータ点の数
    point_num = X_val.size * 10
    x_splined, y_splined = spline1(x_inv, y_inv, point_num)
    dx_dy = diff_calc(x_splined, y_splined, point_num)
    # 改めてスプライン補間によってX軸を揃える
    x_splined_new, X_pdf_val_new = spline1(X_val, X_pdf_val, point_num)
    assert (x_splined == x_splined_new).all()
    Y_pdf_val = X_pdf_val_new * np.abs(dx_dy) # 要素ごとに掛ける
    return x_splined, Y_pdf_val

"""
複数の確率変数に対して変数変換を行う
"""
def trasform_vars(X_vals: np.ndarray, X_pdf_vals: np.ndarray, transform_func) -> tuple[np.ndarray, np.ndarray]:
    assert type(X_pdf_vals[0]) == np.ndarray or type(X_pdf_vals[0]) == list
    if type(X_vals[0]) != np.ndarray and type(X_vals[0]) != list:
        X_vals_2d = np.array([X_vals for _ in range(len(X_pdf_vals))])
    else:
        assert len(X_vals_2d) == len(X_pdf_vals)
    Y_vals = []
    Y_pdf_vals = []
    for X_val, X_pdf_val in zip(X_vals_2d, X_pdf_vals):
        Y_val, Y_pdf_val = trasform_var(X_val, X_pdf_val, transform_func)
        Y_vals.append(Y_val)
        Y_pdf_vals.append(Y_pdf_val)
    return Y_vals, Y_pdf_vals

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
