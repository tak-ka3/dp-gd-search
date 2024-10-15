import numpy as np
from scipy import interpolate
from utils import spline_eq, spline_func, inv_calc, diff_calc, compress_range, calc_prob, visualize, plt_sca
from noise_alg import laplace_func
import matplotlib.pyplot as plt

def trasform_var(X_val: np.ndarray, X_pdf_val: np.ndarray, transform_func) -> tuple[np.ndarray, np.ndarray]:
    """
    変数変換の関数
    Args:
        X_val: 変換前の確率密度関数の範囲を表す一次元配列
        X_pdf_val: 変換前の確率密度関数の値の一次元配列
        transform_func: 変換のための関数
    Return:
        変換後の確率密度関数の範囲を表す一次元配列 (等間隔でなくて良い。無理に等間隔にすることで関数にとって重要な情報が損なわれてしまう)
        変換後の確率密度関数の値の一次元配列
    """
    assert X_val.size == X_pdf_val.size
    Y_val = transform_func(X_val) # Y = F(X)
    # 逆関数を数値的に計算する
    x_inv, y_inv = inv_calc(X_val, Y_val)
    x_splined, y_splined = x_inv, y_inv
    # x_splined = y, y_splined = x
    dx_dy = diff_calc(x_splined, y_splined) # x_splined(=等間隔ではない)をもとにしたdx_dy
    # 改めてスプライン補間によってX軸を揃える
    start_ind, end_ind, X_pdf_val_new = spline_func(X_val, X_pdf_val, y_splined)
    # plt_sca(x_splined[start_ind:end_ind], X_pdf_val_new, "after spline", x_splined[start_ind:end_ind], [laplace_func(np.log(x), 0) for x in x_splined[start_ind:end_ind]])
    Y_pdf_val = X_pdf_val_new * np.abs(dx_dy[start_ind:end_ind]) # 要素ごとに掛ける
    print("after transform, the prob", calc_prob(x_splined[start_ind:end_ind], Y_pdf_val))
    return x_splined[start_ind:end_ind], Y_pdf_val

def trasform_vars(X_vals: np.ndarray, X_pdf_vals: np.ndarray, transform_func) -> tuple[np.ndarray, np.ndarray]:
    """
    複数の確率変数に対して変数変換を行う
    X_vals: 配列の要素である確率変数の範囲を表す二次元配列
    X_pdf_vals: 配列の要素である確率密度関数の値の二次元配列
    """
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

def transform(input_data1: np.ndarray, input_data2: np.ndarray, transform_func, noise_func) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data1: 隣接入力データ
    data2: 隣接入力データ
    noise_func: ノイズを加える関数 (Laplaceメカニズムなど)
    func: ノイズを加えた値に対して施す関数
        - 出力はスカラーorベクター(funcに依存する)も必要
    """
    # TODO: ここのxの範囲を元の確率密度関数が大きい値を取る範囲に限定するのが良さそう。現状確率が0となるような不要な部分も多い
    # 例えばLaplace分布だと確率が0.9以上が入っているのがどの範囲であるかなどを元にしてxの値を決めるとか
    # つまりこのxの値は元の確率変数の確率分布の概形（中心、分散など）に依存する
    # リダクションの場合は要素ごとに入力値が確率分布が変わるので、全ての入力で試す必要がある
    if noise_func == laplace_func:
        sample_num = 1000
        y_sample_data1 = [np.random.laplace(data)  for data in input_data1 for _ in range(sample_num)]
        y_sample_data2 = [np.random.laplace(data)  for data in input_data2 for _ in range(sample_num)]
        range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2))*5, max(max(y_sample_data1), max(y_sample_data2))*5, 5000)

        x1, x2, pdf1, pdf2 = transform_func(range_x, input_data1, input_data2)

        if x1.size == x2.size and (x1 == x2).all():
            return x1, pdf1, pdf2
        else:
            f1 = interpolate.interp1d(x1, pdf1, kind="cubic")
            f2 = interpolate.interp1d(x2, pdf2, kind="cubic")
            x_var = np.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), num=10000)
            pdf1 = f1(x_var)
            pdf2 = f2(x_var)
            return x_var, pdf1, pdf2
    else:
        raise NotImplementedError
