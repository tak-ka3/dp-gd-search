import numpy as np
from scipy import interpolate
from utils import spline_eq, spline_func, inv_calc, diff_calc

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
    Y_pdf_val = X_pdf_val_new * np.abs(dx_dy[start_ind:end_ind]) # 要素ごとに掛ける
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
    return np.array(Y_vals), np.array(Y_pdf_vals)
