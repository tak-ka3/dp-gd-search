import numpy as np
from scipy import interpolate
from utils import spline1, inv_calc, diff_calc

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
