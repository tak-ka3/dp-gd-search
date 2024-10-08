import numpy as np
from scipy import interpolate

"""
ref: https://qiita.com/Ken227/items/aee6c82ec6bab92e6abf
"""
def spline1(x,y,point):
    f = interpolate.interp1d(x, y,kind="cubic")
    # x座標を等間隔にしてくれる
    X = np.linspace(x[0],x[-1],num=point,endpoint=True)
    Y = f(X)
    return X,Y

def inv_calc(x, y):
    x_y_arr = np.array([x, y])
    sorted_coord_indices = np.argsort(x_y_arr[1, :])
    sorted_coord = x_y_arr[:, sorted_coord_indices]
    return sorted_coord[1], sorted_coord[0]

"""
dy/dxを求める関数
TODO: 端の部分の計算は近似の誤差が大きくなることが予想されるので、省いても良いかも。
"""
def diff_calc(x, y, point):
    x_diff = []
    dx2 = ((x[1] - x[0]) * 2).astype(float)
    x_diff = [(y[2] - y[0]) / dx2]
    # x_diff = []
    for i in range(point - 2):
        x_diff.append((y[i+2] - y[i])/ dx2)
    x_diff.append(x_diff[-1])
    return x_diff
