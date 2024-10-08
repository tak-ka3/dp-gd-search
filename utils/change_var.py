import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from searcher import search_all, search_by_threshold

"""
laplace分布を前提とした実装
x_val: 変換前の確率密度関数の範囲を表す配列
pdf_vals: 変換前の確率密度関数の値の二次元配列
pdf_func: 変換前の確率変数が従う確率密度関数
beta: logexpsumのスケールパラメタ
"""
def transform_logexpsum(X_vals, X_pdf_vals, beta=10):
    Y_vals = []
    Y_pdf_vals = []
    for X_val, X_pdf_val in zip(X_vals, X_pdf_vals):
        Y_val, Y_pdf_val = trasform_var(X_val, X_pdf_val, lambda x: np.exp(x*beta))
        Y_vals.append(Y_val)
        Y_pdf_vals.append(Y_pdf_val)
    
    # 合計の確率密度関数を求めるX軸、つまりY_valsの範囲が異なる恐れがある

        Y_pdf_val = np.exp(Y_pdf_val)
        Y_pdf_val = Y_pdf_val / np.sum(Y_pdf_val)
        plt.scatter(Y_val, Y_pdf_val, s=0.5)
    return (1 / beta) * np.log(np.sum(np.exp(beta * arr)))

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
vals: 確率密度関数のxの範囲(=確率変数の範囲)の二次元配列
pdf_vals: 確率密度関数の二次元配列
"""
def transform_sum(vals, pdf_vals):
    if type(vals[0]) != np.ndarray and type(vals[0]) != list:
        vals = np.array([vals for _ in range(len(pdf_vals))])
    # 最初の分布のPDFをセット
    result_pdf = pdf_vals[0]
    val1 = vals[0]
    x_size = val1.size
    conv_x_size = x_size
    conv_x_range = val1
    start_val = val1[0]
    dx = val1[1] - val1[0]

    # 各PDFを順次畳み込み
    for val2, pdf in zip(vals[1:], pdf_vals[1:]):
        # TODO: ifとelseの処理が重複している
        if dx == val2[1] - val2[0]:
            result_pdf = np.convolve(result_pdf, pdf, mode='full') * dx
            start_val = conv_x_range[0] + val2[0]
            conv_x_size = conv_x_size + val2.size - 1
            conv_x_range = np.arange(start_val, start_val + dx*conv_x_size, dx)
            val1 = val2
        else:
            f2 = interpolate.interp1d(val2, pdf, kind="cubic")
            # 最初の確率密度関数に合わせることになる
            x2 = np.arange(val2[0], val2[-1], dx)
            pdf2 = f2(x2)
            result_pdf = np.convolve(result_pdf, pdf2, mode='full') * dx
            start_val = conv_x_range[0] + x2[0]
            conv_x_size = conv_x_size + x2.size - 1
            conv_x_range = np.arange(start_val, start_val + dx*conv_x_size, dx)
            val1 = x2

    return conv_x_range, result_pdf

"""
ノイズを付与後に適用する関数
"""
def test_func(x: np.array):
    # return x**5
    # return np.exp(x)
    return x

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

def laplace_func(x, loc, sensitivity=1, eps=0.1):
    b = sensitivity / eps
    if type(loc) == np.ndarray or type(loc) == list:
        return [np.exp(-np.abs(x - val)/b) / (2*b) for val in loc]
    return np.exp(-np.abs(x - loc)/b) / (2*b)

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
def transform(data1: np.ndarray, data2: np.ndarray, noise_func, func) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: ここのxの範囲を元の確率密度関数が大きい値を取る範囲に限定するのが良さそう。現状確率が0となるような不要な部分も多い
    # 例えばLaplace分布だと確率が0.9以上が入っているのがどの範囲であるかなどを元にしてxの値を決めるとか
    # つまりこのxの値は元の確率変数の確率分布の概形（中心、分散など）に依存する
    # リダクションの場合は要素ごとに入力値が確率分布が変わるので、全ての入力で試す必要がある
    sample_num = 1000
    y_sample_data1 = [np.random.laplace(data)  for data in data1 for _ in range(sample_num)]
    y_sample_data2 = [np.random.laplace(data)  for data in data2 for _ in range(sample_num)]
    range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2))*10, max(max(y_sample_data1), max(y_sample_data2))*10, 1000)

    # laplace_funを修正する
    x_splined1, transformed_pdf1 = trasform_vars(range_x, laplace_func(range_x, loc=data1), lambda x: x)
    x_splined2, transformed_pdf2 = trasform_vars(range_x, laplace_func(range_x, loc=data2), lambda x: x)

    # x_splined1/2はそれぞれ二次元配列
    assert (x_splined1[0] == x_splined2[0]).all()
    x_splined = x_splined1

    # lambda x: x を適用後の確率密度関数を可視化する
    for x_val, pdf1, pdf2 in zip(x_splined, transformed_pdf1, transformed_pdf2):
        plt.scatter(x_val, pdf1, s=0.5, color="blue")
        plt.scatter(x_val, pdf2, s=0.5, color="red")
    plt.show()

    if type(transformed_pdf1[0]) == np.ndarray:
        sum_x1, sum_pdf1 = transform_sum(x_splined, transformed_pdf1)
        sum_x2, sum_pdf2 = transform_sum(x_splined, transformed_pdf2)
        assert (sum_x1 == sum_x2).all()
        sum_x = sum_x1
    else:
        pass

    return sum_x, sum_pdf1, sum_pdf2

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    x_data1 = np.array([1.0, 3.0, 5.0, 7.0])
    x_data2 = np.array([2.0, 4.0, 6.0, 8.0])
    x, y1, y2 = transform(x_data1, x_data2, laplace_func, test_func)
    plt.scatter(x, y1, color="green", s=0.2, label="x1")
    plt.scatter(x, y2, color="orange", s=0.2, label="x2")
    plt.legend()
    plt.show()
    
    eps = search_all(x, y1, y2)

    print("estimated eps: ", eps)
