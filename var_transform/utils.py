import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import matplotlib.pyplot as plt
import yaml
import itertools
from helpers import conditional_execution

def calc_prob(x, pdf):
    """
    Args:
        x: 確率変数の範囲
        pdf: 確率密度関数の値
    Return:
        この範囲における確率を返す
    """
    assert x.size == pdf.size
    dx = [(x[1] + x[0])/2 - x[0]]
    for i in range(1, x.size-1):
        dx.append((x[i+1] + x[i])/2 - (x[i] + x[i-1])/2)
    dx.append(x[-1] - (x[-1] + x[-2])/2)
    return np.sum(pdf * dx)

def compress_range(x, pdf, th=0.9) -> tuple[np.ndarray, np.ndarray]:
    """
    確率変数の範囲を圧縮する関数
    """
    assert x.size == pdf.size
    x_size = x.size
    arg_max = np.argmax(pdf)
    dx = [(x[1] + x[0])/2 - x[0]]
    for i in range(1, x_size-1):
        dx.append((x[i+1] + x[i])/2 - (x[i] + x[i-1])/2)
    dx.append(x[-1] - (x[-1] + x[-2])/2)
    cum_prob = 0
    if arg_max == 0:
        for i in range(x_size):
            cum_prob += pdf[i] * dx[i]
            if cum_prob >= th:
                return x[:i], pdf[:i]
    else:
        cum_prob += pdf[arg_max] * dx[arg_max]
        left_ind = arg_max - 1
        right_ind = arg_max + 1
        left_prob, right_prob = 0, 0
        while True:
            if cum_prob >= th:
                break
            if left_ind - 1 >= 0:
                left_prob = pdf[left_ind - 1] * dx[left_ind - 1]
            else:
                left_prob = 0
            if right_ind + 1 <= x_size - 1:
                right_prob = pdf[right_ind + 1] * dx[right_ind + 1]
            else:
                right_prob = 0
            if left_prob > right_prob:
                left_ind -= 1
                cum_prob += left_prob
            else:
                right_ind += 1
                cum_prob += right_prob
        return x[left_ind:right_ind+1], pdf[left_ind:right_ind+1]

def spline_eq(x,y,point):
    """
    等間隔でスプライン補間を行う関数
    ref: https://qiita.com/Ken227/items/aee6c82ec6bab92e6abf
    """
    f = interpolate.interp1d(x, y,kind="cubic")
    # x座標を等間隔にしてくれる
    X = np.linspace(x[0],x[-1],num=point,endpoint=True)
    Y = f(X)
    return X,Y

def spline_func(x, y, new_x):
    """
    スプライン補間により定められたx座標における値を返す関数
    TODO: start_ind, end_indを返さないような設計に変更する
    """
    f = interpolate.interp1d(x, y,kind="cubic")
    start_ind = 0
    end_ind = len(new_x)
    for i in range(len(new_x)):
        if x[0] <= new_x[i]:
            start_ind = i
            break
    for i in range(len(new_x)-1, -1, -1):
        if x[-1] >= new_x[i]:
            end_ind = i
            break
    assert start_ind <= end_ind
    return start_ind, end_ind, f(new_x[start_ind:end_ind])

def inv_calc(x, y):
    """
    逆関数を求める関数
    """
    x_y_arr = np.array([x, y])
    sorted_coord_indices = np.argsort(x_y_arr[1, :])
    sorted_coord = x_y_arr[:, sorted_coord_indices]
    return sorted_coord[1], sorted_coord[0]

def diff_calc(x, y):
    """
    dy/dxを求める関数
    TODO: 端の部分の計算は近似の誤差が大きくなることが予想されるので、省いても良いかも。
    """
    x_diff = []
    dx2 = ((x[1] - x[0]) * 2).astype(float)
    x_diff = [(y[2] - y[0]) / dx2]
    # x_diff = []
    for i in range(x.size - 2):
        dx2 = (x[i+2] - x[i]).astype(float)
        x_diff.append((y[i+2] - y[i])/ dx2)
    x_diff.append(x_diff[-1])
    return x_diff

def nonuniform_convolution(t_f, t_g, f, g, t_target, integral_way="gauss"):
    """
    Perform convolution of two functions f and g with non-uniformly spaced time points t.
    """
    # Create an interpolation function for both f and g
    f_interp = interpolate.CubicSpline(t_f, f, bc_type='natural', extrapolate=True)
    g_interp = interpolate.CubicSpline(t_g, g, bc_type='natural', extrapolate=True)
    conv_result = np.zeros(len(t_target))

    f_min = min(t_f)
    g_min = min(t_g)

    for i, t_i in enumerate(t_target):
        integral = 0
        integrand = lambda tau: f_interp(tau) * g_interp(t_i - tau) if f_min <= tau and g_min <= t_i - tau else 0
        for j in range(1, len(t_target)):
            # if t_i < t_target[j]:
            #     break
            if integral_way == "gauss":
                integral += quad(integrand, t_target[j-1], t_target[j], limit=100)[0]
            elif integral_way == "trapz":
                integral += (integrand(t_target[j]) + integrand(t_target[j-1])) * (t_target[j] - t_target[j-1]) / 2
            else:
                raise ValueError("integral_way should be gauss or trapz")
        conv_result[i] = integral
    return conv_result

@conditional_execution("DEBUG")
def plt_1d(x, y, title="Plot"):
    """
    1種類の値列をプロットする
    """
    plt.scatter(x, y, color="red", s=0.2)
    plt.title(title)
    plt.show()
    return

@conditional_execution("DEBUG")
def plt_2d(x_list, y_list, labels=None, title="Plot"):
    """
    複数種類の値列をプロットする
    """
    if labels is None:
        labels = [f"y{i}" for i in range(len(y_list))]
    else:
        if type(labels) != list or labels.size != len(y_list):
            raise ValueError("labels should be a list and its size should be the same as the number of y_list")
    for x, y, label in zip(x_list, y_list, labels):
        plt.scatter(x, y, label=label, s=0.5)
    plt.title(title)
    plt.legend()
    # plt.show()
    return

class Settings:
    """
    設定ファイルを読み込むクラス
    """
    def __init__(self, filename):
        self.filename = filename
        self.settings = self._read_settings(filename)

    def _read_settings(self, filename):
        """
        設定ファイルを読み込む関数
        """
        with open("config.yaml") as f:
            settings = yaml.safe_load(f)
        return settings

def read_settings(filename):
    """
    設定ファイルを読み込む関数
    """
    with open(filename, "r") as f:
        lines = f.readlines()

def compute_products(x: np.ndarray) -> np.ndarray:
    """
    x = [
        x1: [x11, x12, x13, ...],
        x2: [x21, x22, x23, ...],
        ...
        xn: [xn1, xn2, xn3, ...]
    ]
    という二次元配列の各行から1つの要素を選ぶ全ての組み合わせに対して積を計算する
    """
    # itertools.productを使って各行から1つの要素を選ぶ全ての組み合わせを生成
    combinations = itertools.product(*x)
    
    # 各組み合わせの積を計算
    result = [np.prod(combination) for combination in combinations]
    
    return np.array(result)