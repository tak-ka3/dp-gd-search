import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from transform import trasform_vars, trasform_var
from noise_alg import laplace_func
from scipy import integrate
from utils import spline_eq, spline_func, calc_prob, compress_range, nonuniform_convolution, plt_sca

def transform_sum(vals, pdf_vals):
    """
    Args:
        vals: 確率密度関数のxの範囲(=確率変数の範囲)の二次元配列
        pdf_vals: 確率密度関数の二次元配列
    Return:
        conv_x_range: 確率変数の合計の範囲の一次元配列
        result_pdf: 確率変数の合計の確率密度関数の値の一次元配列
    """
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

    print(val1[2] - val1[1])
    print(val1[1] - val1[0])
    if val1[2] - val1[1] == val1[1] - val1[0]:
        # 各PDFを順次畳み込み
        for val2, pdf in zip(vals[1:], pdf_vals[1:]):
            # TODO: ifとelseの処理が重複している
            if dx == val2[1] - val2[0]:
                print("start conv 1")
                result_pdf = np.convolve(result_pdf, pdf, mode='full') * dx
                start_val = conv_x_range[0] + val2[0]
                conv_x_size = conv_x_size + val2.size - 1
                print("size: ", len(result_pdf))
                print("size: ", conv_x_size)
                conv_x_range = np.arange(start_val, start_val + dx*conv_x_size, dx)
                print("size(conv_x_range): ", len(conv_x_range))
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
                print("conv_x_range: ", conv_x_range)
                # TODO: conv_x_rangeとresult_pdfのサイズが異なる
                plt_sca(conv_x_range, result_pdf, "convolution")
        assert conv_x_range.size == result_pdf.size
        return conv_x_range, result_pdf
    # 一つの確率密度関数の横軸の確率変数が等間隔ではない場合
    else:
        # TODO: ここで確率が0.9以下の範囲になるように、確率変数を圧縮する。ここの圧縮が適切かどうかのFBを受けて再帰的に調節することも可能性としてある
        # TODO: この圧縮は関数呼び出しの時にTrue/Falseを返すようにする
        comp_val1, comp_result_pdf = compress_range(val1, result_pdf, th=0.9)
        conv_x_size = conv_x_range.size
        dx = conv_x_range[1] - conv_x_range[0]
        result_pdf = comp_result_pdf
        val1 = comp_val1
        # 各PDFを順次畳み込み
        for val2, pdf in zip(vals[1:], pdf_vals[1:]):
            # FFTを使って畳み込みを行う
            # val2, pdf = compress_range(val2, pdf, th=0.9)
            print("start conv")
            conv_result = nonuniform_convolution(val1, val2, result_pdf, pdf, val1)
            print("end conv")
            print("conv_prob: ", calc_prob(val1, conv_result))
            result_pdf = conv_result
        assert val1.size == result_pdf.size
        return val1, result_pdf

def transform_sum_after_func(range_x, input_data1, input_data2, func):
    """
    配列の確率変数にfuncという関数を適用した後に、合計を取るような関数の変換
    """
    # laplace_funを修正する
    x_splined1, transformed_pdf1 = trasform_vars(range_x, laplace_func(range_x, loc=input_data1), func)
    x_splined2, transformed_pdf2 = trasform_vars(range_x, laplace_func(range_x, loc=input_data2), func)
    # x_splined1/2はそれぞれ二次元配列
    # assert (x_splined1[0] == x_splined2[0]).all() # 必ずしもx軸を揃える必要はない

    # lambda x: x を適用後の確率密度関数を可視化する
    for x_val1, x_val2, pdf1, pdf2 in zip(x_splined1, x_splined2, transformed_pdf1, transformed_pdf2):
        plt.scatter(x_val1, pdf1, s=0.5, color="blue")
        plt.scatter(x_val2, pdf2, s=0.5, color="red")
    plt.title("after transform_func before sum")
    plt.show()

    if type(transformed_pdf1[0]) == np.ndarray:
        sum_x1, sum_pdf1 = transform_sum(x_splined1, transformed_pdf1)
        sum_x2, sum_pdf2 = transform_sum(x_splined2, transformed_pdf2)
        print("prob_x1: ", calc_prob(sum_x1, sum_pdf1))
        print("prob_x2: ", calc_prob(sum_x2, sum_pdf2))
    else:
        pass

    return sum_x1, sum_x2, sum_pdf1, sum_pdf2

def transform_linear_sum(range_x, input_data1, input_data2):
    """
    配列の要素に1をかけて合計をとる関数の変換
    """
    return transform_sum_after_func(range_x, input_data1, input_data2, lambda x: x)

def transform_exp_beta_sum(range_x, input_data1, input_data2, beta):
    return transform_sum_after_func(range_x, input_data1, input_data2, lambda x: np.exp(beta*x))

def transform_log(range_x1, range_x2, pdf1, pdf2):
    """
    ある単一の確率変数に対してlog関数を適用する変換
    """
    range_y1, pdf_y1 = trasform_var(range_x1, pdf1, lambda x: np.log(x))
    range_y2, pdf_y2 = trasform_var(range_x2, pdf2, lambda x: np.log(x))

    # x_splined1/2はそれぞれ二次元配列
    plt_sca(range_y1, pdf_y1, title="after transform_log", x2=range_y2, y2=pdf_y2)
    return range_y1, range_y2, pdf_y1, pdf_y2

def transform_exp(range_x1, range_x2, pdf1, pdf2):
    """
    ある単一の確率変数に対してlog関数を適用する変換
    """
    range_y1, pdf_y1 = trasform_var(range_x1, pdf1, lambda x: np.exp(x))
    range_y2, pdf_y2 = trasform_var(range_x2, pdf2, lambda x: np.exp(x))

    # x_splined1/2はそれぞれ二次元配列
    plt_sca(range_y1, pdf_y1, title="after transform_exp", x2=range_y2, y2=pdf_y2)
    return range_y1, range_y2, pdf_y1, pdf_y2

def transform_logsumexp(range_x, input_data1, input_data2, beta=2.0):
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=beta)
    plt_sca(x1, pdf1, title="after transform_exp_beta_sum", x2=x2, y2=pdf2)
    x1_log, x2_log, pdf1_log, pdf2_log =  transform_log(x1, x2, pdf1, pdf2)
    x1_result, pdf1_result = transform_scalar_mul(x1_log, pdf1_log, 1/beta)
    x2_result, pdf2_result = transform_scalar_mul(x2_log, pdf2_log, 1/beta)
    plt_sca(x1_result, pdf1_result, title="after transform_log", x2=x2_result, y2=pdf2_result)
    return x1_result, x2_result, pdf1_result, pdf2_result

def transform_laplace_exp(range_x, input_data):
    x_list, pdf_list = [], []
    for mu in input_data:
        x, pdf = trasform_var(range_x, laplace_func(range_x, loc=mu), lambda x: np.exp(x))
        x_list.append(x)
        pdf_list.append(pdf)
    return x_list, pdf_list

def trasform_arg_max(range_x, input_data1, input_data2, beta=2.0):
    """
    配列の中の最大の要素を持つインデックスを返すアルゴリズム
    """
    print("start transform_arg_max")
    sum_x1, sum_x2, sum_pdf1, sum_pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=1.0)
    print("start transform_laplace_exp")
    x1_list, pdf1_list = transform_laplace_exp(range_x, input_data1)
    x2_list, pdf2_list = transform_laplace_exp(range_x, input_data2)
    size_n = len(x1_list)
    indices = np.linspace(0, 1, size_n)
    x1_sm_list, pdf1_sm_list, x2_sm_list, pdf2_sm_list = [], [], [], []
    x1_sm_list_all = [transform_scalar_mul(*transform_div(x, pdf, sum_x1, sum_pdf1), ind * (size_n-1)) for ind, x, pdf in zip(indices, x1_list, pdf1_list)]
    x2_sm_list_all = [transform_scalar_mul(*transform_div(x, pdf, sum_x2, sum_pdf2), ind * (size_n-1)) for ind, x, pdf in zip(indices, x2_list, pdf2_list)]
    print("start transform_sum")
    for (x1_sm, pdf1_sm), (x2_sm, pdf2_sm) in zip(x1_sm_list_all, x2_sm_list_all):
        x1_sm_list.append(x1_sm)
        pdf1_sm_list.append(pdf1_sm)
        x2_sm_list.append(x2_sm)
        pdf2_sm_list.append(pdf2_sm)
    print(x1_sm_list)
    x1_argmax, pdf1_argmax = transform_sum(x1_sm_list, pdf1_sm_list)
    x2_argmax, pdf2_argmax = transform_sum(x2_sm_list, pdf2_sm_list)
    return x1_argmax, pdf1_argmax, x2_argmax, pdf2_argmax

def transform_noisy_hist(range_x, input_data1, input_data2, beta=2.0):
    """
    ノイズ付きヒストグラムのアルゴリズム
    """
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=beta)
    plt_sca(x1, pdf1, title="after transform_exp_beta_sum", x2=x2, y2=pdf2)
    x1_log, x2_log, pdf1_log, pdf2_log =  transform_log(x1, x2, pdf1, pdf2)
    x1_result, pdf1_result = transform_scalar_mul(x1_log, pdf1_log, 1/beta)
    x2_result, pdf2_result = transform_scalar_mul(x2_log, pdf2_log, 1/beta)
    plt_sca(x1_result, pdf1_result, title="after transform_log", x2=x2_result, y2=pdf2_result)
    return x1_result, x2_result, pdf1_result, pdf2_result

def transform_mul(range_x, pdf_x, range_y, pdf_y, point_num=5000):
    """
    Z = XYの変換
    """
    # TODO: z = x * yの確率変数の範囲の探し方。基本的には端点の積が最大/最小になる
    max_max_z = max(range_x) * max(range_y)
    max_min_z = max(range_x) * min(range_y)
    min_max_z = min(range_x) * max(range_y)
    min_min_z = min(range_x) * min(range_y)
    max_z = max(max_max_z, max_min_z, min_max_z, min_min_z)
    min_z = min(max_max_z, max_min_z, min_max_z, min_min_z)
    range_z = np.linspace(min_z, max_z, point_num)
    f_interp = interpolate.CubicSpline(range_x, pdf_x, bc_type='natural', extrapolate=False)
    g_interp = interpolate.CubicSpline(range_y, pdf_y, bc_type='natural', extrapolate=False)
    pdf_z = []
    min_range_x = min(range_x)
    max_range_x = max(range_x)
    for z in range_z:
        integrand = lambda x: f_interp(z / x) * g_interp(x) / np.abs(x)
        integral = integrate.quad(integrand, min_range_x, max_range_x, limit=100)[0]
        pdf_z.append(integral)
    return range_z, pdf_z

def transform_div(range_x, pdf_x, range_y,  pdf_y, point_num=5000):
    """
    Z = X / Yの変換
    """
    # TODO: z = x / yの確率変数の範囲の探し方。必ずしも正負を考えると端点を考えれば良いわけではない。
    max_max_z = max(range_x) / max(range_y)
    max_min_z = max(range_x) / min(range_y)
    min_max_z = min(range_x) / max(range_y)
    min_min_z = min(range_x) / min(range_y)
    max_z = max(max_max_z, max_min_z, min_max_z, min_min_z)
    min_z = min(max_max_z, max_min_z, min_max_z, min_min_z)
    range_z = np.linspace(min_z, max_z, point_num)
    f_interp = interpolate.CubicSpline(range_x, pdf_x, bc_type='natural', extrapolate=False)
    g_interp = interpolate.CubicSpline(range_y, pdf_y, bc_type='natural', extrapolate=False)
    pdf_z = []
    min_range_x = min(range_x)
    max_range_x = max(range_x)
    for z in range_z:
        integrand = lambda x: f_interp(z * x) * g_interp(x) * np.abs(x)
        integral = integrate.quad(integrand, min_range_x, max_range_x, limit=100)[0]
        pdf_z.append(integral)
    return range_z, pdf_z

def transform_scalar_mul(range_x, pdf_x, a):
    """
    ある確率変数に対してaをかける変換
    """
    range_y = range_x * a
    pdf_y = pdf_x / a
    return range_y, pdf_y

def transform_scalar_add(range_x, pdf_x, a):
    """
    ある確率変数に対してaを足す変換
    """
    range_y = range_x + a
    pdf_y = pdf_x
    return range_y, pdf_y
