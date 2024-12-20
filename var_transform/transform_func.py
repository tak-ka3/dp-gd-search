import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from transform_var import trasform_vars, trasform_var
from noise_alg import laplace_func
from scipy import integrate
from utils import spline_eq, spline_func, calc_prob, compress_range, nonuniform_convolution, plt_1d, plt_2d

def transform_sum(vals, pdf_vals, integral="gauss"):
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

    if val1[2] - val1[1] == val1[1] - val1[0]:
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
                # TODO: conv_x_rangeとresult_pdfのサイズが異なる
                plt_1d(conv_x_range, result_pdf, "convolution")
        assert conv_x_range.size == result_pdf.size
        return conv_x_range, result_pdf
    # 一つの確率密度関数の横軸の確率変数が等間隔ではない場合
    else:
        comp_val1, comp_result_pdf = val1, result_pdf
        conv_x_size = conv_x_range.size
        dx = conv_x_range[1] - conv_x_range[0]
        result_pdf = comp_result_pdf
        val1 = comp_val1
        # 各PDFを順次畳み込み
        for val2, pdf in zip(vals[1:], pdf_vals[1:]):
            # FFTを使って畳み込みを行う
            print("start conv")
            vals = []
            # 全ての確率変数の和の組み合わせを計算する
            for v1 in val1:
                for v2 in val2:
                    vals.append(v1 + v2)
            np_vals = np.unique(np.array(vals))
            split_num = np_vals.size // val1.size
            np_vals = np_vals[::split_num]
            # print("conv_prob1: ", calc_prob(val1, result_pdf))
            # print("conv_prob2: ", calc_prob(val2, pdf))
            # plt.scatter(val1, result_pdf, s=0.2, label="1")
            # plt.scatter(val2, pdf, s=0.2, label="2")
            # plt.title("before conv")
            # plt.legend()
            # plt.show()
            conv_result = nonuniform_convolution(val1, val2, result_pdf, pdf, np_vals, integral_way=integral)
            # plt.scatter(np_vals, conv_result, s=0.2)
            # plt.title("in the middle of conv")
            # plt.show()
            # print("end conv")
            # print("conv_prob: ", calc_prob(np_vals, conv_result))
            # exit()
            result_pdf = conv_result
            val1 = np_vals
        assert val1.size == result_pdf.size
        return val1, result_pdf

def transform_sum_after_func(range_x, input_data1, input_data2, func, integral="gauss"):
    """
    配列の確率変数にfuncという関数を適用した後に、合計を取るような関数の変換
    """
    # laplace_funを修正する
    x_splined1, transformed_pdf1 = trasform_vars(range_x, laplace_func(range_x, loc=input_data1), func)
    x_splined2, transformed_pdf2 = trasform_vars(range_x, laplace_func(range_x, loc=input_data2), func)

    if type(transformed_pdf1[0]) == np.ndarray:
        sum_x1, sum_pdf1 = transform_sum(x_splined1, transformed_pdf1, integral=integral)
        sum_x2, sum_pdf2 = transform_sum(x_splined2, transformed_pdf2, integral=integral)
        # print("prob_x1: ", calc_prob(sum_x1, sum_pdf1))
        # print("prob_x2: ", calc_prob(sum_x2, sum_pdf2))
    else:
        pass

    return sum_x1, sum_x2, sum_pdf1, sum_pdf2

def transform_eq(range_x, input_data1, input_data2, integral="gauss"):
    x1, pdf1 = trasform_vars(range_x, laplace_func(range_x, loc=input_data1), lambda x: x)
    x2, pdf2 = trasform_vars(range_x, laplace_func(range_x, loc=input_data2), lambda x: x)
    return x1, x2, pdf1, pdf2

def transform_exp_beta_sum(range_x, input_data1, input_data2, beta, integral="gauss"):
    return transform_sum_after_func(range_x, input_data1, input_data2, lambda x: np.exp(beta*x), integral=integral)

def transform_log(range_x1, range_x2, pdf1, pdf2):
    """
    ある単一の確率変数に対してlog関数を適用する変換
    """
    range_y1, pdf_y1 = trasform_var(range_x1, pdf1, lambda x: np.log(x))
    range_y2, pdf_y2 = trasform_var(range_x2, pdf2, lambda x: np.log(x))

    # x_splined1/2はそれぞれ二次元配列
    plt_2d([range_y1, range_y2], [pdf_y1, pdf_y2], title="after transform_log")
    return range_y1, range_y2, pdf_y1, pdf_y2

def transform_exp(range_x1, range_x2, pdf1, pdf2):
    """
    ある単一の確率変数に対してlog関数を適用する変換
    """
    range_y1, pdf_y1 = trasform_var(range_x1, pdf1, lambda x: np.exp(x))
    range_y2, pdf_y2 = trasform_var(range_x2, pdf2, lambda x: np.exp(x))

    # x_splined1/2はそれぞれ二次元配列
    plt_2d([range_y1, range_y2], [pdf_y1, pdf_y2], title="after transform_exp")
    return range_y1, range_y2, pdf_y1, pdf_y2

def transform_laplace_exp(range_x, input_data):
    x_list, pdf_list = [], []
    for mu in input_data:
        x, pdf = trasform_var(range_x, laplace_func(range_x, loc=mu), lambda x: np.exp(x))
        x_list.append(x)
        pdf_list.append(pdf)
    return x_list, pdf_list

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
    range_z = []
    for y_ind in range(len(range_y)):
        if range_y[y_ind] == 0:
            continue
        for x_ind in range(len(range_x)):            
            range_z.append(range_x[x_ind] / range_y[y_ind])
    # sorted_range_z = np.array(sorted(range_z)[::len(range_x)])
    np_range_z = np.array(range_z)
    unique_range_z = np.unique(np_range_z)
    split_num = unique_range_z.size // range_x.size
    split_range_z = unique_range_z[::split_num]
    f_interp = interpolate.CubicSpline(range_x, pdf_x, bc_type='natural', extrapolate=False)
    g_interp = interpolate.CubicSpline(range_y, pdf_y, bc_type='natural', extrapolate=False)
    pdf_z = []
    min_range_y = min(range_y)
    max_range_y = max(range_y)
    for z in split_range_z:
        integrand = lambda x: f_interp(z * x) * g_interp(x) * np.abs(x) if min(range_x) <= z*x <= max(range_x) else 0
        integral = integrate.quad(integrand, min_range_y, max_range_y, limit=100)[0]
        pdf_z.append(integral)
    print("div after: ", calc_prob(split_range_z, np.array(pdf_z)))
    return split_range_z, np.array(pdf_z)

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
