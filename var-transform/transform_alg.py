import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from transform import trasform_vars, trasform_var
from noise_alg import laplace_func
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
    plt_sca(range_y1, pdf_y1, title="after transform_log", x2=range_y2, y2=pdf_y2, s=0.5)
    return range_y1, range_y2, pdf_y1, pdf_y2

def transform_logsumexp(range_x, input_data1, input_data2):
    # TODO: 畳み込みによりなぜかpdf1の値が負の値になる
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=1)
    plt_sca(x1, pdf1, title="after transform_exp_beta_sum", x2=x2, y2=pdf2)
    return transform_log(x1, x2, pdf1, pdf2)
