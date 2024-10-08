import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from transform import trasform_vars, trasform_var
from noise_alg import laplace_func

"""
Args:
    vals: 確率密度関数のxの範囲(=確率変数の範囲)の二次元配列
    pdf_vals: 確率密度関数の二次元配列
Return:
    conv_x_range: 確率変数の合計の範囲の一次元配列
    result_pdf: 確率変数の合計の確率密度関数の値の一次元配列
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

def transform_linear_sum(range_x, input_data1, input_data2):
    # laplace_funを修正する
    x_splined1, transformed_pdf1 = trasform_vars(range_x, laplace_func(range_x, loc=input_data1), lambda x: x)
    x_splined2, transformed_pdf2 = trasform_vars(range_x, laplace_func(range_x, loc=input_data2), lambda x: x)

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