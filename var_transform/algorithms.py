import numpy as np
from utils import plt_sca
from transform_func import transform_sum, transform_scalar_mul, transform_div, transform_log, transform_exp_beta_sum, transform_laplace_exp, transform_sum_after_func

def get_alg(func_name):
    return alg_dict[func_name]

def noisy_sum(range_x, input_data1, input_data2, integral="gauss"):
    """
    配列の要素に1をかけて合計をとる関数
    """
    return transform_sum_after_func(range_x, input_data1, input_data2, lambda x: x)

def noisy_max(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=beta, integral=integral)
    plt_sca(x1, pdf1, title="after transform_exp_beta_sum", x2=x2, y2=pdf2)
    x1_log, x2_log, pdf1_log, pdf2_log =  transform_log(x1, x2, pdf1, pdf2)
    x1_result, pdf1_result = transform_scalar_mul(x1_log, pdf1_log, 1/beta)
    x2_result, pdf2_result = transform_scalar_mul(x2_log, pdf2_log, 1/beta)
    plt_sca(x1_result, pdf1_result, title="after transform_log", x2=x2_result, y2=pdf2_result)
    return x1_result, x2_result, pdf1_result, pdf2_result

def noisy_arg_max(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    配列の中の最大の要素を持つインデックスを返すアルゴリズム
    TODO: エラーが出るの修正中
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

def noisy_hist(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    ノイズ付きヒストグラムのアルゴリズム
    TODO: 実装中
    """
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=beta)
    plt_sca(x1, pdf1, title="after transform_exp_beta_sum", x2=x2, y2=pdf2)
    x1_log, x2_log, pdf1_log, pdf2_log =  transform_log(x1, x2, pdf1, pdf2)
    x1_result, pdf1_result = transform_scalar_mul(x1_log, pdf1_log, 1/beta)
    x2_result, pdf2_result = transform_scalar_mul(x2_log, pdf2_log, 1/beta)
    plt_sca(x1_result, pdf1_result, title="after transform_log", x2=x2_result, y2=pdf2_result)
    return x1_result, x2_result, pdf1_result, pdf2_result

alg_dict = {
    "noisy_sum": noisy_sum,
    "noisy_max": noisy_max,
    "noisy_arg_max": noisy_arg_max,
    "noisy_hist": noisy_hist
}
