import numpy as np
from utils import plt_2d
from transform_func import transform_sum, transform_scalar_mul, transform_div, transform_log, transform_exp_beta_sum, transform_laplace_exp, transform_sum_after_func, transform_eq
from matplotlib import pyplot as plt
from noise_alg import laplace_func, laplace_cdf
import itertools

def get_alg(func_name):
    return alg_dict[func_name]

def noisy_sum(range_x, input_data1, input_data2, integral="gauss"):
    """
    配列の要素に1をかけて合計をとる関数
    """
    eps = 0.1 # TODO: アルゴリズム内で使用されるεの値だが、laplace_func内でそれが定義されている
    return transform_sum_after_func(range_x, input_data1, input_data2, lambda x: x)

def noisy_max(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    eps = 0.1 # TODO: アルゴリズム内で使用されるεの値だが、laplace_func内でそれが定義されている
    x1, x2, pdf1, pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=beta, integral=integral)
    plt_2d([x1, x2], [pdf1, pdf2], title="after transform_exp_beta_sum")
    x1_log, x2_log, pdf1_log, pdf2_log =  transform_log(x1, x2, pdf1, pdf2)
    x1_result, pdf1_result = transform_scalar_mul(x1_log, pdf1_log, 1/beta)
    x2_result, pdf2_result = transform_scalar_mul(x2_log, pdf2_log, 1/beta)
    plt_2d([x1_result, x2_result], [pdf1_result, pdf2_result], title="after transform_log")
    return x1_result, x2_result, pdf1_result, pdf2_result

def noisy_arg_max(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    配列の中の最大の要素を持つインデックスを返すアルゴリズム
    TODO: エラーが出るの修正中
    """
    eps = 0.1 # TODO: アルゴリズム内で使用されるεの値だが、laplace_func内でそれが定義されている
    print("start transform_arg_max")
    sum_x1, sum_x2, sum_pdf1, sum_pdf2 = transform_exp_beta_sum(range_x, input_data1, input_data2, beta=1.0, integral=integral)
    print("start transform_laplace_exp")
    x1_list, pdf1_list = transform_laplace_exp(range_x, input_data1)
    x2_list, pdf2_list = transform_laplace_exp(range_x, input_data2)
    size_n = len(x1_list)
    indices = np.linspace(1/size_n, 1, size_n)
    x1_sm_list, pdf1_sm_list, x2_sm_list, pdf2_sm_list = [], [], [], []
    x1_sm_list_all = [transform_scalar_mul(*transform_div(x, pdf, sum_x1, sum_pdf1), ind * size_n) for ind, x, pdf in zip(indices, x1_list, pdf1_list)]
    x2_sm_list_all = [transform_scalar_mul(*transform_div(x, pdf, sum_x2, sum_pdf2), ind * size_n) for ind, x, pdf in zip(indices, x2_list, pdf2_list)]
    print("start transform_sum")
    for (x1_sm, pdf1_sm), (x2_sm, pdf2_sm) in zip(x1_sm_list_all, x2_sm_list_all):
        x1_sm_list.append(x1_sm)
        pdf1_sm_list.append(pdf1_sm)
        x2_sm_list.append(x2_sm)
        pdf2_sm_list.append(pdf2_sm)
    print("x1_sm_list: ", x1_sm_list)
    x1_argmax, pdf1_argmax = transform_sum(x1_sm_list, pdf1_sm_list, integral=integral)
    x2_argmax, pdf2_argmax = transform_sum(x2_sm_list, pdf2_sm_list, integral=integral)
    print("x2_argmax: ", x2_argmax)
    return x1_argmax, x2_argmax, pdf1_argmax, pdf2_argmax

def noisy_max_cdf(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    累積分布関数を用いた最大値出力アルゴリズム
    消費されるのは、eps*|Q|
    """
    eps = 0.1 
    pdf1 = np.zeros(range_x.size)
    for x_i, x in enumerate(range_x):
        prob = 0
        for i in range(0, input_data1.size):
            f = laplace_cdf(x, loc=input_data1[i], eps=eps)
            for j in range(0, input_data1.size):
                if i == j:
                    continue
                f *= laplace_cdf(x, loc=input_data1[j], eps=eps)
            prob += f
        pdf1[x_i] = prob
   
    pdf2 = np.zeros(range_x.size)
    for x_i, x in enumerate(range_x):
        prob = 0
        for i in range(0, input_data2.size):
            f = laplace_cdf(x, loc=input_data2[i], eps=eps)
            for j in range(0, input_data2.size):
                if i == j:
                    continue
                f *= laplace_cdf(x, loc=input_data2[j], eps=eps)
            prob += f
        pdf2[x_i] = prob
    return range_x, range_x, pdf1, pdf2

def noisy_arg_max_cdf(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    累積分布関数を用いた最大値インデックス探索アルゴリズム
    アルゴリズムで消費されるプライバシーコストをeps/2とすると、全体として消費されるプライバシーコストはepsとなる
    """
    eps = 0.1/2
    # 入力のサイズ = インデックスの最大値
    input_length = input_data1.size
    y_range = np.arange(0, input_length)
    # 出力の範囲
    pdf1 = np.zeros(input_length)
    for i in range(input_length):
        prob = 0
        for x in range_x:
            f = laplace_func(x, loc=input_data1[i], eps=eps)
            for j in range(input_length):
                if i == j:
                    continue
                f *= laplace_cdf(x, loc=input_data1[j], eps=eps)
            prob += f
        pdf1[i] = prob

    pdf2 = np.zeros(input_length)
    for i in range(input_length):
        prob = 0
        for x in range_x:
            f = laplace_func(x, loc=input_data2[i], eps=eps)
            for j in range(input_length):
                if i == j:
                    continue
                f *= laplace_cdf(x, loc=input_data2[j], eps=eps)
            prob += f
        pdf2[i] = prob
    return y_range, y_range, pdf1, pdf2
            

def noisy_hist(range_x, input_data1, input_data2, beta=2.0, integral="gauss"):
    """
    ノイズ付きヒストグラムのアルゴリズム
    """
    eps = 0.1 # TODO: アルゴリズム内で使用されるεの値だが、laplace_func内でそれが定義されている
    return transform_eq(range_x, input_data1, input_data2)

def noisy_svt(range_x, input_data1, input_data2, beta=2.0, integral="trapz"):
    """
    ノイズ付きSVTのアルゴリズム
    """
    eps = 0.1
    N = 1
    T = 0.5 # 2.5でも良い
    input_size = len(input_data1)
    ranges = [range_x] * input_size  # 各ループの範囲を定義
    # ノイズ付与後の確率変数の組み合わせを全探索
    input_comb = list(itertools.product(*ranges))
    output_dir_list = []
    for input_data in (input_data1, input_data2):
        output_dir = {}
        sens = 1
        for ita1 in range_x:
            for pr_vars in input_comb:
                output = svt(pr_vars, ita1, N, T)
                ita1_pr = laplace_func(ita1, loc=T, sensitivity=sens, eps=eps/2)
                ita2_pr = 1
                for i, pr_var in enumerate(pr_vars):
                    ita2_pr *= laplace_func(pr_var, loc=input_data[i], sensitivity=sens, eps=eps/(4*N))
                pr = ita1_pr * ita2_pr
                output_tuple = tuple(output)
                if output_tuple not in output_dir:
                    output_dir[output_tuple] = pr
                else:
                    output_dir[output_tuple] += pr
        output_dir_list.append(output_dir)
    
    output_dir1, output_dir2 = output_dir_list
    print(output_dir1)
    exit()
    x1, x2 = np.array(list(output_dir1.keys())), np.array(list(output_dir2.keys()))
    pdf1, pdf2 = np.array(list(output_dir1.values())), np.array(list(output_dir2.values()))
    return x1, x2, pdf1, pdf2

def noisy_output_based_svt(range_x, input_data1, input_data2, beta=2.0, integral="trapz"):
    """
    出力の結果から逆算して、その出力を得るための入力の範囲を先に求め、確率をまとめて計算することで効率化を図る
    """
    eps = 0.1
    N = 1
    T = 0.5 # 2.5でも良い
    # 出力パターンを洗い出す
    # 制約としては1の数がN以下であること。1がN個ある場合はそれ以降は-1になる。0と-1が隣り合うことはない。
    input_size = len(input_data1)
    # TODO: 出力パターンは求めたことにする
    output_patters = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, -1], [0, 0, 1, -1, -1], [0, 1, -1, -1, -1], [1, -1, -1, -1, -1]]
    out0_in1_prs = []
    out0_in2_prs = []
    out1_in1_prs = []
    out1_in2_prs = []
    for _ in range(input_size):
        out0_in1_pr = 0
        out0_in2_pr = 0
        out1_in1_pr = 0
        out1_in2_pr = 0
        # 閾値を動かす
        for ita1 in range_x:
            T_tilde = T + ita1
            # T_tilde未満である確率が0を出力する確率
            for input_data in input_data1:
                ita1_pr = laplace_func(ita1, loc=T, sensitivity=1, eps=eps/2)
                ita2_pr = laplace_cdf(T_tilde, loc=input_data, sensitivity=1, eps=eps/(4*N))
                out0_in1_pr += ita1_pr * ita2_pr
                out1_in1_pr += ita1_pr * (1 - ita2_pr)
            for input_data in input_data2:
                ita1_pr = laplace_func(ita1, loc=T, sensitivity=1, eps=eps/2)
                ita2_pr = laplace_cdf(T_tilde, loc=input_data, sensitivity=1, eps=eps/(4*N))
                out0_in2_pr += ita1_pr * ita2_pr
                out1_in2_pr += ita1_pr * (1 - ita2_pr)
        out0_in1_prs.append(out0_in1_pr)
        out0_in2_prs.append(out0_in2_pr)
        out1_in1_prs.append(out1_in1_pr)
        out1_in2_prs.append(out1_in2_pr)
    
    output_pr_dir1 = {}
    output_pr_dir2 = {}
    for output in output_patters:
        # 出力パターンは要素間の関係性に影響されるが、それぞれの出力が出る確率は独立であると言える
        output_pr1 = 1
        output_pr2 = 1
        for i, val in enumerate(output):
            if val == 0:
                output_pr1 *= out0_in1_prs[i]
                output_pr2 *= out0_in2_prs[i]
            elif val == 1:
                output_pr1 *= out1_in1_prs[i]
                output_pr2 *= out1_in2_prs[i]
            else:
                break
        output_pr_dir1[tuple(output)] = output_pr1
        output_pr_dir2[tuple(output)] = output_pr2
    x1, x2 = np.array(list(output_pr_dir1.keys())), np.array(list(output_pr_dir2.keys()))
    pdf1, pdf2 = np.array(list(output_pr_dir1.values())), np.array(list(output_pr_dir2.values()))
    return x1, x2, pdf1, pdf2

def svt(query, ita1, N=1, T=2.5):
    output = []
    T_tilde = T + ita1
    count = 0
    for q in query:
        if count >= N:
            output.append(-1)
        elif q >= T_tilde:
            count += 1
            output.append(1)
        else:
            output.append(0)
    return output

alg_dict = {
    "noisy_sum": noisy_sum,
    "noisy_max": noisy_max,
    "noisy_arg_max": noisy_arg_max,
    "noisy_hist": noisy_hist,
    "noisy_max_cdf": noisy_max_cdf,
    "noisy_arg_max_cdf": noisy_arg_max_cdf,
    "noisy_svt": noisy_svt,
    "noisy_output_based_svt": noisy_output_based_svt
}
