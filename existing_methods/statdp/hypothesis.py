import numpy as np
from scipy.stats import hypergeom

def hypothesis_test(alg_func, eps, input1, input2, event, iterations=100000):
    """
    Args:
        alg_func: アルゴリズムの関数
        eps: test epsilon
        input1: アルゴリズムの入力1
        input2: アルゴリズムの入力2
        event: イベント
    Return:
        p1: p値
        p2: p値
    """
    result_d1, result_d2 = [], []
    for _ in range(iterations):
        result_d1.append(alg_func(input1))
        result_d2.append(alg_func(input2))
    c1_cnt, c2_cnt = 0, 0
    for row in range(len(result_d1)):
        if event[0] <= result_d1[row] <= event[1]:
            c1_cnt += 1
        if event[0] <= result_d2[row] <= event[1]:
            c2_cnt += 1
    c1_cnt, c2_cnt = (c1_cnt, c2_cnt) if c1_cnt > c2_cnt else (c2_cnt, c1_cnt)
    p1 = calc_pvalue(c1_cnt, c2_cnt, iterations, eps)
    # p2 = calc_pvalue(c2_cnt, c1_cnt, iterations, eps)
    return p1

def calc_pvalue(c1, c2, n, eps):
    """
    Args:
        c1: イベントが発生した回数
        c2: イベントが発生した回数
        n: イテレーション回数
        eps: test epsilon
    Return:
        p値
    """
    c1_tilde = np.random.binomial(c1, 1/np.exp(eps))
    s = c1_tilde + c2
    return 1 - hypergeom.cdf(c1_tilde-1, n*2, s, n)
