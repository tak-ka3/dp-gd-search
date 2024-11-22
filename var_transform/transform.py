import numpy as np
from scipy import interpolate
from noise_alg import laplace_func
import yaml
from settings import Settings
from algorithms import get_alg

def transform(input_data1: np.ndarray, input_data2: np.ndarray, noise_func, settings: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data1: 隣接入力データ
    data2: 隣接入力データ
    noise_func: ノイズを加える関数 (Laplaceメカニズムなど)
    func: ノイズを加えた値に対して施す関数
        - 出力はスカラーorベクター(funcに依存する)も必要
    """
    alg = get_alg(settings.algorithm)
    lower_bound = settings.noisy_var["lower"]
    upper_bound = settings.noisy_var["upper"]
    sample_num = settings.noisy_var["sampling_num"]

    if noise_func == laplace_func:
        # y_sample_data1 = [np.random.laplace(data, scale=1/0.1)  for data in input_data1 for _ in range(sample_num)]
        # y_sample_data2 = [np.random.laplace(data, scale=1/0.1)  for data in input_data2 for _ in range(sample_num)]
        # range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2)), max(max(y_sample_data1), max(y_sample_data2)), 5000)

        dx = (upper_bound - lower_bound) / sample_num
        range_x = np.arange(lower_bound, upper_bound, dx)

        x1, x2, pdf1, pdf2 = alg(range_x, input_data1, input_data2, integral=settings.integral)

        if x1.ndim == x2.ndim == pdf1.ndim == pdf2.ndim == 1:
            if x1.size == x2.size and (x1 == x2).all():
                return x1, pdf1, pdf2
            else:
                f1 = interpolate.interp1d(x1, pdf1, kind="cubic")
                f2 = interpolate.interp1d(x2, pdf2, kind="cubic")
                x_var = np.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), num=10000)
                pdf1 = f1(x_var)
                pdf2 = f2(x_var)
                return x_var, pdf1, pdf2
        elif x1.ndim == x2.ndim == pdf1.ndim == pdf2.ndim == 2: # 出力がスカラ値ではなくベクトルであり、それぞれの要素が独立であり、その確率がわかっている場合
            new_x, new_pdf1, new_pdf2 = np.empty((0, x1[0].size)), np.empty((0, x1[0].size)), np.empty((0, x1[0].size))
            for ind, (x1_item, x2_item) in enumerate(zip(x1, x2)):
                if x1_item.size == x2_item.size and (x1_item == x2_item).all():
                    new_x = np.vstack((new_x, x1_item))
                    new_pdf1 = np.vstack((new_pdf1, pdf1[ind]))
                    new_pdf2 = np.vstack((new_pdf2, pdf2[ind]))
                else:
                    f1 = interpolate.interp1d(x1_item, pdf1[ind], kind="cubic")
                    f2 = interpolate.interp1d(x2_item, pdf2[ind], kind="cubic")
                    x_var = np.linspace(max(x1_item[0], x2_item[0]), min(x1_item[-1], x2_item[-1]), num=10000)
                    pdf1_item = f1(x_var)
                    pdf2_item = f2(x_var)
                    new_x = np.vstack((new_x, x_var))
                    new_pdf1 = np.vstack((new_pdf1, pdf1_item))
                    new_pdf2 = np.vstack((new_pdf2, pdf2_item))
            return new_x, new_pdf1, new_pdf2
        elif x1.ndim == x2.ndim == 2 and pdf1.ndim == pdf2.ndim == 1: # 出力がスカラ値ではなくベクトルであり、出力ベクトルごとに確率がわかっている場合
            if np.array_equal(x1, x2):
                return x1, pdf1, pdf2
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
