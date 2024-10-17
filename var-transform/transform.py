import numpy as np
from scipy import interpolate
from noise_alg import laplace_func
import yaml
from settings import Config, Settings
from algorithms import get_alg

def transform(input_data1: np.ndarray, input_data2: np.ndarray, noise_func, settings: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data1: 隣接入力データ
    data2: 隣接入力データ
    noise_func: ノイズを加える関数 (Laplaceメカニズムなど)
    func: ノイズを加えた値に対して施す関数
        - 出力はスカラーorベクター(funcに依存する)も必要
    """
    config = Config(settings)
    alg = get_alg(config.algorithm)
    lower_bound = config.input["lower"]
    upper_bound = config.input["upper"]
    sample_num = config.input["sampling_num"]

    if noise_func == laplace_func:
        sample_num = 1000
        y_sample_data1 = [np.random.laplace(data, scale=1/0.1)  for data in input_data1 for _ in range(sample_num)]
        y_sample_data2 = [np.random.laplace(data, scale=1/0.1)  for data in input_data2 for _ in range(sample_num)]
        # range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2)), max(max(y_sample_data1), max(y_sample_data2)), 5000) # -100, 100
        # print(min(min(y_sample_data1), min(y_sample_data2)))
        # print(max(max(y_sample_data1), max(y_sample_data2)))
        # exit()

        dx = (upper_bound - lower_bound) / sample_num
        range_x = np.arange(lower_bound, upper_bound, dx)

        x1, x2, pdf1, pdf2 = alg(range_x, input_data1, input_data2, integral=config.integral)

        if x1.size == x2.size and (x1 == x2).all():
            return x1, pdf1, pdf2
        else:
            f1 = interpolate.interp1d(x1, pdf1, kind="cubic")
            f2 = interpolate.interp1d(x2, pdf2, kind="cubic")
            x_var = np.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), num=10000)
            pdf1 = f1(x_var)
            pdf2 = f2(x_var)
            return x_var, pdf1, pdf2
    else:
        raise NotImplementedError
