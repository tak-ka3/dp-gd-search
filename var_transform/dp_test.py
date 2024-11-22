import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from search import search_scalar_all, search_scalar_by_threshold, search_vec_all
from noise_alg import laplace_func
from transform import transform
from settings import Settings
from input_generator import input_generator
import yaml
import os
from datetime import datetime
from utils import compute_products, plt_2d

def dp_test(input_data1: np.ndarray, input_data2: np.ndarray, settings: Settings) -> np.float64:
    x, y1, y2 = transform(input_data1, input_data2, laplace_func, settings)
    
    if x.ndim == y1.ndim == y2.ndim == 1:
        plt_2d([x, x], [y1, y2], title="result of the probability density function")
        if settings.search["way"] == "all":
            eps = search_scalar_all(x, y1, y2)
        elif settings.search["way"] == "threshold":
            th = settings.search["threshold"]
            eps = search_scalar_by_threshold(x, y1, y2, th=th)
    elif x.ndim == y1.ndim == y2.ndim == 2: # 出力がスカラ値ではなく、ベクトルの場合
        # TODO: 出力の確率変数が独立であるとみなして、同時確率密度関数を計算する
        assert y1.shape == y2.shape and y1.ndim == y2.ndim == 2
        pdf1 = compute_products(y1)
        pdf2 = compute_products(y2)
        x_flattened = compute_products(x)
        eps = search_scalar_all(x_flattened, pdf1, pdf2)
        
        # if settings.search["way"] == "all":
        #     eps = search_vec_all(x, y1, y2)
    elif x.ndim == 2 and y1.ndim == y2.ndim == 1: # 出力がスカラ値ではなく、ベクトルの場合 (e.g. Noisy_SVT)
        eps = search_scalar_all(x, y1, y2)
    else:
        raise NotImplementedError
    
    return eps

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    start_time = datetime.now()
    with open("settings.yaml") as f:
        data = yaml.safe_load(f)
    settings = Settings(**data)

    max_eps = 0
    max_input = []
    tmp_eps = []
    input_list = input_generator("1" if settings.algorithm == "noisy_hist" else "inf", size=settings.input["size"])
    # input_list.insert(0, [[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
    for input_data1, input_data2 in input_list:
        eps = dp_test(np.array(input_data1), np.array(input_data2), settings)
        tmp_eps.append(eps)
        if eps > max_eps:
            max_input = [input_data1, input_data2]
            max_eps = eps
        print("tmp eps: ", eps)
    
    # 結果を保存
    exec_time = datetime.now() - start_time
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(f"experiments/{settings.algorithm}", exist_ok=True)
    dir_path = f"experiments/{settings.algorithm}/{formatted_now}"
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(f"{dir_path}/result.png")
    data["result"] = {"eps": max_eps.item(), "time(s)": exec_time.total_seconds()}
    data["input"] = {"data1": max_input[0], "data2": max_input[1], "size": settings.input["size"]}
    with open(f"{dir_path}/result.yaml", "w") as f:
        yaml.dump(data, f, encoding='utf-8', allow_unicode=True)

    # print("tmp eps: ", tmp_eps)
