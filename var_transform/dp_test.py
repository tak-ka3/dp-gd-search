import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from search import search_scalar_all, search_scalar_by_threshold, search_vec_all
from noise_alg import laplace_func
from transform import transform
from settings import Settings
import yaml
import os
from datetime import datetime
from utils import compute_products, plt_2d

def dp_test(input_data1: np.ndarray, input_data2: np.ndarray) -> np.float64:
    start_time = datetime.now()
    with open("settings.yaml") as f:
        data = yaml.safe_load(f)
    settings = Settings(**data)
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
        
        if settings.search["way"] == "all":
            eps = search_vec_all(x, y1, y2)
    else:
        raise NotImplementedError

    # 結果を保存
    exec_time = datetime.now() - start_time
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(f"experiments/{formatted_now}", exist_ok=True)
    plt.savefig(f"experiments/{formatted_now}/result.png")
    data["result"] = {"eps": eps.item(), "time(s)": exec_time.total_seconds()}
    data["input"] = {"data1": input_data1.tolist(), "data2": input_data2.tolist()}
    with open(f"experiments/{formatted_now}/result.yaml", "w") as f:
        yaml.dump(data, f, encoding='utf-8', allow_unicode=True)
    
    return eps

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    x_data1 = np.array([1.0, 3.0, 5.0, 7.0])
    x_data2 = np.array([1.5, 2.5, 5.0, 7.0])
    eps = dp_test(x_data1, x_data2)

    print("estimated eps: ", eps)
