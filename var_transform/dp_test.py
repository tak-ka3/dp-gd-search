import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from search import search_all, search_by_threshold
from noise_alg import laplace_func
from transform import transform
from settings import Settings
import yaml
import os
from datetime import datetime
import shutil

def dp_test(input_data1: np.ndarray, input_data2: np.ndarray) -> np.float64:
    start_time = datetime.now()
    with open("settings.yaml") as f:
        data = yaml.safe_load(f)
    settings = Settings(**data)
    x, y1, y2 = transform(input_data1, input_data2, laplace_func, settings)
    plt.scatter(x, y1, color="green", s=0.2, label="x1")
    plt.scatter(x, y2, color="orange", s=0.2, label="x2")
    plt.legend()
    plt.title("result of the probability density function")
    
    if settings.search["way"] == "all":
        eps = search_all(x, y1, y2)
    elif settings.search["way"] == "threshold":
        th = settings.search["threshold"]
        eps = search_by_threshold(x, y1, y2, th=th)

    # 結果を保存
    exec_time = datetime.now() - start_time
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(f"experiments/{formatted_now}", exist_ok=True)
    plt.savefig(f"experiments/{formatted_now}/result.png")
    data["result"] = {"eps": eps.item(), "time(s)": exec_time.total_seconds()}
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
    x_data2 = np.array([2.0, 4.0, 6.0, 8.0])
    eps = dp_test(x_data1, x_data2)

    print("estimated eps: ", eps)
