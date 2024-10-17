import numpy as np

def search_by_threshold(x, y1, y2) -> np.float64:
    """
    確率密度関数の閾値を求める
    1. 確率密度の上位N%の値を閾値とする
    2. 確率密度の値の最大値*N%の値を閾値とする
    """
    y1_threshold = np.sort(y1)[int(y1.size * 0.1)]
    y2_threshold = np.sort(y2)[int(y2.size * 0.1)]
    # threshold = max(max(y1)*0.01, max(y2)*0.01) # 値が離散的であるので、ほとんど要素を取れない場合も多い
    threshold = max(y1_threshold, y2_threshold)
    # threshold = 1e-10 # ここをプログラムの中で求めたい

    print("x: ", x)
    print("y1: ", y1)
    print("y2: ", y2)
    print("y1_size: ", y1.size)
    print("threshold: ", threshold)

    dx = x[1] - x[0]
    x_size = x.size
    cum_y1 = []
    cum_y2 = []
    tmp_cum_y1 = []
    tmp_cum_y2 = []
    for i in range(x_size):
        if y1[i] > threshold or y2[i] > threshold:
            if len(tmp_cum_y1) == 0:
                tmp_cum_y1.append(y1[i] * dx)
                tmp_cum_y2.append(y2[i] * dx)
            else:
                tmp_cum_y1.append(y1[i] * dx + tmp_cum_y1[-1])
                tmp_cum_y2.append(y2[i] * dx + tmp_cum_y2[-1])
        else:
            if len(tmp_cum_y1) != 0:
                cum_y1.append(tmp_cum_y1)
                cum_y2.append(tmp_cum_y2)
                tmp_cum_y1 = []
                tmp_cum_y2 = []

    if len(tmp_cum_y1) != 0:
        cum_y1.append(tmp_cum_y1)
        cum_y2.append(tmp_cum_y2)

    print("cum_y1: ", cum_y1)
    print("cum_y2: ", cum_y2)
    # 全探索
    max_ratio = 0
    for y1_list, y2_list in zip(cum_y1, cum_y2):
        for i in range(len(y1_list)-1):
            for j in range(i+1, len(y1_list)):
                tmp_ratio = (y1_list[j] - y1_list[i]) / (y2_list[j] - y2_list[i]) if (y1_list[j] - y1_list[i]) >  (y2_list[j] - y2_list[i]) else (y2_list[j] - y2_list[i]) / (y1_list[j] - y1_list[i])
                if max_ratio < tmp_ratio:
                    max_ratio = tmp_ratio
                    print(f"i: {i}, j: {j}, max_ratio: {max_ratio}")
    
    return np.log(max_ratio)

def search_all(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.float64:
    dx = (x[1] + x[0])/2 - x[0]
    print("dx in search_all: ", dx)
    cum_y1 = [y1[0] * dx]
    cum_y2 = [y2[0] * dx]
    for i in range(1, x.size-1):
        dx = (x[i+1] + x[i])/2 - (x[i] + x[i-1])/2
        cum_y1.append(cum_y1[-1] + y1[i] * dx)
        cum_y2.append(cum_y2[-1] + y2[i] * dx)
    dx = x[-1] - (x[-1] + x[-2])/2
    cum_y1.append(cum_y1[-1] + y1[-1] * dx)
    cum_y2.append(cum_y2[-1] + y2[-1] * dx)
    
    # 全探索
    x_size = x.size
    max_ratio = 0
    for i in range(x_size-1):
        for j in range(i+1, x_size):
            tmp_ratio = (cum_y1[j] - cum_y1[i]) / (cum_y2[j] - cum_y2[i]) if (cum_y1[j] - cum_y1[i]) >  (cum_y2[j] - cum_y2[i]) else (cum_y2[j] - cum_y2[i]) / (cum_y1[j] - cum_y1[i])
            if max_ratio < tmp_ratio:
                max_ratio = tmp_ratio
                print(f"i: {i}, j: {j}, max_ratio: {max_ratio}")
                print(f"cum_y1[j] - cum_y1[i]: {cum_y1[j] - cum_y1[i]}, cum_y2[j] - cum_y2[i]: {cum_y2[j] - cum_y2[i]}")

    return np.log(max_ratio)