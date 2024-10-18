import numpy as np

def search_by_threshold(x, y1, y2, th=0.1) -> np.float64:
    """
    二つの確率密度がそれぞれ下位thの割合以下の値である場合は無視して探索する
    """
    y1_threshold = np.sort(y1)[int(y1.size * th)]
    y2_threshold = np.sort(y2)[int(y2.size * th)]
    threshold = max(y1_threshold, y2_threshold)

    max_ratio = 0
    for i in range(x.size):
        if y1[i] > threshold or y2[i] > threshold:
            ratio = y1[i] / y2[i] if y1[i] > y2[i] else y2[i] / y1[i]
            if max_ratio < ratio:
                max_ratio = ratio
    return np.log(max_ratio)

def search_all(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.float64:
    """
    確率密度の比率の最大値を全探索によって求める
    """
    max_ratio = 0
    for i in range(x.size):
        ratio = y1[i] / y2[i] if y1[i] > y2[i] else y2[i] / y1[i]
        if max_ratio < ratio:
            max_ratio = ratio
    return np.log(max_ratio)
