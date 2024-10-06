import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

"""
ノイズを付与後に適用する関数
"""
def test_func(x: np.array):
    # return x**5
    return np.exp(x)
    return x

def spline1(x,y,point):
    f = interpolate.interp1d(x, y,kind="cubic") 
    # x座標を等間隔にしてくれる
    X = np.linspace(x[0],x[-1],num=point,endpoint=True)
    Y = f(X)
    return X,Y

def inv_calc(x, y):
    x_y_arr = np.array([x, y])
    sorted_coord_indices = np.argsort(x_y_arr[1, :])
    sorted_coord = x_y_arr[:, sorted_coord_indices]
    return sorted_coord[1], sorted_coord[0]

def diff_calc(x, y, point):
    x_diff = []
    dx2 = ((x[1] - x[0]) * 2).astype(float)
    x_diff = [(y[2] - y[0]) / dx2]
    # x_diff = []
    for i in range(point - 2):
        x_diff.append((y[i+2] - y[i])/ dx2)
    x_diff.append(x_diff[-1])
    return x_diff

def laplace_func(x, loc=0, sensitivity=1, eps=0.1):
    b = sensitivity / eps
    return np.exp(-np.abs(x - loc)/b) / (2*b)

# 元の確率変数の確率密度関数であるoriginal_fを仮定した変数変換
def var_trasform(y, x_diff, original_f, loc=0):
    if type(loc) == np.ndarray:
        return [original_f(y, loc=val) * np.abs(x_diff) for val in loc]
    else:
        return original_f(y, loc=loc) * np.abs(x_diff)

def convolve_distributions(pdfs, x_range):
    # 最初の分布のPDFをセット
    result_pdf = pdfs[0]

    dx = x_range[1] - x_range[0]

    conv_x_range = x_range

    x_size = x_range.size
    conv_x_size = x_size
    
    # 各PDFを順次畳み込み
    for pdf in pdfs[1:]:
        # 本来であればx_rangeの範囲外からも畳み込むべきだが、それができていない
        # 範囲外の部分が0と見做せるほど小さいと考えるので問題ないか
        # https://deepage.net/features/numpy-convolve.html
        result_pdf = np.convolve(result_pdf, pdf, mode='full') * dx
        start_val = conv_x_range[0] + x_range[0]
        conv_x_size = conv_x_size + x_size - 1
        conv_x_range = np.arange(start_val, start_val + dx*conv_x_size, dx)
    
    return conv_x_range, result_pdf

"""
data1: 隣接入力データ
data2: 隣接入力データ
noise_func: ノイズを加える関数 (Laplaceメカニズムなど)
func: ノイズを加えた値に対して施す関数
    - 出力はスカラーorベクター(funcに依存する)も必要
"""
def transform(data1: np.ndarray, data2: np.ndarray, noise_func, func) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: ここのxの範囲を元の確率密度関数が大きい値を取る範囲に限定するのが良さそう。現状確率が0となるような不要な部分も多い
    # 例えばLaplace分布だと確率が0.9以上が入っているのがどの範囲であるかなどを元にしてxの値を決めるとか
    # つまりこのxの値は元の確率変数の確率分布の概形（中心、分散など）に依存する
    # リダクションの場合は要素ごとに入力値が確率分布が変わるので、全ての入力で試す必要がある
    sample_num = 1000
    y_sample_data1 = [np.random.laplace(data)  for data in data1 for _ in range(sample_num)]
    y_sample_data2 = [np.random.laplace(data)  for data in data2 for _ in range(sample_num)]
    print(min(min(y_sample_data1), min(y_sample_data2))*10)
    print(max(max(y_sample_data1), max(y_sample_data2))*10)
    range_x = np.linspace(min(min(y_sample_data1), min(y_sample_data2))*10, max(max(y_sample_data1), max(y_sample_data2))*10, 1000)
    # xのそれぞれの要素にlaplace(0, 1/0.1)のノイズを加えるとする
    y = func(range_x)
    
    # 逆関数を数値的に計算する
    x_inv, y_inv = inv_calc(range_x, y)
    # スプライン補間の際のデータ点の数
    point_num = range_x.size * 10
    x_splined, y_splined = spline1(x_inv, y_inv, point_num)
    plt.scatter(x_splined, y_splined, s=2, color="red")
    plt.show()

    # X座標ごとに微分を求める
    x_diff = diff_calc(x_splined, y_splined, point_num)
    plt.scatter(x_splined, x_diff, s=0.1)
    plt.show()

    transformed_pdf1 = var_trasform(y_splined, x_diff, noise_func, loc=data1)
    transformed_pdf2 = var_trasform(y_splined, x_diff, noise_func, loc=data2)

    for pdf1, pdf2 in zip(transformed_pdf1, transformed_pdf2):
        plt.scatter(x_splined, pdf1, s=0.5, color="blue")
        plt.scatter(x_splined, pdf2, s=0.5, color="red")
    plt.show()

    # 区分級数法（積分を台形として近似）により確率密度関数から確率を求める


    # 中央値は異なるが、スケールは同じ確率分布に従う確率変数のリダクションを考える
    # そうするとx_splinedの値が確率変数によって異なるので畳み込みがうまくいかない
    # np.convolutionalを使うためにもx_rangeを揃える必要がある
    if type(transformed_pdf1[0]) == np.ndarray:
        sum_x, sum_pdf1 = convolve_distributions(transformed_pdf1, x_splined)
        sum_x, sum_pdf2 = convolve_distributions(transformed_pdf2, x_splined)
    else:
        pass

    # for _ in range(len(sum_pdf1) - len(x_splined)):
    #     x_splined = np.append(x_splined, x_splined[-1] + x_splined[1] - x_splined[0])
        

    return sum_x, sum_pdf1, sum_pdf2

if __name__ == "__main__":
    """
    ノイズ付与前の値（元データセットに対してクエリを施した結果）
    隣接したデータセット同士の出力を用意する
    以下のような場合だと、[-80, 80]だとうまくいった
    """
    x_data1 = np.array([1.0, 3.0, 5.0, 7.0])
    x_data2 = np.array([2.0, 4.0, 6.0, 8.0])
    x, y1, y2 = transform(x_data1, x_data2, laplace_func, test_func)

    # x = x[1:x.size-1]

    plt.scatter(x, y1, color="green", s=0.2, label="x1")
    plt.scatter(x, y2, color="orange", s=0.2, label="x2")
    plt.legend()
    plt.show()

    """
    確率密度関数の閾値を求める
    1. 確率密度の上位N%の値を閾値とする
    2. 確率密度の値の最大値*N%の値を閾値とする
    """
    y1_threshold = np.sort(y1)[int(y1.size * 0.01)]
    y2_threshold = np.sort(y2)[int(y2.size * 0.01)]
    threshold = max(max(y1)*0.01, max(y2)*0.01) # 値が離散的であるので、ほとんど要素を取れない場合も多い
    threshold = max(y1_threshold, y2_threshold)
    # threshold = 1e-10 # ここをプログラムの中で求めたい

    dx = x[1] - x[0]
    x_size = x.size
    cum_y1 = []
    cum_y2 = []
    tmp_cum_y1 = []
    tmp_cum_y2 = []
    for i in range(x_size):
        if y1[i] > threshold and y2[i] > threshold:
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

    # 全探索
    max_ratio = 0
    max_ratio2 = 0
    for y1_list, y2_list in zip(cum_y1, cum_y2):
        for i in range(len(y1_list)-1):
            for j in range(i+1, len(y1_list)):
                tmp_ratio = (y1_list[j] - y1_list[i]) / (y2_list[j] - y2_list[i]) if (y1_list[j] - y1_list[i]) >  (y2_list[j] - y2_list[i]) else (y2_list[j] - y2_list[i]) / (y1_list[j] - y1_list[i])
                if max_ratio < tmp_ratio:
                    max_ratio = tmp_ratio
                    print(f"i: {i}, j: {j}, max_ratio: {max_ratio}")

    # dx = x[1] - x[0]
    # x_size = x.size
    # cum_y1 = [y1[0] * dx]
    # cum_y2 = [y2[0] * dx]
    # for i in range(x_size-1):
    #     cum_y1.append(cum_y1[i] + y1[i+1] * dx)
    #     cum_y2.append(cum_y2[i] + y2[i+1] * dx)
    # print(cum_y1[-1])
    
    # 全探索
    # max_ratio = 0
    # max_ratio2 = 0
    # for i in range(x_size-1):
    #     for j in range(i+1, x_size):
    #         tmp_ratio = (cum_y1[j] - cum_y1[i]) / (cum_y2[j] - cum_y2[i]) if (cum_y1[j] - cum_y1[i]) >  (cum_y2[j] - cum_y2[i]) else (cum_y2[j] - cum_y2[i]) / (cum_y1[j] - cum_y1[i])
    #         if max_ratio < tmp_ratio:
    #             max_ratio = tmp_ratio
    #             print(f"i: {i}, j: {j}, max_ratio: {max_ratio}")
    #             print(f"cum_y1[j] - cum_y1[i]: {cum_y1[j] - cum_y1[i]}, cum_y2[j] - cum_y2[i]: {cum_y2[j] - cum_y2[i]}")

    print("estimated eps: ", np.log(max_ratio))
