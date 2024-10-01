# 任意の確率分布を持つ確率変数の和の確率分布を求める
# つまりそれぞれの確率変数の確率分布がわかっていれば、畳み込みにより和の確率分布を求めることができる
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm, laplace, uniform

# 各確率分布のPDFを生成
def generate_pdf(distribution, x_range, *params):
    return distribution.pdf(x_range, *params)

# 畳み込みを使って複数の確率分布の和の分布を求める関数
def convolve_distributions(pdfs, x_range):
    # 最初の分布のPDFをセット
    result_pdf = pdfs[0]
    
    # 各PDFを順次畳み込み
    for pdf in pdfs[1:]:
        # 本来であればx_rangeの範囲外からも畳み込むべきだが、それができていない
        # 範囲外の部分が0と見做せるほど小さいと考えるので問題ないか
        result_pdf = np.convolve(result_pdf, pdf, mode='same') * (x_range[1] - x_range[0])
    
    return result_pdf

# X軸の範囲を設定
x_range = np.linspace(-10, 10, 1000)

# 確率分布のPDFを生成 (正規分布、ラプラス分布、一様分布)
pdf1 = generate_pdf(norm, x_range, 0, 1)        # 正規分布 N(0, 1)
pdf2 = generate_pdf(laplace, x_range, 0, 1)     # ラプラス分布 L(0, 1)
pdf3 = generate_pdf(uniform, x_range, -2, 4)    # 一様分布 U(-2, 2)

# これらのPDFをリストに格納
pdfs = [pdf1, pdf2, pdf3]

# 畳み込みで和の確率密度関数を計算
result_pdf = convolve_distributions(pdfs, x_range)

print(pdf1)

# 結果をプロット
plt.plot(x_range, result_pdf, label='Sum of P1 + P2 + P3')
plt.title('Sum of Multiple Distributions (PDF)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
