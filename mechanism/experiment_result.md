<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>

# 実験結果
## 実装詳細
- 入力x1, x2それぞれを勾配法によって最適化する。$\^\epsilon = \frac{Pr[x1 \in S]}{Pr[x2 \in S]}$ とするため、分子を最大化させるようにx1を変化させ、分母を最小化させるようにx2を変化させる。
- $|x1_i - x2_i|_{1 \leq i \leq n} \leq 1$という条件を常に満たすように、x1とx2を交互に変化させる。
- 初期の入力は、0~1の範囲からランダムな値を`input_size`だけ集めた配列をx1, x2として用意する。
- 出力集合は出力が離散的な場合と連続的な場合でそれぞれ以下のように用意する
    - 離散的：0~1の範囲からランダムな値を`input_size`だけ集めた配列を入力として出力を`sample_num`だけ生成し、そこでの出力をすべて出力集合の要素として採用（重複は除く）
    - 連続的：0~1の範囲からランダムな値を`input_size`だけ集めた配列を入力として出力を`sample_num`だけ生成し
- 

## 前提条件
## NoisySum
- ideal_eps = 0.1
- est_eps = 0.135, 0.0987, 0.16, 0.107, 0.12, 0.0923

## NoisyMax
- ideal_eps = 0.2
- est_eps = 0.212, 0.213, 0.183

## NoisyArgMax
- ideal_eps = 0.1
- est_eps = 0.00168, 0.0007729

## 

# 現状問題点
- 一部のアルゴリズムについて、精度が悪い (noisyargmaxなど)
    - softmax関数の勾配の小ささだろうか？
    - 必ずしも=にならないことがあるからだろうか？離散的な場合も連続集合と同様に範囲を持たせるのが良いだろうか？{1, 2, 3, 4}なら、[0.5, 1.5), [1.5, 2.5)...のように
- 実行時間が長い
- プログラムを微分可能にするのがある程度大変。
