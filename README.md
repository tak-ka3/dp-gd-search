## プログラム実行方法
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python var_transform/dp_test.py
```

## 実験方法
- 設定をsetting.yamlに記述する
```yaml
# ε推定の対象となるアルゴリズム
algorithm: "noisy_sum" # "noisy_sum", "noisy_max", "noisy_arg_max"
# ノイズ付与時点の確率変数の範囲の下限と上限。その範囲をsampling_numだけ分割してそれぞれを入力の確率変数とする。(TODO: 本来入力によってノイズ付与時点の確率変数の範囲が決まるが、出力結果を決定的にするために静的に決めている)
noisy_var: 
  lower: -30
  upper: 50
  sampling_num: 5000
# 積分の方法
integral: "gauss" # "gauss"(ガウス積分), "trapz"(台形近似)
# εの探索の仕方。allは全探索、thresholdは確率密度の値の下位${threshold}は無視する探索。
search: 
  way: "all" # "all", "threshold"
  threshold: 0.1
```
- 実験結果は`experiments`ディレクトリ配下に格納され、上記のsetting.yamlにresultというkeyが追加され、推定されたεの値(eps)と実行時間(time(s))が格納される

## コード解説
- 基本的に確率変数変換のテスト手法のコードはvar_transformディレクトリ配下に含まれる
