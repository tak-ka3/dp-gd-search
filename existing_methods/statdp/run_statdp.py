from dp_test import counter_example_detection
from datetime import datetime

# p値が有意水準以下である時に、帰無仮説が棄却される（帰無仮説が起こるような場合は少ないと判断する）
# 帰無仮説が成り立つ＝DPが成り立つ

if __name__ == "__main__":
    start_time = datetime.now()
    alg = "noisy_max"
    p_value = 1 - 0.9
    precision = 0.001
    # n_samples_detector = 500000

    init_eps = 0.005 
    print("start while")
    cnt = 0
    while True and cnt < 10:
        pvalue = counter_example_detection(alg, init_eps)
        # print("pvalue: ", pvalue)
        if pvalue + precision > 1.0:
            test_eps = init_eps
            break
        else:
            init_eps *= 2
        cnt += 1

    start_eps = init_eps / 2
    end_eps = init_eps

    # 二分探索
    # p値がp_value未満になり、その差がprecision未満になるまで繰り返す
    cnt = 0
    while end_eps - start_eps > precision:
        test_eps = (start_eps + end_eps) / 2
        pvalue = counter_example_detection(alg, test_eps)
        # print("diff", end_eps - start_eps - precision)
        if pvalue < p_value:
            end_eps = test_eps
        else:
            start_eps = test_eps

    print("estimated eps: ", start_eps)
    exec_time = datetime.now() - start_time
    print("time(s): ", exec_time.total_seconds())
