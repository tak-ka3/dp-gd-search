import numpy as np
import itertools
from hypothesis import calc_pvalue

def event_selector(alg_func, alg_output_type, target_eps, input_list, iterations=1000):
    pvalues = []
    results = [] # (input1, input2, args, event)
    # sample_result = alg_func(input_list[0])
    for input1, input2 in input_list:
        result_d1, result_d2 = [], []
        for _ in range(iterations):
            result_d1.append(alg_func(input1))
            result_d2.append(alg_func(input2))
        event_search_space = []
        combined_result = np.concatenate((result_d1, result_d2))
        unique_result = np.unique(combined_result)
        combined_result.sort()
        # find the densest 70% range
        search_range = int(0.7 * len(combined_result))
        # 最も密度が高い範囲を探す
        search_max = min(range(search_range, len(combined_result)),
                            key=lambda x: combined_result[x] - combined_result[x - search_range])
        search_min = search_max - search_range
        event_search_space = []
        event_search_space = [(-float('inf'), float(alpha)) for alpha in
                  np.linspace(combined_result[search_min], combined_result[search_max], num=10)]
        # all_possible_events = tuple(itertools.product(*event_search_space))
        for event in event_search_space:
            c1_cnt, c2_cnt = 0, 0
            for row in range(len(result_d1)):
                if event[0] <= result_d1[row] <= event[1]:
                    c1_cnt += 1
                if event[0] <= result_d2[row] <= event[1]:
                    c2_cnt += 1
            p1 = calc_pvalue(c1_cnt, c2_cnt, iterations, target_eps)
            p2 = calc_pvalue(c2_cnt, c1_cnt, iterations, target_eps)
            pvalues.append(min(p1, p2))
            results.append((input1, input2, event))
    return results[np.argmin(pvalues)]