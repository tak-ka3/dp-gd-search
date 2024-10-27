from algorithms import noisy_max, alg_dict
from input_generator import input_generator
from event_selector import event_selector
from hypothesis import hypothesis_test

def counter_example_detection(alg_name, target_eps):
    alg_func = alg_dict[alg_name][0]
    adj = alg_dict[alg_name][1]
    alg_output_type = alg_dict[alg_name][2]
    input_list = input_generator(adj)
    event_selector_num = 100000
    input1, input2, event = event_selector(alg_func, alg_output_type, target_eps, input_list, iterations=event_selector_num)
    hypothesis_test_num = 100000
    p_val = hypothesis_test(alg_func, target_eps, input1, input2, event, iterations=hypothesis_test_num)
    return p_val

if __name__ == "__main__":
    # TODO: ここをsettings.yamlに記述するようにする
    alg = "noisy_hist"
    test_eps = 0.2
    print("test eps: ", test_eps)
    p_val = counter_example_detection(alg, test_eps)
    print("p_val: ", p_val)
