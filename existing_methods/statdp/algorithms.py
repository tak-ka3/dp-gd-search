import numpy as np

def noisy_sum(arr, eps=0.1):
    noisy_sum = 0
    for i in range(len(arr)):
        noisy_sum += arr[i] + np.random.laplace(0, 1 / eps)
    return noisy_sum

def noisy_max(arr, eps=0.1):
    noisy_vec = []
    for i in range(len(arr)):
        noisy_vec.append(arr[i] + np.random.laplace(0, 1 / eps))
    return np.max(noisy_vec)

def noisy_hist(arr, eps=0.1):
    noisy_vec = []
    for i in range(len(arr)):
        noisy_vec.append(arr[i] + np.random.laplace(0, 1 / eps))
    return noisy_vec[0]

def noisy_arg_max(arr, eps):
    noisy_vec = []
    for i in range(len(arr)):
        noisy_vec.append(arr[i] + np.random.laplace(0, 1 / (eps/2)))
    return np.argmax(noisy_vec)

# アルゴリズムの関数とそれが想定する隣接性
alg_dict = {
    "noisy_sum": [noisy_sum, "inf", "scalar"],
    "noisy_max": [noisy_max, "inf", "scalar"],
    "noisy_arg_max": [noisy_arg_max, "inf", "scalar"],
    "noisy_hist": [noisy_hist, "1", "vec"]
}