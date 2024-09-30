import torch
import numpy as np
import torch.nn as nn
from enum import Enum
import matplotlib.pyplot as plt
from algorithms import NoisyArgMax, OutputType, NoisySum, NoisyMax, SVT1


lower = 1.5
upper = 2.0

alpha = 10 # 0.01
sample_num = 1000
epochs = 1500
lr = 0.1
input_size = 4
eps = 0.1
# alg = SVT1(eps, t=0.5)
alg = NoisyArgMax(eps)

best_eps = []

tmp_input = torch.rand(input_size, requires_grad=True)
if alg.output_type == OutputType.DISC:
    # NoisyArgMaxに特化した実装
    outputs = [alg.mech(tmp_input).to(int).item() for _ in range(sample_num)]
    outputs = set(outputs)
    outputs = sorted(outputs)
    output_split = [outputs[0] - (outputs[1] - outputs[0]) / 2]
    for i in range(0, len(outputs)-1):
        output_split.append((outputs[i] + outputs[i+1]) / 2)
    output_split.append(outputs[-1] + (outputs[-1] - outputs[-2]) / 2)
    output_cand = [(output_split[j], output_split[j+1]) for j in range(len(output_split)-1)]
elif alg.output_type == OutputType.CONT:
    split_num = 100
    outputs = [alg.mech(tmp_input) for _ in range(sample_num)]
    values = torch.linspace(min(outputs), max(outputs), split_num)
    output_cand = [(values[j], values[j+1]) for j in range(split_num-1)]
for output in output_cand:
    x1 = torch.rand(input_size, requires_grad=True) # 分子の値であり、最大化したい値
    x2 = torch.rand(input_size, requires_grad=True) # 分母の値であり、最小化したい値
    x1_list = []
    x2_list = []
    for i in range(epochs):
        y1_list = torch.stack([alg.mech(x1) for _ in range(sample_num)])
        if alg.output_type == OutputType.DISC:
            # 結局連続的な場合と同じように出力集合に含まれるかの判定を行う
            lower, upper = output
            z1 = torch.sum(torch.sigmoid(alpha * (y1_list - lower) * (upper - y1_list)))
            # z1 = torch.sum(torch.exp(-alpha * torch.pow(y1_list - output, 2)))
        elif alg.output_type == OutputType.CONT:
            lower, upper = output
            z1 = torch.sum(torch.sigmoid(alpha * (y1_list - lower) * (upper - y1_list)))
        z1.backward()
        tmp_lr = torch.full_like(x1, lr)
        next_x1 = []
        for j in range(len(x1)):
            while True:
                tmp_x1 = (x1[j] + tmp_lr[j] * x1.grad[j]).detach()
                if torch.abs(tmp_x1 - x2[j]) <= 1.0:
                    next_x1.append(tmp_x1)
                    break
                else:
                    tmp_lr[j] = tmp_lr[j] / 2
        x1 = torch.stack(next_x1).requires_grad_(True)
        x1_list.append(x1.tolist())

        y2_list = torch.stack([alg.mech(x2) for _ in range(sample_num)])
        if alg.output_type == OutputType.DISC:
            # z2 = torch.sum(torch.exp(-alpha * torch.pow(y2_list - output, 2)))
            lower, upper = output
            z2 = torch.sum(torch.sigmoid(alpha * (y2_list - lower) * (upper - y2_list)))
        elif alg.output_type == OutputType.CONT:
            lower, upper = output
            z2 = torch.sum(torch.sigmoid(alpha * (y2_list - lower) * (upper - y2_list)))
        z2.backward()
        # print("x2: ", x2.grad)
        tmp_lr = torch.full_like(x2, lr)
        next_x2 = []
        for j in range(len(x2)):
            while True:
                tmp_x2 = (x2[j] - tmp_lr[j] * x2.grad[j]).detach()
                if torch.abs(tmp_x2 - x1[j]) <= 1.0:
                    next_x2.append(tmp_x2)
                    break
                else:
                    tmp_lr[j] = tmp_lr[j] / 2
        x2 = torch.stack(next_x2).requires_grad_(True)
        x2_list.append(x2.tolist())

    # 最適化された入力x1, x2を元にεの値を推定する
    est_sample_num = 10000
    est_epoch = 100
    est_eps_list = []
    zero_cnt = 0
    for epoch in range(est_epoch):
        opt_y1 = torch.stack([alg.mech(x1) for _ in range(est_sample_num)])
        opt_y2 = torch.stack([alg.mech(x2) for _ in range(est_sample_num)])
        if alg.output_type == OutputType.DISC:
            # opt_z1 = torch.sum(torch.exp(-alpha * torch.pow(opt_y1 - output, 2)))
            # opt_z2 = torch.sum(torch.exp(-alpha * torch.pow(opt_y2 - output, 2)))
            lower, upper = output
            opt_z1 = torch.sum(torch.sigmoid(alpha * (opt_y1 - lower) * (upper - opt_y1)))
            opt_z2 = torch.sum(torch.sigmoid(alpha * (opt_y2 - lower) * (upper - opt_y2)))
        elif alg.output_type == OutputType.CONT:
            lower, upper = output
            opt_z1 = torch.sum(torch.sigmoid(alpha * (opt_y1 - lower) * (upper - opt_y1)))
            opt_z2 = torch.sum(torch.sigmoid(alpha * (opt_y2 - lower) * (upper - opt_y2)))
        if opt_z1.item() == 0 or opt_z2.item() == 0:
            zero_cnt += 1
            continue
        tmp_est_eps = np.log(opt_z1.item()/opt_z2.item())
        est_eps_list.append(tmp_est_eps)
    # TODO: εの平均値を取るよりも出力集合に含まれる回数の平均値をそれぞれとったほうが良さそう。
    est_eps = np.mean(est_eps_list)
    print("est_eps: ", est_eps)
    print("ideal_eps: ", alg.get_eps(x1))
    best_eps.append(est_eps)

    # 描画
    x1_t = np.array(x1_list).T
    x2_t = np.array(x2_list).T

    epochs_list = [i for i in range(epochs)]
    for k in range(len(x1)):
        plt.plot(epochs_list, x1_t[k], label=f"x1_{k}")
        plt.plot(epochs_list, x2_t[k], label=f"x2_{k}")
    plt.legend()
    plt.show()

    est_epoch_list = [i for i in range(est_epoch - zero_cnt)]
    plt.plot(est_epoch_list, est_eps_list, label="eps")
    plt.legend()
    plt.show()

    plt.hist(est_eps_list, bins=100)
    plt.show()

print("best_eps: ", best_eps)
