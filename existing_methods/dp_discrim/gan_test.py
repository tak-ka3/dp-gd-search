import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# import torchvision
# from torchvision.datasets import MNIST
# from torchvision import transforms
from IPython.display import display

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear_dropout_relu(1, 512), # ある値を見て判断するので1次元
            self._linear_dropout_relu(512, 1024),
            self._linear_relu(1024, 512),
            self._linear_relu(512, 256),
            self._linear_relu(256, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    
    def _linear_dropout_relu(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(0.5),
            nn.ReLU()
        )
    
    def _linear_relu(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    # 入力は出力値
    def forward(self, x):
        y = self.net(x)
        return y # 3クラスそれぞれの確率がベクトルとして出力される
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._linear_relu(nz, 128),
            self._linear_relu(128, 256),
            self._linear_relu(256, 512),
            self._linear(512, nz), # 最後に入力と同じサイズの偽の出力を生成
        )

    def _linear(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )
    
    def _linear_relu(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.net(x)
        return y
    
# ノイズを生成
def make_noise(batch_size):
    return torch.randn(batch_size, nz, device=device) # batch_size * nzの行列を生成

batch_size = 64 # バッチサイズ
nz = 10000 # 潜在変数の次元数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np

real_labels = torch.zeros(batch_size, 1).to(device) # 本物のラベル
fake_labels = torch.ones(batch_size, 1).to(device) # 偽物のラベル
criterion = nn.BCELoss() # 損失関数

def noisy_sum(arr, eps=0.1):
    noisy_sum = 0
    for i in range(len(arr)):
        noisy_sum += arr[i] + np.random.laplace(0, 1 / eps)
    return noisy_sum
alg = noisy_sum
a1 = [1, 3, 5, 7, 9]
a2 = [2, 4, 6, 8, 10]
sample_num = 100000
data1 = [torch.tensor([alg(a1), 1], dtype=torch.float32) for _ in range(sample_num)]
data2 = [torch.tensor([alg(a2), 2], dtype=torch.float32) for _ in range(sample_num)]
data_list = data1 + data2
# print(data1[0])
# print(type(data1))
data = torch.vstack(data_list)
shuffled_data = data[torch.randperm(data.size()[0])]
batch_size = 10000
num_classes = 3
batches = shuffled_data.split(batch_size)

def train(netD, netG, optimD, optimG, n_epochs, write_interval=1):
    # 学習モード
    netD.train()
    netG.train()

    for epoch in range(1, n_epochs+1):
        for batch in batches:
            batch = batch.to(device)

            batch_transposed = batch.transpose(0, 1)
            real_outputs = batch_transposed[0]
            real_labels = batch_transposed[1]
            real_one_hot_labels = torch.nn.functional.one_hot(real_labels.to(torch.int64), num_classes).to(torch.float32)

            # 勾配をリセット
            optimG.zero_grad()

            # Generatorの学習
            batch_size = 2
            z = make_noise(batch_size) # ノイズを生成
            # print(z)
            fake = netG(z) # 偽物を生成 偽物がnz=10000だけ生成される
            # print(fake)
            pred_fake = netD(fake[:-1].T) # 偽物を判定, nzの数のデータのそれぞれのクラスの確率が出力される
            # class_1_or_2 = pred_fake[:, 0] # 偽物であると判断する確率（これを小さくしたい）
            # loss_fake = class_1_or_2.sum() # 偽物の判定に対する誤差
            # lossG = loss_fake # + loss_real # 二つの誤差の和
            lossG = criterion(pred_fake, real_one_hot_labels)
            lossG.backward() # 逆伝播
            optimG.step() # パラメータ更新

            # 出力bと正解ラベルの用意
            batch_transposed = batch.transpose(0, 1)
            real_outputs = batch_transposed[0]
            real_labels = batch_transposed[1]
            real_one_hot_labels = torch.nn.functional.one_hot(real_labels.to(torch.int64), num_classes).to(torch.float32)
            fake = netG(z)
            fake_labels = torch.zeros_like(fake[0])
            fake_one_hot_labels = torch.nn.functional.one_hot(fake_labels.to(torch.int64), num_classes).to(torch.float32)

            # Discriminatorの学習
            optimD.zero_grad()
            pred_real = netD(real_outputs.unsqueeze(1))
            pred_fake = netD(fake[:-1].T)
            loss_real = criterion(pred_real, real_one_hot_labels)
            loss_fake = criterion(pred_fake, fake_one_hot_labels)
            lossD = loss_real + loss_fake
            lossD.backward()
            for name, param in netD.named_parameters():
                if param.grad is not None:
                    print(f"勾配が計算されました: {name}, grad: {param.grad}")
                else:
                    print(f"勾配が計算されていません: {name}")
            optimD.step()

        print(f'{epoch:>3}epoch | lossD: {lossD:.4f}, lossG: {lossG:.4f}')

netD = Discriminator().to(device)
netG = Generator().to(device)
optimD = optim.Adam(netD.parameters(), lr=0.0002)
optimG = optim.Adam(netG.parameters(), lr=0.0002)

n_epochs = 30
train(netD, netG, optimD, optimG, n_epochs)
