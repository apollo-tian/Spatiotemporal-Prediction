# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/13 9:00
# @author: 芜情
# @description:
import matplotlib.pyplot as plt
import torch


# dram a sequence of prediction among one sample
def plot_Moving_MNIST_Prediction(index: int):
    data = torch.load("../results/MovingMNIST/ConvLSTM/prediction.pth")[index]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 11:02d}.png", data[seq, 0], cmap="gray")


if __name__ == '__main__':
    plot_Moving_MNIST_Prediction(100)
