# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 13:46
# @author: 芜情
# @description: Moving Mnist dataset
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from resources.setup import MovingMNIST

__all__ = ["MovingMNISTDataset"]


class MovingMNISTDataset(Dataset):

    def __init__(self, dataset: str):
        if dataset == "train":
            data = np.load(MovingMNIST + r"\moving-mnist-train.npz")
        elif dataset == "validation":
            data = np.load(MovingMNIST + r"\moving-mnist-valid.npz")
        elif dataset == "test":
            data = np.load(MovingMNIST + r"\moving-mnist-test.npz")
        else:
            raise FileNotFoundError(f"the dataset {dataset} in Moving MNIST doesn't exist.")

        self.clips = data["clips"]
        self.dataset = data["input_raw_data"]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        clips = self.clips[:, index, ...]
        input_start, input_len = clips[0]
        labels_start, labels_len = clips[1]

        inputs = self.dataset[input_start:input_start + input_len]
        labels = self.dataset[labels_start:labels_start + labels_len]

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels

    def __len__(self):
        return self.clips.shape[1]
