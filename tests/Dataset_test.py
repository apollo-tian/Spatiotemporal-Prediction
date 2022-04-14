# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 14:31
# @author: 芜情
# @description:
import unittest

from src.utils.data import MovingMNISTDataset


class TestDataset(unittest.TestCase):

    def test_MovingMNISTDataset(self):
        dataset = MovingMNISTDataset("train")
        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][0].max())
