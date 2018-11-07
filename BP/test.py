#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import mnist_loader
import network

print("开始训练，较耗时，请稍等。。。")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 784 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元
net = network.Network([784,30,10])
net.BP_algorithm(training_data, 30, 10, 3.0, test_data)
