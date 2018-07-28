#!/usr/bin/env python
#_*_ coding:utf-8 _*_

#通过直接对k-近邻算法的描述来构建鸢尾花数据集的模型，并利用该模型对鸢尾花类型进行预测

import numpy as np
import random
import operator
from init_data import load_data

def data_split():

    """
    data_split函数的主要作用是将原始数据分为训练数据和测试数据，其中训练数据和测试数据的比例为2：1
    """

    x_data,y_data = load_data()
    x_training = []
    x_test= []
    y_training = []
    y_test = []
    for i in range(len(x_data)):
        if random.random() > 0.67:
            x_training.append(x_data[i])
            y_training.append(y_data[i])
        else:
            x_test.append(x_data[i])
            y_test.append(y_data[i])
    x_training = np.array(x_training)
    y_training = np.array(y_training)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_training,x_test,y_training,y_test)

def euclidean_distance(x_training,row):

    """
    euclidean_distance函数的主要功能是计算每一条测试数据和训练数据集的欧式距离
    :param x_training: 训练数据集
    :param row: 一条测试数据
    :return: 表示欧式距离的矩阵
    """

    dis = np.sum((row - x_training)**2,axis=1)
    dis = np.sqrt(dis)
    return dis

def predict(dis,y_training,k):

    """
    predict函数的主要作用是通过计算所得的欧式距离的集合，从中选取k个距离最小的数据点，统计这k个数据点中各个类别所出现的次数，出现次数最多的类别即为预测值
    :param dis: 表示欧式距离的矩阵
    :param y_training: 训练数据的类别
    :param k: 选取k个距离最近的数据
    :return: 预测值
    """

    dis_sort = np.argsort(dis)#对欧式距离集合进行排序，返回的dis_sort表示的是排序（从小到大）后的数据在原数组中的索引
    statistics = {}#定义字典，用于统计k个数据点中各个类别的鸢尾花出现的次数
    for i in range(k):
        rank = dis_sort[i]
        if y_training[rank] in statistics:
            statistics[y_training[rank]] = statistics[y_training[rank]] + 1
        else:
            statistics[y_training[rank]] = 1
    sort_statis = sorted(statistics.items(), key=operator.itemgetter(1), reverse=True)#对statistics字典按照value进行排序（从大到小）
    y_predict = sort_statis[0][0]
    return y_predict

if __name__ == "__main__":

    x_training, x_test, y_training, y_test = data_split()
    num = 0
    i = 0
    for row in x_test:
        dis = euclidean_distance(x_training,row)
        y_predict = predict(dis,y_training,5)
        if y_predict == y_test[i]:
            num = num + 1
        i = i + 1
    print('The accuracy is {0:1f}%'.format(num/i))








