#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import csv
import numpy as np
import os

#该项目所使用的数据集为UCI数据集中鸢尾花数据集，包含150条数据，分为三类，每类50条数据，每条数据包含4个属性，可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类

def load_data():

    """
    load_data()函数的主要作用是读取csv格式的原始数据，将原始数据分为两个部分，一个属性值的集合，一个类别的集合
    """

    x = []
    y = []
    data_filename = os.path.join('iris.data')#读取当前目录下的数据集
    with open(data_filename,'r') as csvfile:
        dataset = csv.reader(csvfile)
        for row in dataset:
            if row == []: break
            else:
                data = [float(da) for da in row[:-1]]#读取每一行数据中的属性值，并将其转化为float类型
                x.append(data)
                if row[-1] == 'Iris-setosa':#将鸢尾花的三个类别分别用1，2，3来标识
                    y.append(1)
                if row[-1] == 'Iris-versicolor':
                    y.append(2)
                if row[-1] == 'Iris-virginica':
                    y.append(3)
    x_data = np.array(x)#将list类型转化为numpy数组类型
    y_data = np.array(y)
    return (x_data,y_data)

