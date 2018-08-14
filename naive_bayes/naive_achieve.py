#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import numpy as np
from init_data import load_data
import random
from mailfilter.filter import get_data

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


def statictis_y(y_train):

    """
    statictis函数的主要作用是统计训练集中不同类别出现的次数
    :param y_train: 训练集中的类别集合
    :return: 返回一个列表，列表中的每一项是一个元组（类别，出现的次数）
    """
    statictis = {}

    for item_y in y_train:
        if item_y in statictis:
            statictis[item_y] = statictis[item_y] + 1
        else:
            statictis[item_y] = 1
    return list(statictis.items())

def statictis_feature(index,feature,x_train,y_train,y_species):

    """
    statictis_feature函数的主要作用是统计给定特征值和类别的情况下，训练集中符合要求的样本的个数
    :param index: 数据集中的哪一维特征
    :param feature: 特征值
    :param x_train: 训练数据集的特征集合
    :param y_train: 训练集中的类别集合
    :param y_species: 类别
    :return: 返回训练集中特征值和类别都符合要求的样本的个数
    """
    inedx_feature = x_train[:,index]
    num = 0
    list_zip = list(zip(inedx_feature,y_train))
    for i in range(len(list_zip)):
        if list_zip[i][0] == feature and list_zip[i][1] == y_species:
            num = num  + 1
    return num

def calculate_max(x_train,y_train,x,la):

    """
    calculate_max函数的主要作用是预测测试样本类别
    :param x_train: 训练数据集的特征集合
    :param y_train: 训练集中的类别集合
    :param x: 测试样本
    :param la: 拉普拉斯平滑参数
    :return: 测试样本的类别
    """
    statictis = statictis_y(y_train)
    m = [1.0]*len(statictis) #初始化后验概率集合
    for k in range(len(statictis)):
        y_species = statictis[k][0]
        y_num = statictis[k][1]
        for j in range(len(x)):
            num = statictis_feature(j,x[j],x_train,y_train,y_species)
            m[k] = m[k]*(num+la)/(y_num + len(statictis)*la)
        m[k] = m[k]*y_num
    index_max = m.index(max(m))
    print("后验概率:",m,"-------------------","最大值的类别:",statictis[index_max][0])
    return statictis[index_max][0]

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_split() #鸢尾花数据集分类测试
    #x_train, x_test, y_train, y_test = get_data("./mailfilter/train-mails","./mailfilter/test-mails") #邮件过滤数据集测试
    predict = []
    num = 0
    la = 1.0
    for x in x_test:
        test = calculate_max(x_train, y_train, x,la)
        predict.append(test)
    for i in range(len(y_test)):
        if predict[i] == y_test[i]:
            num = num + 1
    print("预测准确率:",num/len(y_test))


