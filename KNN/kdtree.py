#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import numpy as np
from init_data import load_data
import random
import operator
import sys
sys.setrecursionlimit(1000000) #例如这里设置为一百万

def splitdata(data):

    n,m = np.shape(data) #获取numpy矩阵维度
    right_data = []
    left_data = []
    num = 0
    row_mean = np.sum(data,axis=0) / n #求矩阵每一列的平均值
    row_variance = np.sum((data - row_mean)**2,axis=0) #求矩阵每一列的方差
    max_row_variance = np.where(row_variance == np.max(row_variance))[0][0] #方差值最大的那一列的索引
    sort_data = np.argsort(data,axis=0) #data矩阵按照列排序，返回排序后的对应的索引
    split_row = sort_data[:,max_row_variance] #方差值最大的那一列排序后的索引值
    #print(split_row)
    split_index = int(n/2) #中位数
    for line in split_row:
        #print(data[line:,])
        if num > split_index:
            if right_data == []:
                right_data = data[line,:]
                right_data = np.array([right_data])
            else: right_data = np.concatenate((right_data,[data[line,:]]),axis=0)
        elif num < split_index:
            if left_data == []:
                left_data = data[line,:]
                left_data = np.array([left_data])
            else: left_data = np.concatenate((left_data,[data[line,:]]),axis=0)
        num = num + 1
    split_data = data[split_row[split_index]]
    print("split data:",split_data,"----- 切分维度为：",max_row_variance)
    return(max_row_variance,split_data,right_data,left_data) #返回值分别为分割数据的属性，数据分割点，分割后两个部分的数据

class KNode(object):

    def __init__(self,row = None,point = None,right = None,left = None,parent = None):
        self.row = row
        self.point = point
        self.right = right
        self.left = left
        self.parent = parent

def create_tree(dataset,knode):

    length = len(dataset)
    if length == 0:
        return
    row,point,right_data,left_data = splitdata(dataset)
    knode = KNode(row,point)
    knode.right = create_tree(right_data,knode.right)
    knode.left = create_tree(left_data,knode.left)
    return knode

def front_digui(root):
    """利用递归实现树的先序遍历"""
    if root == None:
        return
    print(root.point)
    front_digui(root.left)
    front_digui(root.right)


def euclidean_distance(point1,point2):

    dis = np.sum((point1 - point2) ** 2)
    dis = np.sqrt(dis)
    return dis

def find_KNN(point,kdtree,k):

    current = kdtree
    nodelist = []
    nodek = []
    nodek_point = []
    min_dis = euclidean_distance(point,kdtree.point)
    while current:
        nodelist.append(current)
        dis = euclidean_distance(point,current.point)
        if len(nodek) < k:
            nodek.append(dis)
            nodek_point.append(current.point)
        else:
            max_dis = max(nodek)
            if dis < max_dis:
                index = nodek.index(max_dis)
                del(nodek[index])
                del(nodek_point[index])
                nodek.append(dis)
                nodek_point.append(current.point)
        ind = current.row
        if point[ind] >= current.point[ind]:
            current = current.right
        else:
            current = current.left
    #print(nodek_point)

    while nodelist:

        back_point = nodelist.pop()
        ind = back_point.row
        max_dis = max(nodek)
        if len(nodek) < k or abs(point[ind] - back_point.point[ind])<max_dis:

            if point[ind] <= back_point.point[ind]:
                current = back_point.right
            else:
                current = back_point.left
            if current:
                nodelist.append(current)
                dis = euclidean_distance(point,current.point)
                if max_dis > dis and len(nodek) == k:
                    index = nodek.index((max_dis))
                    del(nodek[index])
                    del (nodek_point[index])
                    nodek.append(dis)
                    nodek_point.append(current.point)
                elif len(nodek) < k:
                    nodek.append(dis)
                    nodek_point.append(current.point)
    return nodek_point

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


def predict(x_train,y_train,x_test,y_test,kdtree,k):

    num = 0
    correct = 0
    for point in x_test:
        nodek_point = find_KNN(point,kdtree,k)
        statistics = {}
        for no_point in nodek_point:
            for index in range(len(x_train)):
                if (x_train[index]==no_point).all(): break
            if y_train[index] in statistics:
                statistics[y_train[index]] = statistics[y_train[index]] + 1
            else:
                statistics[y_train[index]] = 1
        sort_statis = sorted(statistics.items(), key=operator.itemgetter(1),reverse=True)  # 对statistics字典按照value进行排序（从大到小）
        y_predict = sort_statis[0][0]
        if y_predict == y_test[num]:
            correct = correct + 1
        num = num + 1
    return correct/len(y_test)


if __name__ == "__main__":

    x_train, x_test, y_train, y_test = data_split()
    root = KNode()
    kdtree = create_tree(x_train,root)
    correct = predict(x_train,y_train,x_test,y_test,kdtree,5)
    print("准确率：",correct)








