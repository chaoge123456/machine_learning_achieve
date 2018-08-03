#!/usr/bin/env python
#_*_ coding:utf-8 _*_

"""python实现kd树的建立和k近邻查找"""

import numpy as np
from init_data import load_data
import random
import operator
import sys
sys.setrecursionlimit(1000000) #设置递归次数的上限

def splitdata(data):

    """
    splitdata函数的作用是对输入数据集合进行分割，具体规则：求出方差值最大的那一维特征，然后将整个数据集合根据这一维特征进行排序，中位数为分割点
    :param data: 数据集合
    :return: 分割数据的属性，数据分割点，分割后两个部分的数据集合
    """
    n,m = np.shape(data) #获取numpy矩阵维度
    right_data = []
    left_data = []
    num = 0
    row_mean = np.sum(data,axis=0) / n #求矩阵每一列的平均值
    row_variance = np.sum((data - row_mean)**2,axis=0) #求矩阵每一列的方差
    max_row_variance = np.where(row_variance == np.max(row_variance))[0][0] #方差值最大的那一列的索引
    sort_data = np.argsort(data,axis=0) #data矩阵按照列排序，返回排序后的对应的索引
    split_row = sort_data[:,max_row_variance] #方差值最大的那一列排序后的索引值
    split_index = int(n/2) #中位数
    for line in split_row: #将data中的数据分成两个部分，索引排在中位数之前的放进left_data,反之放进right_data
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
        num = num + 1 #用于计数
    split_data = data[split_row[split_index]] #取对应原始数据中的分割点值
    print("分割结点为：",split_data,"--------- 分割维度为：",max_row_variance)
    return(max_row_variance,split_data,right_data,left_data) #返回值分别为分割数据的属性，数据分割点，分割后两个部分的数据

class KNode(object):

    """
    定义节点类
    """
    def __init__(self,row = None,point = None,right = None,left = None,parent = None):
        self.row = row #分割数据集合的特征
        self.point = point #数据分割点
        self.right = right #右子树
        self.left = left #左子树

def create_tree(dataset,knode):

    """
    create_tree函数的主要作用是通过递归的方式来建立kd树
    :param dataset: 数据集合
    :param knode: 根结点
    :return: 返回kd树
    """
    length = len(dataset)
    if length == 0:
        return
    row,point,right_data,left_data = splitdata(dataset)
    knode = KNode(row,point)
    knode.right = create_tree(right_data,knode.right)
    knode.left = create_tree(left_data,knode.left)
    return knode

"""
def front_digui(root):
    #利用递归实现树的先序遍历
    if root == None:
        return
    print(root.point)
    front_digui(root.left)
    front_digui(root.right)

"""

def euclidean_distance(point1,point2):

    """
    计算两点之间的距离
    :param point1:
    :param point2:
    :return: 两点之间的距离
    """
    dis = np.sum((point1 - point2) ** 2)
    dis = np.sqrt(dis)
    return dis

def find_KNN(point,kdtree,k):

    """
    k近邻查找
    :param point: 测试数据点
    :param kdtree: 建立好的kd树
    :param k: k值
    :return: k个近邻点
    """
    current = kdtree #当前节点
    nodelist = [] #搜索路径
    nodek = [] #存储k个近邻点与测试数据点之间的距离
    nodek_point = [] #存储k个近邻点对应的值
    min_dis = euclidean_distance(point,kdtree.point)
    print("---------------------------------------------------------------------------------------")
    while current: #找到测试点所对应的叶子结点，同时将搜索路径中的结点进行k近邻判断
        nodelist.append(current) #将当前结点加入搜索路径
        dis = euclidean_distance(point,current.point)
        if len(nodek) < k: #nodek中不足k个结点时，直接将当前结点加入nodek_point
            nodek.append(dis)
            nodek_point.append(current.point)
            print(current.point,"加入k近邻列表")
        else: #nodek中有k个结点时，删除距离最大的哪个结点，再将该结点加入nodek_point
            max_dis = max(nodek)
            if dis < max_dis:
                index = nodek.index(max_dis)
                print(current.point, "加入k近邻列表;",nodek_point[index],"离开k近邻列表")
                del(nodek[index])
                del(nodek_point[index])
                nodek.append(dis)
                nodek_point.append(current.point)
        ind = current.row #该结点进行分割时的特征
        if point[ind] >= current.point[ind]:
            current = current.right
        else:
            current = current.left

    while nodelist: #回溯寻找k近邻

        back_point = nodelist.pop()
        ind = back_point.row
        max_dis = max(nodek)
        if len(nodek) < k or abs(point[ind] - back_point.point[ind])<max_dis: #如果nodek_point中存储的节点数少于k个，或者测试数据点和当前结点在分割特征维度上的差值的绝对值小于k近邻中的最大距离

            if point[ind] <= back_point.point[ind]: #注意理解这一段判断的代码，因为在之前寻找叶子结点的过程中，我们决定搜索路径的判断方法是大于即搜索右子树，小于即搜索左子树，这里的判断恰恰相反，是为了遍历之前没有搜索的结点
                current = back_point.right
            else:
                current = back_point.left
            if current:
                nodelist.append(current)
                dis = euclidean_distance(point,current.point)
                if max_dis > dis and len(nodek) == k:
                    index = nodek.index((max_dis))
                    print(current.point, "加入k近邻列表;", nodek_point[index], "离开k近邻列表")
                    del(nodek[index])
                    del (nodek_point[index])
                    nodek.append(dis)
                    nodek_point.append(current.point)
                elif len(nodek) < k:
                    nodek.append(dis)
                    nodek_point.append(current.point)
                    print(current.point, "加入k近邻列表")
    return nodek_point

def data_split():

    """
    data_split函数的主要作用是将原始数据分为训练数据和测试数据，其中训练数据和测试数据的比例为2：1
    :return: 
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

    """
    predict函数的主要作用是计算k近邻算法的准确率
    :param x_train: 构建kd树所用的数据集合
    :param y_train:
    :param x_test: 测试数据集合
    :param y_test:
    :param kdtree: 根结点
    :param k: k值
    :return: 预测的准确性
    """
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

    x_train, x_test, y_train, y_test = data_split() #分割数据
    root = KNode() #初始化根结点
    kdtree = create_tree(x_train,root) #建立kd树
    correct = predict(x_train,y_train,x_test,y_test,kdtree,5) #预测准确率
    print("准确率：",correct)








