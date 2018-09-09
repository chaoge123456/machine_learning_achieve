#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import operator
import numpy as np
import random
from init_data import load_data

def Gini_index(y_data):
    '''
    Gini_index函数的主要作用是计算数据集的基尼指数
    :param y_data: 数据集类别
    :return: 返回基尼指数
    '''
    m = len(y_data)
    count = {}
    num = 0.0
    for i in y_data:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    for item in count.keys():
        num = num + pow(1.0 * count[item] / m,2)
    return (1.0-num)

def Gini_D_A(x_data,y_data,feature):
    '''
    Gini_D_A函数的主要作用是计算某一离散特征各个取值的基尼指数，选取最优切分点
    :param x_data: 数据集合
    :param y_data: 数据集类别
    :param feature: 特征
    :return: 该特征的最优切分点
    '''
    Gini_data = list(x_data[:,feature])
    y_data = list(y_data[:])
    m = len(Gini_data)
    Gini = {}
    classfy_data = {}
    for e in range(m):
        if Gini_data[e] not in classfy_data:
            classfy_data[Gini_data[e]] = []
        classfy_data[Gini_data[e]].append(y_data[e])
    for item in classfy_data.keys():
        l1 = len(classfy_data[item])
        r = y_data[:]
        for i in classfy_data[item]:
            r.remove(i)
        l2 = len(r)
        num = 1.0 * l1 / m * Gini_index(classfy_data[item]) + 1.0 * l2 / m * Gini_index(r)
        Gini[item] = num
    sor = sorted(Gini.items(), key=operator.itemgetter(1))
    return sor[0]

def Gini_continuous(x_data,y_data,feature):
    '''
    Gini_continous函数的主要作用是计算某一连续特征各个取值的基尼指数，选取最优切分点
    :param x_data: 数据集合
    :param y_data: 数据集类别
    :param feature: 特征
    :return: 该特征的最优切分点
    '''
    Gini_data = list(x_data[:,feature])
    m = len(Gini_data)
    y_data = list(y_data[:])
    sort_data = sorted(Gini_data)
    Gini = {}
    split_point = []
    for i in range(m-1):
        num = (sort_data[i] + sort_data[i+1]) / 2.0
        split_point.append(num)
    for e in split_point:
        count_y = {0:[],1:[]}
        for k in range(m):
            if Gini_data[k] <= e:
                count_y[0].append(y_data[k])
            else:
                count_y[1].append(y_data[k])
        cal = 1.0 * len(count_y[0]) / m * Gini_index(count_y[0]) + 1.0 * len(count_y[1]) / m * Gini_index(count_y[1])
        Gini[e] = cal
    sor = sorted(Gini.items(), key=operator.itemgetter(1))
    return sor[0]


def choose_feature(x_data,y_data,dis_or_con):
    '''
    choose_feature函数的主要作用是从各个特征的各个切分点中选择基尼指数最小的切分点
    :param x_data: 数据集合
    :param y_data: 数据类别
    :return: 切分点
    '''
    w = np.size(x_data,axis=1)
    count = []
    count_label = {}
    for i in range(w):
        if dis_or_con[i] == 0:
            a = Gini_D_A(x_data,y_data,i)
        else:
            a = Gini_continuous(x_data,y_data,i)
        count.append(a[1])
        count_label[i] = a
    id = count.index(min(count))
    return id,count_label[id][0]

def feature_split(x_data,y_data,w,f,flag):
    '''
    feature_split函数的主要作用是根据切分点对数据进行切分
    :param x_data: 数据集合
    :param y_data: 数据类别
    :param w: 切分特征
    :param f: 切分点
    :return: 切分后的数据集合
    '''
    if flag == 0:
        data = x_data[:,w]
        count_x = {"left":[],"right":[]}
        count_y = {"left":[],"right":[]}
        for i in range(len(data)):
            if data[i] != f:
                count_x["left"].append(x_data[i])
                count_y["left"].append(y_data[i])
            else:
                count_x["right"].append(x_data[i])
                count_y["right"].append(y_data[i])
    else:
        data = x_data[:, w]
        count_x = {"left": [], "right": []}
        count_y = {"left": [], "right": []}
        for i in range(len(data)):
            if data[i] >= f:
                count_x["left"].append(x_data[i])
                count_y["left"].append(y_data[i])
            else:
                count_x["right"].append(x_data[i])
                count_y["right"].append(y_data[i])
    return count_x,count_y

def most_y_data(y_data):
    count = {}
    for i in y_data:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    sor = sorted(count.items(), key=operator.itemgetter(1),reverse=True)
    return sor[0][0]

def del_feature(x_data,feature):
    '''
    del_feature函数的主要作用是删除数据集中指定特征
    :param x_data: 数据集
    :param feature: 特征
    :return: 删除后的数据集
    '''
    x_data = np.delete(x_data,feature,axis=1)
    return x_data

def is_all_same(y_data):
    '''
    is_all_same函数的作用是判断数据集的类别
    :param y_data:数据集类别
    :return:
    '''
    n = len(y_data)
    if type(y_data) == np.ndarray:
        y_data = y_data.tolist()
    if y_data.count(y_data[0]) == n:
        return True
    else:
        return False

def create_tree(x_data,y_data,dis_or_con_data,feature_list_data):

    feature_list = feature_list_data[:]
    dis_or_con = dis_or_con_data[:]
    if dis_or_con == []:
        return most_y_data(y_data)
    if is_all_same(y_data):
        return y_data[0]
    w,f = choose_feature(x_data,y_data,dis_or_con)
    count_x, count_y = feature_split(x_data,y_data,w,f,dis_or_con[w])
    node_name = feature_list[w]
    tree = {(node_name,f):{}}
    del feature_list[w]
    del dis_or_con[w]
    for i in count_x.keys():
        fealist = feature_list[:]
        dis_con = dis_or_con[:]
        count_x_del = del_feature(count_x[i], w)
        tree[(node_name,f)][i] = create_tree(count_x_del, count_y[i], dis_con,fealist)
    return tree

def predict(tree,test_data,dis_or_con,feature_list):

    a = tuple(tree.keys())
    first_dir = a[0]
    feature_index = feature_list.index(first_dir[0])
    classfy_data = test_data[feature_index]
    if dis_or_con[feature_index] == 0:
        if classfy_data == first_dir[1]:
            if type(tree[first_dir]['right']) == dict:
                class_lable = predict(tree[first_dir]['right'], test_data, dis_or_con, feature_list)
            else:
                class_lable = tree[first_dir]['right']
        else:
            if type(tree[first_dir]['left']) == dict:
                class_lable = predict(tree[first_dir]['left'], test_data, dis_or_con, feature_list)
            else:
                class_lable = tree[first_dir]['left']
    else:
        if classfy_data <= first_dir[1]:
            if type(tree[first_dir]['right']) == dict:
                class_lable = predict(tree[first_dir]['right'], test_data, dis_or_con, feature_list)
            else:
                class_lable = tree[first_dir]['right']
        else:
            if type(tree[first_dir]['left']) == dict:
                class_lable = predict(tree[first_dir]['left'], test_data, dis_or_con, feature_list)
            else:
                class_lable = tree[first_dir]['left']
    return class_lable

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

if __name__ == "__main__":

    '''
    x_data = np.array([[1, 0, 0, 1], [1, 0, 0, 2], [1, 1, 0, 2], [1, 1, 1, 0], [1, 0, 0, 0],
                       [2, 0, 0, 0], [2, 0, 0, 1], [2, 1, 1, 1], [2, 0, 1, 2], [2, 0, 1, 2],
                       [3, 0, 1, 2], [3, 0, 1, 1], [3, 1, 0, 1], [3, 1, 0, 2], [3, 0, 0, 0]])
    y_data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    dis_or_con = [0,0,0,0]
    feature_list = ['age','work','home','money']
    mytree = create_tree(x_data,y_data,dis_or_con,feature_list)
    print(mytree)
    p = predict(mytree,[1,1,0,2],dis_or_con,feature_list)
    print(p)
    '''
    pre = 0
    x_training, x_test, y_training, y_test = data_split()
    dis_or_con = [1, 1, 1, 1]
    feature_list = ['age', 'work', 'home', 'money']
    mytree = create_tree(x_training, y_training, dis_or_con, feature_list)
    for item in range(len(x_test)):
        p = predict(mytree, x_test[item], dis_or_con, feature_list)
        if p == y_test[item]:
            pre += 1
    print("准确率为：",pre * 1.0 / len(y_test))






