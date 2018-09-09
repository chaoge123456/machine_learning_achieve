#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import numpy as np
import math
import operator

def emp_entropy(y_data):
    '''
    emp_entropy函数的主要功能是计算数据集的经验熵
    :param y_data: 数据集的类别
    :return: 返回数据集的经验熵
    '''
    count = {}
    emp = 0.0
    m = len(y_data)
    for y in y_data:
        if y in count:
            count[y] += 1
        else:
            count[y] = 1
    for i in count.keys():
        info = (1.0 * count[i] / m)
        emp = emp + info * math.log(info,2)
    return emp

def emp_cond_entropy(x_data,y_data,feature):
    '''
    emp_cond_entropy函数的主要作用是计算经验条件熵
    :param x_data: 数据集
    :param y_data: 数据集类别
    :param feature: 数据集特征特征
    :return: 数据集的经验条件熵
    '''
    count_y = {}
    emp_cond = 0.0
    m = len(y_data)
    fea = x_data[:,feature]
    for i in range(len(fea)):
        if fea[i] in count_y:
            count_y[fea[i]].append(y_data[i])
        else:
            count_y.setdefault(fea[i])
            count_y[fea[i]] = []
            count_y[fea[i]].append(y_data[i])
    for e in count_y.keys():
        l = len(count_y[e])
        emp_cond = emp_cond + (1.0 * l / m) * emp_entropy(count_y[e])
    return emp_cond

def choose_feature(x_data,y_data):
    '''
    choose_feature函数的主要作用是从数据集中选择信息增益最大的特征
    :param x_data: 数据集
    :param y_data: 数据集类别
    :return: 信息增益最大的特征
    '''
    n = np.size(x_data,1)
    count = []
    emp = emp_entropy(y_data)
    for i in range(n):
        emp_cond = emp_cond_entropy(x_data,y_data,i)
        count.append(emp - emp_cond)
    feature = count.index(min(count))
    return feature

def del_feature(x_data,feature):
    '''
    del_feature函数的主要作用是删除数据集中指定特征
    :param x_data: 数据集
    :param feature: 特征
    :return: 删除后的数据集
    '''
    x_data = np.delete(x_data,feature,axis=1)
    return x_data

def node_classfy(y_data):

    count = {}
    for y in y_data:
        if y in count:
            count[y] += 1
        else:
            count[y] = 1
    sorted_y = sorted(count.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_y[0][0]

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

def feature_split(x_data,y_data,feature):
    '''
    feature_split函数的主要作用是按照给定特征对数据集进行分割
    :param x_data:
    :param y_data:
    :param feature:
    :return:
    '''
    count_y = {}
    count_x = {}
    fea = x_data[:, feature]
    for i in range(len(fea)):
        if fea[i] not in count_y:
            count_y.setdefault(fea[i])
            count_x.setdefault(fea[i])
            count_x[fea[i]] = []
            count_y[fea[i]] = []
        count_x[fea[i]].append(x_data[i])
        count_y[fea[i]].append(y_data[i])
    return count_x,count_y


def create_tree(x_data,y_data,feature_list_data):
    '''
    create_tree函数的主要作用是构建决策树
    :param x_data:
    :param y_data:
    :param feature_list:
    :return: 返回决策树
    '''
    feature_list = feature_list_data[:]
    if is_all_same(y_data):
        return y_data[0]
    if len(x_data) == 0:
        return node_classfy(y_data)
    feature = choose_feature(x_data,y_data)
    node_name = feature_list[feature]
    tree = {node_name:{}}
    del feature_list[feature]
    count_x,count_y = feature_split(x_data,y_data,feature)
    for i in count_x.keys():
        fealist = feature_list[:]
        count_x_del = del_feature(count_x[i],feature)
        tree[node_name][i] = create_tree(count_x_del,count_y[i],fealist)
    return tree

def predict(tree,test_data,feature_list):
    '''
    predict函数的主要作用是预测给定测试数据的类别
    :param tree: 决策树
    :param test_data: 测试数据
    :param feature_list: 特征集
    :return: 返回预测结果
    '''
    first_dir = list(tree.keys())[0]
    classfy_data = test_data[feature_list.index(first_dir)]
    for item in tree[first_dir].keys():
        if item == classfy_data:
            if type(tree[first_dir][item]) == dict:
                class_lable = predict(tree[first_dir][item],test_data,feature_list)
            else:
                class_lable = tree[first_dir][item]
    return class_lable

if __name__ == "__main__":
    x_data = np.array([[1, 0, 0, 1], [1, 0, 0, 2], [1, 1, 0, 2], [1, 1, 1, 0], [1, 0, 0, 0],
                       [2, 0, 0, 0], [2, 0, 0, 1], [2, 1, 1, 1], [2, 0, 1, 2], [2, 0, 1, 2],
                       [3, 0, 1, 2], [3, 0, 1, 1], [3, 1, 0, 1], [3, 1, 0, 2], [3, 0, 0, 0]])
    y_data = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    feature_list = ['age','work','home','money']
    mytree = create_tree(x_data,y_data,feature_list)
    print(mytree)
    class_lable = predict(mytree,[1,1,0,2],feature_list)
    print("预测值为:",class_lable)








