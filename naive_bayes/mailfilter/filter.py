#!/usr/bin/env python
#_*_ coding:utf-8 _*_

"""
filter.py的主要作用是提供垃圾邮件的过滤功能，训练数据集为train-mails,测试数据集为test-mails，我们通过对训练数据集进行处理，
提取训练集的邮件中出现频率最高的3000个单词，以这3000个单词出现的次数作为每一封邮件的特征，将数据集转化为3000*n矩阵的形式，然后在利用朴素贝叶斯算法进行训练和预测
"""
import os
import re
import operator
import numpy as np

def split_to_list(filename):

    """
    split_to_ist函数的主要作用提取邮件中的单词，删除·标点符号和特殊字符
    :param filename: 文件名
    :return: 返回单词列表
    """
    keyword = []
    str_split = ''
    with open(filename,'r+') as f:
        str = f.read()
    for s in str:
        if s.isalpha() or s == ' ':
            str_split = str_split + s
    str_list = str_split.split(' ')
    for l in str_list:
        if l != '':
            keyword.append(l)
    return keyword

def exrtact_keywords(train_dir):

    """
    exrtact_keywords函数的主要作用是提取训练集中所有邮件中的单词，然后统计每个单词出现的次数，进行排序，挑选出现次数最多的3000个单词作为特征值
    :param train_dir: 训练数据的目录
    :return: 3000个单词组成的列表
    """
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    keywords = []
    keywords_count = {}
    key = []
    for ema in emails:
        keyword= split_to_list(ema)
        keywords.extend(keyword)
    for w in keywords:
        if w in keywords_count:
            keywords_count[w] = keywords_count[w] + 1
        else:
            keywords_count[w] = 1
    count_list = sorted(keywords_count.items(),key = lambda x:x[1],reverse=True)
    for i in range(3000):
        key.append(count_list[i][0])
    return key

def change(dir,key):

    """
    change函数的主要作用是根据3000个单词特征生成数据
    :param dir:
    :param key:
    :return:
    """
    file_list = [os.path.join(dir,f) for f in os.listdir(dir)]
    n = len(file_list)
    m = len(key)
    x_data = np.zeros((n,m))
    y_data = []
    for f in range(n):
        if 'spmsg' in file_list[f]:
            y_data.append(0)
        else:
            y_data.append(1)
        word = split_to_list(file_list[f])
        for k in range(m):
            if key[k] in word :
                num = word.count(key[k])
                x_data[f][k] = num
    print(y_data)
    return (x_data,y_data)


def get_data(train_dir,test_dir):

    """
    get_data函数的主要作用是获取训练数据和测试数据
    :param train_dir:
    :param test_dir:
    :return:
    """
    key = exrtact_keywords(train_dir)
    x_train, y_train = change(train_dir,key)
    x_test,y_test = change(test_dir,key)
    return(x_train, x_test, y_train, y_test)








