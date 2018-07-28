#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import numpy as np
import random
import operator
from init_data import load_data

def data_split():

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

    dis = np.sum((row - x_training)**2,axis=1)
    dis = np.sqrt(dis)
    return dis

def predict(dis,y_training,k):

    dis_sort = np.argsort(dis)
    statistics = {}
    for i in range(k):
        rank = dis_sort[i]
        if y_training[rank] in statistics:
            statistics[y_training[rank]] = statistics[y_training[rank]] + 1
        else:
            statistics[y_training[rank]] = 1
    sort_statis = sorted(statistics.items(), key=operator.itemgetter(1), reverse=True)
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








