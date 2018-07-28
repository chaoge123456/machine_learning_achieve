#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import csv
import numpy as np
import os

def load_data():

    x = []
    y = []
    data_filename = os.path.join('iris.data')
    with open(data_filename,'r') as csvfile:
        dataset = csv.reader(csvfile)
        for row in dataset:
            if row == []: break
            else:
                data = [float(da) for da in row[:-1]]
                x.append(data)
                if row[-1] == 'Iris-setosa':
                    y.append(1)
                if row[-1] == 'Iris-versicolor':
                    y.append(2)
                if row[-1] == 'Iris-virginica':
                    y.append(3)
    x_data = np.array(x)
    y_data = np.array(y)
    return (x_data,y_data)



