#!/usr/bin/env python
#_*_ coding:utf-8 _*_

#通过sklearn库中所提供的关于k-近邻算法相关的包来实现对鸢尾花数据集的建模与预测

from init_data import load_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

x_data,y_data = load_data() #获取数据
x_training,x_test,y_training,y_test = train_test_split(x_data,y_data,random_state=10) #将数据分为训练集和测试集
estimotor = KNeighborsClassifier() #构造k-近邻分类器
estimotor.fit(x_training,y_training) #训练模型
y_predicted = estimotor.predict(x_test) #用训练的模型进行预测
accuracy = np.mean(y_test == y_predicted)*100 #计算预测结果的准确率
print('The accuracy is {0:1f}%'.format(accuracy))

