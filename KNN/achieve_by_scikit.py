#!/usr/bin/env python
#_*_ coding:utf-8 _*_

from init_data import load_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

x_data,y_data = load_data()
x_training,x_test,y_training,y_test = train_test_split(x_data,y_data,random_state=10)
estimotor = KNeighborsClassifier()
estimotor.fit(x_training,y_training)
y_predicted = estimotor.predict(x_test)
accuracy = np.mean(y_test == y_predicted)*100
print('The accuracy is {0:1f}%'.format(accuracy))

