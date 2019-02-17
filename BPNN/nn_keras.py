'''
Copyright (c) $today.year. Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan. 
Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna. 
Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus. 
Vestibulum commodo. Ut rhoncus gravida arcu. 
'''
# -*- coding: utf-8 -*-
# @Time    : 2019/2/5 18:08
# @Author  : SJL
# @Email   : 1710213777@qq.com
# @File    : nn_keras.py
# @Software: PyCharm

print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import  SGD
from sklearn.preprocessing import OneHotEncoder

# def loadDataSet(fname=''):
#     '''
#     读取以\t分割的字符文件
#     :param fname:文件名
#     :return:
#     '''
#     dataMat=[]
#     labelMat=[]
#     #读取文件
#     fr = open(fname)
#     #行循环
#     for line in fr.readlines():
#         lineArr = line.strip().split()
#         lineData=[1.0]
#         #行元素循环
#         for i,lineArr_data in enumerate(lineArr):
#             #判断是否为label数据
#             if i+1==len(lineArr):
#                 labelMat.append(lineArr_data)
#             else:
#                 lineData.append(lineArr_data)
#         dataMat.append(lineData)
#     return dataMat,labelMat
#
# dataMat,labelMat=loadDataSet('testSet.txt')
# ohe=OneHotEncoder()
#
# x_train=np.array(dataMat[:70])
# y_train=np.array(labelMat[:70])
# x_test=np.array(dataMat[70:])
# y_test=np.array(labelMat[70:])
#
# y_train=ohe.fit_transform(np.mat(y_train).T)
# y_test=ohe.fit_transform(np.mat(y_test).T)

from load_data import my_data

mydata=my_data(10)

x_train, x_test, y_train, y_test=mydata.train_test_split()


model=Sequential()
model.add(Dense(6,input_dim=6))
model.add(Activation("relu"))
model.add(Dense(8))
model.add(Activation("relu"))
model.add(Dense(8))
model.add(Activation("softmax"))
model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1000,batch_size=20,validation_data=(x_test,y_test),)




