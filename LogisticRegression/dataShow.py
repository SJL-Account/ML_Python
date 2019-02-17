#coding:utf-8

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np


def autoNorm(dataSet):
    '''
    用于
    标准化
    '''
    m = dataSet.shape[0]
    # 1*n
    min_values = dataSet.min(0)
    # 1*n
    max_values = dataSet.max(0)

    # 1*n
    differ = max_values - min_values

    # m*n
    new_dataSet = np.zeros((dataSet.shape[0], dataSet.shape[1]))

    # new_value=(old_value-min)/(max-min)
    new_dataSet = (dataSet - np.tile(min_values, (m, 1))) / np.tile(differ, (m, 1))

    return new_dataSet


def sigmoid(inX):
    '''
    回归函数
    :param 需要计算的矩阵
    :return:回归函数计算后的值
    '''
    return 1.0/(1.0+np.exp(-inX))

def gradAscent(dataMatIn,classLales):

    '''
    梯度上升算法的公式
    w:=w+alpha*delta f(x,y)
    :param dataMatIn:
    :param classLales:
    :return:
    '''



    dataMatrix=np.matrix(dataSet)

    labelMatrix=np.matrix(classLales).transpose()

    m, n = np.shape(dataMatrix)

    #回归系数归为1
    weights=np.ones((n,1))

    #上升梯度
    alpha=0.001

    #循环次数
    cycles=500

    for i in range(cycles):
        h=sigmoid(dataMatrix*weights)
        error=(labels-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

dataFrame=pd.read_csv('1200.csv')

labels=np.array(dataFrame)[:,-1]

dataArray=np.array(dataFrame)

dataSet= dataArray[:,3:5]

dataSet=autoNorm(dataSet)

print gradAscent(dataSet,labels)


n=len(dataSet)

xcord0=[];ycord0=[]

xcord1=[];ycord1=[]

for i in range(n):
    if int(labels[i])==0:
        xcord0.append(dataSet[i,0]);ycord0.append(dataSet[i,1])
    else:
        xcord1.append(dataSet[i,0]);ycord1.append(dataSet[i,1])

fig=plt.figure()
ax=fig.add_subplot(111)

ax.scatter(xcord0,ycord0,s=30,c='red',marker='s')
ax.scatter(xcord1,ycord1,s=30,c='green')

#x=range(-3.0)

plt.xlabel('X1');plt.ylabel('X2')

plt.show()

