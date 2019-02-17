'''
Copyright (c) $today.year. Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan. 
Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna. 
Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus. 
Vestibulum commodo. Ut rhoncus gravida arcu. 

# @Time    : 2019/2/2 11:04
# @Author  : SJL
# @Email   : 1710213777@qq.com
# @File    : Logistic_regression.py
# @Software: PyCharm

'''

# -*- coding: utf-8 -*-
print(__doc__)
from numpy import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb

def loadDataSet(fname=''):
    '''
    读取以\t分割的字符文件
    :param fname:文件名
    :return:
    '''
    dataMat=[]
    labelMat=[]
    #读取文件
    fr = open(fname)
    #行循环
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineData=[1.0]
        #行元素循环
        for i,lineArr_data in enumerate(lineArr):
            #判断是否为label数据
            if i+1==len(lineArr):
                labelMat.append(lineArr_data)
            else:
                lineData.append(lineArr_data)
        dataMat.append(lineData)
    return dataMat,labelMat

def sigmoid(X):
    '''
    sigmoid function
    :param X: input x
    :return:
    '''
    return 1.0/(1.0+np.exp(-X))

def gradAscent(dataMat,labelMat):
    '''
    梯度上升方法更新
    :param dataMat:
    :param labelMat:
    :return:
    '''

    # 将数据转化为Matrix格式
    dataMat = np.matrix(dataMat).astype('float64')
    labelMat = np.matrix(labelMat).transpose().astype('float64')

    m,n=dataMat.shape

    # 定义模型参数
    cycles = 500
    alpha = 0.001
    # 定义学习参数
    weights = np.ones((n,1))

    for _ in range(cycles):

        predict_label=sigmoid(dataMat*weights)

        error=labelMat-predict_label

        weights=weights+alpha*dataMat.transpose()*error


    return weights

def stocGradAscent(dataMat,labelMat,numIter):
    '''
    随机梯度上升方法更新
    :param dataMat:
    :param labelMat:
    :return:
    '''

    dataMat=np.array(dataMat).astype('float64')

    m,n=dataMat.shape

    weights=np.ones(n)

    alpha=0.01

    weights_adjust_process=[]


    for _ in range(numIter):

        for i in range(m):

            h=sigmoid(np.sum(dataMat[i]*weights))

            error=float(labelMat[i])-h

            weights=weights+alpha*error*dataMat[i]

            weights_adjust_process.append(weights)

    return weights,weights_adjust_process

def stocGradAscent0(dataMat,labelMat,numIter=150):

    '''
    随机梯度上升方法更新
    :param dataMat:
    :param labelMat:
    :return:
    '''

    dataMat=np.array(dataMat).astype('float64')

    m,n = dataMat.shape

    weights = np.ones(n)

    weights_adjust_process=[]

    for j in range(numIter):

        dataIndex=[l for l in range(m)]

        for i in range(m):

            #TODO 模拟退火
            alpha = 4.0/(1.0+i+j)+0.01

            # 随机索引
            randIndex = int(np.random.uniform(0,len(dataIndex)))

            h = sigmoid(np.sum(dataMat[randIndex]*weights))

            error = float(labelMat[i])-h

            weights = weights + alpha * error * dataMat[i]

            #TODO 了解一下迭代器的知识
            del (dataIndex[randIndex])

            weights_adjust_process.append(weights)

    return weights,weights_adjust_process

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet('testSet.txt')
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

data,label=loadDataSet('testSet.txt')

weights,weights_adjust_process=stocGradAscent0(data,label,150)

#weights,weights_adjust_process=stocGradAscent(data,label,300)


plotBestFit(weights)

ws=[i[0] for i in weights_adjust_process]

plt.plot(ws)

plt.show()

ws=[i[1] for i in weights_adjust_process]

plt.plot(ws)

plt.show()
ws=[i[2] for i in weights_adjust_process]

plt.plot(ws)

plt.show()




