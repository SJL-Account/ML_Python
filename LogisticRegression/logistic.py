#coding:utf-8

import csv
import numpy as np

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

    dataMatrix=np.mat(dataMatIn)

    labelMatrix=np.mat(classLales).transpose()

    m, n = np.shape(dataMatrix)

    #回归系数归为1
    weights=np.ones((n,1))

    #上升梯度
    #alpha=0.001

    #循环次数
    cycles=500

    dataIndex = range(m)
    for i in range(cycles):
        #dataMatix :300*2 weithts 2*1=300*1
        for j in range(n):
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            alpha=4/(1.0+j+i)+0.01
            h=sigmoid(dataMatrix[randIndex]*weights)

            error=(labelMatrix[randIndex]-h)
            # 2*300*300*1

            weights=weights+alpha*(dataMatrix[randIndex].transpose())*error

    #        del(dataIndex[randIndex])
    return weights


def classify(dataset,weights):

    prob= sigmoid(dataset*weights)
    if prob>0.5:
        return  1
    else :
        return  0


f=open('horse.csv')

csv_data= csv.reader(f)


dataset=[]
labels=[]
for data in csv_data:
    n = len(data)
    for i in range(n):
        if data[i]=='?':
            data[i]='0'
    dataset.append( [float(data) for data in data[:28]])
    labels.append(float(data[-1]))

weights=gradAscent(np.array(dataset),labels)


f=open('horsetst.csv')
csv_data= csv.reader(f)
dataset_test=[]
for data in csv_data:
    n = len(data)
    for i in range(n):
        if data[i]=='?':
            data[i]='0'
    a=[float(d) for d in data]
    dataset_test.append(a)
b=0
print classify(dataset_test,weights)
