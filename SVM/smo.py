#coding:utf-8
from numpy import *
import csv
import io

import  matplotlib.pyplot as plt
import pandas as pd

#算法步骤
#1.1 W*X+b可以表示任意空间内的任何一个平面

#1.2 因为我们所要描述的标签的结果范围使{-1,+1},所以经研究决定，使用WX+b这个函数模型值为0时作为超平面来划分边界

#读取一个数据集合，这个数据集合是二维线性可分的


def loadData( ):

    dataFrame = pd.read_csv('1200.csv')

    labels = np.array(dataFrame)[:, -1]

    dataSet = (np.array(dataFrame))[:381,[2,4]]

    m, n = np.shape(dataSet)

    dataArray = np.ones((m, n + 1))

    for i in range(m):
        for j in range(n):
            dataArray[i, j + 1] = dataSet[i, j]

    return dataArray,labels


def plotBestFit(dataSet,labels):


    n=len(dataSet)

    xcord0=[];ycord0=[]

    xcord1=[];ycord1=[]

    for i in range(n):
        if int(labels[i])==0:
            xcord0.append(dataSet[i,1]);ycord0.append(dataSet[i,2])
        else:
            xcord1.append(dataSet[i,1]);ycord1.append(dataSet[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,s=30,c='red',marker='s')
    ax.scatter(xcord1,ycord1,s=30,c='green')

    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

#dataSet,labels=loadData()

#plotBestFit(dataSet,labels)

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas















