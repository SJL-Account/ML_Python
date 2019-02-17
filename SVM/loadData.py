import numpy as np

import io
import  matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat=[]
    lableMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return  np.array(dataMat),np.array(lableMat)

def plot(dataSet,labels):


    n=len(dataSet)

    xcord0=[];ycord0=[]

    xcord1=[];ycord1=[]

    for i in range(n):
        if int(labels[i])==1:
            xcord0.append(dataSet[i,0]);ycord0.append(dataSet[i,1])
        else:
            xcord1.append(dataSet[i,0]);ycord1.append(dataSet[i,1])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,s=30,c='red',marker='s')
    ax.scatter(xcord1,ycord1,s=30,c='green')

    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

dataMat,labelMat=loadDataSet('testSet.txt')

plot(dataMat,labelMat)