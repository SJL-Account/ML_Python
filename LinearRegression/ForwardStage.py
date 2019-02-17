#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import io

def rssError(yMat,yHat):
    return ((yMat-yHat)**2).sum()

def loadDataSet(fileName):
    dataMat=[]
    dataLabel=[]
    numFeat=len(open(fileName).readline().split('\t'))-1

    fr=open(fileName)
    for line in fr.readlines():
        curretLine=[]
        lineArr=line.strip().split('\t')
        for i in range(numFeat):
            curretLine.append(float(lineArr[i]))
        dataMat.append(curretLine)
        dataLabel.append(float(lineArr[-1]))
    return  dataMat,dataLabel

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    #x=(x-x平均值)/x的方差
    m,n=np.shape(xMat)
    reMat=np.zeros((numIt,n))
    ws=np.zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError=np.inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        reMat[i,:]=ws.T

    return reMat

xArr,yArr=loadDataSet('abalone.txt')

ridgeWeights= stageWise(xArr,yArr)


fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()


























