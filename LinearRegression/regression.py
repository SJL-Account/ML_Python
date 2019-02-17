#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import io



def loadDataSet(fileName):
    dataMat=[]
    dataLabel=[]
    fr=open(fileName)
    numFeat=len(fr.readline().split('\t'))-1
    for line in fr.readlines():
        curretLine=[]
        lineArr=line.strip().split('\t')
        for i in range(numFeat):
            curretLine.append(float(lineArr[i]))
        dataMat.append(curretLine)
        dataLabel.append(float(lineArr[-1]))
    return  np.mat(dataMat),np.mat(dataLabel)

def standRegres(xMat,yMat):
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0:
        print "This matix is sigular ,cannont do inverse"
        return
    ws=xTx.I*(xMat.T*yMat)
    return  ws

def lwlr(testPoint,xMat,yMat,k):
    m=xMat.shape[0]
    weight=np.mat(np.eye((m)))
    for i in range(m):
        differMat=testPoint-xMat[i,:];
        #原问题是绝对值，这里写成了平方，能准确吗
        #k是衡量什么的指标
        #weight[i][i]和weight[i,i]有什么区别
        weight[i,i]=np.exp(differMat*differMat.T/(-2*k*k))

    xTx=xMat.T*weight*xMat #n*n

    if np.linalg.det(xTx)==0:
        print "This matix is sigular ,cannont do inverse"
        return
    ws=xTx.I*(xMat.T*weight*yMat)#n*n  *n*1= n*1

    return testPoint*ws


def lwlrTest(testMat,xMat,yMat,k=0.01):
    m = testMat.shape[0]
    yHat=np.zeros(m);
    for i in range(m):
        yHat[i]=(lwlr(testMat[i],xMat,yMat,k))

    return yHat

def rssError(yMat,yHat):
    return ((yMat-yHat)**2).sum()





xMat,yMat=loadDataSet('ex0.txt')


yMat=yMat.T

#ws=standRegres(xMat,yMat)

#xCopy=xMat.copy()

fig=plt.figure()

ax=fig.add_subplot(111)

# 以列为主要单位将数组或者矩阵拆开成一维
ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0])



yHat=lwlrTest(xMat,xMat,yMat)

strInd=xMat[:,1].argsort(0)
xSort=xMat[strInd][:,0,:]
ax.plot(xSort[:,1],yHat[strInd])

plt.show()

#





