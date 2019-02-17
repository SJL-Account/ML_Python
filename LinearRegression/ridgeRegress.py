#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import io



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

def ridgeRegress(xMat,yMat,lam=0.2):
    '''
    w=(XT*X+lambda*I)*XTy

    :return:
    '''


    m = xMat.shape[1];
    xTx=(xMat.T)*xMat
    demon=xTx+lam*(np.eye(m))
    if np.linalg.det(demon)==0.0:
        print 'this maxtrx is sigular ,cannot do inverse'
    ws=demon.I*(xMat.T*yMat)
    return  ws

def ridgeTest (xArr,yArr):
    ''' 
    数据标准化
    :param xArr: 
    :param yArr: 
    :return: 
    '''
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).transpose()
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    #y=y-y的平均值
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    #x=(x-x平均值)/x的方差
    numTestPts=30
    wMat=np.zeros((numTestPts,xMat.shape[1]))

    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return  wMat




xMat,yMat= loadDataSet('abalone.txt')

#print ridgeRegress(xMat,yMat)

ridgeWeights= ridgeTest(xMat,yMat)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
