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

def standRegres(xMat,yMat):
    xMat=np.mat(xMat);yMat=np.mat(yMat).T

    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0:
        print "This matix is sigular ,cannont do inverse"
        return
    ws=xTx.I*(xMat.T*yMat)
    return  ws #n*1

def lwlr(testPoint,xMat,yMat,k):

    m=xMat.shape[0]
    weight=np.mat(np.eye((m)))
    for i in range(m):
        differMat=testPoint-xMat[i,:];
        #原问题是绝对值，这里写成了平方，能准确吗
        #k是衡量什么的指标
        #weight[i][i]和weight[i,i]有什么区别
        weight[i,i]=np.exp(differMat*differMat.T/(-2.0*k*k))

    xTx=xMat.T*(weight*xMat) #n*n

    if np.linalg.det(xTx)==0:
        print "This matix is sigular ,cannont do inverse"
        return
    ws=xTx.I*(xMat.T*(weight*yMat))#n*n  *n*1= n*1

    return testPoint*ws #1*n *  n*1=1


def lwlrTest(testMat,xMat,yMat,k):
    testMat=np.mat(testMat)
    xMat = np.mat(xMat)
    yMat = np.mat(yMat).T
    m = testMat.shape[0]
    yHat=np.zeros(m) #1*m
    for i in range(m):
        yHat[i]=lwlr(testMat[i],xMat,yMat,k)#1*m

    return yHat #1*m ndarray

def rssError(yMat,yHat):
    return ((yMat-yHat)**2).sum()

abX,abY=loadDataSet("abalone.txt")


yHat01=lwlrTest(abX[:99],abX[:99],abY[0:99],0.1)
print rssError(abY[0:99],yHat01)
yHat1=lwlrTest(abX[:99],abX[:99],abY[0:99],1)
print rssError(abY[0:99],yHat1)
yHat10=lwlrTest(abX[:99],abX[:99],abY[0:99],10)
print rssError(abY[0:99],yHat10)

print '----------------------------------------------'

yHat01=lwlrTest(abX[100:199],abX[:99],abY[0:99],0.1)
print rssError(abY[100:199],yHat01)
yHat1=lwlrTest(abX[100:199],abX[:99],abY[0:99],1)
print rssError(abY[100:199],yHat1)
yHat10=lwlrTest(abX[100:199],abX[:99],abY[0:99],10)
print rssError(abY[100:199],yHat10)

print "-----------------------------------------------"

yHat=np.mat(abX[100:199])*standRegres(abX[0:99],abY[0:99])

#print yHat01

print rssError(abY[100:199],yHat)





