#coding:utf-8

from numpy import *
import numpy as np

def rand_cent(data_set, k):
    '''
    生成随机质心点
    :param data_set:
    :param k: k个质心点
    :return:
    '''
    n=data_set.shape[1] #n列
    centroids=np.mat(np.zeros((k,n)))# k行n列
    for j in range(n):
        minj= np.min(data_set[:, j])#从第j列中选出最小值
        maxj= np.max(data_set[:, j])#从第j列中选出最小值
        rangej= float(maxj-minj)#范围
        centroids[:, j]=minj+rangej*np.random.randn(k, 1)  #k行1列
    return  centroids


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

data_set= mat(loadDataSet('1200.txt'))[:,[4,5]]

print rand_cent(data_set,k=2)