#coding:utf-8
import csv
import numpy as np
import pandas as pd
import operator
from read_txt import import_txt


def autoNorm(dataSet):
    '''
    用于
    标准化
    '''
    m=dataSet.shape[0]
    #1*n
    min_values=dataSet.min(0)
    #1*n
    max_values=dataSet.max(0)

    #1*n
    differ=max_values-min_values

    #m*n
    new_dataSet=np.zeros((dataSet.shape[0],dataSet.shape[1]))
    
    #new_value=(old_value-min)/(max-min)
    new_dataSet=(dataSet-np.tile(min_values,(m,1)))/np.tile(differ,(m,1))

    return new_dataSet

def  classify0(inX,dataSet,labels,k):
    '''
     数据分类算法(欧式距离)
    :param inX: 用于分类的输入向量
    :param dataSet: 输入训练样本集
    :param labels: 样本数据类的标签向量
    :param k: 临近数量
    :return: 
    '''
    #dataSet=autoNorm(dataSet)

    #欧式距离求前K

    dataSetSize= dataSet.shape[0]

    #广播复制
    differ_set= (np.tile(inX,(dataSetSize,1))-dataSet)

    #广播平方(m*1024)
    differ_set= differ_set**2

    #广播求和(m*1)
    differ_distance=differ_set.sum(axis=1)

    #广播开方(m*1)
    distances= differ_distance**0.5

    #排序(提取索引)(m*1)
    sorted_distances= distances.argsort()

    classCount={}
    for i in range(k):
        voteIlabel = labels[sorted_distances[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #对类别出现的频次进行排序，从高到低

        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

if __name__=='__main__':
   
   # 测试样本

    #m*7 
    #X=np.array(pd.read_csv('test.csv'))
    
    #m*6 所有行 ，前6列
    #inX=X[:,:6]
    #m*1
    #x_lable=X[:,-1]


#-------------------
    inX,x_lable=import_txt('test.txt',6)

    
#------------------

    #训练样本

    #n*7
    #data_frame=np.array(pd.read_csv('1200.csv'))
    #n*6
    #dataSet=data_frame[:,:6]
    #n*1
    #labels=data_frame[:,-1]
    dataSet, labels = import_txt('1200.txt', 6)
    lable_list=[]
    #print autoNorm(dataSet)

    for label in labels:
        lable_list.append(label)
    m=len(inX)
    e=0.0
    for i in range(m):
        if classify0(inX[i],dataSet,lable_list,3)!=x_lable[i]:
            e+=1.0     
        #print  classify0(inX[i],dataSet,lable_list,3) ,x_lable[i]
    error=e/m
    print u'错误率为：',error