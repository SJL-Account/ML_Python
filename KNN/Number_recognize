# -*- coding: UTF-8 -*-

import operator
import numpy as np
from os import listdir


def createDateSet():


    return

def img2Vecotr(filename):
    '''
    将图像信息转换为一维向量(32*32)->(1*1024)
    :param filename: 文件名
    :return: 一维向量(1*1024)
    '''
    #打开文件
    f=open(filename)

    vector=np.zeros((1,1024))

    #读取每一行(32行)
    for i in range(32):
        lineStr=f.readline()
        #把每一个元素都放到向量vector中去
        for j in  range(32):
            vector[0,32*i+j]=lineStr[j]

    return vector

def  classify0(inX,dataSet,labels,k):
    '''
     数据分类算法(欧式距离)
    :param inX: 用于分类的输入向量
    :param dataSet: 输入训练样本集
    :param labels: 样本数据类的标签向量
    :param k: 临近数量
    :return: 
    '''
    #欧式距离求前K

    dataSetSize= dataSet.shape[0]

    #广播复制
    differ_set= np.tile(inX,(dataSetSize,1))-dataSet

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




def handwriting_recognize():
    #标签集合
    hwLabels=[]

    #训练数据向量
    train_set=np.zeros((1,1024))
    #文件集合
    training_files= listdir('trainingDigits/')
    #文件长度
    m=len(training_files)

    #数据集合
    data_set=np.zeros((m,1024))

    #构造数据集合和标签集合
    for i in range(m):
        file_Str= training_files[i]
        file_nameStr= file_Str.split('.')[0]
        class_name=file_nameStr.split('_')[0]
        #添加标签
        hwLabels.append(class_name)
        #构造数据集合
        data_set[i,:]=(img2Vecotr('trainingDigits/%s' % file_Str))


    #测试文件集合
    test_fileSet=listdir('testDigits/')

    mTest=len(test_fileSet)

    #测试数据
    test_set=np.zeros((1,1024))

    error_num=0.0
    #循环测试数据
    for i in range(mTest):
        file_Str= training_files[i]
        file_nameStr= file_Str.split('.')[0]
        class_name=file_nameStr.split('_')[0]
        #构造测试集合
        test_set=(img2Vecotr('trainingDigits/%s' % file_Str))

        class_result= classify0(inX=test_set,dataSet=data_set,labels= hwLabels,k=3)

        print 'The number is class: %s,the result is class_result: %s ' % (class_name,class_result)

        if(class_result!=class_name):
            error_num+=1.0
    print '错误率为：'+ str(error_num)


if __name__=='__main__':
    handwriting_recognize()


