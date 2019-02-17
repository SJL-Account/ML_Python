#coding:utf-8
'''

此算法为决策树算法的ID3算法,连续型
'''

import numpy as np
import read_txt as rt
from math import log
import operator
import copy
import csv
def create_dataset():

    '''
    输出：数据集合以及类别标签、特征类
    '''
    #dataset = np.loadtxt('1.csv', delimiter=",")

    #数据集合以及类别标签

    dataset=np.array([  [2,1 ,  1 ,'yes'],
                        [2,1,  1 , 'yes'],
                        [2,1 , 0,  'no'],
                        [2,0  ,1 , 'no'],
                        [8,0 , 1 , 'no']])


    #特征类
    features=['a','b','c']
    #features =['GR',	'AC',	'AT10',	'AT20'	,'AT30'	,'AT60'	,'AT90','ID_SY']
    
    return dataset,features

def calcEnt(dataset):
    '''
    计算数据集合的熵
    输入：数据集合
    输出：集合的熵
    '''
    # H=-( n/m*log(n/m,2))

    #np
    #m=dataset的行数
    m=len(dataset)

    #dic
    #键：label,值：label出现的次数
    label_counts={}

    for line_vector in dataset:
        label=line_vector[-1]
        if label not in label_counts.keys():
            label_counts[label]=0
        label_counts[label]+=1
    h=0.0
    for key in label_counts:
        #该label出现的概率
        prob=float(label_counts[key])/m
        h-=prob*log(prob,2)
    return h

def splitDataSet(dataset,axis,value):

    '''
    主要作用是降维度

    dataset:数据集
    axis:特征列
    value:特征值
    return ：np.array 返回拥有特征是value并且去除axis列的数组矩阵
    '''
    reDataSet=[]

    for vector in dataset.tolist():
        if vector[axis]==value:
            #降维
            reduceDataset=vector[:axis]
            reduceDataset.extend(vector[axis + 1:])
            reDataSet.append(reduceDataset)

    return  np.array(reDataSet)


def chooseBestFeatureToSplit(dataset):
    '''
    根据信息量来判断
    :param dataset: 带有标签的数据集合
    :return: 最好的特征值列(对决策影响最大的)
    '''
    #求没有特征值时的熵的大小

    Info=calcEnt(dataset)

    n=len(dataset[0])-1

    best_feature = -1
    best_feature_value = 0.0
    Info_A=0.0
    #循环所有特征类和类中的特征值
    for i in range(n):
        
        #取出每行的特征值
        feature_values= [example[i] for example in dataset]    

        #用set去重
        unique_values=set(feature_values)

        #循环特征值中的去重后值
        for value in unique_values:

            #循环按照特征值分类的集合
            redataset=splitDataSet(dataset,i,value)

            #循环求出熵

            prob=len(redataset)/float(len(dataset))

            Info_A+=prob*calcEnt(redataset)
  
        #Gain=Info()-Inof_A()
        gain =Info-Info_A

        # 比较信息增量的大小、
        if gain >best_feature_value:
            best_feature_value=gain 
            best_feature=i
        
    return  best_feature

def majorityCnt(class_list):

    '''
    当类别已经全部分完时，按照类别出现次数来比较分类结果

    class_list:标签集合
    return :特征类
    '''

    class_counts={}

    for vote in class_list:
        if vote not in class_counts.keys():
            class_counts[vote]=0
        class_counts[vote]+=1
    sorted_class_list=sorted(class_counts.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_list[0][0]

def create_tree(dataset,features):

    '''
    
    :param dataset:数据集合
    :param labels: 特征集合
    :return: 多层数据字典
    '''
    class_labels=[example[-1] for example in dataset]
    features_copy=copy.deepcopy(features)
    #终止条件
    # 1.该特征值下所有的值相同
    if class_labels.count(class_labels[0])==len(class_labels):
        return class_labels[0]
    # 2.分类分完了
    if len(dataset[0])==1:
        return majorityCnt(class_labels)

    #求出最佳分组特征
    best_label=chooseBestFeatureToSplit(dataset)

    #选出最佳特征
    best_feature=features_copy[best_label]

    # 定义一个字典tree
    tree ={best_feature:{}}
    del(features_copy[best_label])

    #提取出最佳特征的所有特征值
    best_values=[example[best_label] for example in dataset]

    #循环所有特征值
    for value in best_values:
        sub_labels=features_copy[:]
        tree[best_feature][value]=create_tree(splitDataSet(dataset,best_label,value),sub_labels)
    
    return tree

# {'no Surfacing': {'1': {'Flipers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}

#测试数据

def classify(tree,features,testVector):
    #获取根节点
    firstKey = tree.keys()[0]
    #获取次节点
    secondDic=tree[firstKey]
    #取出测试向量中首节点的的位置（通过特征字符串我们是无法从测试向量中获取特征值的）
    index= features.index(firstKey)
    class_label=''
    for key in secondDic.keys():
        if str(testVector[index])==key:
            #如果该特征值下面是字典
            if type(secondDic[key]).__name__=='dict':
                class_label=classify(secondDic[key],features,testVector)
            else:
                class_label=secondDic[key]
    return class_label

def load_tree(file_name):
    '''
    保存树
    '''
    import pickle
    fr=open(file_name)
    return pickle.load(fr)
def store_tree(tree,file_name):
    '''
    加载树
    '''
    import pickle
    fw=open(file_name,'w')
    pickle.dump(tree,fw)
    fw.close()

from  treePlotter import *

if __name__=='__main__':
    dataset,features=create_dataset()
    #print calcEnt(dataset)
    tree= create_tree(dataset,features)
    #print classify(tree,features,[1,0])
    #store_tree(tree,'fish')
    #print load_tree('fish')
    print tree#createPlot(tree)



# {'no Surfacing': {'1': {'Flipers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
    
