#coding:utf-8
'''
测试模块
'''

import numpy as np
import read_txt as rt
from math import log
import operator
import copy
import io
import csv


feature_list = []


def create_dataset():

    '''
    输出：数据集合以及类别标签、特征类
    '''
    dataset = np.loadtxt('C138.csv', delimiter=",")

    #数据集合以及类别标签

    dataset1=np.array([  [2.3,1.2 ,  1.0 ,'yes'],
                    [2.5,1.5,  1.5 , 'yes'],
                    [2.7,1.3 , 0.3,  'no'],
                    [2.2,0.9  ,1.3, 'no'],
                    [8.3,0.6 , 1.5 , 'no']])

    #特征类
    #features =['GR','AC','AT10','AT30',	'AT20','AT60'	,'AT90','CODE']
    features =['AC','GR','CILD','DEN','CLL8','RT','AT10','AT30','AT20','AT60','AT90','CODE']

    #features=['A','B','C']'CILD','DEN','CLL8','RT',


    return dataset,features

def calcGini(dataset):
    m=len(dataset)
    label_counts = {}


    for line_vector in dataset:
        label=line_vector[-1]
        if label not in label_counts.keys():
            label_counts[label]=0
        label_counts[label]+=1
    gini=0.0

    for key in label_counts:
        #该label出现的概率
        prob=float(label_counts[key])/m
        gini+=(prob**2)
    return 1-gini

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

def chooseBestPointToSplit(dataset,features,type="entropy"):
    '''
    （连续型随机变量）
    根据信息增益来选取最佳特征列和最佳分割点
    :param dataset: 带有标签的数据集合
    :return: 最好的特征值列(对决策影响最大的),最佳分割点
    '''

    # 算法步骤：
    # 连续型变量找出最佳分割点
    # 循环所有列，循环列的所有值，划分数据集算熵值
    # 取该列信息增益最大的那个
    # 取出所有列中信息最大的那个增益值
    # 返回该值
    feature_dict = {}

    #求没有按照特征分特征值时的熵的大小Info(D)
    Info=calcEnt(dataset)
    gini_D=calcGini(dataset)
    #按照左侧数据分割的信息熵Info_left(D)
    Info_A_left=0.0
    #按照左侧数据分割的信息熵Info_right(D)
    Info_A_right=0.0

    #每行数据的长度（列的长度）
    n=len(dataset[0])-1
    m=len(dataset)
    #最佳特征
    best_feature = -1
    #最佳特征中的最佳分割点
    split_points=0.0
    #最大增益

    #循环所有特征类和类中的特征值

    #对于每列
    for i in range(n):


        #取出每行的特征值
        feature_values= [example[i] for example in dataset]

        #对每列特征值排序
        feature_values.sort()

        #每列特征值的长度

        f_m=len(feature_values)

        #最大信息增益率
        col_max_gain_rate = 0.0
        #最大信息增益
        col_max_gain = 0.0
        #最大基尼增益
        gini_delta_max=0.0
        # 循环该列特征值
        for fi in range(f_m):

            if fi+1<f_m :
                #选取 a1+a2/2 作为预选分割点
                split_point=(feature_values[fi]+feature_values[fi+1])/2

                #根据分割点分别求出左右两侧的熵值
                leftdataset=[]
                rightdataset=[]

                for vector in dataset.tolist():
                    if vector[i] <= split_point:
                        leftdataset.append(vector)
                    else:
                        rightdataset.append(vector)

                # 循环求出熵

                #左侧概率

                prob_left=(fi+1)/float(m)

                #左侧信息熵
                Info_left=prob_left*calcEnt(leftdataset)

                #右侧概率
                prob_right = (m-fi-1)/ float(m)

                #右信息熵
                Info_right= prob_right * calcEnt(rightdataset)

                #根据split_point 分类的信息熵
                col_ent = (Info_left + Info_right)

                #信息增益
                gain=Info-col_ent

                if type=="entropy_rate":
                    #信息增益率
                    split_info=-(prob_left*log(prob_left,2)+prob_right*log(prob_right,2))
                    gain_rate= gain/split_info

                    if gain_rate > col_max_gain_rate:
                        col_max_gain_rate = gain_rate

                elif type=="gini":
                    #基尼指数
                    gini_A=prob_left*(calcGini(leftdataset))+prob_right*(calcGini(rightdataset))
                    gini_delta=gini_D-gini_A
                    if gini_delta > gini_delta_max:
                        gini_delta_max = gini_delta
                else:
                    if gain >col_max_gain:
                        col_max_gain=gain
                        col_split_point=split_point


        feature_name=features[i]

        if type=="entropy":

            feature_dict[feature_name]=col_max_gain
        elif type=="entropy_rate":
            feature_dict[feature_name]=col_max_gain_rate
        elif type=="gini":
            feature_dict[feature_name]=gini_delta_max
        feature_list.append(feature_name)

    return  feature_dict


dataset,features=create_dataset()


feature_dict= chooseBestPointToSplit(dataset,features,type="gini")


for i in feature_list:

    print i,feature_dict[i]