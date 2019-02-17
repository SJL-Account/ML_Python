#coding:utf-8
'''
此算法为决策树算法的ID3算法，测试包的小demo
'''

import numpy as np

def create_dataset():

    '''
    输出：数据集合以及类别标签、特征类
    '''
    dataset = np.loadtxt('demo_data.csv', delimiter=",")

    #特征类
    features =['wml','GR','AC',	'AT10',	'AT20'	,'AT30'	,'AT60'	,'AT90','ID_SY']



    return dataset,features

import decisionTree.ID3


if __name__=='__main__':
    dataset,features=create_dataset()

    treeObj= decisionTree.ID3.Tree()

    treedict= treeObj.create_tree(dataset,features)

    treeObj.store_tree(treedict,'decision_tree')

    treedict= treeObj.load_tree('decision_tree')

    print treedict