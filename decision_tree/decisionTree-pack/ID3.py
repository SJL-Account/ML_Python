
#coding:utf-8
'''
此算法为基于信息增益的连续型变量的的ID3算法
'''
import numpy as np
import read_txt as rt
from math import log
import operator
import copy
import io

class Tree:

    def __int__(self):
        print 'ID3'
    def calcEnt(self,dataset):
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

    def splitDataSet(self,dataset,axis,split_point,value):

        '''
        主要作用是降维度、砍掉不必要的维度

        dataset:数据集
        axis:特征列
        value:特征值
        return ：np.array 返回拥有特征是value并且去除axis列的数组矩阵
        '''
        leftDataSet=[]
        rightDataSet=[]

        if value==1:
            for vector in dataset.tolist():
                if vector[axis] <= split_point:
                    # 降维
                     reduceDataset=vector[:axis]
                     reduceDataset.extend(vector[axis + 1:])
                     leftDataSet.append(reduceDataset)
            return  np.array(leftDataSet)
        elif value==0:
            for vector in dataset.tolist():
                if vector[axis] > split_point:
                    # 降维
                     reduceDataset=vector[:axis]
                     reduceDataset.extend(vector[axis + 1:])
                     rightDataSet.append(reduceDataset)
            return  np.array(rightDataSet)

    def chooseBestPointToSplit(self,dataset):
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


        #求没有按照特征分特征值时的熵的大小Info(D)
        Info=self.calcEnt(dataset)
        #按照左侧数据分割的信息熵Info_left(D)
        Info_A_left=0.0
        #按照左侧数据分割的信息熵Info_right(D)
        Info_A_right=0.0

        #每行数据的长度（列的长度）
        n=len(dataset[0])-1

        #最佳特征
        best_feature = -1
        #最佳特征中的最佳分割点
        split_points=0.0
        #最大增益
        col_max_gain = 0.0

        #循环所有特征类和类中的特征值

        #对于每列
        for i in range(n):

            #取出每行的特征值
            feature_values= [example[i] for example in dataset]

            #对每列特征值排序
            feature_values.sort()

            #每列特征值的长度
            f_m=len(feature_values)


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
                prob_left=len(leftdataset)/float(len(dataset))

                #左侧信息熵 InfoA
                Info_left=prob_left*self.calcEnt(leftdataset)

                #右侧概率
                prob_right = len(rightdataset) / float(len(dataset))

                #右信息熵
                Info_right= prob_right * self.calcEnt(rightdataset)

                #根据split_point 分类的信息熵
                col_ent = (Info_left + Info_right)

                #信息增益
                gain=Info-col_ent

                #信息增益率

                #gain_rate= gain/col_ent



                if gain >col_max_gain:
                    col_max_gain=gain
                    col_split_point=split_point
                    best_feature=i


        return  best_feature,col_split_point

    def majorityCnt(self,class_list):

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

    def create_tree(self,dataset,features):

        '''
        创建决策树
        :param dataset:数据集合
        :param labels: 特征集合
        :return: 多决策树字典
        '''
        #选出dataset的类别集合
        class_labels=[example[-1] for example in dataset]

        #深拷贝 特征
        features_copy=copy.deepcopy(features)

        #终止条件
        # 1.该特征值下所有的值相同
        if class_labels.count(class_labels[0])==len(class_labels):
            #此时的数据

            #class_count=len(class_labels)
            #if class_count>5:
                #print dataset
            #print  class_key[class_labels[0]],str(class_count)
            return class_labels[0]
        # 2.分类分完了
        if len(dataset[0])==1:
            return self.majorityCnt(class_labels)

        n=len(dataset[0])

        continuity_dataset=[]
        discrete_dataset=[]

        #----------------------连续型-------------------------------------------
        #求出最佳分割特征列和该列的最佳分割点
        col_i,split_point=self.chooseBestPointToSplit(dataset)
        #选出最佳特征
        best_feature=features_copy[col_i]
        best_feature=best_feature+'<='+str(split_point)

        # 定义一个字典tree
        tree ={best_feature:{}}

        #删除衡量过的特征列
        del(features_copy[col_i])

        #提取出最佳特征的所有特征值
        best_values=[example[col_i] for example in dataset]

        #递归 ：按照左侧、右侧两侧的值分别分割数据，并剪切掉已经衡量过的特征列，作为下一层树叶的数据集合、特征列
        for value in [1,0]:
            sub_labels=features_copy[:]
            tree[best_feature][value] =self.create_tree(self.splitDataSet(dataset,col_i,split_point,value),sub_labels)
        #--------------------------------连续型-----------------------------------
        return tree

    # {'AT60<=100': {'1': {'Flipers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}

    #测试数据

    def classify(self,tree,features,testVector):
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

    def load_tree(self,file_name):
        '''
        保存树
        '''
        import pickle
        fr=open(file_name)
        return pickle.load(fr)

    def store_tree(self,tree,file_name):
        '''
        加载树
        '''
        import pickle
        fw=open(file_name,'w')
        pickle.dump(tree,fw)
        fw.close()