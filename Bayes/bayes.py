#coding:utf-8

'''
算法描述:朴素贝叶斯分类算法
实例：侮辱类文档的分类

朴素贝叶斯公式：p(wi|c)=p(c|wi)*p(c)/p(wi)

将wi出现的概率视为等可能
'''

import numpy as np


def create_dataset():
    '''
    输出：文档列表，列表包含每个句子中的单词组成的列表
        文档分类标记：0代表非侮辱类，1代表侮辱类
    '''

    doc='my dog has flea problems help please,\
        maybe not take him to dog park stupid,\
        my damlation is so cute I love him,\
        stop posting stupid worthless garbage,\
        mr licks ate my steak how to stop him'
    
    doc_list=[[word.lower() for word in doc.split()] for doc in doc.split(',') ]
    
    class_vec=[0,1,0,1,0]

    return doc_list,class_vec

def create_vocablist(dataset):
    '''
    输入：训练集合
    返回：训练集合中包含的所有单词的列表（词汇表）
    '''

    vocab_set=set([])
    
    for doc in dataset:
        vocab_set=vocab_set|set(doc)
    
    return list(vocab_set)


def setOfWordsVector(input_set,vocab_set):
    '''
    输入：每个文档集合
    输出：长度为词汇表的0，1向量，0代表不存在词汇表中，1代表存在
    
    '''

    #长度为词汇表的0向量
    reVocab_set=[0]*len(vocab_set)
    for word in input_set:
        if word in vocab_set:
            reVocab_set[vocab_set.index(word)]=1
        else:
            print 'no word in vocab_set'
    return reVocab_set



def train_NB(doc_list,vocab_set,class_list):
    '''
    输入：文档向量集合
    输出：侮辱类文档的概率
         侮辱类文档中单词出现的概率
         非侮辱文档中单词出现的概率
    '''
    #文档数
    doc_num=len(doc_list)
    #向量长度
    vec_num=len(vocab_set[0])

    PA= sum(class_list)/float(doc_num)

    p0Vec=np.ones(vec_num)
    p1Vec=np.ones(vec_num)
    p0Count=2.0
    p1Count=2.0
    for i in range(doc_num):
        if class_list[i]==0:
            p0Vec+=vocab_set[i]
            p0Count+=sum(vocab_set[i])
        if class_list[i]==1:
            p1Vec+=vocab_set[i]
            p1Count+=sum(vocab_set[i])
    
    p0Vec=np.log(p0Vec/p0Count)
    p1Vec=np.log(p1Vec/p1Count)

    return PA,p0Vec,p1Vec


def classifyNb(sample,p1Vec,p0Vec,pClass):
    '''
    输入:要分类的样本
        每个单词出现侮辱类词语中出现的概率(已知是侮辱类词语，该单词出现的概率)
        每个单词出现非侮辱类词语中出现的概率(已知是非侮辱类词语，该单词出现的概率)
        侮辱类词语在文档中出现的概率
    输出:样本的类别
        
    '''
    #按照公式写

    #求是已知是Wi是侮辱类词语的概率
    #p(wi|c1)=p(c1|wi)*p(c1)/p(wi)

    p1=sum(sample*p1Vec)+np.log(pClass)
    
    #求是已知是Wi是非侮辱类词语的概率
     #p(wi|c0)=p(c0|wi)*p(c0)/p(wi)
    
    p0=sum(sample*p0Vec)+np.log(1.0-pClass)
    
    if p1>p0:
        return 1
    elif p0>p1:
        return 0

def testingNB():
    dataset,lables=create_dataset()
    vocab_set= create_vocablist(dataset)
    vec_list=[]
    for doc in dataset:
        vec_list.append(setOfWordsVector(doc,vocab_set))

    PA,p0Vec,p1Vec=train_NB(dataset,vec_list,lables) 
    sample=np.array([ 'stupid' ,'worthless'])

    sample_vector=setOfWordsVector(sample,vocab_set)
    print classifyNb(sample_vector,np.array(p1Vec),np.array(p0Vec),PA)
    
    

testingNB()