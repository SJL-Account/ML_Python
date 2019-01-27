#coding:utf-8

import numpy as np

#定义激活函数

#双曲函数
def tanh (x):
    return np.tanh(x)
#导数
def tanh_deriv (x):
    return 1.0-(np.tanh(x)*np.tanh(x))

#逻辑函数

def logistic(x):
    return 1.0/(1.0+np.exp(-x))

def logistic_deriv(x):
    return logistic(x)(1-logistic(x))



#神经网络算法

class NeuralNetwork:

    def __int__(self,layers,activation='tanh'):

        if activation=='tanh':
            #函数赋值，不懂
            self.activation=tanh
            self.activation=tanh_deriv
        elif activation=='logistic':
            self.activation=logistic
            self.activation=logistic_deriv

        #初始化权重:-0.25到0.25之间的随机数

        weight=[]
        for i in range(1,len(layers)-1):  #从第二行一直到最后一行的前一行
            #给前行赋值
            weight.append(2*(np.random.random(layers[i-1]+1,layers[i]+1)-1)*0.25)
            weight.append(2*(np.random.random(layers[i]+1,layers[i+1])-1)*0.25)
            #给后行赋值

'''

        #对于初学者，应该这么写
        for i in range(1,len(layers)-1):
            #i前一行 行的神经元数
            previous_layer= layers[i-1]+1
            
            #i行数
            layer=layers[i]+1
            
            #i后一行
            back_layer=layers[i+1]-1
            
            previous_weight=(2*(np.random.random(previous_layer,layer))-1)*0.25
            back_weight=(2*(np.random.random(layer,back_layer))-1)*0.25    
            
            weight.append(previous_weight)
            weight.append(back_weight)
'''
    def fit(self,X,y,learning_rate=0.2,epochs=1000):
        '''

        :param X:训练样本
        :param y:class lablel
        :param learning_rate:学习曲线斜率
        :param epochs:学习次数
        :return:
        
        '''

        np.atleast_2d(X)
        #前向传送

        input_layer=[]

        hidden_layer=[]

        output_layer=[]


























