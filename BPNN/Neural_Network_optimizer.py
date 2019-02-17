'''
Copyright (c) $today.year. Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan. 
Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna. 
Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus. 
Vestibulum commodo. Ut rhoncus gravida arcu. 

# -*- coding: utf-8 -*-
# @Time    : 2019/2/3 16:33
# @Author  : SJL
# @Email   : 1710213777@qq.com
# @File    : Nerual_Network.py
# @Software: PyCharm
'''

print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb


from load_data import my_data

def LoadDataSet(fname=''):
    '''
    读取以\t分割的字符文件
    :param fname:文件名
    :return:
    '''
    dataMat = []
    labelMat = []

    training_data = []

    # 读取文件
    fr = open(fname)
    # 行循环
    for line in fr.readlines():
        lineArr = line.strip().split()
        data = []
        label = 0
        # 行元素循环
        for i, lineArr_data in enumerate(lineArr):

            # 判断是否为label数据
            if i + 1 == len(lineArr):

                label = float(lineArr_data)

            else:

                data.append([lineArr_data])

        training_data.append((np.array(data).astype('float32'), label))

    return training_data

def sigmoid(X):
    '''
    sigmoid function
    :param X: input x
    :return:
    '''

    return 1.0 / (1.0 + np.exp(-X))

def sigmoid_prime(X):
    '''
    sigmoid function
    :param X: input x
    :return:
    '''

    return sigmoid(X) * (1.0 + sigmoid(X))

def relu(z):
    result = np.array(z)
    result[result < 0] = 0
    return result

def relu_prime(z):
    return (np.array(z) > 0).astype('int')

def mse_loss(pre_value, true_value):

    return 0.5 * (pre_value - true_value) ** 2

def mse_loss_derivative(pre_value, true_value):

    return (pre_value - true_value)

class NeuralNetwork(object):

    def __init__(self, sizes,activation='sigmoid',loss='mse',optimizer='SGD'):

        # 神经网络层数
        self.layer_nums = len(sizes)
        # 中间层单元数
        self.sizes = sizes
        # bias集合
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # weights 集合
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.mini_batch_size=32

        # bias 和weights变化过程
        self.weights_updata_process = []

        self.bias_updata_process = []

        if activation=='sigmoid':

            self.activition_func = sigmoid
            self.activition_prime_func = sigmoid_prime

        elif activation=='relu':
            self.activition_func = relu
            self.activition_prime_func = relu_prime


        self.losses = {'mse':mse_loss,}
        self.losses_derivative = {'mse':mse_loss_derivative,}

        self.loss = self.losses[loss]
        self.loss_derivative = self.losses_derivative[loss]



        self.optimizers={ 'SGD':self.decayed_SGD,'Momentum':self.Momentum,'Adagrad':self.RMSProp,'Adam':self.Adam}
        self.optimizer=self.optimizers[optimizer]

        self.accuracys=[]

        self.losses_process=[]

    def feedforward(self,a):

        '''
        前向传播，用于evaluate
        :param a:
        :return: 前向传播结果
        '''
        # n层网络=n层权重
        for w, b in zip(self.weights, self.biases):
            a = self.activition_func(np.dot(w, a) + b)
        return a

    def train(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        self.optimizer( traing_data, epochs, mini_batch_size, eta, test_data)

    def decayed_SGD(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        m = len(traing_data)

        if test_data != None:

            m_test_data = len(test_data)

        for j in range(epochs):

            # 搅乱顺序
            np.random.shuffle(traing_data)

            # 按照小样本划分训练集
            mini_batchs = [traing_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]
            decay_rate=0.96
            decay_steps=10
            for mini_batch in mini_batchs:

                # 按照一定步长更新w,b
                nabla_b, nabla_w=self.update_mini_batch(mini_batch, eta)

                decayed_learning_rate = (eta/len(mini_batch)) * decay_rate**(j / decay_steps)
                # decayed_learning_rate为每一轮优化时使用的学习率，learning_rate为事先设定的初始学习率，decay_rate为衰减系数，global_step为迭代次数，decay_steps为衰减速度（即迭代多少次进行衰减）
                # 梯度下降更新
                self.weights = [w - decayed_learning_rate*nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - decayed_learning_rate*nb for b, nb in zip(self.biases, nabla_b)]

                self.weights_updata_process.append(self.weights)
                self.bias_updata_process.append(self.biases)

            if test_data == None:

                print('training ', str(j + 1), ' epoch...')

            else:
                if j%10 ==0:

                    print('当前学习率为:',decayed_learning_rate)

                accuracy_number,loss = self.evaluate(test_data)

                print('epoch ', str(j + 1), '/', str(epochs), str(accuracy_number), '/', str(m_test_data),'  loss:',loss)

                self.accuracys.append(accuracy_number/m_test_data)

                self.losses_process.append(loss)

    def Momentum(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        m = len(traing_data)

        if test_data != None:

            m_test_data = len(test_data)

        for j in range(epochs):

            # 搅乱顺序
            np.random.shuffle(traing_data)

            # 按照小样本划分训练集
            mini_batchs = [traing_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]

            beta=0.9

            vdb=[np.zeros(b.shape) for b in self.biases]
            vdw=[np.zeros(w.shape) for w in self.weights]

            for mini_batch in mini_batchs:

                # 按照一定步长更新w,b
                nabla_b, nabla_w=self.update_mini_batch(mini_batch, eta)

                vdb=[(beta*vb+(1.0-beta)*nb) for vb,nb in zip(vdb, nabla_b) ]
                vdw=[(beta*vw+(1.0-beta)*nw) for vw,nw in zip(vdw, nabla_w) ]

                nabla_b=[(eta / len(mini_batch))*vb  for vb in vdb ]
                nabla_w=[(eta / len(mini_batch))*vw  for vw in vdw ]

                # 梯度下降更新
                self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - nb for b, nb in zip(self.biases, nabla_b)]

                self.weights_updata_process.append(self.weights)
                self.bias_updata_process.append(self.biases)

            if test_data == None:

                print('training ', str(j + 1), ' epoch...')

            else:



                accuracy_number,loss = self.evaluate(test_data)

                print('epoch ', str(j + 1), '/', str(epochs), str(accuracy_number), '/', str(m_test_data),'  loss:',loss)

                self.accuracys.append(accuracy_number/m_test_data)

                self.losses_process.append(loss)

    def RMSProp(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        m = len(traing_data)

        if test_data != None:

            m_test_data = len(test_data)

        for j in range(epochs):

            # 搅乱顺序
            np.random.shuffle(traing_data)

            # 按照小样本划分训练集
            mini_batchs = [traing_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]

            beta=0.9
            epsilon=10e-8
            sdb=[np.zeros(b.shape) for b in self.biases]
            sdw=[np.zeros(w.shape) for w in self.weights]

            for mini_batch in mini_batchs:

                # 按照一定步长更新w,b
                nabla_b, nabla_w=self.update_mini_batch(self.mini_batch, eta)

                sdb=[beta*sb+(1.0-beta)*(nb)**2  for sb,nb in zip(sdb, nabla_b) ]
                sdw=[beta*sw+(1.0-beta)*(nw)**2  for sw,nw in zip(sdw, nabla_w) ]

                nabla_b=[nb/(np.sqrt(sb)+epsilon) for sb,nb in zip(sdb, nabla_b)]
                nabla_w=[nw/(np.sqrt(sw)+epsilon) for sw,nw in zip(sdw, nabla_w)]

                # 梯度下降更新
                self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - nb for b, nb in zip(self.biases, nabla_b)]

                self.weights_updata_process.append(self.weights)
                self.bias_updata_process.append(self.biases)

            if test_data == None:

                print('training ', str(j + 1), ' epoch...')

            else:

                accuracy_number,loss = self.evaluate(test_data)

                print('epoch ', str(j + 1), '/', str(epochs),str(accuracy_number), '/', str(m_test_data),'  loss:',loss)

                self.accuracys.append(accuracy_number/m_test_data)

                self.losses_process.append(loss)

    def Adam(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        m = len(traing_data)

        if test_data != None:

            m_test_data = len(test_data)

        for j in range(epochs):

            # 搅乱顺序
            np.random.shuffle(traing_data)

            # 按照小样本划分训练集
            mini_batchs = [traing_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]
            decay_rate=0.96
            decay_steps=10
            beta1=0.9
            beta2=0.999
            epsilon=10e-8
            vdb=[np.zeros(b.shape) for b in self.biases]
            vdw=[np.zeros(w.shape) for w in self.weights]
            sdb=[np.zeros(b.shape) for b in self.biases]
            sdw=[np.zeros(w.shape) for w in self.weights]

            for mini_batch in mini_batchs:

                # 按照一定步长更新w,b
                nabla_b, nabla_w=self.update_mini_batch(mini_batch, eta)

                #decayed_learning_rate = (eta/len(mini_batch)) * decay_rate**(j / decay_steps)

                sdb=[beta2*sb+(1.0-beta2)*(nb)**2  for sb,nb in zip(sdb, nabla_b) ]
                sdw=[beta2*sw+(1.0-beta2)*(nw)**2  for sw,nw in zip(sdw, nabla_w) ]

                vdb=[(beta1*vb+(1.0-beta1)*nb) for vb,nb in zip(vdb, nabla_b) ]
                vdw=[(beta1*vw+(1.0-beta1)*nw) for vw,nw in zip(vdw, nabla_w) ]

                nabla_b=[(eta /len(mini_batch))*vb/(np.sqrt(sb)+epsilon)  for vb,sb in zip(vdb,sdb) ]
                nabla_w=[(eta /len(mini_batch))*vw/(np.sqrt(sw)+epsilon)  for vw,sw in zip(vdw,sdw) ]



                # 梯度下降更新
                self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - nb for b, nb in zip(self.biases, nabla_b)]
                self.weights_updata_process.append(self.weights)
                self.bias_updata_process.append(self.biases)



            if test_data == None:

                print('training ', str(j + 1), ' epoch...')

            else:

                accuracy_number,loss = self.evaluate(test_data)

                print('epoch ', str(j + 1), '/', str(epochs), str(accuracy_number), '/', str(m_test_data),'  loss:',loss)

                self.accuracys.append(accuracy_number/m_test_data)

                self.losses_process.append(loss)



                # if accuracy_number>700:
                #
                #     print('turning sgd....')
                #
                #     self.Momentum( traing_data, epochs-j, mini_batch_size, eta, test_data)
                #
                #     break;

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            # 获取梯度
            nabla_b_delta, nabla_w_delta = self.backprob(x, y)

            # 每个样本更新梯度
            nabla_w = [w + w_delta for w, w_delta in zip(nabla_w, nabla_w_delta)]
            nabla_b = [b + b_delta for b, b_delta in zip(nabla_b, nabla_b_delta)]

        return nabla_b,nabla_w

    def backprob(self, x, y):

        # l(n) 第n层单元数

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播

        activation = x

        # 存储每层的激活值

        activations = [x]

        zs = []

        for w, b in zip(self.weights, self.biases):

            # l(2) x 1= l(2) x n * n x 1

            z = np.dot(w, activation) + b

            zs.append(z)

            # l(2) x 1 = l(2) x 1

            activation = self.activition_func(z)

            activations.append(activation)

        # activations [l(1) x 1,l(2) x 1,l(3) x 1, l(4) x 1]

        # zs [l(1) x 1,l(2) x 1,l(3) x 1, l(4) x 1]

        # 求delta的value

        # l(4),1

        loss_derivative = self.loss_derivative(activations[-1], y)

        # l(4),1

        activition_func_prime = self.activition_prime_func(zs[-1])

        # l(4) x 1 = l(4) x 1 x l(4) x 1

        delta = np.multiply(loss_derivative, activition_func_prime)

        # l(4) x 1 = l(4) x 1

        nabla_b[-1] = delta

        # l(4) x l(3) =l(4) x 1 * 1 x l(3)

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 链式法则，反向传播

        for j in range(2, self.layer_nums):

            z = zs[-j]

            # l(4) x 1

            sp = self.activition_prime_func(z)

            # l(3) x 1 = l(3) x l(4)  * l(4) x 1

            delta = np.dot(self.weights[-j + 1].transpose(), delta) * sp

            # l(3) x 1= l(3) x 1

            nabla_b[-j] = delta

            # l(3) x l(2) = l(3) x 1 *  1 x l(2)

            nabla_w[-j] = np.dot(delta, activations[-j - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):

        # 搅乱顺序
        #np.random.shuffle(test_data)

        m=len(test_data)

        # 按照小样本划分训练集
        mini_batchs = [test_data[k:k + self.mini_batch_size] for k in range(0, m, self.mini_batch_size)]

        accuarcys=[]

        losses=[]

        for mini_batch in mini_batchs:

            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                            for (x, y) in mini_batch]

            loss_results = [self.loss(self.feedforward(x),y).sum() for (x, y) in mini_batch]

            accuarcys.append(sum(int(x == y) for (x, y) in test_results))

            losses.append(sum(loss_results))

        return sum(accuarcys),sum(losses)/len(mini_batchs)

    def predict(self,test_data):


        test_results = [(np.argmax((self.feedforward(x))),y) for (x,y) in test_data]

        return test_results



    def plot_weights(self):
        ...
        # plt.plot([ w[0][1] for w in  self.weights_updata_process])
        # plt.show()

        # plt.plot(self.bias_updata_process)
        # plt.show()

    def plot_accuracy(self):

        plt.plot(self.accuracys)

        plt.show()

    def plot_loss(self):

        plt.plot(self.losses_process)

        plt.show()


print('loading data ...')

import load_mnist

training_data,test_data=load_mnist.load_mnist_data()

print('creating model...')

nn = NeuralNetwork(sizes=[28*28, 28*10,100,  10],activation='sigmoid',optimizer='Adam')

print('fiting...')

nn.train(traing_data=training_data, epochs=100, mini_batch_size=32, eta=0.05, test_data=test_data)

print(nn.predict(test_data))


nn.plot_accuracy()

nn.plot_loss()