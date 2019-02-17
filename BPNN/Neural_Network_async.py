

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from multiprocessing import Pool,Process,Lock
import time

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

    def __init__(self, sizes,activation='sigmoid',loss='mse'):

        # 神经网络层数
        self.layer_nums = len(sizes)
        # 中间层单元数
        self.sizes = sizes
        # bias集合
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # weights 集合
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

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

    def SGD(self, traing_data, epochs, mini_batch_size, eta, test_data=None):

        m = len(traing_data)

        if test_data != None:

            m_test_data = len(test_data)

        for j in range(epochs):

            # 搅乱顺序
            np.random.shuffle(traing_data)

            # 按照小样本划分训练集
            mini_batchs = [traing_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]

            for mini_batch in mini_batchs:
                # 按照一定步长更新w,b
                self.update_mini_batch(mini_batch, eta)

            if test_data == None:

                print('training ', str(j + 1), ' epoch...')

            else:

                accuracy_number,loss = self.evaluate(test_data)

                print('epoch ', str(j + 1), '/', str(epochs), str(accuracy_number), '/', str(m_test_data))

                self.accuracys.append(accuracy_number/m_test_data)

                self.losses_process.append(loss)

    def update_mini_batch(self, mini_batch,nabla_b_list,nabla_w_list):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            # 获取梯度
            nabla_b_delta, nabla_w_delta = self.backprob(x, y)

            nabla_w = [w + w_delta for w, w_delta in zip(nabla_w, nabla_w_delta)]
            nabla_b = [b + b_delta for b, b_delta in zip(nabla_b, nabla_b_delta)]

        nabla_b_list.append(nabla_b)
        nabla_w_list.append(nabla_w)


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

        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]

        loss_results = [self.loss(self.feedforward(x),y).sum() for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results),sum(loss_results)

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

if __name__ =='__main__':

    print('loading data ...')

    mydata = my_data(10)

    X_train, X_test, y_train, y_test = mydata.train_test_split()

    training_data = []
    for i, j in zip(X_train, y_train):
        training_data.append((i.reshape(6, 1), j.reshape(8, 1)))

    test_data = []
    for i, j in zip(X_test, y_test):
        test_data.append((i.reshape(6, 1), j.reshape(8, 1)))

    print('creating model...')

    nn = NeuralNetwork(sizes=[6, 6 * 4, 8 * 4, 8])

    print('fiting...')


    mini_batch_size=32

    eta=0.1

    pool_size=4

    epochs=100

    m = len(training_data)

    if test_data != None:
        m_test_data = len(test_data)

    for j in range(epochs):

        # 搅乱顺序
        np.random.shuffle(training_data)

        # 按照小样本划分训练集
        mini_batchs = [training_data[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]

        mini_batch_len=len(mini_batchs)

        pool_mini_batchs=[mini_batchs[k:k+pool_size]  for k in range(0, mini_batch_len, pool_size)]

        #lock=Lock()

        for j,mini_batchs in enumerate(pool_mini_batchs):

            nabla_b_list=[]
            nabla_w_list=[]
            process=[]
            for i,mini_batch in enumerate(mini_batchs):

                # 按照一定步长更新w,b

                p = Process(target=nn.update_mini_batch, args=[mini_batch,nabla_b_list,nabla_w_list])

                p.start()

                process.append(p)

            while True:

                time.sleep(1)

                if all([not (p.is_alive()) for p in process]):

                    a=sum(nabla_b_list)

                    b=sum(nabla_w_list)

                    _=[p.terminate() for p in process]

                    if test_data == None:

                        print('training ', str(j + 1), ' epoch...')

                    else:

                        accuracy_number, loss = nn.evaluate(test_data)

                        print('epoch ', str(j + 1), '/', str(epochs),'     ', str(accuracy_number))

                    break;





