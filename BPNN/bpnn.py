import numpy as np

class layer:

    def __int__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.weights=np.ones((input_size,output_size))
        self.bias=np.ones((input_size,1))


class NNModel:

    def __int__(self,num_round,mini_batch_size,learning_rate,loss_function):
        self.input_x=self.input_y=...
        
        self.num_round=num_round
        self.batch_size=mini_batch_size
        self.learning_rate=learning_rate
        self.loss_function=self.mse
        self.layers=[]

    def add_layer(self,layer):
        self.layers.append(layer)

    def mse(self,input_y,output_y):

        return ((input_y-output_y)**2).sum()

    def active_funtion(self,node_output):
        '''

        :param node_output: 上次节点向量相乘的结果 ，形式为 output_n

        :return: 输出激活函数处理过的结果 形式为ouput_n
        '''

        return  1/(1+np.exp(-node_output))

    def active_derivative(self):...

    def mse_derivative(self):...

    def gradient_descent(self):


        for layer in self.layers:

            # 1. 根据损失函数计算梯度
            w_descent_direction =-(self.active_derivative*self.mse_derivative*())

            # 3.更新weights

            layer.weights+=self.learning_rate*w_descent_direction

    def forward_propagate(self,x):

        '''
        前向传输产生最后的输出
        :param x: 数据
        :return: 最后的输出结果
        '''
        for i, layer in enumerate(self.layers):
            # input_n-->output_n
            zL = x * layer.weights+layer.bias
            # output_n-->output_n
            aL = self.active_funtion(zL)

            #把上一层的输出当作上一层的输入
            x = aL

        return aL

    def fit(self,input_x,input_y):

        '''
        算法步骤：
        循环
        1.前向传输，产生结果
        2.计算损失函数
        3.求梯度向量
        4.更新参数

        :param input_x:
        :param input_y:
        :return:
        '''
        for i in range(self.batch_size):

            #进行batch数据选择
            input_x=np.ndarray(input_x)[self.batch_size*(i):self.batch_size*(i+1)]
            input_y=np.array(input_y)[self.batch_size*(i):self.batch_size*(i+1)]

            #定义loss
            loss=0

            # 前向传播
            for x in input_x:
                output_y=self.forward_propagate(x)

                loss+=self.loss_function(input_x,input_y)

            print(i,loss)

    def predict(self):...


if __name__=='__main__':

    nn=NNModel(num_round=100,mini_batch_size=100,learning_rate=0.1,loss_function='mse')
    layer1=(100,50)
    nn.add_layer(layer)
    layer2=(50,1)
    nn.fit()
    nn.predict()
