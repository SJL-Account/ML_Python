import numpy as np
import time
from functools import reduce
from load_mnist import load_mnist_img
import matplotlib.pyplot as plt
# https://zhuanlan.zhihu.com/c_162633442
# https://github.com/wuziheng/CNN-Numpy/blob/master/layers/fc.py


def im2col(image, ksize, stride):
    '''
    caffe 中计算卷积的方法

    输入: batch_size,m,n,c
    输出:(m-k*n-k)*(batch_size*k*k*c)

    :param image: 图像
    :param ksize: 核大小
    :param stride: 步长
    :return: 特征形状的向量
    '''
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    # 处理后的形状为 [batch_size,input_channel,(ksize*ksize)]
    image_col = np.array(image_col)

    return image_col

class Conv2d:

    '''
    二维卷积功能类
    '''
    def __init__(self,shape,ouput_channel,ksize=3,stride=1,method='VALID',):
        '''
        卷积类的初始化
        :param shape:输入形状
        :param ouput_channel:输出维度
        :param ksize: 核尺寸
        :param stride: 卷积步长
        :param method: 边缘处理方法
        '''

        '''
        输入数据的shape = [N,W,H,C] N=Batchsize/W=width/H=height/C=channels
        卷积核的尺寸ksize ,个数output_channels, kernel shape [output_channels,k,k,C]
        卷积的步长，基本默认为1.
        卷积的方法，VALID or SAME，即是否通过padding保持输出图像与输入图像的大小不变
        '''
        self.layer_name='conv2d'
        self.shape=shape
        self.batch_size=shape[0]
        self.input_channel=shape[-1]
        self.output_channel=ouput_channel
        self.ksize=ksize
        self.stride=stride
        self.method=method
        self.weights=np.random.standard_normal((ksize,ksize,self.input_channel,self.output_channel))
        self.bias=np.random.standard_normal(self.output_channel)
        # TODO 把same的情况处理一下
        if method=='VALID':
            self.eta=np.zeros((self.batch_size,int((self.shape[1]-ksize+1)/self.stride),int((self.shape[2]-ksize+1)/self.stride),self.output_channel))
        elif method=='SAME':
            ...
        self.output_shape=self.eta.shape
        self.w_gradient=np.zeros(self.weights.shape)
        self.b_gradient=np.zeros(self.bias.shape)

    def forward(self,x):

        # 行权重:变成和图像尺寸相当的形状
        col_weights = self.weights.reshape((-1,self.output_channel))
        # 输出尺寸和delta的尺寸一致
        out_x = np.zeros(self.eta.shape)

        self.col_img=[]

        for i in range(self.batch_size):

            img_i = x[i][np.newaxis, :]
            # TODO
            col_img_i = im2col(img_i,ksize=self.ksize,stride=self.stride)

            # (900, 27) (27, 4)  27代表 3*3核，3通道， 4代表输出通道
            out_x[i] = np.reshape(np.dot(col_img_i,col_weights)+self.bias,self.eta[0].shape)

            self.col_img.append(col_img_i)

        self.col_img = np.array(self.col_img)

        return  out_x

    def backward(self,eta):

        # 1.更新梯度
        self.eta=eta
        # 1.1 更新成 批次*(l+1层特征图长*宽)*(l+1层特征图数量)
        col_eta=eta.reshape([self.batch_size,-1,self.output_channel])
        # 1.2更新参数
        for i in range(self.batch_size):
            self.w_gradient+=(np.dot(self.col_img[i].T,col_eta[i]).reshape(self.weights.shape))
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        if self.method == 'VALID':
            pad_delta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)), 'constant',
                               constant_values=0)

        # 2.传递delta
        # next_delta=当前层的激活值点乘经过180度翻转的kernel
        # 2.1 对kernel进行180度翻转
        flip_weights = np.flipud(np.fliplr(self.weights))
        # 2.2对权重进行转置
        flip_weights = np.swapaxes(flip_weights,2,3)
        # 2.3 reshape 成和col_img 一样的形状
        col_flip_weights = flip_weights.reshape(-1,self.input_channel)
        col_pad_delta = np.array([im2col(pad_delta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batch_size)])
        # img 点乘 经过180度翻转的权重
        next_eta=np.dot(col_pad_delta,col_flip_weights)
        next_eta=np.reshape(next_eta,self.shape)
        return next_eta

    def update_mimi_batch(self,step_size,epoch,decay_rate=0.99,decay_steps=10,):

        #decayed_learning_rate = (step_size / self.batch_size) * decay_rate ** (epoch / decay_steps)

        decay_rate = 0.96
        decay_steps = 10
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8

        self.vdb = np.zeros((self.bias.shape))
        self.vdw = np.zeros((self.weights.shape))
        self.sdb = np.zeros((self.bias.shape))
        self.sdw = np.zeros((self.weights.shape))

        self.sdb = np.multiply(beta2 * self.sdb + (1.0 - beta2), (self.b_gradient) ** 2)
        self.sdw = np.multiply(beta2 * self.sdw + (1.0 - beta2), (self.w_gradient) ** 2)

        self.vdb = np.multiply(beta1 * self.vdb + (1.0 - beta1), self.b_gradient)
        self.vdw = np.multiply(beta1 * self.vdw + (1.0 - beta1), self.w_gradient)

        nabla_b = (step_size ) * self.vdb / (np.sqrt(self.sdb) + epsilon)
        nabla_w = (step_size ) * self.vdw / (np.sqrt(self.sdw) + epsilon)

        # 梯度下降更新
        weight_decay=0.0004
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= nabla_w
        self.bias -= nabla_b

        self.w_gradient=np.zeros(self.weights.shape)
        self.b_gradient=np.zeros(self.bias.shape)

class AvgPooling:

    def __init__(self,shape,ksize,stride):

        self.shape=shape
        self.ksize=ksize
        self.stride=stride
        self.input_channel=shape[-1]
        self.out_channel=shape[-1]
        self.index=np.zeros(shape)
        self.output_shape=[shape[0],int(self.shape[1]/self.stride),int(self.shape[2]/self.stride),self.out_channel]

    def forward(self,x):

        out_x=np.zeros(self.output_shape)

        for b in range(self.shape[0]):
            for c in range(self.input_channel):
                for i in range(0,self.shape[1],self.stride):
                    for j in range(0,self.shape[2],self.stride):
                        out_x[b,int(i/self.stride),int(j/self.stride),c]=np.mean(x[b,i:i+self.ksize,j:j+self.ksize,c])

        return  out_x


    def backward(self,eta):

        # TODO 为什么要乘以index?,改成np.multiply
        next_eta=np.multiply(np.repeat(np.repeat(eta,self.stride,axis=0),self.stride,axis=1),self.index)

        return next_eta/(self.stride*self.stride)

class MaxPooling:

    def __init__(self,shape,ksize,stride):

        self.layer_name='maxpooling'
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.batch_size = shape[0]
        self.input_channel = shape[-1]
        self.out_channel = shape[-1]
        self.index=np.zeros(shape)

        # 对不能整除的进行处理

        self.row_lack=self.stride-self.input_shape[1]%self.stride
        self.col_lack=self.stride-self.input_shape[2]%self.stride
        self.row_lack=0 if (self.row_lack==self.stride) else self.row_lack
        self.col_lack=0 if (self.col_lack==self.stride) else self.col_lack
        self.shape=[0,0,0,0]
        self.shape[0] = self.batch_size
        self.shape[1] = self.input_shape[1]+self.row_lack
        self.shape[2] = self.input_shape[2]+self.col_lack
        self.shape[3] = self.input_channel
        self.output_shape=[shape[0],int(self.shape[1]/self.stride),int(self.shape[2]/self.stride),self.out_channel]

    def forward(self,x):

        # 降维后的维度
        out_x=np.zeros(self.output_shape)

        # 对不能整除的尺寸进行补0运算

        x=np.pad(x,[[0,0],[0,self.row_lack],[0,self.col_lack],[0,0]],mode='constant',constant_values=(0))

        for b in range(self.shape[0]):
            for c in range(self.input_channel):
                for i in range(0,self.shape[1],self.stride):
                    for j in range(0,self.shape[2],self.stride):
                        # 原空间-->下采样空间
                        # i/strid 行走第几步，j/strid 列走第几步
                        out_x[b,int(i/self.stride),int(j/self.stride),c]=np.max(x[b,i:i+self.ksize,j:j+self.ksize,c])
                        # 记住最大的行和列
                        index=np.argmax(x[b,i:i+self.ksize,j:j+self.ksize,c])
                        # index/strid 取第几行 index%strid 取第几列
                        self.index[b,i+int(index/self.stride),j+(index%self.stride),c]=1
        return  out_x

    def backward(self,eta):

        #对原来补0的地方进行删除
        next_eta=np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2)
        next_eta=next_eta[:,:next_eta.shape[1]-self.col_lack,:next_eta.shape[2]-self.col_lack,:]
        return np.multiply(next_eta,self.index)

class FullyConnect:

    def __init__(self,shape,out_num):
        self.layer_name='fullyconnect'
        self.shape=shape
        self.batch_size=shape[0]
        # 求flatten 数量是为了初始化权重
        self.flattn_num=reduce(lambda x,y:x*y,shape[1:])
        self.weights=np.random.standard_normal((self.flattn_num,out_num))
        self.bias=np.random.standard_normal(out_num)
        self.output_shape=[self.batch_size,out_num]
        self.w_gradient=np.zeros(self.weights.shape)
        self.b_gradient=np.zeros(self.bias.shape)

    def forward(self,x):

        # 直接放倒
        self.x = x.reshape([self.batch_size,-1])
        #点乘返回
        return np.dot(self.x,self.weights)+self.bias

    def backward (self,eta):

        # 1.更新梯度
        # 权重梯度=该层单元的激活值点乘下一层传递过来的delta
        for i in range(self.batch_size):
            #input_dim
            col_x = self.x[i][:,np.newaxis]
            #out_dim
            eta_i=eta[i][:,np.newaxis].T
            self.w_gradient+=np.dot(col_x,eta_i)
            self.b_gradient+=eta_i.reshape(self.bias.shape)

        #2.向前传递delta
        next_eta=np.dot(eta,self.weights.T)
        next_eta=next_eta.reshape(self.x.shape)

        return next_eta

    def update_mimi_batch(self, step_size, epoch, decay_rate=0.99, decay_steps=10, ):
        # decayed_learning_rate = (step_size / self.batch_size) * decay_rate ** (epoch / decay_steps)
        #
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8

        self.vdb = np.zeros((self.bias.shape))
        self.vdw = np.zeros((self.weights.shape))
        self.sdb = np.zeros((self.bias.shape))
        self.sdw = np.zeros((self.weights.shape))

        self.sdb = np.multiply(beta2 * self.sdb + (1.0 - beta2), (self.b_gradient) ** 2)
        self.sdw = np.multiply(beta2 * self.sdw + (1.0 - beta2), (self.w_gradient) ** 2)

        self.vdb = np.multiply(beta1 * self.vdb + (1.0 - beta1), self.b_gradient)
        self.vdw = np.multiply(beta1 * self.vdw + (1.0 - beta1), self.w_gradient)

        nabla_b = (step_size ) * self.vdb / (np.sqrt(self.sdb) + epsilon)
        nabla_w = (step_size ) * self.vdw / (np.sqrt(self.sdw) + epsilon)

        # 梯度下降更新
        weight_decay=0.0004
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= nabla_w
        self.bias -= nabla_b

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

class Softmax:
    # TODO http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    def __init__(self,shape):
        self.shape=shape
        self.layer_name='softmax'
        self.batch_size=shape[0]
        self.eta=np.zeros(shape)
        self.softmax=np.zeros(shape)
        self.output_shape=shape

    def cal_loss(self,prediction,label):
        self.label=label
        self.prediction=prediction
        self.loss=0.0
        self.predict(prediction)
        for i in range(self.batch_size):
            self.loss+=np.log(np.exp(prediction[i]).sum())-prediction[i,int(label[i])]
        return self.loss

    def predict(self,prediction):

        exp_prediction=np.zeros(prediction.shape)

        self.softmax=np.zeros(prediction.shape)

        for i in range(self.batch_size):

            # TODO 这里注意一下，防止溢出
            prediction[i,:]-=np.max(prediction[i,:])

            exp_prediction[i] = np.exp(prediction[i])

            self.softmax[i]=exp_prediction[i]/exp_prediction[i].sum()

        return  self.softmax

    def backward(self):

        self.eta=self.softmax.copy()

        for i in range(self.batch_size):

            self.eta[i,int(self.label[i])]-=1

        return self.eta

class Relu():
    def __init__(self, shape):
        self.shape=shape
        self.layer_name='relu'
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    def backward(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta

class Model:

    def __init__(self):

        self.layers=[]
        self.accuracys=[]
        self.train_losses=[]

    def summary(self):

        print('summary :')

        for layer in self.layers:

            if (layer.layer_name=='conv2d')|(layer.layer_name=='fullyconnect'):

                print('name:',layer.layer_name,'   input_shape:',layer.shape,'   output_shape:',layer.output_shape,'  weights:',layer.weights.shape,)
            else:
                print('name:',layer.layer_name,'   input_shape:',layer.shape,'   output_shape:',layer.output_shape, )

            print('-------------------------------------------------------------------------------------------')

    def add(self,layer):

        self.layers.append(layer)

    def forward(self,batch_xs, batch_ys):

        input_=batch_xs

        for layer in self.layers:

            loss = 0.0

            if layer.layer_name=='softmax':

                pre_prob = np.zeros(layer.output_shape)

                delta = np.zeros(layer.output_shape)

                loss = layer.cal_loss(out_,batch_ys)

                pre_prob=layer.softmax


                return loss,pre_prob

            else :

                out_=layer.forward(input_)

                input_=out_

    def backward(self,step_size,epoch,decay_rate,decay_steps):

        m=len(self.layers)

        delta=self.layers[m - 1].backward()

        for i in range(2,m):

            if (self.layers[m - i].layer_name=='conv2d')|(self.layers[m - i].layer_name=='fullyconnect'):
                self.layers[m - i].update_mimi_batch(step_size,epoch,decay_rate,decay_steps)

            delta = self.layers[m - i].backward(delta)

            if self.layers[m-i].layer_name=='fullyconnect':

                delta=delta.reshape(self.layers[m-i-1].output_shape)
    def plot_losses(self):

        plt.plot(self.train_losses)


        plt.savefig('losses.jpg',dpi=100)

        plt.close()

        plt.plot(self.accuracys)

        plt.savefig('accuracy.jpg',dpi=100)

        plt.close()

    def evaluate(self,test_data):

        accuraccy=0.0

        for xs, ys in test_data:

            xs = xs[:, :, :, np.newaxis]

            loss, pre_prob = self.forward(xs, ys)

            accuraccy+=sum([ int(np.argmax(pre)==int(true)) for pre,true in zip(pre_prob,ys)])/len(xs)

        accuraccy=accuraccy/len(test_data)

        return accuraccy,loss

    def train(self,train_data,test_data,mini_batch_size=32,step_size= 5e-4,epochs=100,decay_rate=0.99,decay_steps=10):


        for epoch in range(epochs):

            np.random.shuffle(train_data)
            np.random.shuffle(test_data)

            batch_start=time.time()

            print('epoch :', epoch)

            for i,(xs, ys) in enumerate(train_data):

                xs=xs[:, :, :, np.newaxis]

                start=time.time()

                loss, pre_prob=self.forward(xs,ys)

                self.backward(step_size,epoch,decay_rate,decay_steps)

                self.train_losses.append(loss)

                print('batch-', i, 'current loss:', loss,'  spend-time:',time.time()-start,'s')


            accuracy,loss = self.evaluate(test_data)

            self.accuracys.append(accuracy)

            print('test accuracy:',accuracy,'  test loss:',loss,'  batch-spend-time:',time.time()-batch_start,'s')

            self.plot_losses()

model=Model()

batch_size=64

conv1 = Conv2d(shape=(batch_size, 28, 28, 1), ouput_channel=12,ksize=5,stride=1)

model.add(conv1)

relu1 = Relu(conv1.output_shape)

model.add(relu1)

pool1 = MaxPooling(relu1.output_shape, ksize=2, stride=2)

model.add(pool1)

conv2 = Conv2d(pool1.output_shape, ouput_channel=24,ksize=3,stride=1)

model.add(conv2)

relu2 = Relu(conv2.output_shape)

model.add(relu2)

pool2 = MaxPooling(relu2.output_shape, ksize=2, stride=2)

model.add(pool2)

fc = FullyConnect(pool2.output_shape, out_num=10)

model.add(fc)

softmax=Softmax(fc.output_shape)

model.add(softmax)

model.summary()

train_data, test_data = load_mnist_img(batch_size)

model.train(train_data[:1000],test_data[:100])


