import tensorflow as tf
import pandas as pd
import numpy as np
from  sklearn.preprocessing import  OneHotEncoder
from  sklearn.preprocessing import  StandardScaler
from sklearn.utils import shuffle


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

print('current tf version',tf.VERSION)




class my_data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.i = 0
        self.load_data()

    def load_data(self):

        x_train, x_test, y_train, y_test=self.train_test_split()


        self.x_data = x_train
        self.y_data = y_train

    def train_test_split(self):

        data = pd.read_csv('all_data.txt', delimiter='\t',encoding='gbk')

        #data=shuffle(data)

        test_data = data[data['训练测试分区'] == 0]
        train_data = data[data['训练测试分区'] != 0]
        x_train = train_data[['AC', 'CNL', 'DEN', 'GR', 'PE', 'RLLD']]
        y_train = train_data['code']

        x_test = test_data[['AC', 'CNL', 'DEN', 'GR', 'PE', 'RLLD']]
        y_test = test_data['code']

        ohe = OneHotEncoder()
        std = StandardScaler()
        x_train=std.fit_transform(X=x_train)
        y_train=ohe.fit_transform(np.matrix(y_train).T).toarray()
        x_test=std.fit_transform(X=x_test)
        y_test=ohe.fit_transform(np.matrix(y_test).T).toarray()

        return x_train, x_test, y_train, y_test

    def next_batch(self):
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return re_x, re_y

    def __len__(self):
        return int(len(self.x_data) / self.batch_size)

    def __getitem__(self, i):
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return (re_x, re_y)

    def __next__(self):
        if self.i == len(self):
            raise StopIteration
        re_x = self.x_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        re_y = self.y_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return re_x, re_y

    def __iter__(self):
        return self



input_dim = 6
output_dim = 8
batch_size = 50
epochs = 2000
hidden_num = input_dim*5
hidden_num_2 = input_dim*5
learning_rate=0.01
batch=True
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8

# 1.定义数据
with tf.variable_scope("data"):

    x=tf.placeholder(tf.float32,shape=[None,input_dim],name='input_x')

    y_true=tf.placeholder(tf.float32,shape=[None,output_dim],name='input_y')


#2.定义网络层

with tf.device('/cpu:0'):

    with tf.variable_scope("hidden_1"):
        #[None,input_dim]*[input_dim,hidden_num]+[None,hidden_num]=[None,hidden_num]
        #[input_dim,hidden_num]
        input_hidden_1_weights=tf.Variable(tf.random_normal(shape=[input_dim,hidden_num]),name='input_hidden_1_weights')
        #[None,hidden_num]
        input_hidden_1_bias=tf.Variable(tf.constant(0.0,shape=[hidden_num]),name='input_hidden_1_bias')
        #[None,32]
        hidden_1=tf.nn.relu(tf.matmul(x,input_hidden_1_weights)+input_hidden_1_bias)

with tf.device('/gpu:0'):

    with tf.variable_scope("hidden_2"):

        hidden_1_hidden_2_weights = tf.Variable(tf.random_normal(shape=[hidden_num, hidden_num_2]), name='hidden_1_hidden_2_weights')
        hidden_1_hidden_2_bias = tf.Variable(tf.constant(0.0, shape=[hidden_num_2]),name='hidden_1_hidden_2_bias')
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, hidden_1_hidden_2_weights) + hidden_1_hidden_2_bias)

with tf.variable_scope("output"):

    hidden_2_output_weights = tf.Variable(tf.random_normal(shape=[hidden_num_2, output_dim]), name='hidden_2_output_weights')
    hidden_2_output_bias = tf.Variable(tf.constant(0.0, shape=[output_dim]),name='hidden_2_output_bias')
    y_predict = tf.matmul(hidden_2, hidden_2_output_weights) + hidden_2_output_bias

#3.2定义正则项

with tf.variable_scope("L2_loss"):

    reg_l2=tf.contrib.layers.l2_regularizer(0.1)

#3.1定义损失

with tf.variable_scope("loss"):

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
#4.定义优化策略
with tf.variable_scope('optimize'):

    train_op=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


#5.定义准确率
with tf.variable_scope('accuracy'):

    equal_list=tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))

    accuracy=tf.reduce_mean(tf.cast(equal_list,tf.float32))



tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("input_hidden_1_weights",input_hidden_1_weights)
tf.summary.histogram("input_hidden_1_bias",input_hidden_1_bias)
tf.summary.histogram("hidden_2_output_weights",hidden_2_output_weights)
tf.summary.histogram("hidden_2_output_bias",hidden_2_output_bias)


merged=tf.summary.merge_all()

init_op=tf.global_variables_initializer()


#6.开始运行
with tf.Session(config=config) as sess:

    sess.run(init_op)

    filewriter=tf.summary.FileWriter('./graph',graph=sess.graph)


    for l in range(epochs):

        data = my_data(batch_size=batch_size)

        X_train, X_test, y_train, y_test = data.train_test_split()

        for i,d in enumerate(data):

            x_data=d[0]
            y_data=d[1]

            sess.run(train_op, feed_dict={x: x_data, y_true: y_data})
            # 7.运行是查看准确率


        print('准确率为：%d %f' % (l,sess.run(accuracy, feed_dict={x: X_test, y_true: y_test})))

    summary = sess.run(merged, feed_dict={x: x_data, y_true: y_data})

    filewriter.add_summary(summary, i)


    filewriter.close()

