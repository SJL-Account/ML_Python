#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist  import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

sess=tf.InteractiveSession()

#定义公式
in_units=784
h1_units=300
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
#对于sigmoid函数，在0处最敏感，梯度最大
b1=tf.Variable(tf.zeros([h1_units]))
w2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

y_=tf.placeholder(tf.float32,[None,10])

#隐藏层
hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
#定义损失函数啊和训练器
cross_entropy=tf.reduce_mean (-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
#开始训练
for i in range (3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
    
#结果评估
corret_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(corret_prediction,tf.float32))
#print (accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
print (len(batch_xs))