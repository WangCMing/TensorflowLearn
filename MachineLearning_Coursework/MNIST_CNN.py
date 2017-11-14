#!/usr/bin/python
#coding:utf-8
'''
vanilla版本的CNN实现：
1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
用简单传统的2x2大小的模板做max pooling。
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
FILEPATH = 'output/CNN/CNN_rate=0.01/'
LEARN_RATE = 0.01
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #padding = 'SAME' 输入输出size一致

def max_pool_2x2(x):#size = 2x2x1（长宽为2，深度为1）
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') #步长：长宽为2 深度为1


x_image = tf.reshape(x, [-1,28,28,1])       #为了使输入x能够和W进行卷积，需要对x进行重构

'''
第一层卷积
    构建卷积子W和偏置b
        卷积子W.shape = [5,5,1,32] :大小为5x5，深度为1，输出为32
    reshape输入x为4-D Tensor
    使用relu作为激励函数
    使用max_pool_2x2作为池化操作
'''
with tf.name_scope("Conv1_layer") :
  W_conv1 = weight_variable([5, 5, 1, 32])    #patch =5x5 深度为1 每个像素输出通道为32 size = 28x28x32
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    #进行卷积操作，得到输出，之后使用RELU(cov(x,W)+b)后的输出
  h_pool1 = max_pool_2x2(h_conv1) #输出进行池化操作   14x14x32

'''
第二层卷积
'''
with tf.name_scope("Conv2_layer"):
  W_conv2 = weight_variable([5, 5, 32, 64])   #patch = 5x5 深度为32 输出通道为64 size = 14x14x64
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    #卷积操作：size =7x7x64
  h_pool2 = max_pool_2x2(h_conv2)

'''
图片尺寸减小到7x7（每个像素有64的通道）
我们加入一个有1024个神经元的全连接层，用于处理整个图片
全连接层
  input:7x7x64=reshape=>[1,7*7*64]=matmul:[1,7*7*64]x[7*7*64,1024]=>[1,1024]
  全链接层也需要激励函数！并不是单纯的连接
'''
with tf.name_scope("FC_layer_1"):
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


'''
drop_out
  以1-keep_prob的概率随机抛弃一些输出
'''
with tf.name_scope("DropoutOP"):
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


'''
输出层
  使用softmax，对每个输出计算概率，选取概率最大的
  [1,1024]=matmul:[1,1024]x[1024,10]=>[1,10]
'''
with tf.name_scope("FC_Sotfmax_layer"):
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
训练
  使用交叉熵作为评估函数
  使用AdamOptimizer作为优化算法，目标是最小化交叉熵
'''
with tf.name_scope("train_op"):
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) 
  tf.summary.scalar("cross_entropy",cross_entropy)
  train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)#使用AdamOptimizer优化算法（并非梯度优化算法）
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #y_conv是模型输出,y_是标签
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     #计算正确率百分比
  
  tf.summary.scalar("accuracy",accuracy)

'''
为了方便展示测试集正确率的变化，我们添加额外的accuracy节点
'''
with tf.name_scope("test_op"):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #y_conv是模型输出,y_是标签
  accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction, "float"))   #计算正确率百分比
  tf.summary.scalar("accuracy_test",accuracy_1)

summary_writer = tf.summary.FileWriter(FILEPATH, sess.graph)
summary = tf.summary.merge_all()

sess.run(tf.initialize_all_variables())

for i in range(3000):                                              #模型训练
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={                      #计算第i次迭代时在当前参数下的正确率
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)

    test_accuracy = accuracy_1.eval(feed_dict={
      x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
    })


    summary_str = sess.run(summary, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    summary_writer.add_summary(summary_str, i)
    summary_writer.flush()     
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})   #训练参数

print "test accuracy %g"%accuracy.eval(feed_dict={                  #进行测试
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

