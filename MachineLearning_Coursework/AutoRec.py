#!/usr/bin/python
#coding:utf-8

from __future__ import division, print_function, absolute_import
import Tkinter
import tensorflow as tf
import numpy as np

# Import date
'''
9125个电影
100004个评分
671个用户

输入之前要先对数据进行转换:
    对每个用户的数据都扩展到9125个
    将用户分为450 和221个 前一部分做训练集\后一部分做测试集
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

num_steps = 30000
batch_size = 256

display_step = 100
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)
learning_rate =0.01

FILEPATH ='/home/wcm/TensorflowLearn/output/AE_rate=0.01_RMSPro/'

x = tf.placeholder("float",[None,num_input])
x_reshape = tf.reshape(x,[-1,28,28,1])
tf.summary.image("x",x_reshape)
with tf.name_scope('encoder_layer1'):
    weight = tf.Variable(tf.random_normal([num_input,num_hidden_1]))
    biases = tf.Variable(tf.random_normal([num_hidden_1]))
    encoder_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weight),biases))

with tf.name_scope('encoder_layer2'):
    weight = tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2]))
    biases = tf.Variable(tf.random_normal([num_hidden_2]))
    encoder_layer2 = tf.nn.sigmoid(tf.matmul(encoder_layer1,weight)+biases)

with tf.name_scope('decoder_layer_1'):
    weight = tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1]))
    biases = tf.Variable(tf.random_normal([num_hidden_1]))
    decoder_layer_1 = tf.nn.sigmoid(tf.matmul(encoder_layer2,weight)+biases)

with tf.name_scope('decoder_layer_2'):
    weight = tf.Variable(tf.random_normal([num_hidden_1,num_input]))
    biases = tf.Variable(tf.random_normal([num_input]))
    decoder_layer_2 = tf.nn.sigmoid(tf.matmul(decoder_layer_1,weight)+biases)

y_pre  = decoder_layer_2
y_true = x
y_pre_reshape = tf.reshape(y_pre,[-1,28,28,1])
tf.summary.image("y_pre",y_pre_reshape)
#define loss and optimizer

loss = tf.reduce_mean(tf.pow(y_true-y_pre,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

tf.summary.scalar("loss",loss)

init = tf.global_variables_initializer()

summary = tf.summary.merge_all()


with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(FILEPATH, sess.graph)
    sess.run(init)
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={x: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            summary_str = sess.run(summary, feed_dict={x: batch_x})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()