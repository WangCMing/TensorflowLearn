#!/usr/bin/python
#coding:utf-8

from __future__ import division, print_function, absolute_import
import Tkinter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data 
# Import data
TRAIN_SET,TEST_SET = idate.get_date_set()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

num_input = 784
num_hidden_1 = 512
num_hidden_2 = 256
num_steps

display_step = 100
learning_rate =0.01

FILEPATH ='output/AE/AE_rate=0.1_RMSPro/'

x = tf.placeholder("float",[None,num_input])

with tf.name_scope('encoder_layer1'):
    weight = tf.Variable(tf.random_normal([num_input,num_hidden_1]))
    biases = tf.Variable(tf.random_normal([num_hidden_1]))
    encoder_layer1 = tf.nn.sigmoid(tf.matmul(x,weight)+biases)

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