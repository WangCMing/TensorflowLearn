#!/usr/bin/python
#coding:utf-8
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
#构建Graph
    #构建输入占位符，标签占位符,x是输入,y_是正确的输出 
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
    #构建单层网络基本参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
    #构建单层网络：将输入、网络参数进行组合,y是预测输出
y = tf.nn.softmax(tf.matmul(x,W) + b)
    #构建loss函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#初始化Variable的值
sess.run(tf.initialize_all_variables())

#训练步骤：使用GD，目的：最小化cross_cntropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#训练过程：进行1000次train_step步骤，每次从mnist中获取50个样例进行训练
for i in range(1100): 
  batch = mnist.train.next_batch(50)
#   print("batch[0]:",batch[0],"batch[1]:",batch[1],"\n")   #batch[1].shape = [50,10] 一次batch_size的量
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed_dict 提供数据：x,y_
#   print(sess.run(y,feed_dict={x: batch[0], y_: batch[1]}))    

'''
下面是测试的操作,上面的训练过程完成之后,W和b的值已经确定了下来!!
训练的时候我们使用了交叉熵的概念来计算模型的优劣程度,
但是在测试时,我们需要使用正确率来计算,因此我们创建一个correct_prediction 节点来计算正确率
!!此时,我们重新进行了出入x和y_的定义:mnist.test.images minist.test.labels
此时的x和y_已经改变,不能再看以前的数据了
注意:
    不要混淆,在for中,我们使用GD算法对W b的值进行了训练(只有在optimizer中才会对Variable的值进行训练 ,其他的操作并不会改变他们的值)
    此时我们的输入是for循环中的feed_dict={x: batch[0], y_: batch[1]}.此时还没有correct_prediction accuracy 节点

    之后我们在原有节点上创建了它们,它们这时才加入Graph中,correct_prediction通过y y_来与原有的Graph进行链接,accuracy通过 correct_prediction加入Graph中
    之后我们重新制定了输入x 和y_ 此时没有Optimizer,所以W b的值都不在改变,所以进行了测试操作
'''
#之后的操作还是创建Graph,这部分是在当前的W和b下,计算输入test数据集后的正确率
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #是否正确预测，输出布尔序列
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #根据布尔序列计算百分率

print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})#对test样本进行测试计算
