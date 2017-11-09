import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
#构建Graph
    #构建输入占位符，标签占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
    #构建单层网络基本参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
    #构建单层网络：将输入、网络参数进行组合
y = tf.nn.softmax(tf.matmul(x,W) + b)
    #构建loss函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#初始化Variable的值
sess.run(tf.initialize_all_variables())

#训练步骤：使用GD，目的：最小化cross_cntropy
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#训练过程：进行1000次train_step步骤，每次从mnist中获取50个样例进行训练
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed_dict 提供数据：x,y_

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #是否正确预测，输出布尔序列

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #计算百分率

print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})#对test样本进行测试计算
