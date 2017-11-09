from __future__ import print_function
import tensorflow as tf

#create model parameters
W = tf.Variable([.3],dtype = tf.float32)
b = tf.Variable([-.3],dtype = tf.float32)

#create input and output
x = tf.placeholder(tf.float32)
linear_model = W*x+b
y = tf.placeholder(tf.float32)

#create loss function
loss = tf.reduce_sum(tf.square(linear_model-y))

#create optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)




#create train data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
    #init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    #create training loop
for i in range(1000):
    sess.run(train,{x:x_train,y:y_train}) #placeholder x ,y be placed by x_train y_train

#evaluate training accuracy
curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
print(" W  %s, b %s ,loss:%s"%(curr_W,curr_b,curr_loss))