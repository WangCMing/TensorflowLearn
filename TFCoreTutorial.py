from __future__ import print_function

import tensorflow as tf
import numpy as np
#creat node 
node1 = tf.constant(3.0,dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

##run computational graph
sess = tf.Session()
print(sess.run([node1,node2]))

#value node combine with op node

node3 = tf.add(node1,node2)
print("node3:",node3)
print("sess.run(node3):",sess.run(node3))

#creat variable node
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b # equals to tf.add(a,b)

print(sess.run(adder_node,{a:3,b:4.5}))
print(sess.run(adder_node,{a:[1,3],b:[2,4]}))

# more operator

add_and_triple = adder_node*3
print(sess.run(add_and_triple,{a:3,b:4.5}))

#use Variable node
    # variable node allow to be trained,constructed with a type and initial value
W = tf.Variable([.3],dtype = tf.float32)
b = tf.Variable([-.3],dtype = tf.float32)
x = tf.placeholder(tf.float32)

linear_model  = W*x+b

    # need to be initialized
init = tf.global_variables_initializer()
sess.run(init)
    #run the graph
print(sess.run(linear_model,{x:[1,2,3,4]}))
    #evaluation 
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
    #change the W b to make model better
'''
fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb]) #assign W b 'better' value
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]})) #the loss is zero
'''
#use tf.train API to auto get better [W b]
optimizer = tf.train.GradientDescentOptimizer(0.01)     #create GD algorithm
trian = optimizer.minimize(loss)                        #set the target 
for i in range(1000):                                   #run 1000 epoch
    sess.run(trian,{x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))                                  #Variable set W b can be trained


