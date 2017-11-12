import tensorflow as tf

y = tf.Variable(tf.random_normal([28*28]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = tf.reshape(y,[-1,28,28,1])
    print(sess.run([y]))
    print(sess.run([x]))