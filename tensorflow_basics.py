import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)
print(result)

#Method 1: requires you to close the session explicitly
sess = tf.Session()
print(sess.run(result))
sess.close()

#Method 2
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)
