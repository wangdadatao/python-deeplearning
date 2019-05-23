import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)

sum = tf.add(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum))
