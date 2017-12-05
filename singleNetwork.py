import tensorflow as tf

# https://github.com/ardiya/siamesenetwork-tensorflow
#  https://github.com/ywpkwon/siamese_tf_mnist/blob/master/inference.py -other

def singleNet(input, reuse=False):
    with tf.name_scope("model"):
        with tf.variable_scope("convLayer1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [100, 1],activation_fn=tf.nn.relu, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2,2], padding='SAME')