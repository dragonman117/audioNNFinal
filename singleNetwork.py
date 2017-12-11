import tensorflow as tf

# https://github.com/ardiya/siamesenetwork-tensorflow
# https://github.com/ywpkwon/siamese_tf_mnist/blob/master/inference.py -other

def singleNet(input, reuse=False):
    print("Shape: ",input.shape)
    print("Rank: ", input)
    with tf.name_scope("model"):
        with tf.variable_scope("convLayer1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [600, None],activation_fn=tf.nn.relu, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            # net = tf.layers.conv1d(input, 32, 600, strides=1, padding="same",activation=tf.nn.relu, reuse=reuse)
            #net = tf.layers.max_pooling1d(net, 200, strides=1, padding="same")
        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [300, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [250, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')

        net = tf.contrib.layers.flatten(net)
    return net


def contrastiveLoss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2), 1, keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1-y) * tf.square(tf.maximum((margin-d), 0))
        return tf.reduce_mean(tmp + tmp2) /2