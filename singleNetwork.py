import tensorflow as tf

# https://github.com/ardiya/siamesenetwork-tensorflow
# https://github.com/ywpkwon/siamese_tf_mnist/blob/master/inference.py -other

def singleNet(input, reuse=False):
    with tf.name_scope("model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, 7, activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, 5, activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')
        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, 3, activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')
        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, 1, activation_fn=tf.nn.relu, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')
        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, 1, activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [200,1], padding='SAME')
        net = tf.contrib.layers.flatten(net)
    return net


def contrastiveLoss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2,2), 1, keep_dims=True))
        tmp = y * tf.square(d)
        tmp2 = (1-y) * tf.square(tf.maximum((margin-d), 0))
        return tf.reduce_mean(tmp + tmp2) /2

def contrastiveLossTest(model1, model2, margin, threashold):
    with tf.name_scope("contrastive-loss-test"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2,2), 1, keep_dims=True))
        tmp = tf.square(d)
        tmp2 = tf.square(tf.maximum((margin-d), 0))
        res = (tf.reduce_mean(tmp + tmp2) /2)
        return 0 if res > threashold else 1