import tensorflow as tf
import numpy as np
from singleNetwork import singleNet, constrastiveLoss
from generator import Generator

#Consts
trainIter = 1


def train(trackA, trackB, segs):
    # Prep
    left = tf.placeholder(tf.float32, [None, 44100, 1], name="left")
    right = tf.placeholder(tf.float32, [None, 44100, 1], name="right")
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name="label")
        label = tf.to_float(label)

    margin = 0.2

    left_output = singleNet(left, reuse=False)
    right_output = singleNet(right, reuse=True)

    loss = constrastiveLoss(left_output, right_output, label, margin)

    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    gen = Generator(trackA, trackB, segs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("a")
            _, l, summary_str = sess.run([train_step, loss, merged], feed_dict={left:bLeft, right:bRight, label:bSim})
            print(summary_str)