import tensorflow as tf
import numpy as np
from singleNetwork import singleNet, contrastiveLoss, contrastiveLossTest
from generator import Generator

#Consts
trainIter = 1000


def train(dataset):
    #reset graph
    tf.reset_default_graph()

    #Unpack inputs
    trackA = dataset["specA"]
    trackB = dataset["specB"]
    segs = [dataset["aClassification"], dataset["bClassification"]]

    # Prep
    left = tf.placeholder(tf.float32, [None, 8, 32, 1], name="left")
    right = tf.placeholder(tf.float32, [None, 8, 32, 1], name="right")
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name="label")
        label = tf.to_float(label)

    margin = 0.2
    threashold = 0.005

    gen = Generator(trackA, trackB, segs)

    #Two different networks?
    left_output = singleNet(left, reuse=False)
    right_output = singleNet(right, reuse=True)

    loss = contrastiveLoss(left_output, right_output, label, margin)

    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(0.004, 0.099, use_nesterov=True).minimize(loss, global_step=global_step)

    train_res = contrastiveLossTest(left_output, right_output, margin, threashold )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train A seg (track 1)
        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("a")
            _, l = sess.run([train_step, loss], feed_dict={left:bLeft, right:bRight, label:bSim})
            # writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l) ## I think this string should be what we encode

        # Test A seg (track 1)
        tLeft, tRightArray = gen.getTrain("a")
        for tRight in tRightArray:
            l = sess.run([train_res], feed_dict={left:tLeft, right:tRight})
            print("Predicted: ", l)

    print("Break Between parts A and B")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train B seg (track2)
        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("b")
            _, l = sess.run([train_step, loss], feed_dict={left:bLeft, right:bRight, label:bSim})
            # writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l) ## I think this string should be what we encode

        # TestB seg (track2)
        tLeft, tRightArray = gen.getTrain("b")
        for tRight in tRightArray:
            l = sess.run([train_res], feed_dict={left: tLeft, right: tRight})
            print("Predicted: ", l)