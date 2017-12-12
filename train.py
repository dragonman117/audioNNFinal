import tensorflow as tf
import numpy as np
from singleNetwork import singleNet, contrastiveLoss
from generator import Generator
import os
import matplotlib.pyplot as plt
import datetime

lossCounter = 0
EPOCHS_PER_PLOT = 5
lossCounter = 0
losses = []

GRAPH_DIR = ('./graphs/' + datetime.datetime.now().isoformat(timespec='seconds')).replace(':','-')
if os.path.exists(GRAPH_DIR):
    print('Graph directory for this run already exists. Old graphs will be overwritten.')
os.makedirs(GRAPH_DIR)

#Consts
trainIter = 100


def train(dataset):
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

    gen = Generator(trackA, trackB, segs)

    left_output = singleNet(left, reuse=False)
    right_output = singleNet(right, reuse=True)

    loss = contrastiveLoss(left_output, right_output, label, margin)

    global_step = tf.Variable(0, trainable=False)

    #Maybe a lower momentum?
    train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("a")
            _, lossVal = sess.run([train_step, loss], feed_dict={left:bLeft, right:bRight, label:bSim})
            recordLoss(lossVal)
            # writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, lossVal) ## I think this string should be what I

def recordLoss(loss):
    global lossCounter
    lossCounter += 1
    losses.append(loss)
    if lossCounter % EPOCHS_PER_PLOT == 0:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.savefig(os.path.join(GRAPH_DIR, str(lossCounter) + 'EPOCHLOSS'))
