import tensorflow as tf
import numpy as np
from singleNetwork import singleNet, contrastiveLoss, contrastiveLossTest
from generator import Generator
import os
import datetime
import matplotlib.pyplot as plt

#Consts
trainIter = 10000

lossCounter = 0
EPOCHS_PER_PLOT = 20
losses = []

GRAPH_DIR = ('./graphs/' + datetime.datetime.now().isoformat(timespec='seconds')).replace(':','-')
if os.path.exists(GRAPH_DIR):
    print('Graph directory for this run already exists. Old graphs will be overwritten.')
os.makedirs(GRAPH_DIR)


def train(dataset):
    global lossCounter, losses
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
    threashold = 0.015

    gen = Generator(trackA, trackB, segs)

    left_output = singleNet(left, reuse=False)
    right_output = singleNet(right, reuse=True)

    loss = contrastiveLoss(left_output, right_output, label, margin)

    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(0.004, 0.099, use_nesterov=True).minimize(loss, global_step=global_step)

    train_res = contrastiveLossTest(left_output, right_output, margin )

    train = gen.PredictSet

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        lossCounter = 0
        # Train A seg (track 1)
        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("a")
            _, l = sess.run([train_step, loss], feed_dict={left:bLeft, right:bRight, label:bSim})
            # writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l) ## I think this string should be what we encode
            recordLoss(l, i, 'A')
        # Test A seg (track 1)
        testDat = train[0]
        resSets = []
        for seg in testDat:
            lTest = np.array([seg[0].reshape((8,32,1))])
            rTest = np.array([seg[1].reshape((8,32,1))])
            l = sess.run([train_res], feed_dict={left:lTest, right:rTest})[0]
            l = 0 if l > threashold else 1
            resSets.append([seg[2], l])
            print("Predicted", seg[2], ": ", l)
        predictedClean = gen.asegTrain + filterCleanTimes(resSets)
        writeResToFile(dataset["trackA"][:-4] + "Res.csv", predictedClean)


    print("Break Between parts A and B")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        lossCounter = 0
        # Train B seg (track2)
        for i in range(trainIter):
            bLeft, bRight, bSim = gen.getNextBatch("b")
            _, l = sess.run([train_step, loss], feed_dict={left:bLeft, right:bRight, label:bSim})
            # writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l) ## I think this string should be what we encode
            recordLoss(l, i, 'B')
        # TestB seg (track2)
        testDat = train[1]
        resSets = []
        for seg in testDat:
            lTest = np.array([seg[0].reshape((8, 32, 1))])
            rTest = np.array([seg[1].reshape((8, 32, 1))])
            l = sess.run([train_res], feed_dict={left: lTest, right: rTest})[0]
            l = 0 if l > threashold else 1
            resSets.append([seg[2], l])
            print("Predicted", seg[2], ": ", l)
        predictedClean = gen.asegTrain + filterCleanTimes(resSets)
        writeResToFile(dataset["trackB"][:-4] + "Res.csv", predictedClean)

def recordLoss(loss, iter, segment):
    losses.append(loss)
    if (iter+1) % EPOCHS_PER_PLOT == 0:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.savefig(os.path.join(GRAPH_DIR, segment + str(iter+1) +  'EPOCHLOSS'))


def filterCleanTimes(resSet):
    res = []
    current = None
    for i in range(len(resSet)):
        if not current:
            current = resSet[i]
        if (i+1) < len(resSet) and current[1] == 1:
            if current[0][1] == resSet[(i+1)][0][0] and resSet[(i+1)][1] == 1:
                current[0][1] = resSet[(i+1)][0][0]
            else:
                res.append([current[0][0], current[0][1], "clean"])
                current = None
        else:
            if current[1] == 1:
                res.append([current[0][0], current[0][1], "clean"])
            current = None
    return res

def writeResToFile(outputFile, cleanSegs):
    with open(outputFile, "w") as file:
        for seg in cleanSegs:
            file.write(str(seg)[1:-1].replace("'","") + "\n")