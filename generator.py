from readwav import importWav, soundToNumpy
from pydub import AudioSegment
from pydub.utils import make_chunks
import random
import numpy as np
from importSpec import importSpec
import matplotlib.pyplot as plt

#TODO: Remake make chunks

TRAINING_BATCH_SIZE = 5
SAMPLE_RATE = 16
FREQS = 32
CHUNK_LENGTH = 0.5 #in seconds

#TODO: NegSeg section

class Generator:

    def __init__(self, ta, tb, segs):
        self.batchSize = TRAINING_BATCH_SIZE
        self.sampleRate = SAMPLE_RATE
        self.freqs = FREQS
        self.chunkLength = CHUNK_LENGTH

        #open spetral data
        self.trackA = importSpec(ta).T
        self.trackB = importSpec(tb).T

        self.aseg = segs[0]
        self.bseg = segs[1]

        self.asegTrain = [x for x in segs[0] if x[2] == "clean"]
        self.bsegTrain = [x for x in segs[1] if x[2] == "clean"]

        self.buildTrainSet()

    def makeChunks(self, iterable):
        size =  int(self.sampleRate * self.chunkLength)
        for i in range(0, len(iterable), size):
            yield(iterable[i : i + size])


    def sampleFromMs(self, ms):
        sec = ms / 1000
        return int(np.floor(sec * self.sampleRate + .5))

    def buildTrainSet(self):
        ##Find the positive training points first

        #Initialize with zeros to allow np.append. These will be removed later.
        aPosSpecComb = np.zeros((1, self.freqs))
        bPosSpecComb = np.zeros((1, self.freqs))

        for segtimes in self.asegTrain: # segtimes are tuples of (start_time, end_time) in ms.
            aPosSpecComb = np.append(aPosSpecComb, self.trackA[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)
        aPosSpecComb = aPosSpecComb[1:-1]#remove zeros on first entry

        for segtimes in self.bsegTrain: # segtimes are tuples of (start_time, end_time) in ms.
            bPosSpecComb = np.append(bPosSpecComb, self.trackB[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)
        bPosSpecComb = bPosSpecComb[1:-1]#remove zeros on first entry

        #After this, you have a list of n x 32 timechunks to push through the network.
        #At a sample rate of 32 and a chunk length of .5, chunks are 8 x 32
        self.combSetsA = list(self.makeChunks(aPosSpecComb))
        self.combSetsB = list(self.makeChunks(bPosSpecComb))

        ##Find Negative segments and store them in self.aNeg/self.bNeg
        # Initialize with zeros to allow np.append. These will be removed later.
        aNegSegs = self.findNegSegments(self.aseg)
        bNegSegs = self.findNegSegments(self.bseg)

        aNegSpecComb = np.zeros((1, self.freqs))
        bNegSpecComb = np.zeros((1, self.freqs))

        for segtimes in aNegSegs: # segtimes are tuples of (start_time, end_time) in ms.
            aNegSpecComb = np.append(aNegSpecComb, self.trackA[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)

        for segtimes in bNegSegs: # segtimes are tuples of (start_time, end_time) in ms.
            bNegSpecComb = np.append(bNegSpecComb, self.trackB[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)

        self.aNeg = list(self.makeChunks(aNegSpecComb[1:-1]))#remove zeros on first entry, chunkify, cast to list
        self.bNeg = list(self.makeChunks(bNegSpecComb[1:-1]))#remove zeros on first entry, chunkify, cast to list

        # After this, you have a list of n x 32 timechunks to push through the network.
        # At a sample rate of 32 and a chunk length of .5, chunks are 8 x 32

        #zip everything together in the proper format.
        self.aTrainFin = [[x,x,1] for x in self.combSetsA] + [[x,y,0] for x,y in zip(self.combSetsA, self.aNeg)]
        self.bTrainFin = [[x,x,1] for x in self.combSetsB] + [[x,y,0] for x,y in zip(self.combSetsB, self.bNeg)]
        self.aTrainFin = [x for x in self.aTrainFin if (np.array(x[0]).shape == (8, 32)) and ((np.array(x[1]).shape == (8, 32)))]
        self.bTrainFin = [x for x in self.bTrainFin if (np.array(x[0]).shape == (8, 32)) and ((np.array(x[1]).shape == (8, 32)))]

    def findNegSegments(self, knownSeg):
        start = 0
        #Hijack the shape of self.aseg. Data will not be used asymetrically.
        segs = self.aseg
        if (knownSeg[0][0] == 0):
            start = knownSeg[0][1] + 1
            segs = knownSeg[1:]
        silence = []
        for section in segs:
            silence.append([start, section[0] - 1])
            start = section[1] - 1
        return silence

    def getNextBatch(self, track):
        if track == "a":
            set = random.sample(range(0, len(self.aTrainFin)), self.batchSize)
            raw = [self.aTrainFin[x] for x in set]
            lhs = [np.array(x[0]).reshape((8,32,1)) for x in raw]
            rhs = [np.array(x[1]).reshape((8,32,1)) for x in raw]
            sim = np.array([[x[2]] for x in raw])
            return np.array(lhs), np.array(rhs), sim
        else:
            set = random.sample(range(0, len(self.aTrainFin)), self.batchSize)
            raw = [self.aTrainFin[x] for x in set]
            lhs = [np.array(x[0]).reshape((8, 32, 1)) for x in raw]
            rhs = [np.array(x[1]).reshape((8, 32, 1)) for x in raw]
            sim = np.array([[x[2]] for x in raw])
            return np.array(lhs), np.array(rhs), sim