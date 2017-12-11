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

    def makeChunks(self, iterable, chunkSize):
        for i in range(0, len(iterable), chunkSize):
            yield(iterable[i : i + chunkSize])

    def sampleFromMs(self, ms):
        sec = ms / 1000
        return int(np.floor(sec * self.sampleRate + .5))

    def buildTrainSet(self):
        #Paste the clean tracks together for training
        combinedA = np.zeros((1, self.freqs))
        combinedB = np.zeros((1, self.freqs))

        for segtimes in self.asegTrain: # these are start time/ end time in ms.
            combinedA = np.append(combinedA, self.trackA[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)
        combinedA = combinedA[1:-1]#remove zeros on first entry

        for segtimes in self.bsegTrain: # these are start time/ end time in ms.
            combinedB = np.append(combinedB, self.trackB[self.sampleFromMs(segtimes[0]):self.sampleFromMs(segtimes[1])], axis=0)
        combinedB = combinedB[1:-1]#remove zeros on first entry

        #After this, you have a list of n x 32 timechunks to push through the network.
        #At a sample rate of 32 and a chunk length of .5, chunks are 8 x 32
        self.combSetsA = list(self.makeChunks(combinedA, int(self.sampleRate * self.chunkLength)))
        self.combSetsB = list(self.makeChunks(combinedB, int(self.sampleRate * self.chunkLength)))



        negSeg = self.findNegSegments(self.aseg)
        aChunk = AudioSegment.empty()
        for seg in negSeg:
            aChunk += self.trackA[seg[0]:seg[1]]

        self.aNeg = make_chunks(aChunk, 1000)
        negSeg = self.findNegSegments(self.bseg)
        bChunk = AudioSegment.empty()
        for seg in negSeg:
            bChunk += self.trackB[seg[0]:seg[1]]
        self.bNeg = make_chunks(bChunk, 1000)

        #Asymmetry between combSetsA and combSetsB. Is this on purpose?
        self.aTrainFin = [[x,x,1] for x in self.combSetsA] + [[x,y,0] for x,y in zip(self.combSetsA, self.aNeg)]
        self.bTrainFin = [[x,x,1] for x in self.combSetsB] + [[x,y,0] for x,y in zip(self.combSetsB, self.bNeg)]

    def findNegSegments(self, knownSeg):
        start = 0
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
            lhs = [np.array(x[0].get_array_of_samples()).reshape(44100,1) for x in raw]
            rhs = [np.array(x[1].get_array_of_samples()).reshape(44100,1) for x in raw]
            sim = np.array([[x[2]] for x in raw])
            return lhs, rhs, sim
        else:
            set = random.sample(range(0, len(self.bTrainFin)), self.batchSize)
            raw = [self.aTrainFin[x] for x in set]
            lhs = [np.array(x[0].get_array_of_samples()).reshape(44100, 1) for x in raw]
            rhs = [np.array(x[1].get_array_of_samples()).reshape(44100, 1) for x in raw]
            sim = np.array([[x[2]] for x in raw])
            return lhs, rhs, sim