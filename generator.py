from readwav import importWav, soundToNumpy
from pydub import AudioSegment
from pydub.utils import make_chunks
import random
import numpy as np

class Generator:

    def __init__(self, ta, tb, segs):
        self.trackA = importWav(ta)
        self.trackB = importWav(tb)
        self.aseg = segs[0]
        self.bseg = segs[1]
        self.batchSize = 5
        self.asegTrain = [x for x in segs[0] if x[2] == "clean"]
        self.bsegTrain = [x for x in segs[1] if x[2] == "clean"]
        self.buildTrainSet()

    def buildTrainSet(self):
        combinedA = AudioSegment.empty()
        combinedB = AudioSegment.empty()
        for set in self.asegTrain:
            combinedA += self.trackA[set[0]:set[1]]
        self.combSetsA = make_chunks(combinedA, 1000)[:-1]
        for set in self.bsegTrain:
            combinedB += self.trackB[set[0]:set[1]]
        self.combSetsB = make_chunks(combinedB, 1000)[:-1]
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