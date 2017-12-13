from os.path import join
from prepAudio import prepData
from readwav import importWav
import tensorflow as tf
from generator import Generator
from train import train
import numpy as np


#Global Strings
audioFiles = join("data", "rawAudio")
splitFiles = join("data", "splitAudio")

if __name__ == "__main__":
    dataSets = prepData(audioFiles, splitFiles)
    for set in dataSets:
        train(set)
