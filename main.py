from os.path import join
from prepAudio import prepData
from readwav import importWav
import tensorflow as tf


#Global Strings
audioFiles = join("data", "rawAudio")
splitFiles = join("data", "splitAudio")

if __name__ == "__main__":
    dataSets = prepData(audioFiles, splitFiles)
    # print(dataSets[0])
    # segTime = dataSets[0]["aClassification"][0]
    # print(segTime)
    # audio = importWav(dataSets[0]["trackA"], segTime)
    # print(audio.shape)