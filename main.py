from os.path import join
from prepAudio import prepData

#Global Strings
audioFiles = join("data", "rawAudio")
splitFiles = join("data", "splitAudio")

if __name__ == "__main__":
    prepData(audioFiles, splitFiles)