from os.path import join, exists
from os import makedirs

#Data should exist in the same directory as this file
# - This will build required files for the prep, seperation, and classification
# - This is os Agnostic


if __name__ == "__main__":
    splitAudio = join("data", "splitAudio")
    csv = join("data", "csvFinal")
    audio = join("data", "rawAudio")
    if not exists(splitAudio):
        makedirs(splitAudio)
    if not exists(csv):
        makedirs(csv)
    if not exists(audio):
        makedirs(audio)