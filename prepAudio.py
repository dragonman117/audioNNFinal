from os import listdir, makedirs
from os.path import join, exists
from pydub import AudioSegment, silence, effects

def audioFileGenerator(rootDir):
    files = sorted([f for f in listdir(rootDir)])
    for f in files:
        yield join(rootDir, f)

def splitTracks(file, stepDir):
    dir = join(stepDir, file[14:-4])
    full = AudioSegment.from_file(file)
    trackA = join(dir, "trackA.wav")
    trackB = join(dir, "trackB.wav")
    if not exists(dir):
        makedirs(dir)
        seperated = full.split_to_mono()
        effects.normalize(seperated[0]).export(trackA, format="wav")
        effects.normalize(seperated[1]).export(trackB, format="wav")
    return {"original": file, "trackA": trackA, "trackB":trackB}

def determineSilences(file):
    trackA = AudioSegment.from_file(file["trackA"])
    speakingA = silence.detect_nonsilent(trackA, min_silence_len=500, silence_thresh=-30)
    trackB = AudioSegment.from_file(file["trackB"])
    speakingB = silence.detect_nonsilent(trackB, min_silence_len=500, silence_thresh=-30)
    aClass = [[set[0],set[1], hasConflicts(set, speakingB)] for set in speakingA]
    bClass = [[set[0], set[1], hasConflicts(set, speakingA)] for set in speakingB]
    return aClass, bClass

def hasConflicts(timeRange, track):
    for set in track:
        if set[0] < timeRange[0] and set[1] > timeRange[0]:
            return "dirty"
        elif set[0] < timeRange[1] and set[1] > timeRange[1]:
            return "dirty"
    return "clean"

def writeClassification(filename, track):
    with open(filename, 'w') as file:
        for set in track:
            file.write(str(set)[1:-1] + "\n")

def readClassificationFile(filename):
    with open(filename, 'r') as file:
        res = [[int(x[0]), int(x[1]), x[2]] for x in [line.replace("'", "").replace(" ", "").strip().split(",") for line in file]]
        return res

def loadAudioClassifications(fileSet):
    res = []
    for file in fileSet:
        if not exists(file['trackA'][:-4]+".txt"):
            print("ran")
            trackA, trackB = determineSilences(file)
            writeClassification(file['trackA'][:-4]+".txt", trackA)
            writeClassification(file['trackB'][:-4] + ".txt", trackB)
            file["aClassification"] = trackA
            file["bClassification"] = trackB
            res.append(file)
        else:
            file["aClassification"] = readClassificationFile(file['trackA'][:-4]+".txt")
            file["bClassification"] = readClassificationFile(file['trackB'][:-4]+".txt")
            res.append(file)
    return res

def prepData(srcDir, stepDir):
    files = []
    for file in audioFileGenerator(srcDir):
        files.append(splitTracks(file, stepDir))
    files = loadAudioClassifications(files)
    return files