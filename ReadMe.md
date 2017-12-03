# Diarization Neural Network Final

The purpose of this assignment is to attempt to perform audio diarization
automatically in order to assist researchers in reducing overall
costs and speed up data collection.

### Code Overview

#### prepAudio.py
This file will do most prep work for the neural net. It will split all the audio up into individual tracks (trackA, trackB).
It will then pick out segments with audio, and compair with it's companion track to determin if the audio is clean or dirty,
clean if it does not conflict with the other track or dirty if it is. This analysis is cached in order to speed
up future runs, but initial run could take hours. The result of this is a list of dictionaries. Each dictionary
element has {'original':"string", 'trackA':"string", 'trackB':"string", 'aClassification':[stMilisecond, endMilisecond, 
"classification"], 'bClassification':[stMilisecond, endMilisecond, "classification"]}

#### Repo Rules
1. No Data should be commited!!!! (keep in local machine to help I have created
 a directory that ignores all .wav files)
 
#### To Run
You need to install FFMPEG, google. You will also need to pip install pydub. Some of the prep does require a little bit 
of a specific structure in the data folder, so there is a buildDatafolder.py script to help build this (note it is cross
platformish) this should only need to be run once. Put the audio in the data/rawAudio and csv in data/csvFinal and the 
code will do the rest. After the first run, prepAudio will pull from it's cached files in order to reduce the time of
loading preped data to a few minutes, but note this is cpu/ram intensive.