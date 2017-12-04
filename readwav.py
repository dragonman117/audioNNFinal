# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:16:02 2017

@author: Nirjhar
"""

from pydub import AudioSegment
import numpy as np

# fp = wave.open('Convo Sample.wav')
# nchan = fp.getnchannels()
# N = fp.getnframes()
# dstr = fp.readframes(N*nchan)
# data = numpy.fromstring(dstr, numpy.int16)
# data = numpy.array(numpy.reshape(data, (-1,nchan)))

# https://stackoverflow.com/questions/41109652/how-to-read-ogg-or-mp3-audio-files-in-a-tensorflow-graph

def importWav(filename, segment=None):
    sound = AudioSegment.from_file(filename)
    if segment != None:
        sound = sound[segment[0]:segment[1]]
    return np.array(sound.get_array_of_samples())