# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:16:02 2017

@author: Nirjhar
"""

import wave
import numpy

fp = wave.open('Convo Sample.wav')
nchan = fp.getnchannels()
N = fp.getnframes()
dstr = fp.readframes(N*nchan)
data = numpy.fromstring(dstr, numpy.int16)
data = numpy.array(numpy.reshape(data, (-1,nchan)))