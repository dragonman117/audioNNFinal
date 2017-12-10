from numpy import fft
from readwav import importWav
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import block_reduce
import pickle

SAMPLE_RATE = 44100

def getSpectrum(track, outHz, nOutFreqs, display=False):
    duration = len(track)/SAMPLE_RATE
    reduce = int((SAMPLE_RATE)/outHz)
    f, t, spectrum = signal.spectrogram(track, SAMPLE_RATE, nperseg=reduce, noverlap=0)
    spectrum = np.log10(spectrum)

    freduce = int(len(f)/nOutFreqs)
    spectrum = block_reduce(spectrum, block_size=(freduce, 1), func=np.max)
    spectrum = spectrum[0:nOutFreqs]
    f = f[0::freduce]
    f = f[0:nOutFreqs]

    if display:
        plt.pcolormesh(t, f, spectrum)
        plt.show()

    return spectrum


if __name__ == '__main__':
    wav = importWav('data/splitAudio/HS_D01/trackA.wav')
    spec = getSpectrum(wav, 10, 32, display=True)

