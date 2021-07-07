import scipy.io
import numpy as np
import scipy
import scipy.signal
from wettbewerb import *
from python_speech_features import logfbank
from scipy.ndimage import convolve
from scipy import signal
import pandas as pd
import cv2

class PreprocessingData:

    def __init__(self, fs):
        self.fs = fs

    def trim_data(self, x, length=7.5):
        '''
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        length: float
            length in seconds
        Returns
        -------
        x : List of numpy arrays
            Trimmed data to minimum length of signal data list
        '''
        # min_length = min(x, key=lambda k: k.shape[0]).shape[0]
        min_length = int(length * self.fs)
        for i in range(len(x)):
            x[i] = x[i][0:min_length]
        return x

    def cut_data(self, x, length=2):
        '''
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        length: float
            length in seconds
        Returns
        -------
        x : List of numpy arrays
            Trimmed data to minimum length of signal data list
        '''
        # min_length = min(x, key=lambda k: k.shape[0]).shape[0]
        start_length = int(length * self.fs)
        for i in range(len(x)):
            x[i] = x[i][start_length:]
        return x

    def bandpass_filter(self):
        '''
        Parameters
        ----------

        Returns
        -------
        b, a : float
            Parameters for bandpass filter in scipy package
        '''
        fs = 300 # sampling rate (in Hz)
        freq_pass = np.array([4.0, 35.0]) / (fs / 2.0)
        freq_stop = np.array([0.5, 50.0]) / (fs / 2.0)
        gain_pass = 1
        gain_stop = 20
        filt_order, cut_freq = signal.buttord(freq_pass, freq_stop, gain_pass, gain_stop)
        b, a = signal.butter(filt_order, cut_freq, 'bandpass')
        return b, a

    def zero_padding(self, x, length=60, fs=300):
        '''
        Parameters
        ----------
        x : List of numpy arrays
            List consists of signals, loaded from path
        Returns
        -------
        x : List of numpy arrays
            List of zero-padded arrays for same resolution in fft e.g.
        '''

        for i in range(len(x)):
            if (x[i].shape[0] < length):
                x[i] = np.pad(x[i], (0, int(length*fs) - x[i].shape[0]), 'constant')
            else:
                x[i] = x[i][0:int(length*fs)]
        print(f"Zero Padding erfolgreich abgeschlossen. Die maximale Länge beträgt: {length}")
        return x

    def running_mean(self, x_single, k):
        '''
         Parameters
         ----------
         x_single : Numpy array
            Single data
         k : int
            Factor for the running mean, smoothing factor
         Returns
         -------
         array : numpy array
            Filtered array
         '''

        cumsum = np.cumsum(np.insert(x_single, 0, 0))
        return (cumsum[k:] - cumsum[:-k]) / float(k)


    def denoise_data(self, x):
        '''
         Parameters
         ----------
         x : List of numpy arrays
            List consists of signals, loaded from path
         Returns
         -------
         x : List of numpy arrays
            Denoised signal normalized
         '''

        x_filt = list()
        for i in range(len(x)):
            b, a = self.bandpass_filter()
            x_n = scipy.signal.filtfilt(b, a, x[i], axis=0)
            x_n = x_n / ((max(x_n)-min(x_n))/2)
            x_filt.append(x_n)
        return x_filt

    def _logMelFilterbank(self, signal, parse_param=(0.001, 0.0017, 15)):
        """
        Compute the log Mel-Filterbanks
        Returns a numpy array of shape (600, nfilt) = (600,15)
        """
        wave = signal
        fbank = logfbank(
            wave,
            samplerate=len(wave),
            winlen=float(parse_param[0]),
            winstep=float(parse_param[1]),
            nfilt=int(parse_param[2]),
            nfft=1024
        )
        fbank = cv2.resize(fbank, (15, 600), interpolation=cv2.INTER_CUBIC)
        return fbank