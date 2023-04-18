import numpy as np
import librosa
import pywt

## Feature extractor

def spectral_features(x:np.ndarray, sampling_rate:int, feature_method:str, *, time_window:float, hop_step:float, n_fft:int=None, normalize:bool=False, feature_kwargs:dict={}, **kwargs):
    """Compute the spectral features of a multichannel waveform using the library `librosa`.

    Args
    ----
    x:
        input array with the last dimension being time
    sampling_rate:
        original sampling rate (in Hz) of the input
    feature_method:
        type of feature, one of {'spectrogram', 'melspectrogram', 'mfcc'}
    time_window:
        size of short-time Fourier transform (STFT) window, in second
    hop_step:
        time step of the sliding STFT window, in second
    n_fft:
        size of FFT, by default automatically determined
    normalize:
        whether to normalize the input array
    feature_kwargs:
        keyword arguments for the method defined by `feature_method`, see: https://librosa.org/doc/main/feature.html.
    kwargs:
        additional keyword arguments: {'to_db': boolean, compute power spectrogram in decibel}

    Returns
    -------
    Sxx:
        power spectrogram
    (win_length, hop_length, n_fft):
        real values used by librosa for feature extraction
    """
    sr = float(sampling_rate)

    win_length = int(time_window * sr)
    # hop_length = win_length // 2
    hop_length = int(hop_step * sr)

    if n_fft is None:
        n_fft = 2**(int(np.ceil(np.log2(win_length))))
        # n_fft = 2**(int(np.floor(np.log2(win_length))))

    if normalize:
        y = (x-x.mean())/x.std()
    else:
        y = x

    # Compute the spectral feature
    if feature_method == 'spectrogram':
        S = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, **feature_kwargs)

        try:
            to_db = kwargs.pop('to_db')
        except:
            to_db = False
        Sxx = librosa.power_to_db(np.abs(S)**2) if to_db else np.abs(S)**2

        # time and frequency steps
        Ts = np.arange(S.shape[-1]) * hop_length / sr  # last dimension is time
        Fs = np.arange(S.shape[-2])/S.shape[-2] * sr//2  # before last dimension is frequency

        # Fs, Ts, Sxx = scipy.signal.spectrogram(x, sr, nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft, **kwargs)
        # Sxx = librosa.power_to_db(Sxx) if to_db else Sxx

        # return Sxx, (win_length, hop_length, n_fft), (Ts, Fs)
    elif feature_method == 'melspectrogram':
        # Tensorflow will complain if `sampling_rate` (int) in place of `sr` is used.
        Sxx = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, **feature_kwargs)
    elif feature_method == 'mfcc':
        Sxx = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, **feature_kwargs)
    else:
        raise NameError(f'Unknown feature transform method: {feature_method}')

    return Sxx, (win_length, hop_length, n_fft)


def wavelet_packet_transform(x:np.ndarray, sampling_rate:int, *, level:int=3, wavelet:str='db2'):
    """Compute the Wavelet Packet Transform (WPT).

    Args
    ----
    level:
        Number of decomposition levels
    """
    fs = float(sampling_rate)

    # Perform the wavelet packet transform
    # band_range = [(None, None), (20, 80), (80, 200)] # Frequency range of each sub-band
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=level)
    coeffs = []

    for node in wp.get_level(level, 'freq'):
        coeffs.append(node.data)
        # freq_min, freq_max = band_range[node.level-1]
        # if freq_min is None:
        #     freq_min = 0
        # if freq_max is None:
        #     freq_max = np.inf
        # path = node.path
        # freq = fs / (2 ** (path.count('d')-1))  #
        # if freq >= freq_min and freq <= freq_max:
        #     coeffs.append(node.data)

    return np.asarray(coeffs)


def emd():
    """Empirical Mode Decomposition
    """
    pass



# # Default feature extractors
# _EXTRACTOR_SPEC = lambda x, sr: spectrogram(x, sr, time_window=0.025, hop_step=0.01, to_db=True)[0]

# _EXTRACTOR_MEL = lambda x, sr: melspectrogram(x, sr, time_window=0.025, hop_step=0.01, n_mels=64)[0]


__all__ = ['spectral_features'] # , '_EXTRACTOR_SPEC', '_EXTRACTOR_MEL']