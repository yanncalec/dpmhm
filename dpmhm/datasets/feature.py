import numpy as np
import librosa

## Feature extractor
def spectrogram(x:np.ndarray, sampling_rate:int, *, time_window:float, hop_step:float, n_fft:int=None, to_db:bool=True, normalize:bool=False, **kwargs):
    """Compute the power spectrogram of a multichannel waveform.

    Args
    ----
    x:
        input array with the last dimension being time
    sampling_rate:
        original sampling rate (in Hz) of the input
    time_window:
        size of short-time Fourier transform (STFT) window, in second
    hop_step:
        time step of the sliding STFT window, in second
    n_fft:
        size of FFT, by default automatically determined
    to_db:
        power spectrogram in db

    Returns
    -------
    Sxx:
        power spectrogram
    (win_length, hop_length, n_fft):
        real values used by librosa for feature extraction
    (Ts, Fs):
        time and frequency steps of `Sxx`
    """
    sr = float(sampling_rate)

    win_length = int(time_window * sr)
    # hop_length = win_length // 2
    hop_length = int(hop_step * sr)

    if n_fft is None:
        n_fft = 2**(int(np.ceil(np.log2(win_length))))
        # n_fft = 2**(int(np.floor(np.log2(win_length))))

      # Compute the spectrogram
    if normalize:
        y = (x-x.mean())/x.std()
    else:
        y = x

    S = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, **kwargs)
    Sxx = librosa.power_to_db(np.abs(S)**2) if to_db else np.abs(S)**2

    Ts = np.arange(S.shape[-1]) * hop_length / sr  # last dimension is time
    Fs = np.arange(S.shape[-2])/S.shape[-2] * sr//2  # before last dimension is frequency
    # elif method == 'scipy':
    #   Fs, Ts, Sxx = scipy.signal.spectrogram(x, sr, nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft, **kwargs)
    #   Sxx = librosa.power_to_db(Sxx) if to_db else Sxx

    return Sxx, (win_length, hop_length, n_fft), (Ts, Fs)


def melspectrogram(x, sampling_rate:int, *, time_window:float, hop_step:float, n_fft:int=None, n_mels:int=128, normalize:bool=False, **kwargs):
    """Compute the mel-spectrogram of a multichannel waveform.

    Args
    ----
    x: input
        nd array with the last dimension being time
    sampling_rate: int
        original sampling rate (in Hz) of the input
    time_window: float
        size of short-time Fourier transform (STFT) window, in second
    hop_step: float
        time step of the sliding STFT window, in second
    n_fft: int
        size of FFT, by default automatically determined

    Returns
    -------
    Sxx:
        mel-spectrogram
    (win_length, hop_length, n_fft):
        real values used by librosa for feature extraction
    """
    sr = float(sampling_rate)

    win_length = int(time_window * sr)
    # hop_length = win_length // 2
    hop_length = int(hop_step * sr)

    if n_fft is None:
        n_fft = 2**(int(np.ceil(np.log2(win_length))))

    # Compute the melspectrogram
    if normalize:
        y = (x-x.mean())/x.std()
    else:
        y = x

    Sxx = librosa.feature.melspectrogram(y=y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, **kwargs)

    return Sxx, (win_length, hop_length, n_fft)


# Default feature extractors
_EXTRACTOR_SPEC = lambda x, sr: spectrogram(x, sr, time_window=0.025, hop_step=0.01, to_db=True)[0]

_EXTRACTOR_MEL = lambda x, sr: melspectrogram(x, sr, time_window=0.025, hop_step=0.01, n_mels=64)[0]


__all__ = ['spectrogram', 'melspectrogram', '_EXTRACTOR_SPEC', '_EXTRACTOR_MEL']