<!-- # Feature Extraction &  -->
# Signal Processing Techniques for Vibration Analysis

Over the past few decades, many signal processing techniques have been developed and successfully applied to vibration analysis. Typically, these techniques transform the signal from a raw waveform into a feature space revealing characteristic information of the underlying mechanical system. The output of these transformations can be used as input features to an end-to-end neural network for more advanced representation learning. Some popular techniques widely adopted in vibration analysis include:

- Short Time Fourier Transform (STFT) based: spectrogram / cepstrogram analysis (time-varying spectrum and cepstrum), [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), Gabor transform etc.
- Wavelet Transform (WT) based: Wavelet Packet Transform (WPT), Continuous WT etc.
- [Cyclostationarity](https://en.wikipedia.org/wiki/Cyclostationary_process): cyclic autocorrelation function and cyclic spectrum
- Spectral Kurtosis analysis: spectral Kurtogram
- [Empirical Mode Decomposition](https://en.wikipedia.org/wiki/Hilbert-Huang_transform)


<!-- - Time-frequency analysis: Short Time Fourier Transform (STFT) based, spectrogram / cepstrogram analysis (time-varying spectrum and cepstrum), [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), etc.
- Time-scale analysis: Wavelet Transform (WT) and Wavelet Packet Transform (WPT) based, for analyzing non-stationary / transient signals.
- [Cyclostationarity](https://en.wikipedia.org/wiki/Cyclostationary_process),
- Spectral Kurtosis analysis
- Empirical Mode Decomposition (EMD)
- Envelope analysis
- Linear Predictive Coding (LPC)
- Wiener-Ville Distribution -->

## Time-frequency analysis
Time-frequency analysis allows to reveal a signal's frequency content as it changes over time. Many common time-frequency analysis techniques are based on the Short-Time Fourier Transform (STFT), which involves dividing a signal into short segments and performing a Fourier Transform on each segment. The resulting spectrogram displays the frequency content of the signal at each point in time, allowing for the identification of time-varying frequencies and trends.

### Cepstrum and Mel-frequency scale (refs)
Cepstrum and Mel-frequency cepstral coefficients (MFCC) are two widely used signal processing techniques for analyzing audio and speech signals. In the context of vibration analysis, cepstrum and MFCC are used for various applications such as bearing fault detection, gear fault diagnosis, and condition monitoring of rotating machinery (refs?).

Cepstrum analysis is primarily used in the field of audio signal processing, for tasks such as speech recognition and speaker identification. It is also used in vibration analysis to identify the natural frequencies and modes of mechanical systems. It involves taking the inverse Fourier transform of the logarithm of the magnitude spectrum of a signal, resulting in a signal that contains both the spectral and temporal information of the original signal. The resulting signal is known as the cepstrum, and it can be used to analyze the pitch, harmonics, and other spectral features of the original signal.

MFCC, on the other hand, is a technique used to represent the spectral envelope of an audio signal in a compact form. It involves dividing the frequency spectrum of a signal into a set of frequency bands spaced according to the Mel-scale, which is a non-linear scale that more closely approximates the human perception of pitch. The MFCCs are then computed by taking the discrete cosine transform (DCT) of the log-magnitude spectrum of each frequency band. The resulting coefficients represent the power spectrum of the signal in each frequency band and are used as features for various audio signal processing applications such as speech recognition, music genre classification, and emotion detection.

## Time-scale analysis with wavelets (TODO)
Wavelet Packet Transform (WPT) is a signal processing technique that decomposes a signal into sub-bands using wavelets. Unlike the Discrete Wavelet Transform (DWT), which only provides a dyadic decomposition, WPT allows for more flexible and refined decomposition of the signal.

WPT works by recursively dividing a signal into two sub-bands, one for low frequencies and the other for high frequencies. This is done by applying a wavelet filter to the signal and then downsampling it. The process is then repeated on each of the resulting sub-bands, creating a binary tree structure that is referred to as a wavelet packet tree.

The advantage of WPT over other signal processing techniques is that it provides a high time-frequency resolution, allowing for the analysis of both stationary and non-stationary signals. This makes it particularly useful in the analysis of vibration signals, which often exhibit non-stationary behavior.

Applications of WPT in vibration analysis include fault detection and diagnosis in machinery, as well as the identification of signal features related to specific fault types. WPT has been used in a wide range of applications, from gear fault detection in wind turbines to bearing fault detection in rotating machinery.

In addition to its application in vibration analysis, WPT has also been used in other fields such as image and speech processing, and has shown promising results in the analysis of biomedical signals.


## Cyclostationary signal processing
Vibrational signals produced by rotating machines are remarkably well modelled as [cyclostationary process](https://en.wikipedia.org/wiki/Cyclostationary_process). A stochastic process $x(t)$ is said to be cyclostationary if both the mean $\Exp\bracket{x(t)}$ and the autocorrelation function

\begin{align}
\label{eq:af}
R_x(t,\tau) := \Exp\bracket{x(t)^* x(t+\tau)}
\end{align}

are periodic in $t$ for some period $T_0$, where $x(t)^*$ denotes the complex conjugate.

### Cyclic autocorrelation function & spectrum

Cyclic autocorrelation function (CAF) is the coefficients $\hat R_x^n(\tau)$ of the Fourier series

$$
R_x(t,\tau)=\sum_{n\in\Z} \hat R_x^n(\tau) e^{2\pi i n t/T_0}
$$
<!-- defined as
$$
\hat R_x^{n}(\tau) := \integral{-T_0/2}{T_0/2}{R_{x}(t,\tau) e^{-2\pi i n t/T_0}}{t}
$$ -->

and $n$ is called cyclic frequency. For a cycloergodic process it can be computed by

$$
\hat R_x^{n}(\tau) := \lim_{T\to\infty} \integral{-T/2}{T/2}{x(t)^* x(t+\tau) e^{-2\pi i n t/T_0}}{t}
$$

Taking Fourier transform of CAF gives the cyclic spectrum (or the spectral correlation function):

\begin{align}
\label{eq:SCF}
S^n_x(\omega) := \integralR{\hat R^n_x(\tau) e^{-2\pi i \omega\tau}}{\tau}
\end{align}

which can be used as input feature to a ML model.

## Spectral Kurtosis analysis (TODO)
Spectral kurtosis is a technique used in vibration analysis to highlight frequency bands that contain impulsive noise or transient events. It is based on the kurtosis statistic, which measures the "peakedness" of a distribution.

In spectral kurtosis analysis, the kurtosis value is computed for each frequency bin in the frequency spectrum of a vibration signal. Regions of the spectrum with high kurtosis values are indicative of impulsive noise or transient events. Spectral kurtosis can be used to detect and diagnose faults in rotating machinery, such as bearing defects and gear faults.

A spectral kurtogram is a time-frequency representation of the spectral kurtosis values of a signal. It shows how the spectral kurtosis of a signal changes over time and frequency, providing a way to identify and locate transient events or anomalies in the signal.

To compute the spectral kurtogram, we first divide the signal into a series of overlapping frames and compute the spectral kurtosis of each frame. Then, we plot the spectral kurtosis values as a function of time and frequency, with darker regions indicating higher spectral kurtosis values.


## Empirical Mode Decomposition (TODO)
Empirical Mode Decomposition (EMD) is a signal processing technique used for analyzing non-stationary and nonlinear signals. It is a data-driven approach that decomposes a signal into a set of intrinsic mode functions (IMFs) that capture the signal's different frequency components.

The EMD algorithm works by iteratively decomposing a signal into a series of IMFs and a residual. Each IMF is a signal that has a characteristic oscillatory behavior and represents a frequency component of the original signal. The IMFs are obtained by extracting the local extrema of the signal and interpolating between them using cubic spline functions. The residual is the remaining signal after all the IMFs have been extracted, and it represents the high-frequency noise or other features that cannot be represented by the IMFs.

In vibration analysis, EMD can be used to decompose a vibration signal into its different frequency components and extract useful information about the signal's underlying dynamics. For example, EMD can be used to extract the high-frequency components of a vibration signal that are associated with bearing faults or other types of damage. The resulting IMFs can then be analyzed separately to identify the specific fault characteristic frequencies and diagnose the type and severity of the fault.

EMD has been used in a variety of applications in vibration analysis, including gearbox fault diagnosis, turbine blade crack detection, and rolling element bearing fault diagnosis, among others. It is a useful tool for analyzing complex vibration signals that exhibit non-stationary and nonlinear behavior, and it can provide valuable insights into the underlying physics of the vibration source.