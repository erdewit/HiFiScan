import array
from functools import lru_cache
from typing import NamedTuple, Tuple

import numpy as np
from numba import njit


class XY(NamedTuple):
    """XY coordinate data of arrays of the same length."""

    x: np.ndarray
    y: np.ndarray


class Analyzer:
    """
    Analyze the system response to a chirp stimulus.

    Symbols that are used:

      x: stimulus
      y: response = x * h
      X = FT(x)
      Y = FT(y) = X . H
      H: system transfer function = X / Y
      h: system impulse response = IFT(H)
      h_inv: inverse system impulse response (which undoes h) = IFT(1 / H)

    with:
      *: convolution operator
      FT: Fourier transform
      IFT: Inverse Fourier transform
    """

    MAX_DELAY_SECS = 0.1
    TIMEOUT_SECS = 1.0

    chirp: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rate: int
    secs: float
    fmin: float
    fmax: float
    time: float

    def __init__(
            self, f0: int, f1: int, secs: float, rate: int, ampl: float):
        self.chirp = ampl * geom_chirp(f0, f1, secs, rate)
        self.x = np.concatenate([
            self.chirp,
            np.zeros(int(self.MAX_DELAY_SECS * rate))
        ])
        self.secs = self.x.size / rate
        self.rate = rate
        self.fmin = min(f0, f1)
        self.fmax = max(f0, f1)
        self.time = 0

        # Cache the methods in a way that allows garbage collection of self.
        for meth in ['X', 'Y', 'H', 'H2', 'h', 'h_inv', 'spectrum']:
            setattr(self, meth, lru_cache(getattr(self, meth)))

    def findMatch(self, recording: array.array) -> bool:
        """
        Use correlation to find a match of the chirp in the recording.
        If found, return True and store the system response as ``y``.
        """
        sz = len(recording)
        self.time = sz / self.rate
        if sz >= self.x.size:
            Y = np.fft.fft(recording)
            X = np.fft.fft(np.flip(self.x), n=sz)
            corr = np.fft.ifft(X * Y).real
            idx = int(corr.argmax()) - self.x.size + 1
            if idx >= 0:
                self.y = np.array(recording[idx:idx + self.x.size], 'f')
                return True
        return False

    def timedOut(self) -> bool:
        """See if time to find a match has exceeded the timeout limit."""
        return self.time > self.secs + self.TIMEOUT_SECS

    def freqRange(self, size: int) -> slice:
        """
        Return range slice of the valid frequency range for an array
        of given size.
        """
        nyq = self.rate / 2
        i0 = int(0.5 + size * self.fmin / nyq)
        i1 = int(0.5 + size * self.fmax / nyq)
        return slice(i0, i1 + 1)

    def X(self) -> np.ndarray:
        return np.fft.rfft(self.x)

    def Y(self) -> np.ndarray:
        return np.fft.rfft(self.y)

    def H(self) -> XY:
        """
        Calculate complex-valued transfer function H in the
        frequency domain.
        """
        X = self.X()
        Y = self.Y()
        # H = Y / X
        H = Y * np.conj(X) / (np.abs(X) ** 2 + 1e-3)
        freq = np.linspace(0, self.rate // 2, H.size)
        return XY(freq, H)

    def H2(self, smoothing: float):
        """Calculate smoothed squared transfer function |H|^2."""
        freq, H = self.H()
        H = np.abs(H)
        r = self.freqRange(H.size)
        H2 = np.empty_like(H)
        # Perform smoothing on the squared amplitude.
        H2[r] = smooth(freq[r], H[r] ** 2, smoothing)
        H2[:r.start] = H2[r.start]
        H2[r.stop:] = H2[r.stop - 1]
        return XY(freq, H2)

    def h(self) -> XY:
        """Calculate impulse response ``h`` in the time domain."""
        _, H = self.H()
        h = np.fft.irfft(H)
        h = np.hstack([h[h.size // 2:], h[0:h.size // 2]])
        t = np.linspace(0, h.size / self.rate, h.size)
        return XY(t, h)

    def spectrum(self, smoothing: float = 0) -> XY:
        """
        Calculate the frequency response in the valid frequency range,
        with optional smoothing.

        Args:
          smoothing: Determines the overall strength of the smoothing.
          Useful values are from 0 to around 30.
          If 0 then no smoothing is done.
        """
        freq, H2 = self.H2(smoothing)
        r = self.freqRange(H2.size)
        return XY(freq[r], 10 * np.log10(H2[r]))

    def h_inv(
            self,
            secs: float = 0.05,
            dbRange: float = 24,
            kaiserBeta: float = 5,
            smoothing: float = 0) -> XY:
        """
        Calculate the inverse impulse response.

        Args:
            secs: Desired length of the response in seconds.
            dbRange: Maximum attenuation in dB (power).
            kaiserBeta: Shape parameter of the Kaiser tapering window.
            smoothing: Strength of frequency-dependent smoothing.
        """
        freq, H2 = self.H2(smoothing)
        # Re-sample to halve the number of samples needed.
        n = int(secs * self.rate / 2)
        H = resample(H2, n) ** 0.5
        # Accommodate the given dbRange from the top.
        H /= H.max()
        H = np.fmax(H, 10 ** (-dbRange / 20))

        # Calculate Z, the reciprocal transfer function with added
        # linear phase. This phase will shift and center z.
        Z = 1 / H
        phase = np.exp(Z.size * 1j * np.linspace(0, np.pi, Z.size))
        Z = Z * phase

        # Calculate the inverse impulse response z.
        z = np.fft.irfft(Z)
        z = z[:-1]
        z *= window(z.size, kaiserBeta)
        # Normalize using a fractal dimension for scaling.
        dim = 1.6
        norm = (np.abs(z) ** dim).sum() ** (1 / dim)
        z /= norm
        # assert np.allclose(z[-(z.size // 2):][::-1], z[:z.size // 2])

        t = np.linspace(0, z.size / self.rate, z.size)
        return XY(t, z)

    def correctionFactor(self, invResp: np.ndarray) -> XY:
        """
        Calculate correction factor for each frequency, given the
        inverse impulse response.
        """
        Z = np.abs(np.fft.rfft(invResp))
        Z /= Z.max()
        freq = np.linspace(0, self.rate / 2, Z.size)
        return XY(freq, Z)

    def correctedSpectrum(self, corrFactor: XY) -> Tuple[XY, XY]:
        """
        Simulate the frequency response of the system when it has
        been corrected with the given transfer function.
        """
        freq, H2 = self.H2(0)
        H = H2 ** 0.5
        r = self.freqRange(H.size)

        tf = resample(corrFactor.y, H.size)
        resp = 20 * np.log10(tf[r] * H[r])
        spectrum = XY(freq[r], resp)

        H = resample(H2, corrFactor.y.size) ** 0.5
        rr = self.freqRange(corrFactor.y.size)
        resp = 20 * np.log10(corrFactor.y[rr] * H[rr])
        spectrum_resamp = XY(corrFactor.x[rr], resp)

        return spectrum, spectrum_resamp


@lru_cache
def tone(f: float, secs: float, rate: int):
    """Generate a sine wave."""
    n = int(secs * f)
    secs = n / f
    t = np.arange(0, secs * rate) / rate
    sine = np.sin(2 * np.pi * f * t)
    return sine


@lru_cache
def geom_chirp(f0: float, f1: float, secs: float, rate: int):
    """
    Generate a geometric chirp (with an exponentially changing frequency).

    To avoid a clicking sound at the end, the last sample should be near
    zero. This is done by slightly modifying the time interval to fit an
    integer number of cycli.
    """
    n = int(secs * (f1 - f0) / np.log(f1 / f0))
    k = np.exp((f1 - f0) / n)  # =~ exp[log(f1/f0) / secs]
    secs = np.log(f1 / f0) / np.log(k)

    t = np.arange(0, secs * rate) / rate
    chirp = np.sin(2 * np.pi * f0 * (k ** t - 1) / np.log(k))
    return chirp


@lru_cache
def linear_chirp(f0: float, f1: float, secs: float, rate: int):
    """Generate a linear chirp (with a linearly changing frequency)."""
    n = int(secs * (f1 + f0) / 2)
    secs = 2 * n / (f1 + f0)
    c = (f1 - f0) / secs
    t = np.arange(0, secs * rate) / rate
    chirp = np.sin(2 * np.pi * (0.5 * c * t ** 2 + f0 * t))
    return chirp


def resample(a: np.ndarray, size: int) -> np.ndarray:
    """
    Re-sample the array ``a`` to the given new ``size`` in a way that
    preserves the overall density.
    """
    xp = np.linspace(0, 1, a.size)
    yp = np.cumsum(a)
    x = np.linspace(0, 1, size)
    y = np.interp(x, xp, yp)
    r = size / a.size * np.diff(y, prepend=0)
    return r


@njit
def smooth(freq: np.ndarray, data: np.ndarray, smoothing: float) -> np.ndarray:
    """
    Smooth the data with a smoothing strength proportional to
    the given frequency array and overall smoothing factor.
    The smoothing uses a double-pass exponential moving average (going
    backward and forward).
    """
    if not smoothing:
        return data
    weight = 1 / (1 + freq * 2 ** (smoothing / 2 - 15))
    smoothed = np.empty_like(data)
    prev = data[-1]
    for i, w in enumerate(np.flip(weight), 1):
        smoothed[-i] = prev = (1 - w) * prev + w * data[-i]
    prev = smoothed[0]
    for i, w in enumerate(weight):
        smoothed[i] = prev = (1 - w) * prev + w * smoothed[i]
    return smoothed


@lru_cache
def window(size: int, beta: float) -> np.ndarray:
    """Kaiser tapering window."""
    return np.kaiser(size, beta)


@lru_cache
def taper(y0: float, y1: float, size: int) -> np.ndarray:
    """Create a smooth transition from y0 to y1 of given size."""
    tp = (y0 + y1 - (y1 - y0) * np.cos(np.linspace(0, np.pi, size))) / 2
    return tp
