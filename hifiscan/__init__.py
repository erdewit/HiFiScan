"""'Optimize the frequency response spectrum of an audio system"""

from hifiscan.analyzer import (
    Analyzer, XY, geom_chirp, linear_chirp, resample, smooth, taper,
    tone, window)
from hifiscan.audio import Audio, write_wav
