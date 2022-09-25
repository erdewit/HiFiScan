"""'Optimize the frequency response spectrum of an audio system"""

from hifiscan.analyzer import (
    Analyzer, XY, geom_chirp, linear_chirp, transform_causality, resample,
    smooth, taper, tone, window)
from hifiscan.audio import Audio
from hifiscan.io_ import Sound, read_correction, read_wav, write_wav
