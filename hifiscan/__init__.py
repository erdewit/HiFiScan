"""'Optimize the frequency response spectrum of an audio system"""

from hifiscan.analyzer import (
    Analyzer, XY, geom_chirp, linear_chirp, minimum_phase, resample,
    smooth, taper, tone, window)
from hifiscan.audio import Audio, read_correction, write_wav
