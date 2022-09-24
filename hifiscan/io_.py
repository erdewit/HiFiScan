import wave
from typing import List, NamedTuple, Tuple

import numpy as np


class Sound(NamedTuple):
    data: np.ndarray
    rate: int
    width: int = 4


Correction = List[Tuple[float, float]]


def write_wav(path: str, data: np.ndarray, rate: int, width: int = 4):
    """
    Write n-channel float array with values between -1 and 1 to WAV file

    Params:
      path: Filename of WAV file.
      data: Sound sample data array.
      rate: Sample rate in Hz.
      width: Sample width in bytes.
    """
    if width not in [1, 2, 3, 4]:
        raise ValueError(f'Invalid sample width: {width}')
    data = np.asarray(data)
    ch = 1 if len(data.shape) < 2 else len(data)
    if width == 4:
        arr = np.empty_like(data, np.int32)
        np.rint((2 ** 31 - 1) * data, out=arr, casting='unsafe')
    elif width == 3:
        arr = np.empty_like(data, np.int32)
        np.rint((2 ** 31 - 2 ** 8) * data, out=arr, casting='unsafe')
        arr = arr.flatten(order='F').view(np.uint8)
        # Drop every 4th byte.
        arr = np.vstack([arr[1::4], arr[2::4], arr[3::4]])
    elif width == 2:
        arr = np.empty_like(data, np.int16)
        np.rint((2 ** 15 - 1) * data, out=arr, casting='unsafe')
    else:
        arr = np.empty_like(data, np.int8)
        np.rint((2 ** 7 - 1) * data, out=arr, casting='unsafe')
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(ch)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(arr.tobytes(order='F'))


def read_wav(path: str) -> Sound:
    """
    Read WAV file and return float32 arrays between -1 and 1.
    """
    with wave.open(path, 'rb') as wav:
        ch, width, rate, n, _, _ = wav.getparams()
        frames = wav.readframes(n)
    if width == 4:
        buff = np.frombuffer(frames, np.int32)
        norm = 2 ** 31 - 1
    elif width == 3:
        buff = np.frombuffer(frames, np.uint8)
        uints = buff[0::3].astype(np.uint32) << 8 \
            | buff[1::3].astype(np.uint32) << 16 \
            | buff[2::3].astype(np.uint32) << 24
        buff = uints.view(np.int32)
        norm = 2 ** 31 - 2 ** 8
    elif width == 2:
        buff = np.frombuffer(frames, np.int16)
        norm = 2 ** 15 - 1
    else:
        buff = np.frombuffer(frames, np.int8)
        norm = 2 ** 7 - 1
    data = buff.astype('f')
    data /= np.float32(norm)
    data = data.reshape((-1, ch)).T
    return Sound(data, rate, width)


def read_correction(path: str) -> Correction:
    corr = []
    with open(path, 'r') as f:
        for line in f.readlines():
            try:
                freq, db = line.split()
                corr.append((float(freq), float(db)))
            except ValueError:
                pass
    return corr
