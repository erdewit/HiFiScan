import array
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Deque

import eventkit as ev
import numpy as np
import numpy.typing as npt
import sounddevice as sd


class Audio:
    """
    Bidirectional audio interface, for simultaneous playing and recording.

    Events:
        * recorded(record):
          Emits a new piece of recorded sound as a numpy float array.
    """

    def __init__(self):
        self.recorded = ev.Event()
        self.playQ: Deque[PlayItem] = deque()
        self.stream = sd.Stream(
            channels=2,
            callback=self._onStream)
        self.stream.start()
        self.rate = self.stream.samplerate

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.stream.stop()
        self.stream.close()

    def _onStream(self, in_data, out_data, frames, _time, _status):
        # Note that this is called from a non-main thread.
        out_data.fill(0)
        idx = 0
        while self.playQ and idx < frames:
            playItem = self.playQ[0]
            chunk = playItem.pop(frames - idx).T
            idx2 = idx + len(chunk)
            out_data[idx:idx2, :] = chunk
            idx = idx2
            if not playItem.remaining():
                self.playQ.popleft()
        self.recorded.emit_threadsafe(in_data.copy().T)

    def play(self, sound: npt.ArrayLike):
        """Add a sound to the play queue."""
        sound = np.asarray(sound)
        if len(sound.shape) == 1:
            sound = np.vstack([sound, sound])
        self.playQ.append(PlayItem(sound))

    def cancelPlay(self):
        """Clear the play queue."""
        self.playQ.clear()

    def isPlaying(self) -> bool:
        """Is there sound playing from the play queue?"""
        return bool(self.playQ)

    def record(self) -> AsyncIterator[array.array]:
        """
        Start a recording, yielding the entire recording every time a
        new chunk is added. The recording is a 32-bit float array.
        """
        recording = array.array('f')
        return self.recorded.map(
            lambda rec: recording.extend(0.5 * (rec[0] + rec[1]))) \
            .constant(recording).aiter(skip_to_last=True)


@dataclass
class PlayItem:
    sound: np.ndarray
    index: int = 0

    def remaining(self) -> int:
        return self.sound.size - self.index

    def pop(self, num: int) -> np.ndarray:
        idx = self.index + min(num, self.remaining())
        chunk = self.sound[:, self.index:idx]
        self.index = idx
        return chunk
