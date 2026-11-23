"""영상 읽기 도우미."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class Video:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.n_frames / self.fps if self.n_frames else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_at(self, t: float) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * self.fps)))
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"{t:.1f}초 프레임을 읽지 못했습니다.")
        return frame

    def iterate(self, step: float = 0.5, start: float = 0.0, end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
        """start~end 를 step 초 간격으로 읽는다. 중간 프레임은 grab() 으로 건너뛰어 빠르다."""
        every = max(1, int(round(step * self.fps)))
        i = int(round(start * self.fps))
        last = int(round(end * self.fps)) if end is not None else (self.n_frames or 10**12)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        while i <= last:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield i / self.fps, frame
            for _ in range(every - 1):
                if not self.cap.grab():
                    return
            i += every

    def close(self) -> None:
        self.cap.release()


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}"
