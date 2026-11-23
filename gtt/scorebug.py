"""중계 화면 스코어버그(점수판) 인식과 점수 변화 이벤트 추출.

외부 OCR 엔진 없이 동작한다. 중계사마다 숫자 글꼴이 고정이라, 실제 화면에서 숫자 몇 개를
한 번 등록(템플릿)하면 그 뒤로는 템플릿 매칭만으로 충분히 정확하게 읽힌다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

GLYPH_W, GLYPH_H = 24, 36


# ---------------------------------------------------------------- 설정
@dataclass
class ScorebugConfig:
    """ROI 는 reference_size 해상도 기준 [x, y, w, h]. 다른 해상도 영상이면 비율로 맞춘다."""

    rois: dict[str, list[int]]
    reference_size: tuple[int, int]  # (width, height)
    home_name: str = "HOME"
    away_name: str = "AWAY"
    max_digits: int = 3

    @classmethod
    def load(cls, path: str | Path) -> "ScorebugConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["reference_size"] = tuple(d["reference_size"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def crop(self, frame: np.ndarray, key: str) -> np.ndarray:
        x, y, w, h = self.rois[key]
        fw, fh = frame.shape[1], frame.shape[0]
        sx, sy = fw / self.reference_size[0], fh / self.reference_size[1]
        x0, y0 = int(round(x * sx)), int(round(y * sy))
        x1, y1 = int(round((x + w) * sx)), int(round((y + h) * sy))
        return frame[max(0, y0):y1, max(0, x0):x1]


# ---------------------------------------------------------------- 숫자 분리
def segment_digits(crop: np.ndarray, min_height_ratio: float = 0.4) -> list[np.ndarray]:
    """점수 영역에서 숫자 글리프들을 왼쪽부터 잘라 정규화(GLYPH_W x GLYPH_H)해 돌려준다."""
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = max(1.0, 60.0 / gray.shape[0])  # 작은 글씨는 키워서 이진화 안정
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 글자는 배경보다 적은 쪽 픽셀이다. 흰 글자로 통일.
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:
        bw = 255 - bw
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    H = bw.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_height_ratio * H or h > 0.98 * H or area < 15:
            continue
        if w > 1.5 * h:  # 가로로 긴 덩어리(밑줄·테두리)는 숫자가 아님
            continue
        boxes.append((x, y, w, h, i))
    boxes.sort()
    glyphs = []
    for x, y, w, h, i in boxes:
        g = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        glyphs.append(normalize_glyph(g))
    return glyphs


def normalize_glyph(g: np.ndarray) -> np.ndarray:
    """종횡비를 유지한 채 GLYPH_H 높이에 맞추고 가운데 정렬한다. '1' 이 '7' 처럼 늘어나지 않게."""
    h, w = g.shape
    s = (GLYPH_H - 4) / h
    nw = max(1, min(GLYPH_W - 2, int(round(w * s))))
    r = cv2.resize(g, (nw, GLYPH_H - 4), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_H, GLYPH_W), np.uint8)
    x0 = (GLYPH_W - nw) // 2
    out[2:GLYPH_H - 2, x0:x0 + nw] = r
    return out


# ---------------------------------------------------------------- 템플릿
@dataclass
class DigitTemplates:
    samples: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, digit: str, glyph: np.ndarray) -> None:
        self.samples.setdefault(digit, []).append(glyph.astype(np.uint8))

    def missing(self) -> list[str]:
        return [d for d in "0123456789" if d not in self.samples]

    def classify(self, glyph: np.ndarray) -> tuple[str, float]:
        """(숫자, 신뢰도 0~1). 신뢰도 = 1등과 2등 거리 차이 기반."""
        g = glyph.astype(np.float32) / 255.0
        best: dict[str, float] = {}
        for d, lst in self.samples.items():
            best[d] = min(float(np.mean(np.abs(g - s.astype(np.float32) / 255.0))) for s in lst)
        if not best:
            raise ValueError("템플릿이 비어 있습니다.")
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        d1, e1 = ranked[0]
        e2 = ranked[1][1] if len(ranked) > 1 else 1.0
        conf = float(np.clip((e2 - e1) / max(e2, 1e-6), 0, 1)) * float(np.clip(1 - e1 * 2.5, 0, 1))
        return d1, conf

    def save(self, path: str | Path) -> None:
        arrs = {}
        for d, lst in self.samples.items():
            arrs[f"d{d}"] = np.stack(lst)
        np.savez_compressed(path, **arrs)

    @classmethod
    def load(cls, path: str | Path) -> "DigitTemplates":
        z = np.load(path)
        t = cls()
        for k in z.files:
            for g in z[k]:
                t.add(k[1:], g)
        return t

    @classmethod
    def from_fonts(cls) -> "DigitTemplates":
        """OpenCV 기본 글꼴로 만든 대체 템플릿. 실제 중계 글꼴은 learn() 으로 등록하는 편이 정확하다."""
        t = cls()
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]
        for d in "0123456789":
            for f in fonts:
                for thick in (2, 4, 6):
                    img = np.zeros((80, 60), np.uint8)
                    cv2.putText(img, d, (8, 66), f, 2.2, 255, thick, cv2.LINE_AA)
                    gl = segment_digits(cv2.cvtColor(255 - img, cv2.COLOR_GRAY2BGR))
                    if len(gl) == 1:
                        t.add(d, gl[0])
        return t


def learn(templates: DigitTemplates, crop: np.ndarray, value: str) -> bool:
    """사람이 알려준 값(예: '47')으로 crop 의 글리프를 템플릿에 추가. 글리프 수가 안 맞으면 False."""
    glyphs = segment_digits(crop)
    if len(glyphs) != len(value):
        return False
    for ch, g in zip(value, glyphs):
        templates.add(ch, g)
    return True


@dataclass
class Reading:
    value: int | None
    text: str
    confidence: float


def read_number(crop: np.ndarray, templates: DigitTemplates, max_digits: int = 3, min_conf: float = 0.15) -> Reading:
    glyphs = segment_digits(crop)
    if not glyphs or len(glyphs) > max_digits:
        return Reading(None, "", 0.0)
    out, confs = [], []
    for g in glyphs:
        d, c = templates.classify(g)
        out.append(d)
        confs.append(c)
    text = "".join(out)
    conf = min(confs)
    if conf < min_conf:
        return Reading(None, text, conf)
    return Reading(int(text), text, conf)


# ---------------------------------------------------------------- 점수 변화 이벤트
@dataclass
class ScoreEvent:
    t: float  # 스코어버그가 바뀐 시각(초). 실제 슛은 보통 1~5초 전.
    team: str  # home / away
    delta: int
    home: int
    away: int
    flag: str  # ok / check (1·2·3 이 아닌 증가) / correction (감소)

    def as_row(self) -> dict:
        return {"t_sec": round(self.t, 2), "team": self.team, "delta": self.delta, "home": self.home, "away": self.away, "flag": self.flag}


class ScoreChangeDetector:
    """잡음 섞인 프레임별 판독값에서 "확정된" 점수 변화만 골라낸다.

    - 최근 판독값 창(window) 안에서 같은 새 값이 stable_n 번 이상 읽히고, 마지막 판독도 그 값이면 확정한다.
      중간에 한두 번 잘못 읽혀도(가림·깜빡임) 놓치지 않는다.
    - 1·2·3 이 아닌 변화(예: +10, 감소)는 stable_n_big 번 읽혀야 받아들이고 check/correction 으로 표시한다.
    - 이벤트 시각은 그 새 값이 처음 읽힌 시각.
    """

    def __init__(self, stable_n: int = 3, stable_n_big: int = 6, window: int | None = None):
        self.stable_n = stable_n
        self.stable_n_big = stable_n_big
        self.window = window or max(stable_n_big, 2 * stable_n) + 2
        self.state: tuple[int, int] | None = None
        self.recent: list[tuple[float, tuple[int, int]]] = []

    def push(self, t: float, home: int | None, away: int | None) -> list[ScoreEvent]:
        if home is None or away is None:
            return []
        v = (home, away)
        self.recent.append((t, v))
        self.recent = self.recent[-self.window:]
        if v == self.state:
            return []
        n = sum(1 for _, x in self.recent if x == v)
        if self.state is None:
            if n >= self.stable_n:
                self.state = v
                self.recent.clear()
            return []
        dh, da = v[0] - self.state[0], v[1] - self.state[1]
        normal = (dh in (0, 1, 2, 3)) and (da in (0, 1, 2, 3)) and (dh > 0) != (da > 0)
        if n < (self.stable_n if normal else self.stable_n_big):
            return []
        t0 = next(tt for tt, x in self.recent if x == v)
        events = []
        for team, d in (("home", dh), ("away", da)):
            if d == 0:
                continue
            flag = "ok" if normal else ("correction" if d < 0 else "check")
            events.append(ScoreEvent(t0, team, d, v[0], v[1], flag))
        self.state = v
        self.recent.clear()
        return events
