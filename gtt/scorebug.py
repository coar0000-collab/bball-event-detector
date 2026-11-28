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
def _split_blob(g: np.ndarray, k: int) -> list[np.ndarray]:
    """붙은 숫자 덩어리를 k 개로 나눈다. 대략 같은 폭으로 자르되, 자르는 자리는 근처에서 글자 픽셀이 가장 적은 열."""
    h, w = g.shape
    if k <= 1:
        return [g]
    cols = (g > 0).sum(axis=0).astype(np.float32)
    cuts, prev = [], 0
    for j in range(1, k):
        c0 = int(round(j * w / k))
        lo, hi = max(prev + 1, c0 - w // (4 * k)), min(w - 1, c0 + w // (4 * k))
        c = lo + int(np.argmin(cols[lo:hi + 1])) if hi > lo else c0
        cuts.append(c)
        prev = c
    parts = []
    for a, b in zip([0] + cuts, cuts + [w]):
        part = g[:, a:b]
        ys = np.where(part.any(axis=1))[0]
        xs = np.where(part.any(axis=0))[0]
        if len(ys) and len(xs):
            parts.append(part[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1])
    return parts


def segment_digits(crop: np.ndarray, min_height_ratio: float = 0.4, templates: "DigitTemplates | None" = None,
                   n_expected: int | None = None) -> list[np.ndarray]:
    """점수 영역에서 숫자 글리프들을 왼쪽부터 잘라 정규화(GLYPH_W x GLYPH_H)해 돌려준다.

    붙어 버린 숫자(예: '44' 처럼 가로획이 옆 숫자에 닿음)는 한 덩어리로 잡히므로 나눈다.
    몇 개로 나눌지는 templates 가 있으면 나눈 조각들이 가장 확실하게 읽히는 쪽, n_expected(등록할 때 아는 자릿수)가
    있으면 그 수에 맞게, 둘 다 없으면 폭으로 정한다.
    """
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
        if w > 3.6 * h:  # 아주 가로로 긴 덩어리(밑줄·테두리)는 숫자가 아님. 숫자 2~3개가 붙은 것은 남긴다
            continue
        boxes.append((x, y, w, h, i))
    boxes.sort()
    groups = [([i], x, y, x + w, y + h) for x, y, w, h, i in boxes]
    if n_expected is not None and len(groups) > n_expected:
        # 한 숫자가 두 조각으로 끊겨 잡힘 (실제 중계 글꼴의 '4': 사선과 세로획이 떨어져 '11' 처럼 읽힘)
        # → 가로 간격이 가장 좁은 이웃끼리 합쳐 개수를 맞춘다
        while len(groups) > n_expected:
            gaps = [groups[j + 1][1] - groups[j][3] for j in range(len(groups) - 1)]
            j = int(np.argmin(gaps))
            a_, b_ = groups[j], groups[j + 1]
            groups[j:j + 2] = [(a_[0] + b_[0], min(a_[1], b_[1]), min(a_[2], b_[2]), max(a_[3], b_[3]), max(a_[4], b_[4]))]
    blobs = [np.isin(lab[y0:y1, x0:x1], ids).astype(np.uint8) * 255 for ids, x0, y0, x1, y1 in groups]
    ks = [1] * len(blobs)
    if n_expected is not None and 0 < len(blobs) < n_expected:
        # 등록할 때: 모자란 만큼 가장 넓은 덩어리를 나눈다
        j = int(np.argmax([b.shape[1] for b in blobs]))
        ks[j] = n_expected - len(blobs) + 1
    elif n_expected is None:
        for j, b in enumerate(blobs):
            h, w = b.shape
            opts = [k for k in (1, 2, 3) if 0.3 * h <= w / k <= 1.3 * h]
            if len(opts) <= 1:
                ks[j] = opts[0] if opts else 1
                continue
            if templates is not None and templates.samples:
                def conf(k):
                    parts = _split_blob(b, k)
                    return min(templates.classify(normalize_glyph(p))[1] for p in parts) if len(parts) == k else -1.0
                scored = {k: conf(k) for k in opts}
                best = max(scored, key=scored.get)
                ks[j] = 1 if 1 in scored and scored[1] >= scored[best] - 0.05 else best
            else:
                ks[j] = 1 if w <= 1.25 * h else int(np.clip(round(w / (0.9 * h)), 1, 3))
    glyphs = []
    for b, k in zip(blobs, ks):
        glyphs += [normalize_glyph(part) for part in _split_blob(b, k)]
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
    glyphs = segment_digits(crop, n_expected=len(value))
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
    glyphs = segment_digits(crop, templates=templates)
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


def _cost(templates: DigitTemplates, glyph: np.ndarray, digit: str) -> float:
    """글리프와 그 숫자 템플릿들 중 가장 가까운 것의 평균 차이 (0 = 똑같음)."""
    lst = templates.samples.get(digit)
    if not lst:
        return 1.0
    g = glyph.astype(np.float32) / 255.0
    return min(float(np.mean(np.abs(g - s.astype(np.float32) / 255.0))) for s in lst)


def read_number_near(crop: np.ndarray, templates: DigitTemplates, prev: int | None, max_digits: int = 3,
                     min_conf: float = 0.15, max_step: int = 3, max_cost: float = 0.12) -> Reading:
    """직전에 확정된 점수(prev)를 알 때 읽기. 점수는 한 번에 3점(정정이면 -3점)보다 많이 바뀌지 않는다.

    그냥 읽은 값이 prev±max_step 밖이면(실제 중계: 22→24 의 '4' 가 두 조각으로 끊겨 '114' 로 읽혀 +92), prev±max_step 의
    값마다 그 자릿수로 글리프를 나눠(끊긴 조각은 합침) 그 숫자 템플릿과 얼마나 닮았는지 보고, 모든 자리가 max_cost 안으로
    닮은 값 중 가장 닮은 것을 고른다. 닮은 값이 없으면 그냥 읽은 값을 그대로 돌려준다 — 다음 경기로 넘어가 0:0 이 되는
    것처럼 정말 크게 바뀐 경우는 ScoreChangeDetector 가 여러 번 읽힌 뒤에 check/correction 으로 받는다."""
    r = read_number(crop, templates, max_digits, min_conf)
    if prev is None or (r.value is not None and abs(r.value - prev) <= max_step):
        return r
    if not r.text:
        return r  # 글자가 안 보임 (스코어버그가 사라진 화면)
    best: tuple[float, int] | None = None
    for c in range(max(0, prev - max_step), prev + max_step + 1):
        txt = str(c)
        glyphs = segment_digits(crop, n_expected=len(txt))
        if len(glyphs) != len(txt):
            continue
        cost = max(_cost(templates, g, ch) for g, ch in zip(glyphs, txt))
        if best is None or cost < best[0]:
            best = (cost, c)
    if best is not None and best[0] <= max_cost:
        return Reading(best[1], str(best[1]), float(np.clip(1 - best[0] / max_cost, 0, 1)) * 0.5)
    return r


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
