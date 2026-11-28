"""점수판이 정답인 성공 슛으로 '발끝이 3점 라인에 닿는 기준'을 경기(체육관·카메라)마다 학습해 실패 슛에 쓴다.

찍은 점(두 발 가운데)에서 3점 라인 바깥 가장자리까지 거리 m 에 대해, 3점 ⇔ m > offset 이다.
offset 의 기본값은 발끝 길이(spec.foot_reach, 0.14m)지만 실제로는 찍는 버릇, 카메라 각도에 따른 쏠림, 신발 크기가 섞인다.
점수판이 +2/+3 을 알려 주는 성공 슛마다 offset 이 들어갈 수 있는 범위가 좁혀진다:
  +3 이었던 슛 → offset < m,  +2 였던 슛 → offset ≥ m
모든 성공 슛을 만족하는 구간 [lo, hi) 가 '학습된 기준'이다. 구간 안쪽에 떨어지는 실패 슛은 정직하게 '판정 불가'로 둔다.
영상이 쌓일수록 구간이 좁아져 판정 불가가 줄어든다 (fit 결과를 JSON 으로 모아 여러 경기를 합칠 수 있다).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .court import FIBA, CourtSpec, three_point_margin
from .scorecheck import check, shot_points


@dataclass
class LineModel:
    lo: float  # offset 하한 (+2 였던 슛 중 가장 바깥)
    hi: float  # offset 상한 (+3 이었던 슛 중 가장 안쪽)
    offset: float
    n2: int
    n3: int
    errors: int = 0  # 어떤 offset 으로도 설명 못 한 성공 슛 수 (찍기 실수·추적 오류 의심)
    samples: list = field(default_factory=list)

    def decide(self, m: float, sigma: float = 0.0) -> int | None:
        """찍은 점의 라인 거리 m(+ 찍기 오차 sigma) → 3, 2, 또는 None(판정 불가)."""
        if m - sigma > max(self.hi, self.offset):
            return 3
        if m + sigma <= min(self.lo, self.offset):
            return 2
        return None

    def as_dict(self) -> dict:
        return {"lo": round(self.lo, 3), "hi": round(self.hi, 3), "offset": round(self.offset, 3), "n2": self.n2, "n3": self.n3,
                "errors": self.errors}


def margin_mid(s: dict, spec: CourtSpec = FIBA) -> float | None:
    try:
        x, y = float(s["court_x"]), float(s["court_y"])
    except (KeyError, TypeError, ValueError):
        return None
    side = s.get("side") if s.get("side") in ("left", "right") else None
    return three_point_margin(x, y, spec, side)


def labeled_samples(events: list[dict], shots: list[dict], spec: CourtSpec = FIBA, window: float = 2.0) -> list[tuple[float, int]]:
    """점수판과 짝지어진 성공 필드골 중 라인에서 window(m) 안쪽만: (m, 점수판 점수)."""
    out = []
    for p in check(events, shots).pairs:
        if len(p.shots) != 1 or p.bug_points not in (2, 3) or shot_points(p.shots[0]) == 1:
            continue
        m = margin_mid(p.shots[0], spec)
        if m is not None and abs(m) <= window:
            out.append((m, p.bug_points))
    return out


def fit(samples: list[tuple[float, int]], prior: float = FIBA.foot_reach, wide: float = 2.0) -> LineModel:
    m2 = [m for m, p in samples if p == 2]
    m3 = [m for m, p in samples if p == 3]
    lo = max(m2) if m2 else -wide
    hi = min(m3) if m3 else wide
    errors = 0
    if lo >= hi:  # 서로 엇갈림: 오류가 가장 적은 offset
        cands = sorted({m for m, _ in samples})
        best = min(cands, key=lambda o: sum((m > o) != (p == 3) for m, p in samples))
        errors = sum((m > best) != (p == 3) for m, p in samples)
        lo = hi = best
    offset = prior if lo <= prior < hi else (lo + hi) / 2 if (m2 and m3) else (min(prior, hi - 1e-3) if m3 else max(prior, lo))
    return LineModel(lo=lo, hi=hi, offset=offset, n2=len(m2), n3=len(m3), errors=errors, samples=samples)
