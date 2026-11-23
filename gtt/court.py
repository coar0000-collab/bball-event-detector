"""코트 규격과 슛 위치 → 점수(2점/3점)·구역 판정.

코트 좌표계 (단위: m)
- x: 코트 길이 방향, 왼쪽 베이스라인 = 0, 오른쪽 베이스라인 = length (FIBA 28)
- y: 코트 폭 방향, 아래쪽 사이드라인 = 0, 위쪽 사이드라인 = width (FIBA 15)
- 화면에서 어느 쪽이 "왼쪽"인지는 캘리브레이션 때 정한 랜드마크 이름으로 결정된다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CourtSpec:
    """코트 규격. 기본값은 FIBA(2010년 이후) 규격."""

    name: str = "FIBA"
    length: float = 28.0
    width: float = 15.0
    rim_from_baseline: float = 1.575  # 베이스라인 → 림 중심
    three_radius: float = 6.75  # 림 중심 → 3점 아크
    three_corner_from_sideline: float = 0.9  # 코너 3점 직선과 사이드라인 거리
    key_width: float = 4.9  # 페인트존 폭
    key_length: float = 5.8  # 베이스라인 → 자유투 라인
    ft_circle_radius: float = 1.8

    @property
    def corner_offset(self) -> float:
        """림 중심선(y=width/2)에서 코너 3점 직선까지의 거리."""
        return self.width / 2 - self.three_corner_from_sideline

    @property
    def corner_straight_end(self) -> float:
        """베이스라인에서 코너 직선이 아크와 만나는 지점까지의 거리."""
        c = self.corner_offset
        return self.rim_from_baseline + math.sqrt(max(self.three_radius**2 - c**2, 0.0))

    def rim(self, side: str) -> tuple[float, float]:
        y = self.width / 2
        if side == "left":
            return (self.rim_from_baseline, y)
        return (self.length - self.rim_from_baseline, y)


FIBA = CourtSpec()
# 참고용: NBA 규격 (피트 → m)
NBA = CourtSpec(
    name="NBA",
    length=28.65,
    width=15.24,
    rim_from_baseline=1.6,
    three_radius=7.24,
    three_corner_from_sideline=0.91,
    key_width=4.88,
    key_length=5.79,
)
SPECS = {"FIBA": FIBA, "NBA": NBA}


@dataclass
class ShotCall:
    points: int  # 2 또는 3
    zone: str  # PAINT / MID / 3P_CORNER / 3P_WING / 3P_TOP
    zone_kr: str  # 기록지 표기: 2점 / 코너 / 45도 / 탑
    side: str  # 어느 골대를 향한 슛인지: left / right
    hand_side: str  # 림 기준 슈터 위치: L / R (베이스라인에서 코트를 바라볼 때)
    dist_to_rim: float  # m
    margin: float  # 3점 라인까지의 거리 (+ 바깥/3점, - 안쪽/2점), m
    near_line: bool  # |margin| < 임계값이면 사람이 검수
    court_xy: tuple[float, float] = field(default=(0.0, 0.0))
    # 자유투 자리(자유투 라인 바로 뒤, 레인 폭 안). 위치만으로는 자유투와 그 자리 점프슛(2점)을 못 가르므로
    # 실제 자유투인지는 상황(멈춘 시계, 레인에 늘어선 선수 등)이나 라벨로 정한다.
    ft_spot: bool = False
    # 화면에서 발 위치를 몇 px 잘못 찍었을 때 3점 라인까지 거리가 얼마나 흔들리는지 (m). 멀고 비스듬한 곳일수록 크다.
    margin_sigma: float = 0.0

    def as_dict(self) -> dict:
        return {
            "points": self.points,
            "zone": self.zone,
            "zone_kr": self.zone_kr,
            "side": self.side,
            "hand_side": self.hand_side,
            "dist_to_rim_m": round(self.dist_to_rim, 2),
            "margin_m": round(self.margin, 2),
            "near_line": self.near_line,
            "court_x": round(self.court_xy[0], 2),
            "court_y": round(self.court_xy[1], 2),
            "ft_spot": self.ft_spot,
            "margin_sigma_m": round(self.margin_sigma, 3),
        }


def three_point_margin(x: float, y: float, spec: CourtSpec = FIBA, side: str | None = None) -> float:
    """3점 라인까지의 부호 있는 거리(m). 양수 = 3점 구역, 음수 = 2점 구역.

    side 가 None 이면 하프라인 기준으로 가까운 골대를 고른다.
    """
    if side is None:
        side = "left" if x <= spec.length / 2 else "right"
    rx, ry = spec.rim(side)
    # 왼쪽 골대 기준으로 정규화: d = 베이스라인으로부터의 거리
    d_baseline = x if side == "left" else spec.length - x
    dy = y - ry
    c = spec.corner_offset
    if d_baseline <= spec.corner_straight_end:
        # 코너 직선 구간: 직선(|dy| = c)과의 거리
        return abs(dy) - c
    dx = d_baseline - spec.rim_from_baseline
    return math.hypot(dx, dy) - spec.three_radius


def classify_shot(
    x: float,
    y: float,
    spec: CourtSpec = FIBA,
    side: str | None = None,
    near_line_m: float = 0.3,
) -> ShotCall:
    """슈터 발 위치(코트 좌표, m)로 2점/3점과 구역을 판정한다.

    x, y 는 릴리스 순간 슈터 두 발의 중간점(바운딩 박스 하단 중앙)을 쓴다.
    side 는 공격 방향(어느 림을 향해 쏘는지). 모르면 None → 가까운 림.
    """
    if side is None:
        side = "left" if x <= spec.length / 2 else "right"
    rx, ry = spec.rim(side)
    d_baseline = x if side == "left" else spec.length - x
    dx = d_baseline - spec.rim_from_baseline  # 림에서 코트 안쪽 방향 +
    dy = y - ry
    dist = math.hypot(dx, dy)
    margin = three_point_margin(x, y, spec, side)
    # 베이스라인에서 코트를 바라볼 때 왼손/오른손 쪽. 왼쪽 골대에서는 y 가 작은 쪽이 오른쪽.
    if side == "left":
        hand = "R" if dy < 0 else "L"
    else:
        hand = "L" if dy < 0 else "R"

    if margin > 0:
        points = 3
        angle = math.degrees(math.atan2(abs(dy), max(dx, 1e-6)))  # 0 = 정면, 90 = 베이스라인 방향
        if d_baseline <= spec.corner_straight_end:
            zone, zone_kr = "3P_CORNER", "코너"
        elif angle < 22.5:
            zone, zone_kr = "3P_TOP", "탑"
        else:
            zone, zone_kr = "3P_WING", "45도"
    else:
        points = 2
        in_key = d_baseline <= spec.key_length and abs(dy) <= spec.key_width / 2
        zone, zone_kr = ("PAINT", "2점") if in_key else ("MID", "2점")

    return ShotCall(
        points=points,
        zone=zone,
        zone_kr=zone_kr,
        side=side,
        hand_side=hand,
        dist_to_rim=dist,
        margin=margin,
        near_line=abs(margin) < near_line_m,
        court_xy=(x, y),
        ft_spot=is_ft_spot(x, y, spec, side),
    )


def is_ft_spot(x: float, y: float, spec: CourtSpec = FIBA, side: str | None = None,
               behind_m: float = 1.0, front_m: float = 0.3, half_width_m: float = 1.2) -> bool:
    """자유투 쏘는 자리인지 (자유투 라인에서 뒤로 behind_m, 앞으로 front_m, 레인 가운데에서 좌우 half_width_m 안)."""
    if side is None:
        side = "left" if x <= spec.length / 2 else "right"
    d_baseline = x if side == "left" else spec.length - x
    behind = d_baseline - spec.key_length  # 자유투 라인 뒤쪽이 +
    return -front_m <= behind <= behind_m and abs(y - spec.width / 2) <= half_width_m


def landmarks(spec: CourtSpec = FIBA) -> dict[str, tuple[float, float]]:
    """캘리브레이션에 쓰는 코트 랜드마크 (이름 → 코트 좌표 m).

    이름 규칙: L_ = 왼쪽 골대 쪽, R_ = 오른쪽 골대 쪽, top = y 가 큰 사이드라인 쪽, bot = y 가 작은 쪽.
    화면에서 보이는 점 4개 이상(같은 직선 위에 3개 이상 몰리지 않게)을 찍으면 된다.
    """
    L, W = spec.length, spec.width
    kw = spec.key_width / 2
    c = spec.corner_offset
    m = W / 2
    ce = spec.corner_straight_end
    arc_top_x = spec.rim_from_baseline + spec.three_radius
    pts = {
        "L_baseline_bot": (0.0, 0.0),
        "L_baseline_top": (0.0, W),
        "L_key_baseline_bot": (0.0, m - kw),
        "L_key_baseline_top": (0.0, m + kw),
        "L_ft_bot": (spec.key_length, m - kw),
        "L_ft_top": (spec.key_length, m + kw),
        "L_ft_center": (spec.key_length, m),
        "L_corner3_baseline_bot": (0.0, m - c),
        "L_corner3_baseline_top": (0.0, m + c),
        "L_corner3_arc_bot": (ce, m - c),
        "L_corner3_arc_top": (ce, m + c),
        "L_arc_top": (arc_top_x, m),
        "mid_sideline_bot": (L / 2, 0.0),
        "mid_sideline_top": (L / 2, W),
        "center": (L / 2, m),
        "center_circle_bot": (L / 2, m - spec.ft_circle_radius),
        "center_circle_top": (L / 2, m + spec.ft_circle_radius),
    }
    # 오른쪽 골대 쪽은 x 를 뒤집어서 만든다.
    for k, (x, y) in list(pts.items()):
        if k.startswith("L_"):
            pts["R_" + k[2:]] = (L - x, y)
    return pts


def court_polylines(spec: CourtSpec = FIBA, arc_steps: int = 48) -> dict[str, list[tuple[float, float]]]:
    """오버레이·라인 정합에 쓰는 코트 라인 (코트 좌표 폴리라인)."""
    L, W = spec.length, spec.width
    m = W / 2
    kw = spec.key_width / 2
    c = spec.corner_offset
    ce = spec.corner_straight_end
    lines: dict[str, list[tuple[float, float]]] = {
        "boundary": [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
        "halfcourt": [(L / 2, 0), (L / 2, W)],
    }
    for side in ("L", "R"):
        def fx(x: float, side: str = side) -> float:
            return x if side == "L" else L - x

        lines[f"{side}_key"] = [
            (fx(0), m - kw), (fx(spec.key_length), m - kw), (fx(spec.key_length), m + kw), (fx(0), m + kw)
        ]
        rimx = spec.rim_from_baseline
        a0 = math.asin(c / spec.three_radius)  # 코너 직선과 만나는 각
        arc = []
        for i in range(arc_steps + 1):
            a = -a0 + (2 * a0) * i / arc_steps  # 림에서 코트 안쪽(x+) 기준 각도, -a0..a0
            ya = m + spec.three_radius * math.sin(a)
            xa = rimx + spec.three_radius * math.cos(a)
            arc.append((fx(xa), ya))
        three = [(fx(0), m - c), (fx(ce), m - c)] + arc + [(fx(ce), m + c), (fx(0), m + c)]
        lines[f"{side}_three"] = three
    return lines
