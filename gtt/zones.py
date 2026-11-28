"""코트 규격 좌표(m)만으로 슛 위치를 세분한 구역·거리대로 나누고, 구역별 시도·성공을 센다.

카메라 각도와 상관없이 코트 좌표(x, y)만 쓴다. 화면 → 코트 좌표 변환은 track-court/click-shots 가 한다.
구역은 공격하는 림 기준이고 좌/우는 베이스라인에서 코트를 바라볼 때 기준이다.

2점 (8구역): 림 근처(1.25m 안), 페인트, 미드 베이스라인 좌/우, 미드 45도 좌/우, 미드 정면
3점 (5구역): 코너 좌/우, 45도 좌/우, 탑
거리대: 0~1.25, 1.25~3, 3~5, 5~3점 라인, 3점 라인~8, 8m 이상 (3점 라인 기준은 classify_shot 과 같다)
"""
from __future__ import annotations

import math

from .court import FIBA, CourtSpec, classify_shot

ZONES_2 = ["림 근처", "페인트", "미드 베이스라인 좌", "미드 45도 좌", "미드 정면", "미드 45도 우", "미드 베이스라인 우"]
ZONES_3 = ["코너 좌", "45도 좌", "탑", "45도 우", "코너 우"]
BANDS = ["0~1.25m", "1.25~3m", "3~5m", "5m~3점 라인", "3점 라인~8m", "8m 이상"]


def zone_of(x: float, y: float, spec: CourtSpec = FIBA, side: str | None = None) -> dict:
    """코트 좌표 → {points, zone, band, dist_m, angle_deg, hand}."""
    c = classify_shot(x, y, spec, side)
    rx, ry = spec.rim(c.side)
    d_base = x if c.side == "left" else spec.length - x
    dx, dy = d_base - spec.rim_from_baseline, y - ry
    dist = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dy), max(dx, 1e-6)))  # 0 = 정면, 90 = 베이스라인 쪽
    lr = "좌" if c.hand_side == "L" else "우"
    if c.points == 3:
        zone = {"3P_CORNER": f"코너 {lr}", "3P_WING": f"45도 {lr}", "3P_TOP": "탑"}[c.zone]
    elif dist < 1.25:
        zone = "림 근처"
    elif c.zone == "PAINT":
        zone = "페인트"
    elif angle >= 60:
        zone = f"미드 베이스라인 {lr}"
    elif angle >= 22.5:
        zone = f"미드 45도 {lr}"
    else:
        zone = "미드 정면"
    if c.points == 3:
        band = "3점 라인~8m" if dist < 8 else "8m 이상"
    else:
        band = "0~1.25m" if dist < 1.25 else "1.25~3m" if dist < 3 else "3~5m" if dist < 5 else "5m~3점 라인"
    return {"points": c.points, "zone": zone, "band": band, "dist_m": round(dist, 2), "angle_deg": round(angle, 1), "hand": c.hand_side}


def _made(r: dict) -> bool:
    return str(r.get("result", "")).upper() in ("MADE", "성공", "1", "TRUE")


def tally(shots: list[dict], spec: CourtSpec = FIBA) -> dict:
    """슛 기록(court_x, court_y 가 있는 필드골) → 팀별 구역·거리대 시도/성공. 자유투와 좌표 없는 슛은 뺀다."""
    out = {"zone": {}, "band": {}, "skipped": 0}
    for r in shots:
        if str(r.get("zone", "")).upper() == "FT":
            continue
        try:
            x, y = float(r["court_x"]), float(r["court_y"])
        except (KeyError, TypeError, ValueError):
            out["skipped"] += 1
            continue
        side = r.get("side") if r.get("side") in ("left", "right") else None
        z = zone_of(x, y, spec, side)
        team = r.get("team", "home")
        for kind in ("zone", "band"):
            cell = out[kind].setdefault((team, z[kind]), [0, 0])
            cell[0] += 1
            cell[1] += int(_made(r))
    return out
