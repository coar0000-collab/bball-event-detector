"""코트로 2점/3점을 확실히 못 정한 슛만 점수판으로 보정한다 (하이브리드).

- 코트로 확실한 슛: 코트 판정 그대로 (점수 출처 '코트').
- 라인 근처(사람 판정 없음)이거나 코트 좌표가 없는(발이 가려짐·추적 실패) **성공** 슛: 짝지은 점수판 변화가 +2/+3 이면
  그 점수를 쓴다 (출처 '점수판'). 위치(코너/45도/탑)는 코트 좌표의 방향으로 정한다 — 방향은 거리보다 오차에 강하다.
- 실패한 슛은 점수판이 안 바뀌므로 보정할 수 없다 → 코트 추정값을 두고 출처 '판정 불가(실패 슛)'.
- 코트로 확실했는데 점수판과 다르면 코트 값을 두고 '코트≠점수판'으로 표시해 사람이 본다.

'코트 점수 vs 점수판' 일치율(코트 단독 지표)은 보정 전 슛으로 따로 계산하므로 이 보정이 지표를 부풀리지 않는다.
"""
from __future__ import annotations

from .court import FIBA, CourtSpec
from .scorecheck import check, shot_points


def _true(v) -> bool:
    return str(v).lower() in ("true", "1")


def uncertain(s: dict) -> bool:
    if shot_points(s) is None:
        return True
    if str(s.get("zone", "")).upper() == "FT":
        return False
    return _true(s.get("near_line")) and not s.get("line_call")


def three_zone_by_direction(x: float, y: float, spec: CourtSpec = FIBA, side: str | None = None) -> str:
    """거리와 상관없이 방향만으로 3점 구역(코너/45도/탑). 점수판이 3점이라고 할 때 위치 이름을 붙이는 데 쓴다."""
    import math

    if side not in ("left", "right"):
        side = "left" if x <= spec.length / 2 else "right"
    d_base = x if side == "left" else spec.length - x
    if d_base <= spec.corner_straight_end:
        return "코너"
    dx, dy = d_base - spec.rim_from_baseline, y - spec.width / 2
    return "탑" if math.degrees(math.atan2(abs(dy), max(dx, 1e-6))) < 22.5 else "45도"


def resolve(events: list[dict], shots: list[dict], max_lag: float = 20.0) -> tuple[list[dict], dict]:
    out = [dict(s) for s in shots]
    for s in out:
        s["points_source"] = "판정 불가(실패 슛)" if uncertain(s) else "코트"
    res = check(events, out, max_lag=max_lag)
    idx = {id(s): s for s in out}
    stats = {"코트": 0, "점수판 보정": 0, "학습 기준": 0, "판정 불가": 0, "코트≠점수판": 0}
    for p in res.pairs:
        if len(p.shots) != 1:
            continue
        s = idx[id(p.shots[0])]
        d = p.bug_points
        if s["points_source"] != "코트" and d in (2, 3):
            s["court_points_guess"] = s.get("points", "")
            s["points"] = d
            s["points_source"] = "점수판"
            try:
                x, y = float(s["court_x"]), float(s["court_y"])
                s["zone_kr"] = three_zone_by_direction(x, y, side=s.get("side")) if d == 3 else "2점"
            except (KeyError, TypeError, ValueError):
                s["zone_kr"] = "3점(위치 모름)" if d == 3 else "2점"
        elif s["points_source"] == "코트" and shot_points(s) != d and d in (1, 2, 3):
            s["points_source"] = "코트≠점수판"
    # 점수판이 정답인 성공 슛으로 '발끝 라인 기준'을 이 영상에서 학습해, 판정 불가인 실패 슛에 쓴다
    from .linefit import fit, labeled_samples, margin_mid

    model = fit(labeled_samples(events, shots))
    stats["라인 기준"] = model.as_dict()
    for s in out:
        if s["points_source"] != "판정 불가(실패 슛)" or str(s.get("result", "")).upper() in ("MADE", "성공"):
            continue
        m = margin_mid(s)
        if m is None or (model.n2 + model.n3) == 0:
            continue
        try:
            sig = float(s.get("margin_sigma_m") or 0)
        except ValueError:
            sig = 0.0
        d = model.decide(m, sig)
        if d is not None:
            s["court_points_guess"] = s.get("points", "")
            s["points"] = d
            s["points_source"] = "학습 기준"
            if d == 3:
                s["zone_kr"] = three_zone_by_direction(float(s["court_x"]), float(s["court_y"]), side=s.get("side"))
            else:
                s["zone_kr"] = "2점"
    for s in out:
        if s["points_source"] == "판정 불가(실패 슛)" and str(s.get("result", "")).upper() in ("MADE", "성공"):
            s["points_source"] = "판정 불가(점수판 짝 없음)"
        k = s["points_source"]
        stats["점수판 보정" if k == "점수판" else "판정 불가" if k.startswith("판정 불가") else k] += 1
    return out, stats
