import math

import pytest

from gtt.court import FIBA, NBA, classify_shot, court_polylines, landmarks, three_point_margin

R = FIBA.rim_from_baseline
M = FIBA.width / 2


def test_top_three():
    c = classify_shot(R + 7.5, M, side="left")
    assert c.points == 3 and c.zone == "3P_TOP" and c.zone_kr == "탑"


def test_midrange_two():
    c = classify_shot(R + 6.0, M + 1.0, side="left")
    assert c.points == 2 and c.zone == "MID"


def test_paint():
    c = classify_shot(R + 1.5, M, side="left")
    assert c.points == 2 and c.zone == "PAINT"


def test_corner_three_and_corner_two():
    assert classify_shot(1.0, 0.5, side="left").zone == "3P_CORNER"
    assert classify_shot(1.0, 1.2, side="left").points == 2
    assert classify_shot(1.0, FIBA.width - 0.5, side="left").zone_kr == "코너"


def test_wing_45():
    a = math.radians(45)
    c = classify_shot(R + 7.2 * math.cos(a), M + 7.2 * math.sin(a), side="left")
    assert c.points == 3 and c.zone_kr == "45도"


def test_right_basket_mirrors_left():
    l = classify_shot(R + 7.0, M + 2.0, side="left")
    r = classify_shot(FIBA.length - (R + 7.0), M + 2.0, side="right")
    assert (l.points, l.zone) == (r.points, r.zone)
    assert l.margin == pytest.approx(r.margin)


def test_nearest_basket_when_side_unknown():
    assert classify_shot(26.0, M).side == "right"
    assert classify_shot(3.0, M).side == "left"


def test_three_line_points_have_zero_margin():
    for side, key in (("left", "L_three"), ("right", "R_three")):
        for x, y in court_polylines(FIBA)[key]:
            if x in (0.0, FIBA.length):
                continue  # 베이스라인 끝점
            assert abs(three_point_margin(x, y, FIBA, side)) < 1e-6


def test_near_line_flag():
    c = classify_shot(R + FIBA.three_radius + 0.1, M, side="left", foot="toe")
    assert c.points == 3 and c.near_line
    c = classify_shot(R + FIBA.three_radius - 0.1, M, side="left", foot="toe")
    assert c.points == 2 and c.near_line


def test_toe_touching_line_band_is_two():
    """3점 라인 띠(바깥 가장자리 6.75m, 폭 5cm)에 발끝이 조금이라도 닿으면 2점 (FIBA: 라인은 2점 구역)."""
    top = R + FIBA.three_radius
    assert classify_shot(top + 0.01, M, side="left", foot="toe").points == 3      # 발끝이 라인 바로 뒤
    c = classify_shot(top - 0.02, M, side="left", foot="toe")                    # 발끝이 라인 띠 위
    assert c.points == 2 and c.on_line
    assert classify_shot(top, M, side="left", foot="toe").points == 2            # 바깥 가장자리에 딱 닿음
    c = classify_shot(top - 0.2, M, side="left", foot="toe")
    assert c.points == 2 and not c.on_line                                       # 라인 안쪽
    # 두 발 가운데를 찍으면 림 쪽 발끝은 foot_reach(0.14m) 앞: 가운데가 라인 뒤 0.1m 여도 발끝은 라인 위 → 2점
    c = classify_shot(top + 0.1, M, side="left")
    assert c.points == 2 and c.on_line and c.margin == pytest.approx(0.1 - FIBA.foot_reach)
    assert classify_shot(top + FIBA.foot_reach + 0.01, M, side="left").points == 3
    # 코너도 같은 규칙: 사이드라인에서 0.9m(라인 바깥 가장자리)
    corner = FIBA.width - FIBA.three_corner_from_sideline
    assert classify_shot(1.0, corner + 0.01, side="left", foot="toe").points == 3
    assert classify_shot(1.0, corner - 0.03, side="left", foot="toe").on_line


def test_three_line_edges():
    from gtt.court import three_line_edges

    e = three_line_edges(FIBA)
    for (x, y) in e["L_outer"][3:-3]:
        assert abs(three_point_margin(x, y, FIBA, "left")) < 1e-6
    for (x, y) in e["L_inner"][3:-3]:
        assert three_point_margin(x, y, FIBA, "left") == pytest.approx(-FIBA.line_width)


def test_nba_line_is_deeper():
    x = R + 7.0
    assert classify_shot(x, M, FIBA, "left").points == 3
    assert classify_shot(x, M, NBA, "left").points == 2


def test_landmarks_symmetric():
    lm = landmarks()
    assert lm["R_ft_bot"] == pytest.approx((FIBA.length - FIBA.key_length, M - FIBA.key_width / 2))
    assert len([k for k in lm if k.startswith("R_")]) == len([k for k in lm if k.startswith("L_")])
