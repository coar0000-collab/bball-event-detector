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
    c = classify_shot(R + FIBA.three_radius + 0.1, M, side="left")
    assert c.points == 3 and c.near_line
    c = classify_shot(R + FIBA.three_radius - 0.1, M, side="left")
    assert c.points == 2 and c.near_line


def test_nba_line_is_deeper():
    x = R + 7.0
    assert classify_shot(x, M, FIBA, "left").points == 3
    assert classify_shot(x, M, NBA, "left").points == 2


def test_landmarks_symmetric():
    lm = landmarks()
    assert lm["R_ft_bot"] == pytest.approx((FIBA.length - FIBA.key_length, M - FIBA.key_width / 2))
    assert len([k for k in lm if k.startswith("R_")]) == len([k for k in lm if k.startswith("L_")])
