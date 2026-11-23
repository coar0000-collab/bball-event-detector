import cv2
import numpy as np
import pytest

from gtt import homography as HG
from gtt.court import FIBA, classify_shot, landmarks


def make_scene(shift=(0.0, 0.0), seed=0):
    """중계 카메라처럼 비스듬히 본 합성 코트 영상과 정답 H(코트→화면)."""
    top, T = HG.render_topdown(FIBA, px_per_m=40)
    rng = np.random.default_rng(seed)
    # ORB 가 잡을 수 있도록 관중석·바닥 무늬 같은 질감을 넣는다 (코트와 함께 움직임).
    tex = rng.integers(0, 60, top.shape[:2], dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (5, 5), 0)
    top = cv2.add(top, cv2.merge([tex, tex, tex]))
    HG.draw_court(top, T, FIBA, (255, 255, 255), 3)
    h, w = top.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [1130, 120], [1400, 700], [-120, 700]])
    dst += np.float32(shift)
    Hcam = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(top, Hcam, (1280, 720), borderValue=(40, 40, 40))
    return frame, Hcam @ T


def test_fit_from_landmarks_recovers_court_coords():
    frame, Htrue = make_scene()
    lm = landmarks(FIBA)
    names = ["L_baseline_bot", "L_baseline_top", "L_ft_bot", "L_ft_top", "mid_sideline_bot", "mid_sideline_top", "L_corner3_arc_top"]
    rng = np.random.default_rng(1)
    pts = {n: tuple(HG.court_to_image(Htrue, *lm[n]) + rng.normal(0, 1.5, 2)) for n in names}
    H, err = HG.fit_homography(pts, FIBA)
    assert err < 5
    for x, y in [(8.0, 7.5), (2.0, 1.0), (10.0, 12.0)]:
        u, v = HG.court_to_image(Htrue, x, y)
        xe, ye = HG.image_to_court(H, u, v)
        assert np.hypot(xe - x, ye - y) < 0.2


def test_needs_four_points():
    with pytest.raises(ValueError):
        HG.fit_homography({"L_baseline_bot": (0, 0), "L_baseline_top": (1, 1), "center": (2, 2)})


@pytest.mark.parametrize("offset", [(7, -6), (-12, 8)])
def test_refine_snaps_to_lines(offset):
    frame, Htrue = make_scene()
    # 선수들이 라인 일부를 가린 상황
    rng = np.random.default_rng(3)
    for _ in range(10):
        x, y = rng.integers(100, 1150), rng.integers(150, 600)
        cv2.rectangle(frame, (int(x), int(y)), (int(x) + 30, int(y) + 90), (30, 30, 30), -1)
    Hbad = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], float) @ Htrue
    Href, score = HG.refine_homography(frame, Hbad, FIBA)
    pts = np.array([[5.8, 5.05], [5.8, 9.95], [14, 0], [0, 15], [8.3, 7.5]])
    e_ref = np.linalg.norm(HG.project(Href, pts) - HG.project(Htrue, pts), axis=1).mean()
    assert score > 0.8
    assert e_ref < 1.5


def test_tracker_follows_pan():
    f0, H0 = make_scene()
    trk = HG.CourtTracker(H0, f0, FIBA)
    for k in range(1, 6):
        fk, Hk = make_scene(shift=(-18.0 * k, 4.0 * k))
        r = trk.update(fk)
        assert r.status == "ok"
    pts = np.array([[5.8, 5.05], [10, 3], [14, 7.5]])
    err = np.linalg.norm(HG.project(r.H, pts) - HG.project(Hk, pts), axis=1).mean()
    assert err < 4


def test_tracker_detects_cut():
    f0, H0 = make_scene()
    trk = HG.CourtTracker(H0, f0, FIBA)
    blank = np.full_like(f0, 30)
    assert trk.update(blank).status == "cut"


def test_shot_from_pixels():
    frame, Htrue = make_scene()
    u, v = HG.court_to_image(Htrue, 1.575 + 7.4, 7.5)
    x, y = HG.image_to_court(Htrue, u, v)
    assert classify_shot(x, y, side="left").points == 3
