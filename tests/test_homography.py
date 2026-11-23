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


def _broadcast(t, mirror=False):
    """합성 중계 한 장면 (bench 와 같은 카메라). mirror=True 면 반대쪽 골대 쪽 화면(가운데선 대칭으로 패닝)."""
    import cv2

    from gtt import bench

    top, T = bench._court_texture(0)
    Hc = bench._camera(t, top.shape[1::-1])
    if mirror:  # 화면 가운데를 코트 반대쪽으로 옮긴 카메라
        S = np.array([[1, 0, 2 * 640 - 2 * HG.court_to_image(Hc @ T, 14, 7.5)[0]], [0, 1, 0], [0, 0, 1]], float)
        Hc = S @ Hc
    frame = cv2.warpPerspective(top, Hc, (1280, 720), borderValue=(40, 40, 40))
    return frame, Hc @ T


def test_tracker_cut_to_other_basket_needs_anchor():
    """실제 중계 126초처럼 반대쪽 골대로 확 넘어간 뒤: 기준 화면이 없으면 엉뚱하게 맞추지 않고, 있으면 정확히 다시 찾는다."""
    from gtt import cli

    fa, Ha = _broadcast(3.0)
    fb, Hb = _broadcast(3.0, mirror=True)
    fb2, Hb2 = _broadcast(3.5, mirror=True)
    pts = np.array([[5.8, 5.05], [22.2, 9.95], [14, 7.5], [26.4, 7.5]])

    def err(H, Ht):
        return float(np.linalg.norm(HG.project(H, pts) - HG.project(Ht, pts), axis=1).mean())

    # 기준 화면 없이: 다시 찾기 실패는 괜찮지만, 틀린 H 를 내면 안 된다
    trk = HG.CourtTracker(Ha, fa, FIBA)
    r = trk.update(fb2)
    if r.H is not None:
        assert err(r.H, Hb2) < 3
    # 반대쪽 골대 화면을 기준(앵커)으로 넣으면 다시 찾는다
    trk = HG.CourtTracker(Ha, fa, FIBA)
    trk.add_anchor(fb, Hb)
    r = trk.update(fb2)
    if r.H is None:
        Hr, sc = trk.relocalize(fb2)
        assert Hr is not None and trk.last_inliers >= 80
        r = HG.TrackResult(Hr, sc, "recovered")
    assert err(r.H, Hb2) < 3


def _one_line_scene():
    """줌인해서 흰 선(사이드라인) 하나만 보이는 화면. 나머지 라인은 지운다."""
    import cv2

    frame, Htrue = make_scene()
    mask = np.zeros(frame.shape[:2], np.uint8)
    seg = HG.project(Htrue, HG.densify([(0, 15), (14, 15)], 0.2)).astype(np.int32)
    cv2.polylines(mask, [seg], False, 255, 9)
    return np.where(mask[..., None] > 0, frame, np.full_like(frame, 90)), Htrue


@pytest.mark.parametrize("offset", [(7, -6), (-12, 8)])
def test_refine_does_not_collapse_onto_one_line(offset):
    """실제 중계에서 선 하나만 보일 때 코트 전체가 그 선으로 쪼그라든 H 가 일치율 0.9~1.0 으로 통과했다 (예전 코드: 370~480 px 틀림)."""
    one, Htrue = _one_line_scene()
    Hbad = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], float) @ Htrue
    Hr, score, (n_vis, n_good, _) = HG.refine_homography(one, Hbad, FIBA, return_support=True)
    assert 0.8 <= HG.key_area(Hr) / HG.key_area(Hbad) <= 1.25
    pts = np.array([[5.8, 5.05], [5.8, 9.95], [14, 0], [8.3, 7.5]])
    assert np.linalg.norm(HG.project(Hr, pts) - HG.project(Htrue, pts), axis=1).mean() < 20
    # 보이는 라인 줄 대부분이 안 맞으므로 추적기는 좌표를 내지 않는다
    assert n_vis >= 10 and n_good < 0.6 * n_vis


def test_tracker_does_not_emit_on_one_line_frame():
    f0, H0 = make_scene()
    one, _ = _one_line_scene()
    trk = HG.CourtTracker(H0, f0, FIBA)
    r = trk.update(one)
    assert r.H is None and r.status in ("coast", "cut")


def test_line_support_full_scene():
    frame, Htrue = make_scene()
    _, score, (n_vis, n_good, cross) = HG.refine_homography(frame, Htrue, FIBA, return_support=True)
    assert score > 0.8 and n_good >= 0.9 * n_vis and cross == 1


def _parallel_lines_scene():
    """먼쪽 사이드라인과 그와 평행한 다른 종목 선 두 줄만 보이는 화면 (실제 영상 31s·87s 처럼 키가 화면 밖)."""
    import cv2

    frame, Htrue = make_scene()
    mask = np.zeros(frame.shape[:2], np.uint8)
    for y in (0.0, 1.2):
        seg = HG.project(Htrue, HG.densify([(0, y), (28, y)], 0.2)).astype(np.int32)
        cv2.polylines(mask, [seg], False, 255, 7)
    return np.where(mask[..., None] > 0, np.full_like(frame, 235), np.full_like(frame, 90)), Htrue


def test_parallel_lines_only_is_not_supported():
    """평행한 가로선만 맞으면 선 방향으로 미끄러진 H 도 똑같이 맞는다 → 좌표를 내면 안 된다."""
    frame, Htrue = _parallel_lines_scene()
    slid = Htrue @ np.array([[1, 0, 3.0], [0, 1, 0], [0, 0, 1]])  # 사이드라인 방향으로 3m 밀린 틀린 H
    dist = cv2_dist(frame)
    for H in (Htrue, slid):
        n_vis, n_good, cross = HG.line_support(dist, H, FIBA)
        assert cross == 0
    trk = HG.CourtTracker(Htrue, make_scene()[0], FIBA)
    assert not trk.supported(HG.line_support(dist, slid, FIBA))


def cv2_dist(frame):
    import cv2

    return cv2.distanceTransform((HG.line_mask(frame) == 0).astype(np.uint8), cv2.DIST_L2, 5)


def test_sane_rejects_collapsed_court():
    """코트가 흰 선 하나로 찌그러진 H(실제 영상: 키 넓이 0~60px²)는 캘리브레이션과 비교해 거른다. 줌은 통과."""
    frame, Htrue = make_scene()
    trk = HG.CourtTracker(Htrue, frame, FIBA)
    assert trk.sane(Htrue, frame.shape)
    zoom = np.array([[2.0, 0, -640], [0, 2.0, -360], [0, 0, 1]]) @ Htrue  # 화면 가운데로 2배 줌
    assert trk.sane(zoom, frame.shape)
    squash = np.array([[1.0, 0, 0], [0, 0.02, 360 * 0.98], [0, 0, 1]]) @ Htrue  # 세로로 납작 → 거의 한 줄
    assert not trk.sane(squash, frame.shape)
    assert HG.view_aspect(squash, frame.shape) < 0.05 < 0.5 < HG.view_aspect(Htrue, frame.shape)
    shrink = np.array([[0.2, 0, 512], [0, 0.2, 288], [0, 0, 1]]) @ Htrue  # 점 하나로 쪼그라듦(넓이 1/25)
    assert not trk.sane(shrink, frame.shape)
    assert not trk.sane(np.full((3, 3), np.nan), frame.shape)
