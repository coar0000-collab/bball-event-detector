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


def test_bridge_gap_fills_only_when_both_ways_agree():
    """좌표가 빈 짧은 구간: 앞뒤 정상 화면에서 화면 이동으로 따라와 서로 맞으면 채우고, 뒤 H 가 틀렸으면 채우지 않는다."""
    import cv2

    scenes = [make_scene(shift=(-15.0 * k, 3.0 * k)) for k in range(6)]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f, _ in scenes]
    Hs = [H for _, H in scenes]
    filled = HG.bridge_gap(grays, Hs[0], Hs[-1], FIBA)
    assert len(filled) == 4 and all(H is not None for H in filled)
    for H, Ht in zip(filled, Hs[1:-1]):
        assert HG.h_distance(H, Ht, grays[0].shape) < 2
    wrong_end = np.array([[1, 0, 12.0], [0, 1, -8.0], [0, 0, 1]]) @ Hs[-1]
    assert all(H is None for H in HG.bridge_gap(grays, Hs[0], wrong_end, FIBA))


def test_click_snaps_to_painted_three_line():
    """코트 전체 H 가 3점 라인 근처에서 몇 px 어긋나 있어도, 발 근처에 보이는 칠해진 라인에 맞춰 판정한다.
    발끝이 라인 띠 바로 뒤(3점)인데 H 가 라인을 바깥으로 6px 밀어 계산하면 2점으로 나오던 경우."""
    frame, Htrue = make_scene()
    R0 = 1.575
    # 탑에서 림 쪽 발끝이 라인 바깥 가장자리 뒤 0.06m → 두 발 가운데는 그보다 foot_reach 뒤
    x = R0 + FIBA.three_radius + 0.06 + FIBA.foot_reach
    u, v = HG.court_to_image(Htrue, x, 7.5)
    good = HG.classify_click(Htrue, u, v, FIBA, side="left")
    assert good.points == 3
    shift = np.array([[1, 0, 6.0], [0, 1, 0], [0, 0, 1]]) @ Htrue  # 화면에서 오른쪽(탑에서는 라인 바깥쪽)으로 6px 어긋난 H
    assert HG.classify_click(shift, u, v, FIBA, side="left").points == 2     # 보정 없이: 틀림
    snapped = HG.classify_click(shift, u, v, FIBA, side="left", frame=frame)
    assert snapped.points == 3 and snapped.line_snap_px is not None and abs(snapped.line_snap_px + 6) < 1.5
    # 라인이 다 가려지면(수비수) 보정하지 않고 표시만
    import cv2

    # 발 근처(110px)가 가려져도 더 먼 라인 조각(150px 까지)으로 보정한다. 아크를 따라 바깥 방향이 돌아가도 발 자리 값으로 맞춘다
    part = frame.copy()
    cv2.circle(part, (int(u), int(v)), 110, (30, 30, 30), -1)
    c = HG.classify_click(shift, u, v, FIBA, side="left", frame=part)
    # (이 합성 화면은 3점 라인을 띠 가운데가 아니라 6.75m 에 칠해 1px 바깥에 있다 → 먼 조각에서는 그 차이가 조금 커진다)
    assert c.line_snap_px is not None and abs(c.line_snap_px + 6) < 2.0 and c.points == 3
    hidden = frame.copy()
    cv2.circle(hidden, (int(u), int(v)), 170, (30, 30, 30), -1)
    c = HG.classify_click(shift, u, v, FIBA, side="left", frame=hidden)
    assert c.line_snap_px is None


def _bug(frame, x0=420, y0=640, w=440, h=60, digits="47  52"):
    """화면 아래 스코어버그: 어두운 상자 + 흰 테두리 + 흰 숫자 (라인처럼 보이는 흰 가는 획)."""
    cv2 = __import__("cv2")
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (30, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 2)
    cv2.putText(frame, digits, (x0 + 30, y0 + 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    return frame


def test_graphics_mask_finds_static_overlay_only():
    """카메라가 움직이는 화면들에서 그대로인 스코어버그만 그래픽으로 잡는다 (코트 라인은 아님)."""
    frames = [_bug(make_scene(shift=(dx, dy), seed=0)[0]) for dx, dy in [(0, 0), (60, 10), (-80, 20), (120, -10), (-30, 30), (90, 15)]]
    g = HG.graphics_mask(frames)
    assert g is not None
    assert g[650:700, 430:850].mean() > 0.9   # 스코어버그
    assert g[:600].mean() < 0.01               # 코트
    # 카메라가 전혀 안 움직이면 판단하지 않는다
    assert HG.graphics_mask([frames[0]] * 6) is None


def test_refine_ignores_graphics_and_flags_blocked_line():
    """스코어버그 흰 글자는 라인 정합에 쓰지 않고, 그 뒤에 가려진 3점 라인 근처 슛은 사람 판정(line_hidden)으로 넘긴다."""
    frame, Htrue = make_scene()
    # 오른쪽 아래 코너 쪽 3점 라인이 지나가는 자리에 스코어버그를 얹는다
    from gtt.court import FIBA as F

    x, y = 1.575 + F.three_radius * np.cos(np.radians(60)), 7.5 + F.three_radius * np.sin(np.radians(60))
    u, v = HG.court_to_image(Htrue, x, y)
    x0, y0 = int(u) - 170, int(v) - 160  # 발 근처(150px 안) 라인을 다 덮는 큰 그래픽
    fb = _bug(frame.copy(), x0=x0, y0=y0, w=340, h=320)
    ignore = np.zeros(frame.shape[:2], bool)
    ignore[max(0, y0 - 6):y0 + 326, max(0, x0 - 6):x0 + 346] = True
    # 맞는 H 에서 시작하면 그래픽을 빼고 맞춰도 그대로, 일치율은 가려진 점을 세지 않아 더 높다
    H1, s_ign = HG.refine_homography(fb, Htrue, F, ignore=ignore)
    _, s_all = HG.refine_homography(fb, Htrue, F)
    assert HG.h_distance(H1, Htrue, frame.shape) < 1.5 and s_ign >= s_all
    # 발이 라인 바로 안쪽(2점), 라인은 스코어버그 뒤 → 계산만 믿지 않고 사람 판정
    xi, yi = 1.575 + (F.three_radius - 0.4) * np.cos(np.radians(60)), 7.5 + (F.three_radius - 0.4) * np.sin(np.radians(60))
    uf, vf = HG.court_to_image(Htrue, xi, yi)
    c = HG.classify_click(Htrue, uf, vf, F, side="left", frame=fb, ignore=ignore)
    assert c.line_snap_px is None and c.line_hidden and c.near_line
    # 그래픽이 없는 화면이면 라인을 직접 보고 확실하게 판정
    c2 = HG.classify_click(Htrue, uf, vf, F, side="left", frame=frame, ignore=None)
    assert not c2.line_hidden and c2.points == 2


def test_relocalize_tries_anchor_when_best_keyframe_is_wrong():
    """특징점이 가장 많이 맞는 키프레임의 H 가 틀렸어도, 기준 화면(앵커)에서 옮겨 온 후보로 다시 찾는다
    (실제 영상 195~206s: 왼쪽 기준 화면에서 맞추면 0.81 인데 계속 lost)."""
    f0, H0 = make_scene()
    f1, H1 = make_scene(shift=(40, 15))
    trk = HG.CourtTracker(H0, f0, FIBA, use_static_mask=False)
    # 지금 화면과 똑같은 화면을 '틀린 H(옆으로 45px)' 로 키프레임에 넣어 둔다 → 특징점은 이쪽이 훨씬 많이 맞는다
    bad = np.array([[1, 0, 45.0], [0, 1, 0], [0, 0, 1]]) @ H1
    trk._add_keyframe(cv2_gray(f1), bad)
    H, sc = trk.relocalize(f1)
    assert H is not None and sc >= 0.8
    assert HG.h_distance(H, H1, f1.shape) < 2.0


def cv2_gray(f):
    import cv2

    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def test_wide_snap_that_flips_call_goes_to_human():
    """코트 변환이 18px 어긋나 넓힌 창(±28px)에서 라인을 찾아 2점→3점으로 바꿨다면, 다른 선을 잡았을 수도 있으니 사람 판정.
    (실제 영상 골밑 슛에서 17~23px 보정이 나옴.) 판정이 안 바뀌는 큰 보정이나 ±12px 안 보정은 그대로 믿는다."""
    frame, Ht = make_scene()
    F = FIBA

    def call(dx, marg):
        Hb = np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1.0]]) @ Ht
        u, v = HG.court_to_image(Ht, 1.575 + F.three_radius + marg + F.foot_reach, 7.5)
        return HG.classify_click(Hb, u, v, F, side="left", frame=frame)

    c = call(18, 0.3)
    assert c.points == 3 and abs(c.line_snap_px) > 12 and c.near_line
    c = call(14, 0.5)  # 큰 보정이지만 판정은 그대로 3점
    assert c.points == 3 and not c.near_line
    c = call(8, 0.5)  # 작은 보정
    assert c.points == 3 and abs(c.line_snap_px) <= 12 and not c.near_line


def _plain_scene(shift, seed=0):
    """make_scene 과 같은 바닥 무늬·카메라인데 코트 라인이 하나도 안 보이는 화면 (라인 정합으로는 확인 불가)."""
    import cv2

    top, T = HG.render_topdown(FIBA, px_per_m=40)
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 60, top.shape[:2], dtype=np.uint8), (5, 5), 0)
    top = cv2.add(np.full_like(top, top[3, 3]), cv2.merge([tex, tex, tex]))
    h, w = top.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [1130, 120], [1400, 700], [-120, 700]]) + np.float32(shift)
    Hcam = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(top, Hcam, (1280, 720), borderValue=(40, 40, 40)), Hcam @ T


def test_consensus_when_lines_not_visible():
    """라인이 안 보여 라인 정합으로 확인 못 하는 화면도, 서로 다른 시각의 기준 화면 둘이 특징점만으로 같은 자리를 가리키면 받는다.
    하나뿐이거나 둘이 다른 자리를 가리키면(한쪽 H 가 틀림) 받지 않는다."""
    f0, H0 = make_scene()
    f1, H1 = make_scene(shift=(-60, 10))
    f2, H2 = _plain_scene((-30, 5))
    trk = HG.CourtTracker(H0, f0, FIBA, use_static_mask=False)
    _, sc = trk.relocalize(f2)
    assert sc < 0.65 and trk.consensus(f2.shape) is None  # 기준 화면 하나뿐
    trk.n = 20
    trk._add_keyframe(cv2_gray(f1), H1)
    trk.relocalize(f2)
    Hc = trk.consensus(f2.shape)
    assert Hc is not None and HG.h_distance(Hc, H2, f2.shape) < 1.0
    # 같은 화면을 틀린 H(옆으로 45px)로 저장한 키프레임: 둘이 엇갈리면 받지 않는다
    trk2 = HG.CourtTracker(H0, f0, FIBA, use_static_mask=False)
    trk2.n = 20
    trk2._add_keyframe(cv2_gray(f1), np.array([[1, 0, 45.0], [0, 1, 0], [0, 0, 1]]) @ H1)
    trk2.relocalize(f2)
    assert trk2.consensus(f2.shape) is None


def test_consensus_rejects_when_visible_lines_disagree():
    """기준 화면 둘이 같은 틀린 자리를 가리켜도(둘 다 같은 틀린 H 로 저장됨), 화면에 흰 선이 보이는데 그 선과 크게 어긋나면
    consensus 로 받지 않는다 (실제 B-LAB holdout 528·529s: 코트가 비틀린 H 를 받아 레이업을 3점으로 판정)."""
    f0, H0 = make_scene()
    f1, H1 = make_scene(shift=(-60, 10))
    f3, H3 = make_scene(shift=(-30, 5))
    wrong = np.array([[0.9, 0.0, 1.8], [0.0, 0.88, 1.2], [0, 0, 1]])  # 코트 좌표에서 줄이고 옮긴 틀린 대응 (두 화면 모두 같게)
    trk = HG.CourtTracker(H0, f0, FIBA, use_static_mask=False)
    trk.keyframes, trk.anchors = [], []
    for n, (f, H) in enumerate([(f1, H1 @ wrong), (f0, H0 @ wrong)]):
        trk.n = 20 * (n + 1)
        trk._add_keyframe(cv2_gray(f), H)
    trk.relocalize(f3)
    assert trk.consensus(f3.shape) is not None  # 선을 안 보면 둘이 같은 자리라 받아 버린다
    assert trk.consensus(f3.shape, frame=f3) is None and trk.consensus_rejected == 1
    assert not HG.lines_contradict(f3, H3, FIBA)  # 맞는 H 는 통과
    f2, _ = _plain_scene((-30, 5))
    assert not HG.lines_contradict(f2, H3, FIBA)  # 흰 선이 없는 화면은 판단하지 않음
