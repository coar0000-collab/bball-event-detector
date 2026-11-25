import cv2
import numpy as np

from gtt import homography as HG
from gtt.court import FIBA
from gtt.lens import Lens, LensVideo, convert_h, fit_k1, fit_on_canvas


def distorted_scene(k1, shift=(0.0, 0.0), seed=0, thick=3):
    """렌즈가 휜(배럴) 중계 화면과, 휨을 편 화면(canvas)에서의 정답 H."""
    top, T = HG.render_topdown(FIBA, px_per_m=40)
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 60, top.shape[:2], dtype=np.uint8), (5, 5), 0)
    top = cv2.add(top, cv2.merge([tex, tex, tex]))
    HG.draw_court(top, T, FIBA, (255, 255, 255), thick)
    h, w = top.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    from test_homography import _DST0, pan_h

    Hcam = pan_h(shift) @ cv2.getPerspectiveTransform(src, _DST0)  # 카메라를 제자리에서 돌린 화면
    lens = Lens(k1, 1280, 720)
    P = np.array([[1, 0, lens.pad], [0, 1, lens.pad], [0, 0, 1.0]])
    cw, ch = lens.canvas_size
    ideal = cv2.warpPerspective(top, P @ Hcam, (cw, ch), borderValue=(40, 40, 40))
    xs, ys = np.meshgrid(np.arange(1280.0), np.arange(720.0))
    pu = lens.undistort_pts(np.stack([xs.ravel(), ys.ravel()], 1))
    raw = cv2.remap(ideal, pu[:, 0].reshape(720, 1280).astype(np.float32), pu[:, 1].reshape(720, 1280).astype(np.float32),
                    cv2.INTER_LINEAR)
    return raw, Hcam @ T, lens, P @ Hcam @ T


def test_points_roundtrip_and_identity():
    lens = Lens(-0.15, 1280, 720)
    assert lens.pad > 0
    p = np.array([[0, 0], [1279, 719], [640, 360], [100, 650]], float)
    assert np.abs(lens.distort_pts(lens.undistort_pts(p)) - p).max() < 1e-6
    ident = Lens(0.0, 1280, 720)
    assert ident.is_identity and ident.pad == 0
    f = np.zeros((720, 1280, 3), np.uint8)
    assert ident.undistort(f) is f and ident.invalid_mask() is None


def test_undistort_frame_matches_points():
    """편 화면에서 코트 점의 위치 = 원본 화면 점을 undistort_pts 로 옮긴 곳."""
    raw, _, lens, Hc = distorted_scene(-0.12)
    canvas = lens.undistort(raw)
    assert canvas.shape[:2] == (lens.canvas_size[1], lens.canvas_size[0])
    # 정답 H 로 편 화면에서 맞춘 라인 일치율이 높다 (휨을 제대로 폈다)
    _, sc = HG.refine_homography(canvas, Hc, FIBA, iterations=0, ignore=lens.invalid_mask())
    assert sc > 0.95


def test_fit_k1_recovers_barrel_and_fixes_edges():
    """휜 화면 한 장과 거기에 맞춘 H 로 k1 을 찾고, 편 화면에서는 가장자리 라인까지 맞는다."""
    raw, Hideal, _, _ = distorted_scene(-0.12)
    H = Hideal
    for _ in range(4):
        H, sc_raw = HG.refine_homography(raw, H, FIBA)
    res_raw, _ = HG.line_residual(raw, H, FIBA)
    best, table = fit_k1([(raw, H)], FIBA, ks=[-0.2, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04])
    assert abs(best - (-0.12)) <= 0.011
    _, _, sc, res, _ = fit_on_canvas(raw, H, Lens(best, 1280, 720), FIBA)
    assert sc > 0.95 > sc_raw and res < 0.5 * res_raw


def test_convert_h_and_lens_video():
    raw, _, lens, Hc = distorted_scene(-0.1)
    # 원본 화면 H(휨 없이 맞춘 근사)를 편 화면용으로 옮기면 정답과 몇 px 안
    Hraw = np.linalg.inv(np.array([[1, 0, lens.pad], [0, 1, lens.pad], [0, 0, 1.0]])) @ Hc  # 휨 무시한 근사
    H2 = convert_h(Hraw, lens, FIBA)
    assert HG.h_distance(H2, Hc, (lens.canvas_size[1], lens.canvas_size[0])) < 25

    class V:
        fps, duration, width, height = 30.0, 1.0, 1280, 720

        def frame_at(self, t):
            return raw

        def iterate_between(self, **kw):
            yield 0.0, raw, [raw]

    lv = LensVideo(V(), lens)
    assert lv.frame_at(0).shape[1] == lens.canvas_size[0] and lv.fps == 30.0
    t, f, between = next(lv.iterate_between())
    assert f.shape == between[0].shape == (lens.canvas_size[1], lens.canvas_size[0], 3)
