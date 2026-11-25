import numpy as np

from gtt import calib_edit as CE
from gtt import homography as HG
from gtt.court import FIBA, landmarks
from test_homography import make_scene


def _wrong(H):
    return H @ np.array([[1.04, 0.0, -0.4], [0.03, 0.97, 0.3], [0, 0, 1]])  # 코트 좌표에서 조금 틀어진 캘리브레이션


def test_solve_h_points_recovers_h():
    _, H = make_scene()
    lm = landmarks(FIBA)
    pins = [CE.Pin("pt", np.array(lm[n]), HG.project(H, [lm[n]])[0]) for n in
            ("L_baseline_bot", "L_baseline_top", "L_ft_bot", "L_ft_top", "mid_sideline_bot")]
    H2 = CE.solve_h(pins)
    assert HG.h_distance(H2, H, (720, 1280)) < 0.5


def test_drag_lines_onto_true_lines():
    """틀어진 노란 선을 실제 선 위로 끌어 놓으면(곧은 선은 그 선 위 아무 곳으로) 코트 전체가 맞는 자리로 간다."""
    _, Htrue = make_scene()
    H0 = _wrong(Htrue)
    ed = CE.Editor(H0, (720, 1280), FIBA)
    assert HG.h_distance(ed.H, Htrue, (720, 1280)) > 8
    lm = landmarks(FIBA)
    # 모서리 4곳을 끌어 놓기 (잡을 때는 지금 그려진 노란 점 자리, 놓을 때는 실제 자리)
    for n in ("L_baseline_bot", "L_baseline_top", "L_ft_top", "mid_sideline_bot"):
        u0, v0 = HG.project(ed.H, [lm[n]])[0]
        assert ed.start(u0, v0)
        u1, v1 = HG.project(Htrue, [lm[n]])[0]
        ed.drag(u1, v1)
        ed.release()
    assert HG.h_distance(ed.H, Htrue, (720, 1280)) < 1.0
    # 되돌리기·처음으로
    ed.undo()
    assert len(ed.pins) == 3
    ed.reset()
    assert np.allclose(ed.H, ed.H0)


def test_line_drag_moves_line_but_slides_free():
    """곧은 선 중간을 잡아 끌면 '그 선이 그 자리를 지나게' — 선 방향 성분은 묶지 않는다."""
    _, Htrue = make_scene()
    H0 = _wrong(Htrue)
    ed = CE.Editor(H0, (720, 1280), FIBA)
    X = np.array([8.0, 0.0])  # 아래 사이드라인 위 (모서리에서 먼 곳)
    u0, v0 = HG.project(ed.H, [tuple(X)])[0]
    assert ed.start(u0, v0) and ed.active.kind == "line"
    # 실제 사이드라인 위의 다른 자리(선 방향으로 30px 옆)로 놓아도 사이드라인은 실제 선에 붙는다
    a, b = HG.project(Htrue, [(6.0, 0.0), (10.0, 0.0)])
    t = (b - a) / np.linalg.norm(b - a)
    tgt = HG.project(Htrue, [tuple(X)])[0] + 30 * t
    ed.drag(*tgt)
    ed.release()
    p = HG.project(ed.H, [tuple(X)])[0]
    n = np.array([-t[1], t[0]])
    assert abs(float((p - HG.project(Htrue, [tuple(X)])[0]) @ n)) < 1.0  # 선에 수직 방향 오차 1px 안
    assert ed.remove_near(*p) and not ed.pins


def test_render_and_magnifier_shapes():
    f, H = make_scene()
    ed = CE.Editor(H, f.shape[:2], FIBA)
    img = ed.render(f, cursor=(640.0, 360.0))
    assert img.shape == f.shape
    mg = CE.magnifier(img, 5.0, 5.0)
    assert mg.shape == (320, 320, 3)
