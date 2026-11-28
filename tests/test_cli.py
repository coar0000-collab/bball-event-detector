"""합성 영상으로 명령줄 도구 전체 흐름을 확인한다 (창을 띄우지 않는 명령만)."""
import csv
import json

import cv2
import numpy as np
import pytest

from gtt import cli
from gtt import homography as HG
from gtt import scorebug as SB
from gtt.court import FIBA, landmarks
from gtt.video import Video

from test_homography import make_scene

FPS = 10


def _draw_bug(frame, home, away):
    cv2.rectangle(frame, (440, 650), (840, 700), (40, 20, 10), -1)
    cv2.putText(frame, f"{home:d}", (480, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{away:d}", (720, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    video = d / "game.avi"
    # 0~4초 10:8, 4~8초 12:8 (홈 2점), 8~12초 12:11 (원정 3점). 카메라는 천천히 오른쪽으로 팬.
    scores = [(10, 8)] * 40 + [(12, 8)] * 40 + [(12, 11)] * 40
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (1280, 720))
    Hs = []
    for i, (h, a) in enumerate(scores):
        frame, H = make_scene(shift=(-1.5 * i, 0.0))
        _draw_bug(frame, h, a)
        w.write(frame)
        Hs.append(H)
    w.release()
    cfg = SB.ScorebugConfig(rois={"home": [460, 655, 110, 42], "away": [700, 655, 110, 42]}, reference_size=(1280, 720),
                            home_name="KU", away_name="YU")
    cfg.save(d / "scorebug.json")
    # 템플릿: 0~9 가 다 나오도록 등록
    t = SB.DigitTemplates()
    for val in ["10", "23", "45", "67", "89"]:
        img = np.zeros((720, 1280, 3), np.uint8)
        _draw_bug(img, int(val), int(val))
        assert SB.learn(t, cfg.crop(img, "home"), val)
    t.save(d / "digits.npz")
    # 캘리브레이션: 첫 프레임의 랜드마크 화면 좌표 (사람이 클릭한 것처럼 약간 흔들어서)
    lm = landmarks(FIBA)
    rng = np.random.default_rng(0)
    names = ["L_baseline_bot", "L_baseline_top", "L_ft_bot", "L_ft_top", "mid_sideline_bot", "mid_sideline_top"]
    pts = {n: [float(c) for c in np.array(HG.court_to_image(Hs[0], *lm[n])) + rng.normal(0, 1.0, 2)] for n in names}
    H0, _ = HG.fit_homography(pts, FIBA)
    (d / "calib.json").write_text(json.dumps({"spec": "FIBA", "t_sec": 0.0, "image_points": pts, "H_court_to_image": H0.tolist()}))
    return d, video, Hs


def test_scorebug_read_cli(synthetic):
    d, video, _ = synthetic
    out = d / "events.csv"
    cli.main(["scorebug-read", "--video", str(video), "--config", str(d / "scorebug.json"), "--templates", str(d / "digits.npz"),
              "--step", "0.5", "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert [(r["team_name"], r["delta"], r["flag"]) for r in rows] == [("KU", "2", "ok"), ("YU", "3", "ok")]
    assert abs(float(rows[0]["t_sec"]) - 4.0) <= 0.5


def test_track_and_shot_cli(synthetic, capsys):
    d, video, Hs = synthetic
    track = d / "track.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    z = np.load(track)
    assert all(s in ("calib", "ok", "recovered") for s in z["status"])
    # 10초 시점에서 탑 3점 위치 (림에서 7.4m) 픽셀을 찍으면 3점이어야 한다
    i = 10 * FPS
    u, v = HG.court_to_image(Hs[i], 1.575 + 7.4, 7.5)
    capsys.readouterr()
    cli.main(["shot", "--track", str(track), "--t", "10", "--u", str(u), "--v", str(v), "--side", "left"])
    res = json.loads(capsys.readouterr().out)
    assert res["points"] == 3 and res["zone_kr"] == "탑"
    assert abs(res["court_x"] - (1.575 + 7.4)) < 0.3
    u, v = HG.court_to_image(Hs[i], 1.575 + 4.0, 9.0)
    cli.main(["shot", "--track", str(track), "--t", "10", "--u", str(u), "--v", str(v)])
    assert json.loads(capsys.readouterr().out)["points"] == 2


def test_track_start_differs_from_calib(synthetic):
    """캘리브레이션은 6초 화면에서 했는데 0초부터 추적: 6초의 H 를 0초 화면에 그대로 붙이면 90 px 어긋난다.
    기준 화면(6초)과 맞춰 0초 화면의 H 를 찾아야 한다 (실제 영상 --start 0, calib t=120 에서 생긴 문제)."""
    d, video, Hs = synthetic
    c = json.loads((d / "calib.json").read_text())
    i6 = 6 * FPS
    c6 = {"spec": "FIBA", "t_sec": 6.0, "H_court_to_image": Hs[i6].tolist()}
    (d / "calib6.json").write_text(json.dumps(c6))
    track = d / "track6.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib6.json"), "--start", "0", "--end", "8",
              "--step", "0.5", "--out", str(track)])
    z = np.load(track)
    pts = np.array([[5.8, 5.05], [5.8, 9.95], [14, 7.5]])
    for t, H, s in zip(z["times"], z["H"], z["status"]):
        if not np.all(np.isfinite(H)):
            continue
        Ht = Hs[int(round(t * FPS))]
        assert np.linalg.norm(HG.project(H, pts) - HG.project(Ht, pts), axis=1).mean() < 3, (t, s)
    assert z["status"][0] == "recovered"
    assert sum(np.all(np.isfinite(H)) for H in z["H"]) >= len(z["H"]) - 1


def test_extra_calib_inserted_at_its_time(synthetic):
    """샘플 간격(0.5초)에 없는 시각(6.3초)에 맞춘 캘리브레이션도 그 시각 그대로 결과에 들어간다 (실제 영상 204.5s·309.5s)."""
    d, video, Hs = synthetic
    (d / "calib63.json").write_text(json.dumps({"spec": "FIBA", "t_sec": 6.3, "H_court_to_image": Hs[63].tolist()}))
    (d / "calib_no_t.json").write_text(json.dumps({"spec": "FIBA", "H_court_to_image": Hs[70].tolist()}))
    track = d / "track63.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--calib", str(d / "calib63.json"),
              "--calib", str(d / "calib_no_t.json"), "--end", "8", "--step", "0.5", "--out", str(track)])
    z = np.load(track)
    i = int(np.argmin(np.abs(z["times"] - 6.3)))
    assert abs(z["times"][i] - 6.3) < 1e-6 and z["status"][i] == "calib"
    assert np.allclose(z["H"][i], Hs[63])
    assert list(z["times"]) == sorted(z["times"])
    assert sum(s == "calib" for s in z["status"]) == 2  # 시각 없는 파일은 빠진다


def test_court_lookup_skips_missing_sample(tmp_path):
    """가장 가까운 샘플이 좌표 없음(coast)이면 tol 안의 다음 가까운 샘플을 쓴다 (실제 영상 309.5s)."""
    nan = np.full((3, 3), np.nan)
    track = tmp_path / "t.npz"
    np.savez(track, times=np.array([309.2, 309.4, 309.6]), H=np.stack([np.eye(3) * 2, nan, np.eye(3) * 3]),
             line_score=np.array([0.9, 0.48, 0.9]), status=np.array(["ok", "coast", "ok"]))
    look = cli._CourtLookup(None, str(track))
    H, dt = look.nearest(309.45)
    assert H[0, 0] == 3 and abs(dt - 0.15) < 1e-9
    assert look.at(309.45, tol=0.1) is None


def test_overlay_cli(synthetic):
    d, video, _ = synthetic
    out = d / "ov.png"
    cli.main(["overlay", "--video", str(video), "--t", "0", "--calib", str(d / "calib.json"), "--out", str(out)])
    assert cv2.imread(str(out)) is not None


def test_demo_runs_end_to_end(tmp_path):
    from gtt.demo import run
    res = run(tmp_path / "demo", seconds=20)
    assert res["점수 변화"].startswith("정답 2개 중 2개 맞음")  # 6초 +2, 14초 +3
    assert all(s["맞음"] for s in res["슛 판정 상세"])
    assert (tmp_path / "demo" / "overlay_grid.png").exists()


def test_lens_fit_and_track_with_lens(tmp_path, capsys):
    """렌즈가 휜 영상: lens-fit 으로 k1 을 찾고, track-court --lens 로 편 화면에서 추적하고, 원본 화면에서 찍은 발 위치를
    shot 이 편 화면으로 옮겨 판정한다. 화면 왼쪽 아래 가장자리, 라인 바로 바깥(+0.06m) 슛: 휜 화면에 맞춘 H 만으로는
    (라인 보정을 못 하면) -0.02m 로 2점이 나오지만, 휨을 펴면 맞는다."""
    from test_lens import distorted_scene

    from gtt import homography as HG
    from gtt.court import FIBA
    from gtt.lens import Lens

    k1 = -0.12
    video = tmp_path / "bent.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
    truth = []
    for i in range(30):  # 3초, 천천히 오른쪽으로 팬
        raw, Hideal, lens, Hc = distorted_scene(k1, shift=(-3.0 * i, 0.0))
        cv2.rectangle(raw, (500, 660), (780, 712), (30, 20, 20), -1)  # 스코어버그 (움직이지 않음)
        cv2.putText(raw, "47 52", (540, 700), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        w.write(raw)
        truth.append((Hideal, Hc))
    w.release()
    # 사람이 원본(휜) 화면에서 맞춘 캘리브레이션 = 휨을 무시한 H 를 라인에 맞춘 것
    raw0 = Video(str(video)).frame_at(0.0)
    H = truth[0][0]
    for _ in range(4):
        H, _ = HG.refine_homography(raw0, H, FIBA)
    (tmp_path / "calib.json").write_text(json.dumps({"spec": "FIBA", "t_sec": 0.0, "H_court_to_image": H.tolist()}))
    lens_json = tmp_path / "lens.json"
    cli.main(["lens-fit", "--video", str(video), "--calib", str(tmp_path / "calib.json"), "--out", str(lens_json)])
    got = Lens.load(lens_json)
    assert abs(got.k1 - k1) <= 0.021, got
    track = tmp_path / "t.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(tmp_path / "calib.json"), "--lens", str(lens_json),
              "--step", "0.5", "--out", str(track)])
    z = np.load(track)
    look = cli._CourtLookup(None, str(track))
    assert look.lens is not None and abs(look.lens.k1 - got.k1) < 1e-9
    good = [i for i, H in enumerate(z["H"]) if np.all(np.isfinite(H))]
    assert len(good) >= 5
    # 편 화면 정답 H 와 비교 (정답은 k1 로 편 화면, 찾은 k1 과 조금 다를 수 있어 원본 화면 점으로 비교)
    for i in good:
        t = float(z["times"][i])
        Hideal = truth[int(round(t * 10))][0]
        for x, y in [(1.575 + 6.0, 7.5), (3.0, 2.0), (8.0, 13.0)]:
            u_true, v_true = Lens(k1, 1280, 720).distort_pts([HG.court_to_image(Hideal, x, y) + np.array([Lens(k1, 1280, 720).pad] * 2)])[0]
            uc, vc = HG.court_to_image(z["H"][i], x, y)
            ur, vr = look.to_raw(uc, vc)
            assert np.hypot(ur - u_true, vr - v_true) < 3.0, (t, x, y)
    # 화면 아래쪽 45도 3점 라인 바로 바깥(림 쪽 발끝 +0.12m)에서 쏜 슛: 원본 화면에서 찍은 두 발 가운데
    lens_true = Lens(k1, 1280, 720)
    th = np.radians(75)
    d = FIBA.three_radius + 0.06 + FIBA.foot_reach
    x, y = 1.575 + d * np.cos(th), 7.5 + d * np.sin(th)
    u, v = lens_true.distort_pts([np.array(HG.court_to_image(truth[0][0], x, y)) + lens_true.pad])[0]
    assert HG.classify_click(H, u, v, FIBA, side="left").points == 2  # 휨 무시: 틀림
    capsys.readouterr()
    cli.main(["shot", "--track", str(track), "--t", "0", "--u", str(u), "--v", str(v), "--side", "left"])
    out = json.loads(capsys.readouterr().out)
    assert out["points"] == 3 and abs(out["margin_m"] - 0.06) < 0.04
    cli.main(["shot", "--track", str(track), "--t", "0", "--u", str(u), "--v", str(v), "--video", str(video), "--side", "left"])
    out = json.loads(capsys.readouterr().out)
    assert out["points"] == 3 and abs(out["margin_m"] - 0.06) < 0.04


def _punch_holes(src, dst, lo, hi):
    """추적 결과에서 lo~hi 초 샘플을 '좌표 없음(coast)'으로 바꾼 사본 (실제 중계: 스코어버그 바로 위 슛 장면에서 3~4초씩 놓침)."""
    z = dict(np.load(src))
    gap = (z["times"] > lo) & (z["times"] < hi)
    H = z["H"].copy()
    H[gap] = np.nan
    st = z["status"].astype("<U10")
    st[gap] = "coast"
    z["H"], z["status"] = H, st
    np.savez(dst, **z)


def test_shot_bridges_missing_frames_from_both_sides(synthetic, capsys, tmp_path):
    """슛 프레임 앞뒤 3초가 좌표 없음이어도, 앞(4초)·뒤(8초) 정상 화면에서 각각 화면 이동으로 따라온 두 H 가 발 자리에서
    만나면 그것으로 판정한다. 영상이 없으면(이동을 못 따라가면) 예전처럼 좌표 없음."""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    holes = tmp_path / "holes.npz"
    _punch_holes(track, holes, 4.4, 7.6)
    i = 60  # 6.0초
    capsys.readouterr()
    for x, y, pts in ((1.575 + 7.4, 7.5, 3), (1.575 + 4.0, 9.0, 2)):
        u, v = HG.court_to_image(Hs[i], x, y)
        cli.main(["shot", "--track", str(holes), "--video", str(video), "--t", "6.0", "--u", str(u), "--v", str(v),
                  "--side", "left"])
        res = json.loads(capsys.readouterr().out)
        assert res["h_source"] == "two-way" and res["points"] == pts, res
        assert np.hypot(res["court_x"] - x, res["court_y"] - y) < 0.15
        assert res["h_agree_m"] < 0.1
    with pytest.raises(SystemExit):
        cli.main(["shot", "--track", str(holes), "--t", "6.0", "--u", str(u), "--v", str(v)])
    # 샘플 사이 프레임(6.3초)도 가까운 샘플 H 를 그대로 쓰지 않고 화면 이동만큼 옮겨서 쓴다
    u, v = HG.court_to_image(Hs[63], 1.575 + 7.4, 7.5)
    cli.main(["shot", "--track", str(track), "--video", str(video), "--t", "6.3", "--u", str(u), "--v", str(v), "--side", "left"])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] in ("two-way", "near") and abs(res["court_x"] - (1.575 + 7.4)) < 0.1


def test_shot_inside_cut_has_no_court(tmp_path, capsys):
    """슛 시각이 화면 전환(리플레이·광고) 안이면 앞뒤에서 이어 올 수 없다 → 지어내지 않고 좌표 없음."""
    video = tmp_path / "cut.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (1280, 720))
    rng = np.random.default_rng(3)
    Hs = []
    for i in range(80):
        frame, H = make_scene(shift=(-1.5 * i, 0.0))
        if 40 <= i < 50:  # 4~5초: 전혀 다른 화면
            frame = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (7, 7), 0)
        w.write(frame)
        Hs.append(H)
    w.release()
    (tmp_path / "calib.json").write_text(json.dumps({"spec": "FIBA", "t_sec": 0.0, "H_court_to_image": Hs[0].tolist()}))
    track = tmp_path / "t.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(tmp_path / "calib.json"), "--step", "0.5", "--out", str(track)])
    u, v = HG.court_to_image(Hs[45], 1.575 + 7.4, 7.5)
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli.main(["shot", "--track", str(track), "--video", str(video), "--t", "4.5", "--u", str(u), "--v", str(v)])
    assert "none" in str(e.value)


def test_rejudge_cli(synthetic, tmp_path, capsys):
    """click-shots CSV 를 다시 판정: 자유투·사람 라인 판정은 지키고, 좌표 없던 슛도 이어 와서 판정한다."""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    holes = tmp_path / "holes.npz"
    _punch_holes(track, holes, 4.4, 7.6)
    rows = []
    for t, x, y, zone, line_call in ((3.0, 1.575 + 7.4, 7.5, "", ""), (6.0, 1.575 + 4.0, 9.0, "", ""),
                                     (6.5, 5.8 + 0.3, 7.5, "FT", ""), (10.0, 1.575 + 7.4, 7.5, "", "on")):
        u, v = HG.court_to_image(Hs[int(round(t * FPS))], x, y)
        rows.append({"t_sec": t, "team": "home", "result": "MADE", "u": round(u, 1), "v": round(v, 1), "zone": zone,
                     "points": "", "line_call": line_call, "court_x": "99", "h_source": "old"})  # 예전 판정 값
    src = tmp_path / "shots.csv"
    cli._write_csv(str(src), rows)
    out = tmp_path / "re.csv"
    cli.main(["rejudge", "--shots", str(src), "--track", str(holes), "--video", str(video), "--side", "left", "--out", str(out)])
    got = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert [g["points"] for g in got] == ["3", "2", "1", "2"]
    assert [g["court_points"] for g in got] == ["3", "2", "2", "3"]  # 사람 판정·자유투 전 코트만의 점수
    assert got[1]["h_source"] == "two-way" and got[0]["h_source"] in ("two-way", "track")
    assert all(g["court_x"] != "99" for g in got)  # 예전 좌표가 남지 않는다
    tr = np.load(track)
    assert tr["H_pred"].shape == tr["H"].shape  # 진단용 예측 위치도 샘플마다 저장


def test_graphics_box_with_calib_flags_hidden_line(tmp_path, capsys):
    """캘리브레이션(--calib)으로 판정할 때도 --graphics-box 로 준 스코어버그 자리는 라인으로 쓰지 않는다.
    실제 B-LAB 447s: 추적 결과가 아니라 캘리브레이션으로 판정하니 그래픽 자리가 비어, 가려진 3점 라인 대신 스코어버그 윗변을
    라인으로 잡아(+1.9px) 2점으로 확정했다. 이제 라인이 가려진 것으로 보고 사람 판정(near_line)으로 넘긴다."""
    from test_homography import _bug

    from gtt.court import FIBA as F

    frame, Htrue = make_scene()
    x, y = 1.575 + F.three_radius * np.cos(np.radians(60)), 7.5 + F.three_radius * np.sin(np.radians(60))
    u, v = HG.court_to_image(Htrue, x, y)
    x0, y0 = int(u) - 170, int(v) - 160
    fb = _bug(frame.copy(), x0=x0, y0=y0, w=340, h=320)
    cv2.line(fb, (x0, y0), (x0 + 340, y0), (255, 255, 255), 2)  # 그래픽 윗변의 흰 테두리
    video = tmp_path / "still.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (1280, 720))
    for _ in range(20):
        w.write(fb)
    w.release()
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"spec": "FIBA", "t_sec": 0.0, "H_court_to_image": Htrue.tolist()}))
    xi, yi = 1.575 + (F.three_radius - 0.4) * np.cos(np.radians(60)), 7.5 + (F.three_radius - 0.4) * np.sin(np.radians(60))
    uf, vf = HG.court_to_image(Htrue, xi, yi)
    box = tmp_path / "graphics.json"
    box.write_text(json.dumps({"boxes": [[(x0 - 6) / 2, (y0 - 6) / 2, 352 / 2, 332 / 2]], "reference_size": [640, 360]}))
    capsys.readouterr()
    cli.main(["shot", "--calib", str(calib), "--video", str(video), "--t", "1.0", "--u", str(uf), "--v", str(vf),
              "--side", "left", "--graphics-box", str(box)])
    res = json.loads(capsys.readouterr().out)
    assert res["line_hidden"] and res["near_line"] and res["line_snap_px"] == ""
    m = cli._graphics_boxes([str(box), "0,0,10,10"], (720, 1280))
    assert m[y0, x0] and m[5, 5] and not m[y0 - 20, x0 - 20]


def test_shot_multi_view_when_no_good_sample_nearby(synthetic, capsys, tmp_path):
    """앞뒤 정상 샘플이 이어 오기 창(--window) 밖에 있으면(실제 B-LAB: 20~80초씩 정상 샘플 없음), 앞뒤 2분 안의 정상 화면들과
    직접 맞춰 서로 다른 시각의 화면 여러 장이 같은 답을 낼 때만 쓴다."""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    holes = tmp_path / "holes.npz"
    _punch_holes(track, holes, 3.9, 9.1)
    i = 65
    u, v = HG.court_to_image(Hs[i], 1.575 + 7.4, 7.5)
    capsys.readouterr()
    cli.main(["shot", "--track", str(holes), "--video", str(video), "--t", "6.5", "--u", str(u), "--v", str(v),
              "--side", "left", "--window", "2"])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] == "multi-view" and res["h_votes"] >= 2, res
    assert res["points"] == 3 and abs(res["court_x"] - (1.575 + 7.4)) < 0.15


def test_track_around_shots_only(synthetic, tmp_path):
    """--around: 경기 전체 대신 슛 앞뒤 창만 추적해 npz 하나로 합친다 (실제 KBA 92분 전체 추적이 하루 가까이 걸림)."""
    d, video, Hs = synthetic
    shots = tmp_path / "s.csv"
    cli._write_csv(str(shots), [{"t_sec": 3.0, "u": 1, "v": 1}, {"t_sec": 9.5, "u": 1, "v": 1}, {"t_sec": 3.5, "u": "", "v": ""}])
    assert cli._around_windows(str(shots), 1.5, 1.0) == [(1.5, 4.5), (8.0, 10.5)]  # 겹친 창은 합침
    out = tmp_path / "around.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5",
              "--around", str(shots), "--before", "1.5", "--after", "1.0", "--out", str(out)])
    z = np.load(out)
    t = z["times"]
    assert np.all(((t >= 1.5) & (t <= 4.5)) | ((t >= 8.0) & (t <= 10.5)) | (t == 0.0))
    assert np.all(np.diff(t) > 0) and z["H_pred"].shape == z["H"].shape
    st = z["status"].astype(str)
    good = np.isin(st, cli._GOOD) & (t >= 8.0)
    assert good.sum() >= 3
    i = int(np.flatnonzero(good)[0])
    Htrue = Hs[int(round(t[i] * FPS))]
    assert HG.h_distance(z["H"][i], Htrue, (720, 1280)) < 4.0


def test_gym_spec_json(tmp_path):
    """체육관별 규격 JSON: 칠한 3점 라인이 FIBA 와 다른 체육관(옛 6.25m, 두 줄 등)은 그 라인으로 판정한다."""
    from gtt import court as C

    p = tmp_path / "gym.json"
    p.write_text(json.dumps({"base": "FIBA", "three_radius": 7.0}), encoding="utf-8")
    sp = cli._spec(str(p))
    assert sp.name == "gym" and sp.three_radius == 7.0 and sp.key_width == C.FIBA.key_width
    x, y = sp.rim_from_baseline + 6.9, sp.width / 2  # 탑, 림에서 6.9m
    assert C.classify_shot(x, y, C.FIBA, side="left").points == 3
    assert C.classify_shot(x, y, sp, side="left").points == 2
    assert cli._spec("fiba") is C.FIBA
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"three_radiu": 7.0}), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli._spec(str(bad))


def test_calib_edit_saves(synthetic, tmp_path, monkeypatch):
    """calib-edit: 창에서 S 를 누른 것처럼 저장하면 H·원본 화면 좌표 image_points 가 들어가고, 덮어쓸 때 처음 한 번 백업한다."""
    from gtt import calib_edit as CE

    d, video, Hs = synthetic
    calib = tmp_path / "c.json"
    calib.write_text((d / "calib.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(CE, "run", lambda frame, H, spec, save, *a, **k: save(H) or True)
    cli.main(["calib-edit", "--video", str(video), "--calib", str(calib)])
    o = json.loads(calib.read_text(encoding="utf-8"))
    assert o["edited_with"] == "calib-edit" and len(o["image_points"]) >= 4 and "lens" not in o
    assert (tmp_path / "c.bak.json").exists()
    H2, _ = HG.fit_homography({k: tuple(v) for k, v in o["image_points"].items()}, FIBA)
    assert HG.h_distance(H2, np.array(o["H_court_to_image"]), (720, 1280)) < 0.5


def test_shot_skips_h_the_camera_cannot_make(synthetic, capsys, tmp_path):
    """슛 프레임의 추적 H 가 코트를 5m 옮긴 틀린 H(consensus)면, 캘리브레이션 카메라 자리에서는 나올 수 없는 화면이라 버리고
    앞뒤 정상 화면에서 이어 온다 (버린 이유는 h_reject). 맞는 추적 샘플은 모두 카메라 자리와 몇 px 안."""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    z = dict(np.load(track))
    assert len(z["cam_C"]) == 1 and tuple(z["frame_shape"]) == (720, 1280)
    assert np.nanmax(z["cam_px"]) < 8
    i = int(np.argmin(np.abs(z["times"] - 6.0)))
    H = z["H"].copy()
    H[i] = H[i] @ np.array([[1, 0, 5.0], [0, 1, 0], [0, 0, 1]])
    st = z["status"].astype("<U10")
    st[i] = "consensus"
    z["H"], z["status"] = H, st
    bad = tmp_path / "bad.npz"
    np.savez(bad, **z)
    x, y = 1.575 + 7.4, 7.5
    u, v = HG.court_to_image(Hs[60], x, y)
    capsys.readouterr()
    cli.main(["shot", "--track", str(bad), "--video", str(video), "--t", "6.0", "--u", str(u), "--v", str(v), "--side", "left",
              "--max-cam-px", "8"])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] != "track" and res["points"] == 3 and res["h_reject"].startswith("track:카메라"), res
    assert np.hypot(res["court_x"] - x, res["court_y"] - y) < 0.15 and res["h_cam_px"] < 8
    # 진단: 카메라 자리 검사 전에 만든 npz 처럼 cam_px 가 없어도 다시 재서 틀린 샘플 구간을 보여 준다
    z.pop("cam_px"), z.pop("cam_C"), z.pop("frame_shape")
    np.savez(bad, **z)
    cli.main(["cam-check", "--track", str(bad), "--video", str(video), "--out", str(tmp_path / "cam.csv")])
    out = capsys.readouterr()
    assert "넘는 구간 1개" in out.out and "consensus" in out.out and "카메라 자리 (캘리브레이션 1)" in out.err


def test_shot_skips_h_that_puts_foot_in_other_half(synthetic, capsys, tmp_path):
    """공격 방향(--side)을 알 때, 슛 프레임 H 가 발을 하프라인 너머(반대쪽 골대 근처)로 옮기면 그 H 는 버리고 앞뒤에서 이어 온다.
    (실제 B-LAB holdout 383·414·447·529s: 틀린 H 로 '탑 3점, 라인까지 11~18m' — 발이 반대쪽 림 근처였다)
    공격 방향을 모르면 가까운 림으로 판정하므로 이 검사는 하지 않는다."""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    z = dict(np.load(track))
    i = int(np.argmin(np.abs(z["times"] - 6.0)))
    H = z["H"].copy()
    H[i] = H[i] @ np.array([[1, 0, -14.0], [0, 1, 0], [0, 0, 1]])  # 발이 14m 오른쪽(반대쪽 골대 3m 앞)으로 옮겨지는 틀린 H
    z["H"] = H
    bad = tmp_path / "bad.npz"
    np.savez(bad, **z)
    x, y = 1.575 + 7.4, 7.5
    u, v = HG.court_to_image(Hs[60], x, y)
    capsys.readouterr()
    cli.main(["shot", "--track", str(bad), "--video", str(video), "--t", "6.0", "--u", str(u), "--v", str(v), "--side", "left"])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] != "track" and res["points"] == 3 and res["h_reject"].startswith("track:반대쪽코트"), res
    assert np.hypot(res["court_x"] - x, res["court_y"] - y) < 0.15
    cli.main(["shot", "--track", str(bad), "--video", str(video), "--t", "6.0", "--u", str(u), "--v", str(v)])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] == "track" and res["side"] == "right" and res["points"] == 2  # 방향을 모르면 못 거름
    # 진단: 그 샘플만 흰 선과 어긋난다. 사이드라인·하프라인↔베이스라인은 옮겨도 겹쳐 줄 60% 가 '맞음' 이지만, 키·아크처럼
    # 자리가 하나뿐인 줄은 1/3 만 맞는다 → consensus 였다면 버림 (화면 가운데가 보는 코트 x 도 14m 옮겨짐)
    out_csv = tmp_path / "lines.csv"
    cli.main(["line-check", "--track", str(bad), "--video", str(video), "--status", "", "--start", "5.4", "--end", "6.6",
              "--out", str(out_csv)])
    with open(out_csv, encoding="utf-8-sig") as f:
        rows = {round(float(r["t_sec"]), 1): r for r in csv.DictReader(f)}
    assert rows[6.0]["contradict"] == "1" and float(rows[6.0]["distinct_frac"]) < 0.5
    assert rows[5.5]["contradict"] == rows[6.5]["contradict"] == "0" and float(rows[5.5]["distinct_frac"]) > 0.9
    assert float(rows[6.0]["center_x"]) - float(rows[5.5]["center_x"]) > 10


def test_shot_drops_far_court_h_even_if_lines_fit(synthetic, capsys, tmp_path):
    """코트는 180° 돌려도 흰 선이 똑같아서 키·아크까지 흰 선과 다 맞는 H 라도, 공격 방향(--side)으로 보아 발이 반대쪽 코트로
    가면 버리고 앞뒤에서 이어 온다. 돌려 쓰지 않는다.
    (실제 B-LAB holdout 383·414·447·529s: 그런 H 는 쌍둥이가 아니라 팬·줌이 틀린 H 였다 — 돌리면 529s 레이업이 6m)"""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    z = dict(np.load(track))
    i = int(np.argmin(np.abs(z["times"] - 6.0)))
    H = z["H"].copy()
    H[i] = H[i] @ np.array([[-1.0, 0, 28.0], [0, -1.0, 15.0], [0, 0, 1]])  # 흰 선은 다 맞는 180° 쌍둥이
    z["H"] = H
    bad = tmp_path / "twin.npz"
    np.savez(bad, **z)
    capsys.readouterr()
    for x, y, pts in ((1.575 + 7.4, 7.5, 3), (1.575 + 4.0, 9.0, 2)):
        u, v = HG.court_to_image(Hs[60], x, y)
        cli.main(["shot", "--track", str(bad), "--video", str(video), "--t", "6.0", "--u", str(u), "--v", str(v),
                  "--side", "left"])
        res = json.loads(capsys.readouterr().out)
        assert res["h_source"] not in ("track", "none") and "+" not in res["h_source"], res
        assert res["points"] == pts and res["h_reject"].startswith("track:반대쪽코트"), res
        assert np.hypot(res["court_x"] - x, res["court_y"] - y) < 0.15


def test_shot_one_way_when_shot_frame_is_blurry(synthetic, capsys, tmp_path, monkeypatch):
    """슛 뒤로는 좌표가 없고(놓침) 슛 순간 화면은 흐려 라인 정합이 안 되며 다른 화면들과도 못 맞출 때, 1초 전 정상 샘플에서 화면
    이동으로 따라온 H 로 그린 키·아크가 슛 화면 흰 선과 맞으면 쓴다 ("one-way"). 이어 온 시간만큼 라인 근처 기준이 넓어진다.
    (실제 KBA D201 181·529.6·1038.5s: 1~2초 전까지 정상, 슛 직전 팬·줌에서 놓침)"""
    d, video, Hs = synthetic
    track = tmp_path / "tr.npz"
    cli.main(["track-court", "--video", str(video), "--calib", str(d / "calib.json"), "--step", "0.5", "--out", str(track)])
    holes = tmp_path / "holes.npz"
    _punch_holes(track, holes, 5.2, 30.0)
    real_refine = HG.refine_homography
    monkeypatch.setattr(HG, "refine_homography",
                        lambda frame, H, *a, **k: (H, 0.3, (6, 1, 0)) if k.get("return_support") else (H, 0.3))
    monkeypatch.setattr(cli, "_multi_view", lambda *a, **k: (None, a[9]))
    x, y = 1.575 + 7.4, 7.5
    u, v = HG.court_to_image(Hs[62], x, y)
    capsys.readouterr()
    cli.main(["shot", "--track", str(holes), "--video", str(video), "--t", "6.2", "--u", str(u), "--v", str(v), "--side", "left"])
    res = json.loads(capsys.readouterr().out)
    assert res["h_source"] == "one-way" and res["points"] == 3 and abs(res["h_gap_s"] - 1.2) < 0.01, res
    assert np.hypot(res["court_x"] - x, res["court_y"] - y) < 0.15
    assert res["margin_sigma_m"] >= 0.15 * 1.2 - 1e-6
    # 3초(--one-way-s) 넘게 떨어지면 쓰지 않는다 (여기선 4.7초)
    _punch_holes(track, holes, 1.9, 30.0)
    with pytest.raises(SystemExit) as e:
        cli.main(["shot", "--track", str(holes), "--video", str(video), "--t", "6.2", "--u", str(u), "--v", str(v),
                  "--side", "left"])
    assert "none" in str(e.value)
    monkeypatch.setattr(HG, "refine_homography", real_refine)
