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
