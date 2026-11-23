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
