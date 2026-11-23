"""`gtt demo`: 실제 중계 영상 없이 전체 흐름을 한 번에 돌려 보는 점검용 데모.

합성 중계 영상(팬·줌 카메라, 움직이는 선수, 점수가 바뀌는 스코어버그)을 만들고,
사람이 창에서 하던 단계(스코어버그 영역 지정, 숫자 등록, 코트 점 찍기)는 정답 좌표로 대신 채운 뒤
나머지 명령(scorebug-read, track-court, overlay, shot)을 실제 명령줄과 똑같이 실행한다.
정답과 비교한 정확도와 단계별 걸린 시간, 40분 경기 기준 예상 시간을 요약한다.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import homography as HG
from . import scorebug as SB
from .court import FIBA, landmarks

W, H_IMG, FPS = 1280, 720, 10
BUG_ROIS = {"home": [460, 655, 110, 42], "away": [700, 655, 110, 42]}
# (초, 팀, 점수) — 자유투 2개는 6초 간격 (1.5초보다 가까우면 +2 한 번으로 합쳐진다)
SCORE_PLAN = [(6, "home", 2), (14, "away", 3), (21, "home", 1), (27, "home", 1), (33, "away", 2), (40, "home", 3), (49, "away", 2)]
START = (10, 8)


def _draw_bug(frame, home: int, away: int) -> None:
    cv2.rectangle(frame, (440, 650), (840, 700), (40, 20, 10), -1)
    cv2.putText(frame, f"{home:d}", (480, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{away:d}", (720, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


def _court_texture(seed: int = 0):
    top, T = HG.render_topdown(FIBA, px_per_m=40)
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 60, top.shape[:2], dtype=np.uint8), (5, 5), 0)
    top = cv2.add(top, cv2.merge([tex, tex, tex]))
    HG.draw_court(top, T, FIBA, (255, 255, 255), 3)
    return top, T


def _camera(t: float, size) -> np.ndarray:
    """t초의 카메라: 좌우로 천천히 팬하고 살짝 줌 (탑다운 그림 → 화면)."""
    w, h = size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [1130, 120], [1400, 700], [-120, 700]])
    zoom = 1.0 + 0.12 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 40))
    dst = (dst - [640, 400]) * zoom + [640, 400] + [-140 * math.sin(2 * math.pi * t / 30), 6 * math.sin(2 * math.pi * t / 17)]
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def make_video(path: Path, seconds: float = 60.0, start: tuple[int, int] = START) -> list[np.ndarray]:
    """합성 중계 영상을 쓰고, 프레임별 정답 H(코트 m → 화면 px)를 돌려준다."""
    top, T = _court_texture()
    rng = np.random.default_rng(7)
    players = [(rng.uniform(1, 27), rng.uniform(1, 14), rng.uniform(-1.5, 1.5), rng.uniform(-1, 1)) for _ in range(10)]
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H_IMG))
    if not w.isOpened():
        raise RuntimeError(f"영상 파일을 쓸 수 없습니다: {path}")
    truth = []
    home, away = start
    plan = list(SCORE_PLAN)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        while plan and plan[0][0] <= t:
            _, team, d = plan.pop(0)
            home, away = (home + d, away) if team == "home" else (home, away + d)
        Hc = _camera(t, top.shape[1::-1])
        frame = cv2.warpPerspective(top, Hc, (W, H_IMG), borderValue=(40, 40, 40))
        Ht = Hc @ T
        for x0, y0, vx, vy in players:  # 선수: 코트 위를 움직이는 어두운 사각형 (라인을 가림)
            x = 1 + (x0 + vx * t - 1) % 26
            y = 1 + (y0 + vy * t - 1) % 13
            u, v = HG.court_to_image(Ht, x, y)
            cv2.rectangle(frame, (int(u) - 14, int(v) - 70), (int(u) + 14, int(v)), (35, 30, 30), -1)
        _draw_bug(frame, home, away)
        w.write(frame)
        truth.append(Ht)
    w.release()
    return truth


def _truth_events(seconds: float) -> list[tuple[float, str, int]]:
    return [(float(t), team, d) for t, team, d in SCORE_PLAN if t < seconds - 2]


def build_scorebug_profile(out: Path) -> None:
    """scorebug-setup / scorebug-learn 창에서 사람이 하던 일(영역 지정, 숫자 등록)을 정답으로 대신한다."""
    cfg = SB.ScorebugConfig(rois=BUG_ROIS, reference_size=(W, H_IMG), home_name="HOME", away_name="AWAY")
    cfg.save(out / "scorebug.json")
    tpl = SB.DigitTemplates()
    for val in ["10", "23", "45", "67", "89"]:
        img = np.zeros((H_IMG, W, 3), np.uint8)
        _draw_bug(img, int(val), int(val))
        SB.learn(tpl, cfg.crop(img, "home"), val)
    tpl.save(out / "digits.npz")


def build_calib(video: Path, truth: list[np.ndarray], path: Path) -> None:
    """calibrate 창에서 코트 점을 찍는 일을 대신한다: 정답 위치에 손 떨림 정도의 오차를 더하고, --refine 처럼 흰 라인에 맞춘다."""
    lm = landmarks(FIBA)
    rng = np.random.default_rng(0)
    names = ["L_baseline_bot", "L_baseline_top", "L_ft_bot", "L_ft_top", "mid_sideline_bot", "mid_sideline_top", "L_corner3_arc_top"]
    pts = {n: [float(c) for c in np.array(HG.court_to_image(truth[0], *lm[n])) + rng.normal(0, 1.5, 2)] for n in names}
    H0, _ = HG.fit_homography(pts, FIBA)
    from .video import Video
    v = Video(video)
    H0, _ = HG.refine_homography(v.frame_at(0.0), H0, FIBA)
    v.close()
    path.write_text(json.dumps({"spec": "FIBA", "t_sec": 0.0, "image_points": pts, "H_court_to_image": H0.tolist()},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def run(out_dir: str | Path, seconds: float = 60.0) -> dict:
    from . import cli

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    res: dict = {"영상 길이(초)": seconds}

    def step(name, fn):
        t0 = time.perf_counter()
        r = fn()
        timing[name] = round(time.perf_counter() - t0, 2)
        return r

    video = out / "demo_game.avi"
    truth = step("합성 영상 만들기", lambda: make_video(video, seconds))

    build_scorebug_profile(out)

    ev_csv = out / "score_events.csv"
    step("scorebug-read", lambda: cli.main(["scorebug-read", "--video", str(video), "--config", str(out / "scorebug.json"),
                                            "--templates", str(out / "digits.npz"), "--step", "0.5", "--out", str(ev_csv)]))
    with open(ev_csv, encoding="utf-8-sig") as f:
        events = list(csv.DictReader(f))
    want = _truth_events(seconds)
    got = [(float(e["t_sec"]), e["team"], int(e["delta"])) for e in events]
    hit = sum(any(team == gt and d == gd and abs(t - tt) <= 1.0 for t, team, d in got) for tt, gt, gd in want)
    res["점수 변화"] = f"정답 {len(want)}개 중 {hit}개 맞음, 읽은 이벤트 {len(got)}개"

    build_calib(video, truth, out / "calib.json")

    track = out / "court_track.npz"
    step("track-court", lambda: cli.main(["track-court", "--video", str(video), "--calib", str(out / "calib.json"),
                                          "--step", "0.5", "--out", str(track)]))
    z = np.load(track)
    probe = np.array([[5.8, 5.05], [5.8, 9.95], [14, 0], [14, 15], [8.3, 7.5], [22.2, 7.5]])
    errs = []
    for t, Hk in zip(z["times"], z["H"]):
        i = min(int(round(t * FPS)), len(truth) - 1)
        errs.append(float(np.linalg.norm(HG.project(Hk, probe) - HG.project(truth[i], probe), axis=1).mean()))
    statuses = {s: int((z["status"] == s).sum()) for s in set(z["status"].tolist())}
    res["코트 추적"] = f"평균 오차 {np.mean(errs):.1f}px, 최대 {np.max(errs):.1f}px ({len(errs)}프레임, 상태 {statuses})"

    # 3) 오버레이 4장 → 한 장으로
    tiles = []
    for t in (0, seconds * 0.3, seconds * 0.6, seconds * 0.9):
        p = out / f"overlay_{int(t):02d}s.png"
        step(f"overlay {int(t)}s", lambda p=p, t=t: cli.main(["overlay", "--video", str(video), "--t", str(t), "--track", str(track), "--out", str(p)]))
        img = cv2.imread(str(p))
        cv2.putText(img, f"t={int(t)}s", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360)))
    cv2.imwrite(str(out / "overlay_grid.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))

    # 4) 슛 위치 판정 (원래는 click-shots 창에서 슈터 발을 클릭)
    rim = 1.575
    cases = [("코너 3점", (1.0, 0.5), 3, "코너"), ("45도 3점", (rim + 7.2 * math.cos(math.pi / 4), 7.5 + 7.2 * math.sin(math.pi / 4)), 3, "45도"),
             ("탑 3점", (rim + 7.3, 7.5), 3, "탑"), ("미드레인지 2점", (rim + 4.5, 9.5), 2, "2점"), ("3점 라인 바로 안쪽", (rim + 6.5, 7.5), 2, "2점")]
    t_shot = seconds * 0.6
    i = int(t_shot * FPS)
    shots = []
    for label, (x, y), want_pts, want_zone in cases:
        u, v = HG.court_to_image(truth[i], x, y)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["shot", "--track", str(track), "--t", str(t_shot), "--u", f"{u:.1f}", "--v", f"{v:.1f}", "--side", "left"])
        r = json.loads(buf.getvalue())
        shots.append({"상황": label, "정답": f"{want_pts}점 {want_zone}", "판정": f"{r['points']}점 {r['zone_kr']}",
                      "맞음": r["points"] == want_pts and r["zone_kr"] == want_zone, "3점 라인까지(m)": r.get("margin")})
    (out / "shots.json").write_text(json.dumps(shots, ensure_ascii=False, indent=1), encoding="utf-8")
    res["슛 판정"] = f"{sum(s['맞음'] for s in shots)}/{len(shots)} 맞음"

    # 시간: 영상 1초당 → 40분(2400초) 경기 예상
    per_sec = {k: v / seconds for k, v in timing.items() if k in ("scorebug-read", "track-court")}
    res["걸린 시간(초)"] = timing
    res["40분 경기 예상(분)"] = {k: round(v * 2400 / 60, 1) for k, v in per_sec.items()}
    res["슛 판정 상세"] = shots
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
