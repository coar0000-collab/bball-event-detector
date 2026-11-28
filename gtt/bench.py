"""`gtt bench`: 실제 중계에서 흔한 어려운 장면을 넣은 합성 중계로 정확도를 잰다.

`gtt demo` 보다 거칠게 만든다.
- 빠른 팬과 하프코트까지 들어가는 줌, 팬 속도에 비례한 모션 블러, 밝기 변화, 압축 잡음
- 화면 구석의 방송사 로고(움직이지 않는 그림)
- 리플레이 2번: 다른 각도의 클로즈업으로 화면 전환, 그동안 스코어버그 숨김, 리플레이 중 득점
- 스코어버그 숫자가 잠깐 잘못 나오는 깜빡임, 2초 간격 연속 득점

정답과 비교해 점수 변화 정밀도·재현율, 코트 추적 정확도(본 카메라 프레임), 리플레이에서 잘못 낸 코트 좌표,
리플레이 뒤 복구까지 걸린 시간을 보고한다. 같은 씨앗(seed)이면 같은 영상이 나오므로 개선 전후를 비교할 수 있다.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import homography as HG
from . import scorebug as SB
from .court import FIBA, classify_shot, court_polylines, landmarks
from .demo import BUG_ROIS, FPS, H_IMG, W, _draw_bug

# 리플레이 구간. REPLAY_DELAY 가 있으면 '몇 초 전 장면'을 그 득점 자리를 중심으로 다시 보여 준다(놓친 득점 리플레이).
# 없으면 앞서 지나간 장면의 다른 각도 클로즈업. 리플레이 동안 스코어버그는 숨는다.
REPLAYS = [(17.0, 20.0), (22.0, 24.0), (38.0, 42.5), (70.0, 74.0)]
REPLAY_DELAY = [None, 4.0, None, None]
RIM = 1.575
R3 = 6.75


def _top(margin):  # 탑: 림 정면, 3점 라인에서 margin(m)
    return (RIM + R3 + margin, 0.0)


def _wing(margin, sign=1):  # 45도
    d = R3 + margin
    return (RIM + d * math.cos(math.pi / 4), sign * d * math.sin(math.pi / 4))


# (릴리스 시각, 종류, 공격 림 기준 림 쪽 발끝 위치 (베이스라인에서 거리, 림 중심선에서 옆으로), 성공 여부)
# 발끝이 3점 라인 바깥 가장자리 뒤면 3점, 라인 띠에 닿거나 안쪽이면 2점. 아래 _feet 로 '두 발 가운데'로 바꾼다.
# 라인 바로 안팎(±0.1~0.3m) 슛을 일부러 많이 넣었다. 공격 방향은 그 순간 화면 가운데에 더 가까운 림.
SHOTS = [
    (4.5, "FG", (0.5, -6.85), True),     # 코너 3점 (사이드라인 쪽 0.25m 바깥)
    (11.0, "FG", _wing(0.3), True),      # 45도 3점
    (15.0, "FG", (4.0, 1.0), False),     # 페인트 실패
    (18.5, "FG", (3.0, -1.5), True),     # 리플레이가 나가는 동안 난 골밑 득점: 본 화면에는 안 나오고, 22~24초 리플레이로 다시 나온다
    (24.5, "FG", _top(-0.15), True),     # 3점 라인 바로 안쪽 (2점)
    (29.5, "FG", _top(0.12), True),      # 3점 라인 바로 바깥 (3점)
    (34.0, "FG", _wing(-0.2, -1), True), # 45도 라인 바로 안쪽 (2점)
    (45.5, "FT", (6.2, 0.0), True),      # 자유투 2개
    (50.5, "FT", (6.2, 0.0), True),
    (55.5, "FG", (6.4, 0.4), True),      # 자유투 자리 점프슛 (2점)
    (60.5, "FG", (1.2, 6.6), True),      # 코너 라인 바깥 가장자리에 발끝이 딱 닿음 (2점) — 위치만으로는 못 가르는 경우
    (65.0, "FG", _top(1.2), False),      # 먼 3점 실패
    (78.5, "FG", _wing(0.08, -1), True), # 45도 라인 위 바로 바깥 (3점)
    (84.0, "FG", (5.0, 3.0), True),      # 미드레인지
]


def _feet(p):
    """위 위치는 '림 쪽 발끝' 기준. 사람이 찍는 '두 발 가운데'는 발끝에서 foot_reach 만큼 라인 바깥쪽(림 반대쪽)."""
    d, lat = p
    if d <= FIBA.corner_straight_end:
        return (d, lat + math.copysign(FIBA.foot_reach, lat))
    r = math.hypot(d - RIM, lat)
    return (d + FIBA.foot_reach * (d - RIM) / r, lat + FIBA.foot_reach * lat / r)


SHOTS = [(t, k, _feet(p) if k == "FG" else p, made) for t, k, p, made in SHOTS]
GLITCHES = [(9.0, 9.3), (29.0, 29.3), (51.0, 51.3)]  # 이 동안 홈 점수가 +10 으로 잘못 보임
SCORE_LAG = 1.5  # 슛 → 스코어버그 반영
FAST_PANS = [(0.0, 3.0), (52.0, 54.0), (60.0, 61.5)]  # 속공 따라가는 빠른 팬 (블러 심함)
# 하프코트를 가로지르는 휙 패닝 (실제 B-LAB 중계 126초에서 추적이 깨진 장면): 1초 동안 반대쪽 골대 쪽으로 넘어갔다가 돌아온다
WHIPS = [(52.0, 53.0, 60.0, 61.0)]  # (넘어가기 시작, 다 넘어감, 돌아오기 시작, 다 돌아옴)
START = (31, 27)


def _in(t, spans):
    return any(a <= t < b for a, b in spans)


def _camera(t: float, size) -> np.ndarray:
    w, h = size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[150, 120], [1130, 120], [1400, 700], [-120, 700]])
    zoom = 1.0 + 0.45 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 22))
    cx = 640 + 260 * math.sin(2 * math.pi * t / 14)
    for a, b, c, d in WHIPS:  # 휙 패닝: 화면 가운데를 코트 반대쪽(가운데선 기준 대칭)으로 옮긴다
        w = min(1.0, max(0.0, (t - a) / (b - a))) if t < c else 1.0 - min(1.0, max(0.0, (t - c) / (d - c)))
        if a <= t < d:
            w = 0.5 - 0.5 * math.cos(math.pi * w)  # 부드럽게 가속·감속
            cx = cx + w * (1280 - 2 * cx)
    center = np.float32([cx, 420])
    dst = (dst - center) * zoom + [640, 400] + [0, 10 * math.sin(2 * math.pi * t / 11)]
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _replay_camera(t: float, size, k: int, focus_px=None) -> np.ndarray:
    """리플레이: 다른 위치·각도의 클로즈업. focus_px(탑다운 그림 px)가 있으면 그곳을 화면 가운데 근처에 둔다."""
    w, h = size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = focus_px if focus_px is not None else (w * (0.2 + 0.55 * k), h * 0.5)
    a = math.radians(10 + 5 * k)
    s = 2.4
    pts = []
    for x, y in src:
        x, y = (x - cx) * s, (y - cy) * s * 0.7
        pts.append([x * math.cos(a) - y * math.sin(a) + 640 + 40 * t, x * math.sin(a) + y * math.cos(a) + 360])
    return cv2.getPerspectiveTransform(src, np.float32(pts))


def _logo(frame):
    x0, y0 = 1080, 24
    cv2.rectangle(frame, (x0, y0), (x0 + 170, y0 + 64), (20, 20, 160), -1)
    for i in range(0, 170, 12):
        cv2.line(frame, (x0 + i, y0), (x0 + i + 30, y0 + 64), (60, 60, 220), 3)
    cv2.putText(frame, "B-LIVE", (x0 + 12, y0 + 46), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


def _resolve_shots(seconds: float, size, T) -> list[dict]:
    """슛마다 공격 림(그 순간 화면 가운데에 가까운 쪽)과 팀, 코트 좌표를 정한다. 왼쪽 림 공격 = 홈."""
    out = []
    for t, kind, (db, dy), made in SHOTS:
        if t >= seconds - SCORE_LAG - 1:
            continue
        Ht = _camera(t, size) @ T
        ul = HG.court_to_image(Ht, RIM, 7.5)[0]
        ur = HG.court_to_image(Ht, FIBA.length - RIM, 7.5)[0]
        side = "left" if abs(ul - W / 2) <= abs(ur - W / 2) else "right"
        x = db if side == "left" else FIBA.length - db
        y = 7.5 + (dy if side == "left" else -dy)
        pts = 1 if kind == "FT" else classify_shot(x, y, FIBA, side=side).points
        out.append({"t": t, "kind": kind, "made": made, "side": side, "team": "home" if side == "left" else "away",
                    "x": x, "y": y, "points": pts, "replay": _in(t, REPLAYS)})
    return out


def _replay_focus(k: int, size, T) -> tuple | None:
    """놓친 득점 리플레이가 비출 자리(탑다운 그림 px): 그 리플레이가 보여 주는 시간대에 난 성공 슛 위치."""
    d = REPLAY_DELAY[k]
    if d is None:
        return None
    a, b = REPLAYS[k]
    for s in _resolve_shots(1e9, size, T):
        if s["made"] and a - d <= s["t"] <= b - d:
            return tuple(HG.court_to_image(T, s["x"], s["y"]))
    return None


def _frame_camera(t: float, size, T) -> tuple[np.ndarray, int | None, float]:
    """t초 화면의 카메라(탑다운 그림 → 화면), 리플레이 번호(본 화면이면 None), 화면에 나오는 장면의 시각."""
    k = next((j for j, (a, b) in enumerate(REPLAYS) if a <= t < b), None)
    if k is None:
        return _camera(t, size), None, t
    return _replay_camera(t - REPLAYS[k][0], size, k, _replay_focus(k, size, T)), k, t - (REPLAY_DELAY[k] or 0.0)


def make_video(path: Path, seconds: float = 90.0, seed: int = 0) -> tuple[list, list, list]:
    """어려운 합성 중계를 쓰고 (프레임별 정답 H 또는 None(리플레이), 정답 점수 이벤트, 슛 목록)을 돌려준다."""
    top, T = _court_texture(seed)
    size = top.shape[1::-1]
    shots = _resolve_shots(seconds, size, T)
    # 스코어버그: 성공 슛 SCORE_LAG 초 뒤에 반영. 리플레이 중이면 끝나고 보인다.
    changes = sorted((s["t"] + SCORE_LAG, s["team"], s["points"]) for s in shots if s["made"])
    events = [(float(next((b for a, b in REPLAYS if a <= tc < b), tc)), team, d) for tc, team, d in changes]
    rng = np.random.default_rng(seed + 7)
    players = [(rng.uniform(1, 27), rng.uniform(1, 14), rng.uniform(-2, 2), rng.uniform(-1.2, 1.2)) for _ in range(9)]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H_IMG))
    if not wr.isOpened():
        raise RuntimeError(f"영상 파일을 쓸 수 없습니다: {path}")
    wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 60)
    truth = []
    home, away = START
    plan = list(changes)
    prev_c = None
    for i in range(int(seconds * FPS)):
        t = i / FPS
        while plan and plan[0][0] <= t:
            _, team, d = plan.pop(0)
            home, away = (home + d, away) if team == "home" else (home, away + d)
        Hc, replay, ts = _frame_camera(t, size, T)
        frame = cv2.warpPerspective(top, Hc, (W, H_IMG), borderValue=(40, 40, 40))
        Ht = Hc @ T
        # ts: 화면에 나오는 장면의 시각 (놓친 득점 리플레이면 몇 초 전)
        movers = [(1 + (x0 + vx * ts - 1) % 26, 1 + (y0 + vy * ts - 1) % 13) for x0, y0, vx, vy in players]
        # 슈터: 릴리스 1초 전부터 0.5초 뒤까지 그 자리에 선다
        movers += [(s["x"], s["y"]) for s in shots if s["t"] - 1.0 <= ts <= s["t"] + 0.5]
        for x, y in movers:
            u, v = HG.court_to_image(Ht, x, y)
            hgt = 70 * (2.4 if replay is not None else 1.0)
            cv2.rectangle(frame, (int(u) - int(hgt / 5), int(v - hgt)), (int(u) + int(hgt / 5), int(v)), (35, 30, 30), -1)
        c = HG.court_to_image(Ht, 14.0, 7.5)  # 가로 모션 블러: 1/60초 노출 동안 움직인 만큼 (빠른 팬 구간은 3배)
        if replay is None and prev_c is not None:
            speed = abs(c[0] - prev_c[0]) * FPS  # px/초
            k = int(min(25, speed / 60 * (3.0 if _in(t, FAST_PANS) else 1.0)))
            if k >= 3:
                ker = np.zeros((k, k), np.float32)
                ker[k // 2, :] = 1.0 / k
                frame = cv2.filter2D(frame, -1, ker)
        prev_c = c if replay is None else None
        frame = np.clip(frame.astype(np.float32) * (1 + 0.12 * math.sin(2 * math.pi * t / 9)) + rng.normal(0, 4, frame.shape),
                        0, 255).astype(np.uint8)
        if replay is not None:
            cv2.putText(frame, "REPLAY", (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 220, 255), 4, cv2.LINE_AA)
        else:
            _draw_bug(frame, home + 10 if _in(t, GLITCHES) else home, away)
        _logo(frame)
        wr.write(frame)
        truth.append(None if replay is not None else Ht)
    wr.release()
    return truth, events, shots


def _score_metrics(got, want, tol=1.5):
    used = set()
    tp = 0
    for tt, gt, gd in want:
        for j, (t, team, d) in enumerate(got):
            if j not in used and team == gt and d == gd and abs(t - tt) <= tol:
                used.add(j)
                tp += 1
                break
    fp = [got[j] for j in range(len(got)) if j not in used]
    return tp, fp


def _court_texture(seed: int = 0):
    """합성 코트 바닥 그림. 3점 라인은 실제 규격처럼 바깥 가장자리 6.75m 에서 안쪽으로 5cm 폭 띠로 칠한다
    (다른 라인은 데모와 같이 그린다)."""
    from .court import three_line_edges

    top, T = HG.render_topdown(FIBA, px_per_m=40)
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 60, top.shape[:2], dtype=np.uint8), (5, 5), 0)
    top = cv2.add(top, cv2.merge([tex, tex, tex]))
    for name, poly in court_polylines(FIBA).items():
        if name.endswith("_three"):
            continue
        uv = HG.project(T, HG.densify(poly, 0.25))
        cv2.polylines(top, [np.round(uv).astype(np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA)
    w_px = max(1, int(round(FIBA.line_width * 40)))
    for k in ("L_center", "R_center"):
        uv = HG.project(T, HG.densify(three_line_edges(FIBA)[k], 0.05))
        cv2.polylines(top, [np.round(uv).astype(np.int32)], False, (255, 255, 255), w_px, cv2.LINE_AA)
    return top, T


_texture = lru_cache(maxsize=4)(_court_texture)


def _camera_truth(t: float, seed: int) -> tuple[np.ndarray, bool]:
    """t초 화면의 정답 H(코트 m → 화면 px)와 리플레이 여부. 리플레이도 같은 코트를 다른 각도로 찍은 화면이다."""
    top, T = _texture(seed)
    size = top.shape[1::-1]
    Hc, k, _ = _frame_camera(t, size, T)
    return Hc @ T, k is not None


def _display_times(s: dict) -> list[float]:
    """슛 릴리스 장면이 화면에 나오는 시각: 본 화면(그 순간이 리플레이가 아니면) 먼저, 그다음 그 장면을 다시 보여 주는 리플레이."""
    shows = [] if _in(s["t"], REPLAYS) else [s["t"]]
    shows += [s["t"] + d for (a, b), d in zip(REPLAYS, REPLAY_DELAY) if d is not None and a <= s["t"] + d < b]
    return shows


def _calibrate_at(video: Path, t: float, seed: int, path: Path, names=None) -> bool:
    """사람이 calibrate 창에서 하던 일: t초 화면에 보이는 코트 랜드마크를 손 떨림(1.5px)만큼 틀리게 찍고 흰 라인에 맞춘다."""
    from .video import Video

    lm = landmarks(FIBA)
    rng = np.random.default_rng(seed + int(t * 10))
    Ht, _ = _camera_truth(t, seed)
    pts = {}
    for n in names or lm:
        u, v = HG.court_to_image(Ht, *lm[n])
        if 0 <= u < W and 0 <= v < H_IMG:
            pts[n] = [float(u + rng.normal(0, 1.5)), float(v + rng.normal(0, 1.5))]
    if len(pts) < 4:
        return False
    H0, _ = HG.fit_homography(pts, FIBA)
    vv = Video(video)
    H0, _ = HG.refine_homography(vv.frame_at(t), H0, FIBA)
    vv.close()
    path.write_text(json.dumps({"spec": "FIBA", "t_sec": t, "image_points": pts, "H_court_to_image": H0.tolist()}), encoding="utf-8")
    return True


def _label_shots(shots, truth, track, noise_px: float, ft_rule: str, seed: int, review: bool = True,
                 frame_at=None) -> list[dict]:
    """사람(나중에는 검출 모델)이 슛마다 발 위치를 화면에서 찍었다고 보고 코트로 판정한다.

    noise_px: 발 위치를 찍을 때의 화면 오차(표준편차, px).
    ft_rule: "label" = 자유투인지는 사람이 f 로 표시 / "spot" = 자유투 자리에서 쏘면 무조건 자유투로 본다(위치만).
    review: 라인 근처로 뜬 슛은 click-shots 에서처럼 앞뒤 프레임에서 발을 더 찍어(Shift+클릭) 평균한다.
    리플레이 중에 난 슛은 리플레이 화면에서 찍는다(replay=True). 화면 밖이면 못 찍는다.
    frame_at: 시각 → 화면. 주면 click-shots 처럼 발 근처에 보이는 3점 라인에 맞춰 보정한다.
    """
    from .cli import _CourtLookup
    from .video import fmt_time

    look = _CourtLookup(None, str(track))
    rng = np.random.default_rng(seed + 100)
    rows = []

    def click(t, s):
        Ht, rep = _camera_truth(t, seed)
        u, v = HG.court_to_image(Ht, s["x"], s["y"])
        if not (0 <= u < W and 0 <= v < H_IMG):
            return None
        return u + rng.normal(0, noise_px), v + rng.normal(0, noise_px), rep

    for s in shots:
        c = None
        for td in _display_times(s):
            c = click(td, s)
            if c is not None:
                break
        if c is None:  # 어디에도 안 나온 슛은 찍을 수 없다
            continue
        u, v, rep = c
        lag = td - s["t"]  # 리플레이면 실제 슛보다 늦게 나온다
        H = look.at(td, tol=0.05)
        if H is None:
            # 릴리스 프레임에 코트가 없으면(click-shots 의 NO COURT) 슈터가 아직 서 있는 앞뒤 프레임에서 찍는다
            for dt in (-0.5, 0.5, -1.0):
                c2 = click(td + dt, s)
                H2 = look.at(td + dt, tol=0.05)
                if c2 is not None and c2[2] == rep and H2 is not None:
                    u, v, _ = c2
                    H = H2
                    break
        row = {"t_sec": td, "time": fmt_time(td), "team": s["team"], "result": "MADE" if s["made"] else "MISSED",
               "u": round(u, 1), "v": round(v, 1), "replay": rep}
        if H is None:
            row.update({"points": "", "zone": "", "zone_kr": "", "near_line": "", "clicks": 0})
        else:
            fr = frame_at(td) if frame_at else None
            clicks = [(H, u, v, fr)] if fr is not None else [(H, u, v)]
            call = HG.classify_click(H, u, v, FIBA, side=s["side"], frame=fr)
            if review and call.near_line:
                # 슈터는 릴리스 1초 전부터 0.5초 뒤까지 그 자리에 있다 → 같은 화면 종류(본/리플레이)의 앞뒤 추적 프레임에서 더 찍기
                for dt in (-1.0, -0.5, 0.5):
                    t2 = s["t"] + lag + dt
                    H2 = look.at(t2, tol=0.05)
                    c2 = click(t2, s)
                    if H2 is not None and c2 is not None and c2[2] == rep:
                        f2 = frame_at(t2) if frame_at else None
                        clicks.append((H2, c2[0], c2[1], f2) if f2 is not None else (H2, c2[0], c2[1]))
                call = HG.classify_clicks(clicks, FIBA, side=s["side"])
            row.update(call.as_dict())
            row["clicks"] = len(clicks)
            if (ft_rule == "label" and s["kind"] == "FT") or (ft_rule == "spot" and call.ft_spot):
                row.update({"points": 1, "zone": "FT", "zone_kr": "자유투"})
        rows.append(row)
    return rows


def run(out_dir: str | Path, seconds: float = 90.0, seed: int = 0, step: float = 0.5) -> dict:
    from . import cli

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    video = out / "bench_game.avi"
    truth, want, shots = make_video(video, seconds, seed)

    # 스코어버그 설정·숫자 (사람이 창에서 하던 단계)
    cfg = SB.ScorebugConfig(rois=BUG_ROIS, reference_size=(W, H_IMG), home_name="HOME", away_name="AWAY")
    cfg.save(out / "scorebug.json")
    tpl = SB.DigitTemplates()
    for val in ["10", "23", "45", "67", "89"]:
        img = np.zeros((H_IMG, W, 3), np.uint8)
        _draw_bug(img, int(val), int(val))
        SB.learn(tpl, cfg.crop(img, "home"), val)
    tpl.save(out / "digits.npz")
    ev_csv = out / "score_events.csv"
    cli.main(["scorebug-read", "--video", str(video), "--config", str(out / "scorebug.json"), "--templates", str(out / "digits.npz"),
              "--step", str(step), "--out", str(ev_csv)])
    with open(ev_csv, encoding="utf-8-sig") as f:
        got = [(float(e["t_sec"]), e["team"], int(e["delta"])) for e in csv.DictReader(f)]
    tp, fp = _score_metrics(got, want)
    res: dict = {"seed": seed, "점수 변화": {"정답": len(want), "맞음": tp, "놓침": len(want) - tp, "잘못 읽음": len(fp),
                                           "잘못 읽은 것": [f"{t:.1f}s {team} +{d}" for t, team, d in fp]}}

    # 코트: 첫 프레임에서 점 찍기 + 라인 정합 (calibrate --refine)
    _calibrate_at(video, 0.0, seed, out / "calib.json", names=("L_baseline_bot", "L_baseline_top", "L_ft_bot", "L_ft_top",
                                                               "mid_sideline_bot", "mid_sideline_top", "L_corner3_arc_top", "center"))
    track = out / "court_track.npz"
    calibs = ["--calib", str(out / "calib.json")]
    cli.main(["track-court", "--video", str(video), *calibs, "--step", str(step), "--out", str(track)])
    # 득점 장면인데 코트가 안 잡힌 화면(예: 확 당겨 찍은 리플레이)은 사람이 그 화면을 한 번 캘리브레이션해서 기준으로 넣고 다시 돈다
    from .cli import _CourtLookup
    look = _CourtLookup(None, str(track))
    extra = 0
    for s in shots:
        if not s["made"]:
            continue
        for td in _display_times(s)[:1]:
            rep = _camera_truth(td, seed)[1]
            same = [dt for dt in (0.0, -0.5, 0.5, -1.0) if _camera_truth(td + dt, seed)[1] == rep]  # 같은 화면(본/리플레이)만
            def on_screen(t, s=s):  # 그 화면에 슈터 발이 보이는가
                u, v = HG.court_to_image(_camera_truth(t, seed)[0], s["x"], s["y"])
                return 0 <= u < W and 0 <= v < H_IMG

            # 발이 보이면서 코트 좌표도 있는 화면이 하나도 없으면 (사람이) 그 시각으로 캘리브레이션
            if all(look.at(td + dt, tol=0.05) is None or not on_screen(td + dt) for dt in same):
                c = out / f"calib_extra{extra + 1}.json"
                if _calibrate_at(video, td, seed, c):
                    extra += 1
                    calibs += ["--calib", str(c)]
    if extra:
        cli.main(["track-court", "--video", str(video), *calibs, "--step", str(step), "--out", str(track)])
    z = np.load(track)
    probe = np.array([[5.8, 5.05], [5.8, 9.95], [14, 0], [14, 15], [8.3, 7.5], [22.2, 7.5], [1.575, 7.5], [26.425, 7.5]])
    cnt = {"정확(<3px)": 0, "약간 어긋남(3~8px)": 0, "틀림(>=8px)": 0, "좌표 없음": 0}
    replay_wrong = replay_right = 0
    top, T = _court_texture(seed)
    size = top.shape[1::-1]
    errs = []
    recover = []
    for t, Hk in zip(z["times"], z["H"]):
        i = min(int(round(t * FPS)), len(truth) - 1)
        has = bool(np.all(np.isfinite(Hk)))
        if truth[i] is None:
            # 리플레이도 같은 코트를 다른 각도로 찍은 것이라, 좌표를 냈다면 그 화면에서 맞는지만 본다
            if has:
                Hr = _frame_camera(i / FPS, size, T)[0] @ T
                pt = HG.project(Hr, probe)
                vis = (pt[:, 0] > -50) & (pt[:, 0] < W + 50) & (pt[:, 1] > -50) & (pt[:, 1] < H_IMG + 50)
                e = float(np.linalg.norm(HG.project(Hk, probe[vis]) - pt[vis], axis=1).mean()) if vis.any() else 99.0
                replay_wrong += e >= 8
                replay_right += e < 8
            continue
        if not has:
            cnt["좌표 없음"] += 1
            continue
        # 화면 안에 보이는 점들로만 오차를 잰다
        pt = HG.project(truth[i], probe)
        vis = (pt[:, 0] > -50) & (pt[:, 0] < W + 50) & (pt[:, 1] > -50) & (pt[:, 1] < H_IMG + 50)
        e = float(np.linalg.norm(HG.project(Hk, probe[vis]) - pt[vis], axis=1).mean()) if vis.any() else 0.0
        errs.append(e)
        cnt["정확(<3px)" if e < 3 else "약간 어긋남(3~8px)" if e < 8 else "틀림(>=8px)"] += 1
    for a, b in REPLAYS:
        if b >= seconds:
            continue
        after = [(t, Hk) for t, Hk in zip(z["times"], z["H"]) if t >= b]
        rec = None
        for t, Hk in after:
            i = min(int(round(t * FPS)), len(truth) - 1)
            if truth[i] is not None and np.all(np.isfinite(Hk)):
                pt = HG.project(truth[i], probe)
                if np.linalg.norm(HG.project(Hk, probe) - pt, axis=1).mean() < 3:
                    rec = round(float(t - b), 1)
                    break
        recover.append(rec)
    n_main = sum(cnt.values())
    res["코트 추적"] = {"본 카메라 샘플": n_main, "추가 캘리브레이션(사람)": extra, **cnt, "중앙값 오차(px)": round(float(np.median(errs)), 2) if errs else None,
                     "리플레이 중 틀린 좌표": int(replay_wrong), "리플레이 중 맞은 좌표": int(replay_right), "리플레이 뒤 복구(초)": recover}

    # 핵심 지표: 점수판을 보지 않고 코트(슛 위치)로만 정한 점수가 점수판(OCR 결과) 변화와 맞는가
    from . import scorecheck as SC
    events = SC.read_csv(ev_csv)
    court = {}
    from .video import Video

    vid = Video(str(video))
    fcache: dict = {}

    def frame_at(t):  # click-shots 처럼 찍은 화면에 보이는 3점 라인으로 보정
        k = round(t, 3)
        if k not in fcache:
            try:
                fcache[k] = vid.frame_at(t)
            except ValueError:
                fcache[k] = None
        return fcache[k]

    for noise in (0.0, 3.0, 6.0):
        for rule, review in (("label", False), ("label", True), ("spot", True)):
            rows = _label_shots(shots, truth, track, noise, rule, seed, review=review, frame_at=frame_at)
            sm = SC.check(events, rows).summary()
            key = (f"발 위치 오차 {noise:g}px, 자유투 {'라벨' if rule == 'label' else '위치만'}, "
                   f"라인 근처 {'다시 찍어 평균' if review else '한 번만'}")
            court[key] = {k: sm[k] for k in ("점수판 변화", "일치", "일치율(전체 변화 중)", "확실한 판정", "확실한 판정 일치",
                                             "라인 근처 슛이 낀 변화", "라인 근처에서 틀림", "슛 없이 바뀐 점수", "점수 변화 없는 성공 슛",
                                             "점수 비교")}
            if noise == 3.0 and rule == "label" and review:
                cli._write_csv(str(out / "shots_court.csv"), rows)
    res["코트로 계산한 점수 vs 점수판"] = court
    (out / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
