"""명령줄 도구. `gtt <명령> --help` 로 옵션을 볼 수 있다.

화면 창이 뜨는 명령(scorebug-setup, scorebug-learn, calibrate, click-shots)은
opencv-python(GUI 포함) 이 설치된 데스크톱에서 실행해야 한다.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import court as C
from . import homography as HG
from . import scorebug as SB
from .video import Video, fmt_time

WIN_MAX_W = 1600


# ---------------------------------------------------------------- 공용
def _fit_window(img: np.ndarray) -> tuple[np.ndarray, float]:
    s = min(1.0, WIN_MAX_W / img.shape[1])
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


def _spec(name: str) -> C.CourtSpec:
    """FIBA·NBA 또는 체육관별 규격 JSON 경로 ({"base": "FIBA", "name": "...", "three_radius": 6.75, ...}).
    아마추어 체육관은 3점 라인이 옛 규격(6.25m)으로 남아 있거나 두 줄로 칠해진 곳이 있어, 경기에 쓰는 실제 칠한 라인에 맞춘다."""
    if name.upper() in C.SPECS:
        return C.SPECS[name.upper()]
    p = Path(name)
    if not p.exists():
        sys.exit(f"코트 규격 '{name}' 을 모릅니다 (FIBA, NBA 또는 규격 JSON 경로)")
    d = json.loads(p.read_text(encoding="utf-8"))
    base = C.SPECS[str(d.pop("base", "FIBA")).upper()]
    fields = {f.name for f in dataclasses.fields(C.CourtSpec)}
    bad = sorted(set(d) - fields)
    if bad:
        sys.exit(f"{name}: 모르는 항목 {bad} (쓸 수 있는 것: {sorted(fields)})")
    d.setdefault("name", p.stem)
    return dataclasses.replace(base, **{k: (v if k == "name" else float(v)) for k, v in d.items()})


def _write_csv(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀에서 한글 안 깨짐
        if not rows:
            f.write("")
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))  # 예전 CSV 에 없던 열도 받기
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def _put(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- grab
def cmd_grab(a):
    v = Video(a.video)
    frame = v.frame_at(a.t)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, frame)
    print(f"저장: {a.out} ({v.width}x{v.height}, {fmt_time(a.t)})")


# ---------------------------------------------------------------- scorebug
def cmd_scorebug_setup(a):
    v = Video(a.video)
    frame = v.frame_at(a.t)
    shown, s = _fit_window(frame)
    rois = {}
    for key in ["home", "away"] + (["clock", "period"] if a.with_clock else []):
        print(f"[{key}] 점수 숫자만 꼭 맞게 드래그한 뒤 Enter (취소 c). 팀 이름·로고는 빼세요.")
        r = cv2.selectROI(f"select {key} score", shown, showCrosshair=False)
        cv2.destroyAllWindows()
        if r[2] == 0 or r[3] == 0:
            print(f"{key} 건너뜀")
            continue
        rois[key] = [int(round(x / s)) for x in r]
    if "home" not in rois or "away" not in rois:
        sys.exit("home, away 영역은 꼭 필요합니다.")
    cfg = SB.ScorebugConfig(rois=rois, reference_size=(v.width, v.height), home_name=a.home, away_name=a.away)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cfg.save(a.out)
    print(f"저장: {a.out}")


def cmd_scorebug_learn(a):
    v = Video(a.video)
    cfg = SB.ScorebugConfig.load(a.config)
    tmpl = SB.DigitTemplates.load(a.templates) if Path(a.templates).exists() else SB.DigitTemplates()
    times = [float(t) for t in a.times.split(",")] if a.times else list(
        np.linspace(60, max(120, v.duration - 60), a.n))
    for t in times:
        frame = v.frame_at(t)
        crops = [cfg.crop(frame, "home"), cfg.crop(frame, "away")]
        preview = np.hstack([cv2.resize(c, (int(c.shape[1] * 120 / max(1, c.shape[0])), 120)) for c in crops])
        cv2.imshow("scorebug (home | away) - type values in console", preview)
        cv2.waitKey(300)
        ans = input(f"{fmt_time(t)}  홈 점수 원정 점수 (예: 47 52, 안 보이면 Enter): ").split()
        if len(ans) != 2:
            continue
        for crop, val in zip(crops, ans):
            if not SB.learn(tmpl, crop, val):
                print(f"  '{val}' 글자 수와 잘린 글자 수가 달라 건너뜀 (ROI 를 다시 잡아야 할 수도 있음)")
    cv2.destroyAllWindows()
    tmpl.save(a.templates)
    miss = tmpl.missing()
    print(f"저장: {a.templates}" + (f"  아직 없는 숫자: {' '.join(miss)} → 다른 시각으로 한 번 더 실행하세요" if miss else "  0~9 모두 등록됨"))


def cmd_scorebug_read(a):
    v = Video(a.video)
    cfg = SB.ScorebugConfig.load(a.config)
    if a.templates and Path(a.templates).exists():
        tmpl = SB.DigitTemplates.load(a.templates)
        if tmpl.missing():
            print(f"경고: 실제 화면에서 등록하지 않은 숫자 {tmpl.missing()} → 글꼴 대체 템플릿으로 보충합니다. "
                  f"중계 글꼴과 달라 잘못 읽을 수 있으니, 그 숫자가 나오는 화면에서 scorebug-learn 으로 등록하세요.")
            fb = SB.DigitTemplates.from_fonts()
            for d in tmpl.missing():
                for g in fb.samples.get(d, []):
                    tmpl.add(d, g)
    else:
        print("경고: 템플릿이 없어 기본 글꼴 템플릿을 씁니다. scorebug-learn 으로 등록하면 훨씬 정확합니다.")
        tmpl = SB.DigitTemplates.from_fonts()
    det = SB.ScoreChangeDetector(stable_n=a.stable_n)
    events, raw = [], []
    last_print = 0.0
    for t, frame in v.iterate(step=a.step, start=a.start, end=a.end):
        # 확정된 점수를 알면 그 근처 값으로만 읽는다 (한 번에 +4 이상 바뀌는 판독은 잘못 읽은 것)
        ph, pa = det.state if det.state is not None else (None, None)
        rh = SB.read_number_near(cfg.crop(frame, "home"), tmpl, ph, cfg.max_digits)
        ra = SB.read_number_near(cfg.crop(frame, "away"), tmpl, pa, cfg.max_digits)
        raw.append({"t_sec": round(t, 2), "home": rh.value, "away": ra.value, "home_conf": round(rh.confidence, 3), "away_conf": round(ra.confidence, 3)})
        for e in det.push(t, rh.value, ra.value):
            row = e.as_row()
            row["team_name"] = cfg.home_name if e.team == "home" else cfg.away_name
            row["time"] = fmt_time(e.t)
            events.append(row)
            print(f"{row['time']}  {row['team_name']} +{e.delta}  ({e.home}:{e.away}) {e.flag}")
        if t - last_print > 300:
            last_print = t
            print(f"... {fmt_time(t)} 처리 중", file=sys.stderr)
    from .scorecheck import resolve_corrections

    before = [(e["flag"], e["delta"]) for e in events]
    events = resolve_corrections(events)
    # 읽는 동안 찍은 줄은 정정 정리 전이라, 정리로 바뀐 변화는 따로 알려 준다
    for (f0, d0), e in zip(before, events):
        if (f0, d0) != (e["flag"], e["delta"]):
            note = {"revert": "잠깐 내렸다 되돌림 → 득점 아님", "reverted": "뒤에 지움 → 득점 아님",
                    "merged": "앞 득점을 고친 것"}.get(e["flag"], e.get("note", ""))
            print(f"정정 정리: {e['time']}  {e['team_name']} {d0:+d} {f0} → {int(e['delta']):+d} {e['flag']} ({note})")
    _write_csv(a.out, events)
    if a.raw:
        _write_csv(a.raw, raw)
    ok = sum(1 for e in events if e["flag"] == "ok")
    print(f"저장: {a.out}  (점수 변화 {len(events)}건, 정상 {ok}건, 확인 필요 {len(events) - ok}건)")


# ---------------------------------------------------------------- calibrate
def cmd_calibrate(a):
    spec = _spec(a.spec)
    frame = cv2.imread(a.image) if a.image else Video(a.video).frame_at(a.t)
    lm = C.landmarks(spec)
    order = [
        "L_baseline_bot", "L_corner3_baseline_bot", "L_key_baseline_bot", "L_key_baseline_top", "L_corner3_baseline_top",
        "L_baseline_top", "L_ft_bot", "L_ft_top", "L_corner3_arc_bot", "L_corner3_arc_top", "L_arc_top",
        "mid_sideline_bot", "mid_sideline_top", "center", "center_circle_bot", "center_circle_top",
    ]
    order += ["R_" + n[2:] for n in order if n.startswith("L_")]
    pts: dict[str, tuple[float, float]] = {}
    if a.out and Path(a.out).exists() and not a.fresh:
        pts = {k: tuple(v) for k, v in json.loads(Path(a.out).read_text(encoding="utf-8"))["image_points"].items()}
    idx = 0
    top, T = HG.render_topdown(spec, px_per_m=28)
    shown, s = _fit_window(frame)
    H = None
    state = {"click": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / s, y / s)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    print(__CALIB_HELP)
    while True:
        name = order[idx]
        if state["click"] is not None:
            pts[name] = state["click"]
            state["click"] = None
            idx = min(idx + 1, len(order) - 1)
            name = order[idx]
        view = shown.copy()
        if H is not None:
            HG.draw_court(view, np.diag([s, s, 1.0]) @ H, spec, (0, 255, 255), 1)
        for k, (u, v) in pts.items():
            cv2.circle(view, (int(u * s), int(v * s)), 5, (0, 0, 255), -1)
            _put(view, k, (int(u * s) + 6, int(v * s) - 6), 0.45)
        _put(view, f"[{idx + 1}/{len(order)}] click: {name}   n/p: next/prev  u: undo  f: fit  s: save  q: quit", (10, 28), 0.6, (0, 255, 255))
        ref = top.copy()
        for k, (x, y) in lm.items():
            u, v = HG.court_to_image(T, x, y)
            col = (0, 0, 255) if k == name else ((0, 200, 0) if k in pts else (120, 120, 120))
            cv2.circle(ref, (int(u), int(v)), 6 if k == name else 4, col, -1)
        cv2.imshow("calibrate", view)
        cv2.imshow("court reference (red = current)", ref)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            idx = min(idx + 1, len(order) - 1)
        elif key == ord("p"):
            idx = max(idx - 1, 0)
        elif key == ord("u"):
            pts.pop(name, None)
        elif key == ord("f") or key == ord("s"):
            try:
                H, err = HG.fit_homography(pts, spec)
                if a.refine:
                    H, score = HG.refine_homography(frame, H, spec)
                    print(f"라인 정합 일치율 {score:.2f}")
                print(f"맞춤 완료: 점 {len(pts)}개, 평균 재투영 오차 {err:.1f}px. 노란 선이 실제 라인에 겹치는지 보세요.")
            except ValueError as e:
                print(e)
                H = None
            if key == ord("s") and H is not None:
                out = {"spec": spec.name, "source": a.image or a.video, "t_sec": a.t, "frame_size": [frame.shape[1], frame.shape[0]],
                       "image_points": {k: list(map(float, v)) for k, v in pts.items()}, "H_court_to_image": H.tolist()}
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"저장: {a.out}")
        elif key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


def cmd_calib_edit(a):
    """캘리브레이션 JSON 을 열어 노란 코트 선을 마우스로 끌어 고친다. --lens 를 주면 렌즈 휨을 편 화면에서 고치고 그 화면 H 로 저장."""
    from . import calib_edit as CE

    spec = _spec(a.spec)
    H, d = _load_calib(a.calib)
    raw_v = Video(a.video)
    t = a.t if a.t is not None else float(d.get("t_sec") or 0.0)
    raw = raw_v.frame_at(t)
    lens = None
    if a.lens:
        from .lens import Lens, convert_h

        lens = Lens.load(a.lens)
        if lens.is_identity:
            lens = None
    if lens is not None:
        frame = lens.undistort(raw)
        if not (d.get("lens") and abs(float(d["lens"].get("k1", 0.0)) - lens.k1) < 1e-9):
            pts = d.get("image_points") or {}
            Hc = None
            if len(pts) >= 4:
                try:
                    Hc, _ = HG.fit_homography({k: tuple(lens.undistort_pts([v])[0]) for k, v in pts.items()}, spec)
                except ValueError:
                    Hc = None
            H = Hc if Hc is not None else convert_h(H, lens, spec)
    else:
        frame = raw
        if d.get("lens") and float(d["lens"].get("k1", 0.0)) != 0.0:
            sys.exit(f"{a.calib} 은 렌즈 휨을 편 화면 H 입니다. 같은 --lens 를 주세요.")
    out = Path(a.out or a.calib)
    if out.exists() and a.out is None:
        bak = out.with_name(out.stem + ".bak.json")
        if not bak.exists():
            bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"원래 파일 백업: {bak}")

    def save(Hn, strokes=None):
        lm = C.landmarks(spec)
        names = list(lm)
        uv = HG.project(Hn, [lm[n] for n in names])
        h, w = frame.shape[:2]
        keep = [i for i in range(len(names)) if 0 <= uv[i][0] < w and 0 <= uv[i][1] < h]
        img = uv[keep]
        if lens is not None:
            img = lens.distort_pts(img)  # image_points 는 늘 원본 화면 좌표 (다른 렌즈 값으로도 다시 맞출 수 있게)
        pts = {names[i]: [round(float(p[0]), 2), round(float(p[1]), 2)] for i, p in zip(keep, img) if np.all(np.isfinite(p))}
        o = dict(d)
        o.update({"spec": spec.name, "t_sec": t, "H_court_to_image": (Hn / Hn[2, 2]).tolist(), "image_points": pts,
                  "edited_with": "calib-edit"})
        if strokes is not None:
            o["drawn_lines"] = strokes  # 그은 선 (편집 화면 좌표, 다시 볼 때)
        if lens is not None:
            o["lens"] = lens.as_dict()
        else:
            o.pop("lens", None)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {out}")

    print("그리기: 화면의 흰 선 위를 마우스로 따라 그으세요 (곧은 선은 처음부터 끝까지 끌기, 3점 아크는 곡선을 따라 끌기).\n"
          "  그은 선마다 가장 가까운 코트 선 이름이 붙고(틀리면 Tab), 그은 선들에 코트 전체가 맞춰집니다. 4줄 이상 그으세요\n"
          "  (베이스라인, 사이드라인, 자유투 라인, 키 변, 3점 아크처럼 방향이 다른 선). 오른쪽 클릭: 그은 선 지우기.\n"
          "M: 끌기 모드(노란 선을 잡아 끌기)와 바꾸기, Z 되돌리기, R 처음으로, S 저장, Q 닫기 (저장 안 했으면 한 번 더 Q).")
    CE.run(frame, H, spec, save, WIN_MAX_W, title=f"calib-edit {Path(a.calib).name}")


__CALIB_HELP = """
캘리브레이션 방법
 1) 'court reference' 창의 빨간 점이 지금 찍을 지점입니다. 영상 창에서 그 지점을 클릭하세요.
 2) 화면에 안 보이는 지점은 n 으로 건너뛰세요 (p: 이전, u: 지우기).
 3) 4개 이상 찍고 f 를 누르면 노란 코트 라인이 겹쳐 보입니다. 실제 라인과 맞으면 s 로 저장.
    → 같은 직선 위의 점만 찍으면 안 됩니다. 베이스라인 점 + 자유투 라인 점처럼 섞어 주세요.
"""


# ---------------------------------------------------------------- track / overlay / shot
def _load_calib(path: str) -> tuple[np.ndarray, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(d["H_court_to_image"], dtype=np.float64), d


def _sample_frames(v, start: float, end: float, n: int = 12) -> list:
    """구간 전체에서 띄엄띄엄 n 장 (방송 그래픽 찾기용). 못 읽는 프레임은 건너뛴다."""
    out = []
    for k in range(n):
        try:
            out.append(v.frame_at(start + (end - start) * k / n))
        except ValueError:
            pass
    return out


def _graphics_boxes(specs, shape) -> np.ndarray | None:
    """--graphics-box 값들(원본 화면 x,y,w,h 또는 {"boxes": [[x,y,w,h], ...], "reference_size": [w,h]} JSON 파일)을
    원본 화면 크기(shape)의 마스크로. 스코어버그처럼 늘 같은 자리에 있는 그래픽을 확실히 빼고 싶을 때 쓴다
    (자동 찾기는 카메라가 거의 안 움직이는 구간에서 '못 찾음'이 된다: 실제 B-LAB 앞 구간)."""
    if not specs:
        return None
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    for sp_ in specs:
        boxes, ref = [], None
        if str(sp_).lower().endswith(".json"):
            d = json.loads(Path(sp_).read_text(encoding="utf-8"))
            boxes, ref = d.get("boxes", []), d.get("reference_size")
        else:
            boxes = [[float(x) for x in str(sp_).split(",")]]
        sx, sy = (w / ref[0], h / ref[1]) if ref else (1.0, 1.0)
        for b in boxes:
            if len(b) != 4:
                sys.exit(f"--graphics-box 는 x,y,w,h 네 값이어야 합니다: {b}")
            x0, y0 = int(np.floor(b[0] * sx)), int(np.floor(b[1] * sy))
            x1, y1 = int(np.ceil((b[0] + b[2]) * sx)), int(np.ceil((b[1] + b[3]) * sy))
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m if m.any() else None


def _graphics_raw(raw_v, start: float, end: float, boxes, auto: bool = True) -> np.ndarray | None:
    """원본 화면 기준 방송 그래픽 자리 = 자동으로 찾은 곳(구간에서 띄엄띄엄 뽑은 화면) ∪ --graphics-box."""
    g = HG.graphics_mask(_sample_frames(raw_v, start, end)) if auto else None
    b = _graphics_boxes(boxes, (raw_v.height, raw_v.width))
    if g is None:
        return b
    return g if b is None else (g | b)


def _calib_on_canvas(H: np.ndarray, d: dict, frame_raw: np.ndarray, lens, spec, ignore) -> tuple[np.ndarray, float | None]:
    """원본 화면에서 맞춘 캘리브레이션 H 를 렌즈 휨을 편 화면(canvas) 용으로 바꾼다. (H, 일치율)

    사람이 찍은 점(image_points)이 있으면 그 점을 펴서 다시 맞추고, 없으면 H 를 옮긴 뒤 흰 라인에 다시 맞춘다."""
    from .lens import convert_h

    if lens is None or lens.is_identity:
        return H, None  # 휨 보정 없음(k1=0): 원본 화면 그대로
    if d.get("lens") and abs(float(d["lens"].get("k1", 0.0)) - lens.k1) < 1e-9:
        return H, None  # 이미 편 화면에서 맞춘 캘리브레이션
    Hc = None
    pts = d.get("image_points") or {}
    if len(pts) >= 4:
        und = {k: tuple(lens.undistort_pts([v])[0]) for k, v in pts.items()}
        try:
            Hc, _ = HG.fit_homography(und, spec)
        except Exception:  # noqa: BLE001 — 점이 이상하면 H 를 옮기는 쪽으로
            Hc = None
    if Hc is None:
        Hc = convert_h(H, lens, spec)
    canvas = lens.undistort(frame_raw)
    Hc, sc = HG.refine_homography(canvas, Hc, spec, ignore=ignore)
    for _ in range(4):
        H2, s2 = HG.refine_homography(canvas, Hc, spec, ignore=ignore)
        if s2 <= sc + 0.002:
            break
        Hc, sc = H2, s2
    return Hc, sc


def cmd_lens_fit(a):
    from . import lens as LN

    spec = _spec(a.spec)
    v = Video(a.video)
    samples = []
    for c in a.calib:
        H, d = _load_calib(c)
        if d.get("t_sec") is None:
            print(f"경고: {c} 에 t_sec 가 없어 뺌", file=sys.stderr)
            continue
        samples.append((v.frame_at(float(d["t_sec"])), H, f"{Path(c).name} ({float(d['t_sec']):.1f}s)"))
    if a.track:
        z = np.load(a.track, allow_pickle=False)
        ok = [i for i, (st, sc) in enumerate(zip(z["status"], z["line_score"]))
              if str(st) in ("ok", "recovered") and float(sc) >= a.min_score]
        if ok:
            pick = [ok[int(j)] for j in np.linspace(0, len(ok) - 1, min(a.n_extra, len(ok)))]
            for i in dict.fromkeys(pick):
                t = float(z["times"][i])
                samples.append((v.frame_at(t), z["H"][i], f"추적 {t:.1f}s ({float(z['line_score'][i]):.2f})"))
    if not samples:
        sys.exit("렌즈 휨을 잴 화면이 없습니다 (--calib 에 t_sec 가 있는 캘리브레이션, 또는 --track).")
    end = v.duration or 600.0
    graphics = _graphics_raw(v, 0.0, end, a.graphics_box)
    print(f"화면 {len(samples)}장: " + ", ".join(n for _, _, n in samples))
    print("방송 그래픽 자리: " + (f"{int(graphics.sum())}px 제외" if graphics is not None else "못 찾음(전부 사용)"))
    best, table = LN.fit_k1([(f, H) for f, H, _ in samples], spec, graphics_raw=graphics, log=print)
    r0 = dict((k, r) for k, r, _ in table).get(0.0)
    rb = dict((k, r) for k, r, _ in table)[best]
    lens = LN.Lens(best, v.width, v.height)
    print(f"\n가장 잘 맞는 k1 = {best:+.3f}  (라인까지 평균 {rb:.3f}px"
          + (f", 휨 보정 없음 {r0:.3f}px" if r0 is not None else "") + f")  편 화면 여백 {lens.pad}px")
    if r0 is not None and r0 - rb < 0.05:
        print("휨 보정으로 거의 나아지지 않습니다 → k1=0 (보정 안 함) 으로 저장")
        lens = LN.Lens(0.0, v.width, v.height, 0)
    for f, H, name in samples:
        _, _, sc, res, _ = LN.fit_on_canvas(f, H, lens, spec, graphics)
        print(f"  {name}: 일치율 {sc:.3f}, 라인까지 {res:.3f}px")
    lens.save(a.out)
    print(f"저장: {a.out}")


def _around_windows(path: str, before: float, after: float, start=None, end=None) -> list[tuple[float, float]]:
    """슛 CSV(t_sec)의 시각마다 [t-before, t+after] 창을 만들고 겹치는 창은 합친다."""
    from .scorecheck import read_csv

    ts = []
    for r in read_csv(path):
        try:
            ts.append(float(r["t_sec"]))
        except (KeyError, TypeError, ValueError):
            continue
    wins: list[list[float]] = []
    for t in sorted(ts):
        w0, w1 = max(0.0, t - before), t + after
        if start is not None:
            w0 = max(w0, start)
        if end is not None:
            w1 = min(w1, end)
        if w1 <= w0:
            continue
        if wins and w0 <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], w1)
        else:
            wins.append([w0, w1])
    return [(w0, w1) for w0, w1 in wins]


def _track_around(a):
    """경기 전체 대신 슛 시각 앞뒤 창만 추적한다. 추적기 하나로 창들을 차례로 돌아, 앞 창에서 잡은 화면(키프레임)이 뒤 창을
    다시 찾을 때 기준이 된다. 창이 바뀔 때는 화면 전환처럼 놓친 상태로 보고 기준 화면·키프레임과 다시 맞춘다.
    (실제 KBA D201: 92분 전체를 0.2초 간격으로 돌리면 하루 가까이 걸렸고, 창마다 따로 새로 시작하면 창 시작 화면이
    캘리브레이션 화면과 카메라 자세가 달라 대부분 창에서 좌표가 0~1개였다)"""
    import copy

    wins = _around_windows(a.around, a.before, a.after, a.start, a.end)
    if not wins:
        sys.exit(f"{a.around} 에서 t_sec 를 하나도 못 읽었습니다.")
    print(f"슛 앞뒤 창 {len(wins)}개, 모두 {sum(w1 - w0 for w0, w1 in wins):.0f}초를 추적합니다", file=sys.stderr)
    # 방송 그래픽·안 변하는 자리는 짧은 창 하나(자유투처럼 카메라가 멈춘 장면)로 찾으면 코트까지 잡히므로, 창 전체에 걸쳐 고른 화면으로 찾는다
    raw_v = Video(a.video)
    mids = [(w0 + w1) / 2 for w0, w1 in wins]
    pick = [mids[int(i)] for i in np.unique(np.linspace(0, len(mids) - 1, min(24, len(mids))).round())]
    if len(pick) < 12:
        pick = list(np.linspace(wins[0][0], wins[-1][1], 12))
    raw_samples = []
    for t in pick:
        try:
            raw_samples.append(raw_v.frame_at(float(t)))
        except ValueError:
            pass
    b = copy.copy(a)
    b.around, b.windows, b.raw_samples = None, wins, raw_samples
    b.start, b.end = wins[0][0], wins[-1][1]
    return cmd_track_court(b)


def cmd_track_court(a):
    if getattr(a, "around", None):
        return _track_around(a)
    spec = _spec(a.spec)
    calibs = a.calib if isinstance(a.calib, list) else [a.calib or "configs/calib.json"]
    H0, d = _load_calib(calibs[0])
    raw_v = Video(a.video)
    lens = None
    if a.lens:
        from .lens import Lens, LensVideo

        lens = Lens.load(a.lens)
        if (lens.width, lens.height) != (raw_v.width, raw_v.height):
            sys.exit(f"렌즈 파일 해상도 {lens.width}x{lens.height} 가 영상 {raw_v.width}x{raw_v.height} 와 다릅니다.")
        print(f"렌즈 휨 보정 k1={lens.k1:+.3f} (편 화면 {lens.canvas_size[0]}x{lens.canvas_size[1]})", file=sys.stderr)
    v = LensVideo(raw_v, lens) if lens is not None else raw_v
    t_calib = float(d.get("t_sec") or 0.0)
    start = a.start if a.start is not None else t_calib
    end = a.end if a.end is not None else v.duration
    if end <= start:  # 길이를 모르는 영상
        end = start + 600
    # 방송 그래픽(스코어버그·로고) 찾기: 구간 전체에서 띄엄띄엄 12장을 뽑아 안 변하는 곳을 특징점에서 빼고(overlay),
    # 라인 정합·라인 보정에서도 뺀다(ignore: 그래픽의 흰 글자를 라인으로 쓰지 않고, 가려진 라인을 '틀림'으로 세지 않음)
    raw_samples = getattr(a, "raw_samples", None) or _sample_frames(raw_v, start, end)
    graphics_raw = HG.graphics_mask(raw_samples) if not a.no_graphics_mask else None
    boxes = _graphics_boxes(a.graphics_box, (raw_v.height, raw_v.width))
    if boxes is not None:
        graphics_raw = boxes if graphics_raw is None else (graphics_raw | boxes)
    if lens is not None:
        from .lens import canvas_ignore

        ignore = canvas_ignore(lens, graphics_raw)
        samples = [lens.undistort(f) for f in raw_samples]
    else:
        ignore = graphics_raw
        samples = raw_samples
    print("방송 그래픽 자리: " + (f"{int(graphics_raw.sum())}px 라인 정합에서 제외" if graphics_raw is not None else "못 찾음"),
          file=sys.stderr)
    frame_c = v.frame_at(t_calib)  # 캘리브레이션 H 는 이 화면에서 맞춘 것
    frame0 = frame_c if abs(start - t_calib) < 1e-3 else v.frame_at(start)
    if lens is not None:
        H0, sc0 = _calib_on_canvas(H0, d, raw_v.frame_at(t_calib), lens, spec, ignore)
        print(f"캘리브레이션 {calibs[0]} → " + ("휨 보정 없음(k1=0), 그대로 씀" if sc0 is None
                                              else f"편 화면에서 다시 맞춤 (일치율 {sc0:.2f})"), file=sys.stderr)
    sm = HG.StaticMask.from_frames(samples)
    overlay = sm.mask(frame0.shape[:2])
    if overlay is not None and lens is not None and lens.invalid_mask() is not None:
        overlay = overlay.copy()
        overlay[cv2.dilate(lens.invalid_mask().astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0  # 검정 여백 경계는 특징점 아님
    # 첫 기준(앵커)은 반드시 캘리브레이션한 그 화면과 그 H 로 만든다
    trk = HG.CourtTracker(H0, frame_c, spec, refine_every=a.refine_every, overlay_mask=overlay, min_score=a.min_score,
                          ignore=ignore, max_cam_px=getattr(a, "max_cam_px", 0.0),
                          consensus_line_frac=getattr(a, "consensus_line_frac", 0.25))
    # 다른 화면(예: 반대쪽 골대)에서 맞춘 캘리브레이션은 다시 찾을 때의 기준(앵커)으로 쓴다
    # 사람이 맞춘 화면은 그 시각의 정답이기도 하다: (시각, H, 회색 화면). 추적 결과에 그 시각으로 그대로 넣고,
    # 그 직후 샘플에서 추적이 놓쳐 있으면 이 화면에서 이어 붙여 다시 잡는다.
    cal_list = [(t_calib, H0, cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY))]
    for c in calibs[1:]:
        Hc, dc = _load_calib(c)
        if dc.get("t_sec") is None:
            # 시각이 없으면 어느 화면의 H 인지 모른다 → 엉뚱한 화면과 짝지어지지 않게 뺀다
            print(f"경고: {c} 에 t_sec(캘리브레이션한 화면 시각)가 없어 뺌", file=sys.stderr)
            continue
        tc = float(dc["t_sec"])
        try:
            fc = v.frame_at(tc)
        except ValueError:
            print(f"경고: {c} 의 시각 프레임을 못 읽어 기준에서 뺌", file=sys.stderr)
            continue
        if lens is not None:
            Hc, scc = _calib_on_canvas(Hc, dc, raw_v.frame_at(tc), lens, spec, ignore)
            print(f"캘리브레이션 {c} → " + ("휨 보정 없음(k1=0), 그대로 씀" if scc is None
                                         else f"편 화면에서 다시 맞춤 (일치율 {scc:.2f})"), file=sys.stderr)
        trk.add_anchor(fc, Hc)
        cal_list.append((tc, Hc, cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)))
    for tc, _, _ in cal_list:
        print(f"캘리브레이션 {fmt_time(tc)} ({tc:.2f}s) → 그 시각 좌표로 넣음", file=sys.stderr)
    print(f"기준 화면(캘리브레이션) {len(trk.anchors)}개", file=sys.stderr)
    _print_cams(trk.cams)

    def strong(Hr, sc):
        return Hr is not None and trk.last_inliers >= 80 and sc >= trk.min_score

    if frame0 is frame_c:
        times, Hs, scores, status = [start], [H0], [1.0], ["calib"]
        last_good, last_good_gray = H0, cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    else:
        # 시작 시각이 캘리브레이션 시각과 다르면 캘리브레이션 H 를 시작 화면에 그대로 쓰면 안 된다(카메라가 움직였을 수 있음).
        # 기준 화면들과 맞춰 시작 화면의 H 를 찾고, 못 찾으면 놓친 상태로 시작해 다음 샘플부터 다시 찾는다.
        Hr, sc = trk.relocalize(frame0)
        if strong(Hr, sc) or (Hr is not None and sc >= a.recover_score):
            trk.reset(Hr, frame0)
            times, Hs, scores, status = [start], [Hr], [sc], ["recovered"]
            last_good, last_good_gray = Hr, cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        else:
            trk.reset(H0, frame0)
            trk.lost = True
            times, Hs, scores, status = [start], [np.full((3, 3), np.nan)], [0.0], ["lost"]
            last_good, last_good_gray = H0, cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
            print(f"시작 화면({fmt_time(start)})을 기준 화면과 맞추지 못해 놓친 상태로 시작합니다", file=sys.stderr)

    prev_good_gray = last_good_gray if status[0] != "lost" else None
    support = [tuple(trk.last_support) + (0,) * (3 - len(trk.last_support))]
    # 좌표를 못 낸 짧은 구간(coast)을 앞뒤 정상 화면 사이에서 메우기 위해, 마지막 정상 화면 뒤의 샘플 화면을 모아 둔다
    gap: list | None = [] if prev_good_gray is not None else None  # [(샘플 번호, 회색 화면)], None = 메우기 포기
    n_bridged = 0
    n_consensus = 0
    lost_run = 0  # 연달아 놓친 샘플 수
    # 진단용: 샘플마다 추적기가 들고 있던 위치(좌표를 못 낸 coast 도 화면 이동으로 예측한 H). 확인 안 된 값이라 판정에는 안 쓴다
    preds = [Hs[0].copy()]
    wins = getattr(a, "windows", None)

    def samples():
        if not wins:
            for x in v.iterate_between(step=a.step, start=start + a.step, end=a.end):
                yield (False,) + x
            return
        for k, (w0, w1) in enumerate(wins):
            for j, x in enumerate(v.iterate_between(step=a.step, start=w0 + (a.step if k == 0 else 0.0), end=w1)):
                yield (k > 0 and j == 0,) + x

    for new_win, t, frame, between in samples():
        if t <= start:
            continue
        if new_win:
            # 다음 슛 창으로 건너뜀: 화면 이동이 끊겼으니 놓친 상태로 보고 기준 화면·키프레임(앞 창들에서 쌓인 것 포함)과 다시 맞춘다
            trk.lost = True
            lost_run = 0
            gap = None
        r = trk.update(frame, between)
        if r.status == "coast" and trk.coast_n % a.reloc_every == 0:
            # 좌표를 못 내는 동안 주기적으로 기준 화면·키프레임과 맞춰 본다: 틀린 곳(반대쪽 골대 등)에 붙어 있었다면
            # 여기서 바로잡히고, 라인이 약해 못 잡던 화면도 기준 화면에서 옮겨 와 다시 맞추면 잡히는 경우가 많다
            Hr, sc = trk.relocalize(frame)
            if strong(Hr, sc) or (Hr is not None and sc >= max(a.recover_score, 0.75)):
                trk.reset(Hr, frame)
                r = HG.TrackResult(Hr, sc, "recovered")
            elif a.consensus and (Hc := trk.consensus(frame.shape, frame=frame)) is not None:
                # 라인으로는 확인 못 해도 서로 다른 시각의 기준 화면 둘 이상이 같은 자리를 가리킨다
                trk.reset(Hc, frame)
                r = HG.TrackResult(Hc, sc, "consensus")
                n_consensus += 1
        elif r.status == "cut" or (r.status == "lost" and (lost_run < 5 or lost_run % a.lost_every == 0)):
            # 화면 전환(리플레이·광고) 뒤: 기준 화면·키프레임과 맞춰 다시 찾는다
            # (오래 놓쳐 있으면 --lost-every 샘플마다만: 클로즈업·광고가 길게 이어질 때 매 샘플 다시 찾기가 가장 느림)
            Hr, sc = trk.relocalize(frame)
            Hc = trk.consensus(frame.shape, frame=frame) if a.consensus else None
            if not strong(Hr, sc) and (Hr is None or sc < a.recover_score) and Hc is not None:
                trk.reset(Hc, frame)
                r = HG.TrackResult(Hc, sc, "consensus")
                n_consensus += 1
            elif not strong(Hr, sc) and (Hr is None or sc < a.recover_score):
                # 마지막으로 잘 맞던 화면과 거의 같은 화면으로 돌아왔을 때만 그 H 에서 라인 정합 (멀리 이동한 화면에 억지로 맞추지 않음)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                F, n = HG.frame_to_frame(last_good_gray, gray, trk.orb, trk.matcher, mask=overlay, return_inliers=True)
                Hr, sc = None, 0.0
                if F is not None and n >= 80:
                    Hr, sc, sup = HG.refine_homography(frame, F @ last_good, spec, return_support=True, ignore=ignore)
                    if not trk.supported(sup) or not trk.sane(Hr, frame.shape):
                        Hr, sc = None, 0.0
            if strong(Hr, sc) or (Hr is not None and sc >= a.recover_score):
                trk.reset(Hr, frame)
                r = HG.TrackResult(Hr, sc, "recovered")
        if r.status not in ("ok", "recovered"):
            # 바로 앞에 사람이 맞춘 화면이 있으면 그 화면에서 화면 이동으로 이어 붙여 다시 잡는다
            for tc, Hc, gc in cal_list:
                if 0 < t - tc <= a.step + 1e-6:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    F, n = HG.frame_to_frame(gc, gray, trk.orb, trk.matcher, mask=overlay, return_inliers=True)
                    if F is None or n < 80:
                        continue
                    Hr, sc, sup = HG.refine_homography(frame, F @ Hc, spec, return_support=True, ignore=ignore)
                    if sc >= a.min_score and trk.supported(sup) and trk.sane(Hr, frame.shape):
                        trk.reset(Hr, frame)
                        r = HG.TrackResult(Hr, sc, "recovered")
                        break
        if r.status in ("ok", "recovered"):
            last_good, last_good_gray = r.H, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lost_run = lost_run + 1 if r.status == "lost" else 0
        times.append(t)
        Hs.append(r.H if r.H is not None else np.full((3, 3), np.nan))
        preds.append(trk.H.copy() if not trk.lost else np.full((3, 3), np.nan))
        scores.append(r.line_score)
        status.append(r.status)
        support.append(tuple(trk.last_support) + (0,) * (3 - len(trk.last_support)))
        if a.bridge > 0:
            if r.status == "coast":
                if gap is not None:
                    gap.append((len(times) - 1, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    if t - times[gap[0][0] - 1] > a.bridge:
                        gap = None
            elif r.status in ("ok", "recovered"):
                if gap:
                    i0 = gap[0][0] - 1  # 구간 앞의 정상 샘플
                    if np.all(np.isfinite(Hs[i0])) and status[i0] in ("ok", "recovered", "calib"):
                        grays = [prev_good_gray] + [g for _, g in gap] + [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
                        filled = HG.bridge_gap(grays, Hs[i0], r.H, spec, trk.orb, trk.matcher, mask=overlay, ok=trk.sane)
                        for (j, _), Hj in zip(gap, filled):
                            if Hj is not None:
                                Hs[j], status[j] = Hj, "bridged"
                                n_bridged += 1
                gap = []
                prev_good_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gap = None  # 화면 전환·놓침: 이동이 끊겼으니 메우지 않는다
        if len(times) % int(max(1, 60 / a.step)) == 0:
            print(f"{fmt_time(t)} 상태={r.status} 라인일치={r.line_score:.2f}", file=sys.stderr)
    # 사람이 맞춘 화면을 그 시각 그대로 넣는다 (같은 프레임 샘플이 있으면 바꾸고, 없으면 끼워 넣는다)
    half = 0.5 / (v.fps or 30.0)
    for tc, Hc, _ in cal_list:
        if not (times[0] - half <= tc <= times[-1] + half):
            continue
        i = int(np.argmin(np.abs(np.array(times) - tc)))
        if abs(times[i] - tc) <= half:
            Hs[i], status[i], scores[i] = Hc, "calib", 1.0
        else:
            times.append(tc)
            Hs.append(Hc)
            preds.append(Hc)
            scores.append(1.0)
            status.append("calib")
            support.append((0, 0, 0))
    order = np.argsort(np.array(times), kind="stable")
    times, Hs, scores = [times[i] for i in order], [Hs[i] for i in order], [scores[i] for i in order]
    preds = [preds[i] for i in order]
    status, support = [status[i] for i in order], [support[i] for i in order]
    # support: 샘플마다 (보이는 라인 줄, 맞은 줄, 방향 엇갈림) — 틀린 정합을 찾을 때 본다
    extra = {}
    if lens is not None:
        extra["lens"] = json.dumps(lens.as_dict())  # H 는 렌즈 휨을 편 화면(canvas) 좌표 → 원본 화면 점은 lens 로 옮겨 쓴다
    if ignore is not None:
        extra["ignore"] = np.packbits(ignore.astype(bool))
        extra["ignore_shape"] = np.array(ignore.shape)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # 샘플마다 '캘리브레이션 카메라 자리에서 돌리기·당기기로 흉내 낸 차이(px)' — 큰 값은 비틀린 H (진단·판정에서 거름)
    cam_px = np.array([trk.camera_px(H, frame0.shape) if np.all(np.isfinite(H)) else np.nan for H in Hs])
    if trk.cams:
        extra["cam_C"] = np.stack(trk.cams)
    np.savez_compressed(a.out, times=np.array(times), H=np.stack(Hs), H_pred=np.stack(preds), line_score=np.array(scores),
                        status=np.array(status),
                        support=np.array(support, dtype=int).reshape(-1, 3), spec=spec.name, cam_px=cam_px,
                        frame_shape=np.array(frame0.shape[:2]), **extra)
    ok = sum(s in _GOOD for s in status)
    print(f"저장: {a.out}  (" + (f"슛 앞뒤 창 {len(wins)}개, " if wins else "") + f"프레임 {len(times)}개 중 코트 좌표 있음 {ok}개, 빠른 패닝 이어 붙임 {trk.chained}번, "
          f"앞뒤 정상 화면 사이 메움 {n_bridged}개, 기준 화면 여러 개 일치로 찾음 {n_consensus}개"
          f" (흰 선과 어긋나 버림 {trk.consensus_rejected}개, 카메라 자리와 안 맞아 버림 {trk.camera_rejected}개))")
    _print_cam_stats(status, cam_px, getattr(a, "max_cam_px", 0.0) or 8.0)


def _print_cams(cams) -> None:
    """캘리브레이션 화면들에서 푼 카메라 자리. 같은 중계 카메라면 서로 1~2m 안이어야 한다 (멀면 캘리브레이션 하나가 틀렸거나 다른 카메라)."""
    for k, C in enumerate(cams):
        print(f"카메라 자리 (캘리브레이션 {k + 1}): x={C[0]:.1f} y={C[1]:.1f} 높이={C[2]:.1f} m", file=sys.stderr)
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            print(f"  캘리브레이션 {i + 1}↔{j + 1} 카메라 자리 차이 {np.linalg.norm(cams[i] - cams[j]):.2f} m", file=sys.stderr)


def _print_cam_stats(status, cam_px, max_px: float) -> None:
    st = np.array(status).astype(str)
    cam_px = np.asarray(cam_px, float)
    for s_ in sorted(set(st)):
        r = cam_px[(st == s_) & np.isfinite(cam_px)]
        if len(r):
            print(f"  {s_}: 카메라 차이 중앙 {np.median(r):.1f}px, 90% {np.percentile(r, 90):.1f}px, "
                  f"{max_px:g}px 넘음 {int((r > max_px).sum())}/{len(r)}", file=sys.stderr)


class _CourtLookup:
    """시각 → 코트 변환. 추적 결과(npz)에 렌즈 휨 보정(lens)이 있으면 H 는 편 화면(canvas) 좌표이므로
    화면은 undistort(), 원본 화면에서 찍은 점은 to_canvas() 로 옮겨 쓴다. ignore: 라인 정합에서 뺀 자리(방송 그래픽 등)."""

    def __init__(self, calib: str | None, track: str | None):
        self.fixed = _load_calib(calib)[0] if calib else None
        self.track = np.load(track, allow_pickle=False) if track else None
        self.lens = None
        self.ignore = None
        if self.track is not None:
            if "lens" in self.track.files:
                from .lens import Lens

                d = json.loads(str(self.track["lens"]))
                self.lens = Lens(float(d["k1"]), int(d["width"]), int(d["height"]), int(d["pad"]))
            if "ignore" in self.track.files:
                shp = tuple(int(x) for x in self.track["ignore_shape"])
                self.ignore = np.unpackbits(self.track["ignore"])[: shp[0] * shp[1]].reshape(shp).astype(bool)

    def add_graphics(self, raw_mask: np.ndarray | None) -> None:
        """원본 화면 기준 그래픽 마스크(--graphics-box·자동 찾기)를 ignore 에 더한다 (렌즈 보정이면 편 화면으로 옮겨서)."""
        if raw_mask is None:
            return
        if self.lens is not None:
            from .lens import canvas_ignore

            m = canvas_ignore(self.lens, raw_mask)
        else:
            m = raw_mask
        if self.ignore is not None and self.ignore.shape != m.shape:
            return
        self.ignore = m if self.ignore is None else (self.ignore | m)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.lens.undistort(frame) if self.lens is not None else frame

    def to_canvas(self, u: float, v: float) -> tuple[float, float]:
        if self.lens is None:
            return u, v
        p = self.lens.undistort_pts([[u, v]])[0]
        return float(p[0]), float(p[1])

    def to_raw(self, u: float, v: float) -> tuple[float, float]:
        if self.lens is None:
            return u, v
        p = self.lens.distort_pts([[u, v]])[0]
        return float(p[0]), float(p[1])

    def nearest(self, t: float, tol: float = 0.6) -> tuple[np.ndarray | None, float]:
        """(H, 그 H 가 계산된 시각과의 차이 초). 고정 캘리브레이션이면 차이 0.

        tol 안에서 좌표가 있는(NaN 이 아닌) 샘플 중 가장 가까운 것. 가장 가까운 샘플이 coast 라도 바로 옆 샘플이
        ok 면 그것을 쓴다 (실제 영상 309.5s: 309.4s 는 coast, 309.6s 는 ok)."""
        if self.track is not None:
            times = self.track["times"]
            Hs = self.track["H"]
            d = np.abs(times - t)
            good = (d <= tol) & np.all(np.isfinite(Hs.reshape(len(Hs), -1)), axis=1)
            if not good.any():
                return None, 0.0
            i = int(np.flatnonzero(good)[np.argmin(d[good])])
            return Hs[i], float(times[i] - t)
        return self.fixed, 0.0

    def at(self, t: float, tol: float = 0.6) -> np.ndarray | None:
        return self.nearest(t, tol)[0]

    def shape(self, v) -> tuple[int, int]:
        """H 가 쓰는 화면 크기 (높이, 폭): 렌즈 휨 보정이면 편 화면(canvas)."""
        if self.track is not None and "frame_shape" in self.track.files:
            return tuple(int(x) for x in self.track["frame_shape"])
        if self.lens is not None:
            w, h = self.lens.canvas_size
            return int(h), int(w)
        return int(v.height), int(v.width)

    def cams(self, spec, shape) -> list:
        """캘리브레이션 화면들의 카메라 자리 (추적 때 저장한 것, 없으면 추적 결과의 calib 샘플에서 푼다)."""
        if not hasattr(self, "_cams"):
            self._cams = []
            if self.track is not None:
                if "cam_C" in self.track.files:
                    self._cams = [np.asarray(c, float) for c in self.track["cam_C"]]
                else:
                    st = self.track["status"].astype(str)
                    for H in self.track["H"][st == "calib"]:
                        p = HG.camera_pose(H, shape, spec)
                        if p is not None and not any(np.linalg.norm(p[2] - c) < 0.05 for c in self._cams):
                            self._cams.append(p[2])
        return self._cams

    def cam_px(self, H, spec, shape) -> float:
        cams = self.cams(spec, shape)
        if H is None or not cams:
            return float("nan")
        rs = [r for r in (HG.camera_fit_px(H, shape, C, spec) for C in cams) if not np.isnan(r)]
        return float(min(rs)) if rs else float("nan")


_GOOD = ("ok", "recovered", "calib", "bridged", "consensus")
_CALL_KEYS = ("points", "zone", "zone_kr", "side", "hand_side", "dist_to_rim_m", "margin_m", "near_line", "court_x", "court_y",
              "ft_spot", "margin_sigma_m", "on_line", "foot_point", "line_snap_px", "line_hidden")
_VERIFIED = ("ok", "recovered", "calib")  # 그 화면 흰 라인으로 확인된 것 (multi-view 의 기준 화면으로 씀)


def _read_motions(v, look, i0: int, i1: int, keep: set, every: int, mask=None, orb=None, matcher=None):
    """프레임 i0~i1 을 차례로 읽으며, every 간격 프레임과 keep 프레임 사이의 화면 변환을 구한다.
    반환: (남긴 프레임 번호들, 이웃끼리의 변환들 (len-1, 실패는 None)). 화면은 렌즈 보정 화면(canvas) 기준."""
    idx = sorted(set(range(i0, i1 + 1, max(1, every))) | {i0, i1} | {k for k in keep if i0 <= k <= i1})
    kept, motions = [], []
    sub = max(1, int(round(v.fps / 30.0)))  # 빠른 패닝이면 1/30초 간격 사이 프레임으로 이어 붙인다
    v.cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    prev, between = None, []
    want = set(idx)
    for i in range(i0, i1 + 1):
        ok, frame = v.cap.read()
        if not ok:
            break
        if i not in want:
            if (i - (kept[-1] if kept else i0)) % sub == 0:
                between.append(cv2.cvtColor(look.undistort(frame), cv2.COLOR_BGR2GRAY))
            continue
        g = cv2.cvtColor(look.undistort(frame), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            motions.append(HG.pair_motion(prev, g, between, orb, matcher, mask=mask))
        kept.append(i)
        prev, between = g, []
    return kept, motions


def _court_at_shot(look, v, t: float, spec, uv=None, window: float = 8.0, step: float = 0.1, agree_m: float = 0.2,
                   near_s: float = 0.6, min_score: float = 0.65, max_cam_px: float = 0.0,
                   off_court_m: float = 1.0, side: str | None = None,
                   far_m: float = 1.5) -> tuple[np.ndarray | None, dict]:
    """슛 시각 t 의 코트 변환 (H, 정보). 추적 결과에 그 프레임 좌표가 없으면(coast·lost) 앞뒤 정상 샘플에서 화면 이동으로 이어 온다.

    - 그 프레임에 좌표가 있으면 그대로 ("track").
    - 앞(window 초 안 마지막 정상)과 뒤(첫 정상)에서 각각 화면 이동만으로 t 까지 따라온 두 H 가 발 자리에서 agree_m 안으로
      같은 곳을 가리키면 둘의 평균 ("two-way"). 서로 따로 구한 두 길이 만나야 하므로 한쪽 이동 계산이 틀린 경우는 걸러진다.
      (실제 B-LAB: 스코어버그 바로 위 슛 장면에서 라인이 적게 보여 3~4초씩 좌표를 못 냈다 — 18건 중 6건)
    - 한쪽뿐이거나 둘이 어긋나면: near_s 초 안의 가까운 쪽 이동은 그대로 믿고 ("near"), 그보다 멀면 그 H 를 그 화면 흰 라인에
      다시 맞춰 추적과 같은 기준(일치율 min_score, 보이는 줄 대부분 맞고 방향 엇갈림)을 통과할 때만 ("verified").
    - 그래도 안 되면 앞뒤 2분 안의 정상 화면들과 직접 맞춰 여러 장이 같은 답을 낼 때만 ("multi-view", _multi_view).
    - 아니면 None.
    정보: h_source, h_gap_s(가장 가까운 정상 샘플까지 초), h_agree_m(두 길·여러 화면의 발 자리 차이 m), h_votes,
    h_cam_px(고른 H 가 캘리브레이션 카메라 자리에서 얼마나 흉내 나는가 px), h_reject(버린 출처와 이유).

    어느 출처든 (1) 발 자리가 코트 밖 off_court_m 넘게 나가거나 (2) 공격 방향(side)을 알 때 발 자리가 그 림에서 하프라인
    너머 far_m 보다 멀거나 (3) max_cam_px > 0 이고 캘리브레이션 화면들의 카메라 자리에서 돌리기·당기기로 그 안으로 만들 수
    없는 H 면 버리고 다음 출처로 간다. (실제 B-LAB holdout: 틀린 H 가 슛 4개를 '탑 3점, 라인까지 11~18m' 로 냈다 —
    발이 반대쪽 골대 근처로 옮겨진 것, 둘은 우연히 점수만 맞음. 하프라인 너머 버저비터는 좌표 없음 → 사람 판정)"""
    info = {"h_source": "none", "h_gap_s": "", "h_agree_m": "", "h_votes": "", "h_cam_px": "", "h_reject": ""}
    if look.track is None:
        info["h_source"] = "calib"
        return look.fixed, info
    times, Hs, st = look.track["times"], look.track["H"], look.track["status"]
    fin = np.all(np.isfinite(Hs.reshape(len(Hs), -1)), axis=1)
    good = fin & np.isin(st.astype(str), _GOOD)
    shape = look.shape(v)
    saved = look.track["cam_px"] if "cam_px" in look.track.files and "cam_C" in look.track.files else None
    rejects: list[str] = []
    cache: dict[int, bool] = {}

    def sample_ok(i: int) -> bool:
        """추적 샘플 i 의 H 가 카메라 자리와 맞는가 (모르면 통과)."""
        if max_cam_px <= 0:
            return True
        if i not in cache:
            r = float(saved[i]) if saved is not None else look.cam_px(Hs[i], spec, shape)
            cache[i] = not (np.isfinite(r) and r > max_cam_px)
        return cache[i]

    def accept(src: str, Hx) -> bool:
        why = None
        if uv is not None:
            x, y = HG.image_to_court(Hx, *uv)
            if not (-off_court_m <= x <= spec.length + off_court_m and -off_court_m <= y <= spec.width + off_court_m):
                why = f"코트밖({x:.1f},{y:.1f})"
            elif side in ("left", "right"):
                rx, ry = spec.rim(side)
                d = float(np.hypot(x - rx, y - ry))
                if d > spec.length / 2 - spec.rim_from_baseline + far_m:
                    why = f"반대쪽코트(공격 림까지 {d:.1f}m)"
        r = look.cam_px(Hx, spec, shape) if max_cam_px > 0 else float("nan")
        if why is None and np.isfinite(r) and r > max_cam_px:
            why = f"카메라{r:.0f}px"
        if why is not None:
            rejects.append(f"{src}:{why}")
            info["h_reject"] = "; ".join(rejects)
            return False
        info["h_cam_px"] = round(r, 1) if np.isfinite(r) else ""
        return True

    half = 0.5 / (v.fps or 30.0)
    exact = np.flatnonzero(fin & (np.abs(times - t) <= half))
    if len(exact):
        i = int(exact[np.argmin(np.abs(times[exact] - t))])
        if accept("track", Hs[i]):
            info.update(h_source="track", h_gap_s=0.0)
            return Hs[i], info
    before = np.flatnonzero(good & (times < t) & (times >= t - window))
    after = np.flatnonzero(good & (times > t) & (times <= t + window))
    ia = next((int(i) for i in before[::-1] if sample_ok(int(i))), None)
    ib = next((int(i) for i in after if sample_ok(int(i))), None)
    if len(before) and ia != int(before[-1]):
        rejects.append(f"앞 샘플 {times[int(before[-1])]:.1f}s:카메라")
    if len(after) and ib != int(after[0]):
        rejects.append(f"뒤 샘플 {times[int(after[0])]:.1f}s:카메라")
    info["h_reject"] = "; ".join(rejects)
    mask = None
    if look.ignore is not None:
        mask = np.where(cv2.dilate(look.ignore.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, 0, 255).astype(np.uint8)
    orb = HG.make_orb()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if ia is None and ib is None:
        return _multi_view(look, v, t, look.undistort(v.frame_at(t)), spec, uv, mask, orb, matcher, info, agree_m=agree_m,
                           sample_ok=sample_ok, accept=accept)
    gaps = [abs(t - times[i]) for i in (ia, ib) if i is not None]
    info["h_gap_s"] = round(float(min(gaps)), 2)
    fps = v.fps or 30.0
    it = int(round(t * fps))
    i0 = int(round(times[ia] * fps)) if ia is not None else it
    i1 = int(round(times[ib] * fps)) if ib is not None else it
    kept, motions = _read_motions(v, look, i0, i1, {it}, int(round(step * fps)), mask, orb, matcher)
    if it not in kept:
        return None, info
    k = kept.index(it)
    Hf = Hb = None
    if ia is not None:
        Hf = Hs[ia] / Hs[ia][2, 2]
        for F in motions[:k]:
            if F is None:
                Hf = None
                break
            Hf = F @ Hf
            Hf = Hf / Hf[2, 2]
    if ib is not None:
        Hb = Hs[ib] / Hs[ib][2, 2]
        for F in reversed(motions[k:]):
            if F is None:
                Hb = None
                break
            Hb = np.linalg.inv(F) @ Hb
            Hb = Hb / Hb[2, 2]
    if Hf is not None and Hb is not None:
        if uv is not None:
            pf = np.array(HG.image_to_court(Hf, *uv))
            pb = np.array(HG.image_to_court(Hb, *uv))
            dm = float(np.linalg.norm(pf - pb))
        else:
            dm = HG.h_distance(Hf, Hb, (v.height, v.width), spec) * 0.05
        info["h_agree_m"] = round(dm, 3)
        Hm = 0.5 * (Hf + Hb)
        Hm = Hm / Hm[2, 2]
        if dm <= agree_m and accept("two-way", Hm):
            info["h_source"] = "two-way"
            return Hm, info
    # 한쪽 길만: 가까우면 그대로, 멀면 그 화면 라인에 다시 맞춰 확인
    cands = sorted([(abs(t - times[i]), Hx) for i, Hx in ((ia, Hf), (ib, Hb)) if i is not None and Hx is not None],
                   key=lambda c: c[0])
    if cands and cands[0][0] <= near_s and accept("near", cands[0][1]):
        info["h_source"] = "near"
        return cands[0][1], info
    frame = look.undistort(v.frame_at(t))
    for _, Hx in cands:
        Hr, sc, sup = HG.refine_homography(frame, Hx, spec, return_support=True, ignore=look.ignore)
        for _ in range(3):
            H2, s2, sup2 = HG.refine_homography(frame, Hr, spec, return_support=True, ignore=look.ignore)
            if s2 <= sc + 0.005:
                break
            Hr, sc, sup = H2, s2, sup2
        if (sc >= min_score and HG.support_ok(sup) and HG.h_distance(Hr, Hx, frame.shape, spec) <= 10.0
                and accept("verified", Hr)):
            info["h_source"] = "verified"
            return Hr, info
    return _multi_view(look, v, t, frame, spec, uv, mask, orb, matcher, info, agree_m=agree_m,
                       sample_ok=sample_ok, accept=accept)


def _multi_view(look, v, t: float, frame: np.ndarray, spec, uv, mask, orb, matcher, info: dict, span: float = 120.0,
                n_src: int = 30, min_inliers: int = 60, agree_m: float = 0.2, min_votes: int = 2, min_sep: float = 2.0,
                sample_ok=None, accept=None):
    """슛 프레임을 추적이 잘 된 다른 화면들(앞뒤 span 초 안, 서로 min_sep 초 이상 떨어진 n_src 장 + 사람 캘리브레이션 화면)과
    직접 특징점으로 맞춘다. 중계 카메라는 한 자리에서 돌고 당기기만 하므로 두 화면 사이는 관중석까지 통째로 하나의 화면 변환이다.
    서로 다른 시각의 화면 min_votes 장 이상에서 옮겨 온 H 가 발 자리에서 한 후보의 agree_m 안에 모일 때만 쓴다
    ("multi-view"). 모인 것끼리 가장 먼 거리(h_agree_m, 최대 2×agree_m)만큼 라인 근처 기준을 넓힌다(_judge). 따로따로 맞춘 것이 같은 답을 내야 하므로 한 화면과 잘못 맞은 경우는 걸러진다.
    (실제 B-LAB: 스코어버그 근처 장면에서 20~80초씩 정상 샘플이 없어 앞뒤 8초 이어 오기로는 6건을 못 살렸다)"""
    times, Hs, st = look.track["times"], look.track["H"], look.track["status"].astype(str)
    fin = np.all(np.isfinite(Hs.reshape(len(Hs), -1)), axis=1)
    good = fin & np.isin(st, _VERIFIED)
    near_idx = [int(i) for i in np.flatnonzero(good & (np.abs(times - t) <= span)) if sample_ok is None or sample_ok(int(i))]
    near_idx.sort(key=lambda i: abs(times[i] - t))
    picks: list[int] = []
    for i in near_idx:
        if len(picks) >= n_src:
            break
        if all(abs(times[i] - times[j]) >= min_sep for j in picks):
            picks.append(i)
    picks += [int(i) for i in np.flatnonzero(fin & (st == "calib")) if int(i) not in picks]
    if not picks:
        return None, info
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    A0 = abs(HG.key_area(Hs[picks[0]], spec)) or 1.0
    cands = []  # (시각, 맞은 점, H, 발 자리 코트 좌표)
    for i in picks:
        try:
            g = cv2.cvtColor(look.undistort(v.frame_at(float(times[i]))), cv2.COLOR_BGR2GRAY)
        except ValueError:
            continue
        F, n = HG.frame_to_frame(g, gray, orb, matcher, mask=mask, return_inliers=True)
        if F is None or n < min_inliers:
            continue
        Hc = F @ Hs[i]
        Hc = Hc / Hc[2, 2]
        A = HG.key_area(Hc, spec)
        if not np.isfinite(A) or abs(A) < 0.1 * A0 or A * HG.key_area(Hs[i], spec) <= 0:
            continue  # 코트가 쪼그라들거나 뒤집힌 변환
        p = np.array(HG.image_to_court(Hc, *uv)) if uv is not None else HG.project(Hc, [[14.0, 7.5]])[0]
        cands.append((float(times[i]), n, Hc, p))
    info["h_votes"] = 0
    if len(cands) < min_votes:
        return None, info
    best = None
    for c in cands:
        grp = [d for d in cands if np.linalg.norm(d[3] - c[3]) <= agree_m]
        ts = sorted({round(d[0], 1) for d in grp})
        distinct = sum(1 for k, x in enumerate(ts) if k == 0 or x - ts[k - 1] >= min_sep - 1e-6)
        key = (distinct, sum(d[1] for d in grp))
        if best is None or key > best[0]:
            best = (key, grp)
    (votes, _), grp = best
    info["h_votes"] = votes
    if votes < min_votes:
        return None, info
    w = np.array([d[1] for d in grp], float)
    Hm = sum(wi * d[2] for wi, d in zip(w, grp)) / w.sum()
    Hm = Hm / Hm[2, 2]
    P = np.stack([d[3] for d in grp])
    spread = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=2)))  # 같은 답을 낸 화면끼리 가장 먼 거리 (최대 2×agree_m)
    if accept is not None and not accept("multi-view", Hm):
        return None, info
    info["h_source"] = "multi-view"
    info["h_agree_m"] = round(spread, 3)
    info["h_gap_s"] = round(float(min(abs(d[0] - t) for d in grp)), 2)
    return Hm, info


def _judge(look, v, t: float, u_raw: float, v_raw: float, spec, side=None, window: float = 8.0, max_cam_px: float = 0.0,
           off_court_m: float = 1.0):
    """원본 화면 발 위치 하나를 판정 (shot·rejudge 공용). 반환: (ShotCall 또는 None, H 정보, 편 화면 좌표)."""
    u, vv = look.to_canvas(u_raw, v_raw)
    if v is not None:
        H, info = _court_at_shot(look, v, t, spec, uv=(u, vv), window=window, max_cam_px=max_cam_px,
                                 off_court_m=off_court_m, side=side)
    else:
        H = look.at(t)
        info = {"h_source": "track" if H is not None else "none", "h_gap_s": "", "h_agree_m": "", "h_votes": ""}
    if H is None:
        return None, info, (u, vv)
    frame = look.undistort(v.frame_at(t)) if v is not None else None
    call = HG.classify_click(H, u, vv, spec, side=side, frame=frame, ignore=look.ignore)
    if info.get("h_agree_m") not in ("", None) and info["h_source"] in ("two-way", "multi-view"):
        # 두 길이 발 자리에서 어긋난 만큼 위치가 흔들릴 수 있다 → 라인 근처 기준을 넓힌다
        call.margin_sigma = float(np.hypot(call.margin_sigma, float(info["h_agree_m"])))
        call.near_line = call.near_line or abs(call.margin) < max(0.3, 2.0 * call.margin_sigma)
    return call, info, (u, vv)


def cmd_cam_check(a):
    """추적 결과(npz)의 H 마다 '캘리브레이션 카메라 자리에서 돌리기·당기기로 흉내 낸 차이(px)' 를 재서, 상태별 분포와
    기준(--max-cam-px)을 넘는 구간을 출력한다. 카메라 자리 검사 전에 만든 npz 도 된다 (그때는 --video 로 화면 크기를 앎)."""
    spec = _spec(a.spec)
    look = _CourtLookup(None, a.track)
    tr = look.track
    v = Video(a.video) if a.video else None
    if v is None and "frame_shape" not in tr.files and look.lens is None:
        sys.exit("화면 크기를 알 수 없습니다: --video 를 주세요.")
    shape = look.shape(v)
    cams = look.cams(spec, shape)
    if not cams:
        sys.exit("카메라 자리를 못 풀었습니다 (calib 샘플이 없거나 실제 카메라로 풀리지 않음).")
    _print_cams(cams)
    times, Hs, st = tr["times"], tr["H"], tr["status"].astype(str)
    if "cam_px" in tr.files and "cam_C" in tr.files:
        cam_px = np.asarray(tr["cam_px"], float)
    else:
        cam_px = np.array([look.cam_px(H, spec, shape) if np.all(np.isfinite(H)) else np.nan for H in Hs])
    _print_cam_stats(st, cam_px, a.max_cam_px)
    over = np.isfinite(cam_px) & (cam_px > a.max_cam_px)
    spans, k = [], 0
    idx = np.flatnonzero(np.isfinite(cam_px))
    while k < len(idx):
        if not over[idx[k]]:
            k += 1
            continue
        j = k
        while j + 1 < len(idx) and over[idx[j + 1]]:
            j += 1
        seg = idx[k:j + 1]
        spans.append(seg)
        k = j + 1
    print(f"{a.max_cam_px:g}px 넘는 구간 {len(spans)}개" + (" (앞 40개)" if len(spans) > 40 else ""))
    for seg in spans[:40]:
        kinds = {}
        for i in seg:
            kinds[st[i]] = kinds.get(st[i], 0) + 1
        print(f"  {times[seg[0]]:.1f}~{times[seg[-1]]:.1f}s  {len(seg)}개 {kinds}  카메라 차이 "
              f"{np.nanmin(cam_px[seg]):.0f}~{np.nanmax(cam_px[seg]):.0f}px")
    if a.out:
        _write_csv(a.out, [{"t_sec": round(float(times[i]), 3), "status": st[i],
                            "cam_px": "" if not np.isfinite(cam_px[i]) else round(float(cam_px[i]), 2)}
                           for i in range(len(times))])
        print(f"저장: {a.out}")


def cmd_line_check(a):
    """추적 결과(npz)의 샘플마다 그 H 로 그린 코트 라인이 화면 흰 선과 얼마나 맞는지 (consensus 기준 정하기·틀린 H 찾기 진단).
    줄마다 8px(--tol) 안 30% 이상 겹치면 '맞은 줄'. center_x 는 화면 가운데가 보는 코트 x (하프라인 = 길이/2)."""
    spec = _spec(a.spec)
    v = Video(a.video)
    look = _lookup(a, v)
    tr = look.track
    times, Hs, st = tr["times"], tr["H"], tr["status"].astype(str)
    want = set(a.status.split(",")) if a.status else None
    rows = []
    for i in range(len(times)):
        if (want and st[i] not in want) or not np.all(np.isfinite(Hs[i])):
            continue
        if a.start is not None and times[i] < a.start or a.end is not None and times[i] > a.end:
            continue
        frame = look.undistort(v.frame_at(float(times[i])))
        e = HG.line_evidence(frame, Hs[i], spec, look.ignore, tol=a.tol)
        frac = e["n_good"] / e["n_vis"] if e["n_vis"] else float("nan")
        fd = e["n_good_d"] / e["n_vis_d"] if e["n_vis_d"] else float("nan")
        rows.append({"t_sec": round(float(times[i]), 3), "status": st[i], "n_vis": e["n_vis"], "n_good": e["n_good"],
                     "good_frac": round(frac, 3) if np.isfinite(frac) else "", "cross": e["cross"],
                     "n_vis_d": e["n_vis_d"], "n_good_d": e["n_good_d"],
                     "distinct_frac": round(fd, 3) if np.isfinite(fd) else "",
                     "contradict": int(HG.lines_contradict(frame, Hs[i], spec, look.ignore, frac=a.frac)),
                     "density": round(e["density"], 4), "area": round(e["area"], 3),
                     "center_x": round(e["center_x"], 2) if np.isfinite(e["center_x"]) else ""})
    for s_ in sorted({r["status"] for r in rows}):
        fr = np.array([r["good_frac"] for r in rows if r["status"] == s_ and r["good_frac"] != ""], float)
        fd = np.array([r["distinct_frac"] for r in rows if r["status"] == s_ and r["distinct_frac"] != ""], float)
        nc = sum(r["contradict"] for r in rows if r["status"] == s_)
        if len(fr):
            print(f"{s_}: {len(fr)}개, 맞은 줄 비율 중앙 {np.median(fr):.2f}, 10% {np.percentile(fr, 10):.2f}, "
                  + ", ".join(f"<{q:.2f} {int((fr < q).sum())}개" for q in (0.25, 0.4, 0.5, 0.6))
                  + (f" / 키·아크 줄 맞은 비율 중앙 {np.median(fd):.2f}, <0.5 {int((fd < 0.5).sum())}개" if len(fd) else "")
                  + f" / 흰 선과 어긋남(버릴 것) {nc}개")
    _write_csv(a.out, rows)
    print(f"저장: {a.out} ({len(rows)}개)")


def cmd_overlay(a):
    spec = _spec(a.spec)
    v = Video(a.video)
    look = _lookup(a, v)
    frame = look.undistort(v.frame_at(a.t))  # 렌즈 휨 보정한 추적이면 편 화면에 그린다
    H = look.at(a.t)
    if a.pred and look.track is not None and "H_pred" in look.track.files:
        # 진단: 좌표를 못 낸 샘플(coast)에서 추적기가 화면 이동으로 예측하던 위치를 빨간 선으로 그린다
        i = int(np.argmin(np.abs(look.track["times"] - a.t)))
        Hp = look.track["H_pred"][i]
        if np.all(np.isfinite(Hp)):
            HG.draw_court(frame, Hp, spec, (0, 0, 255), 1)
            print(f"예측 위치(빨강): {look.track['times'][i]:.2f}s 상태 {look.track['status'][i]} 라인 일치율 "
                  f"{float(look.track['line_score'][i]):.2f} 줄 {tuple(int(x) for x in look.track['support'][i])}")
        if H is None:
            cv2.imwrite(a.out, frame)
            print(f"저장: {a.out} (확인된 좌표 없음, 예측 위치만)")
            return
    if H is None:
        sys.exit("이 시각의 코트 변환이 없습니다 (화면 전환 구간일 수 있음).")
    if a.refine:
        H, sc = HG.refine_homography(frame, H, spec, ignore=look.ignore)
        print(f"라인 정합 일치율 {sc:.2f}")
    HG.draw_court(frame, H, spec, (0, 255, 255), 2)
    cv2.imwrite(a.out, frame)
    print(f"저장: {a.out}")


def _lookup(a, video=None) -> "_CourtLookup":
    """추적 결과·캘리브레이션 + 그래픽 자리(추적에 저장된 것 ∪ --graphics-box, 영상이 있고 추적에 없으면 자동으로 찾음)."""
    look = _CourtLookup(a.calib, a.track)
    if video is not None:
        auto = look.ignore is None and not getattr(a, "no_graphics_mask", False)
        look.add_graphics(_graphics_raw(video, 0.0, video.duration or 600.0, getattr(a, "graphics_box", None), auto=auto))
    return look


def cmd_shot(a):
    """원본 화면에서 찍은 발 위치(--u --v) 하나를 판정. --video 를 주면 그 화면에 보이는 3점 라인으로 보정하고(click-shots 와 같음),
    추적 결과에 그 프레임 좌표가 없으면 앞뒤 정상 화면에서 화면 이동으로 이어 온다. 렌즈 휨 보정이 있으면 편 화면에서 판정한다."""
    spec = _spec(a.spec)
    v = Video(a.video) if a.video else None
    look = _lookup(a, v)
    t = a.t or 0.0
    call, info, (u, vv) = _judge(look, v, t, a.u, a.v, spec, side=a.side, window=a.window, max_cam_px=a.max_cam_px,
                                 off_court_m=a.off_court_m)
    if call is None:
        sys.exit("코트 변환이 없습니다 (" + json.dumps(info, ensure_ascii=False) + ")")
    out = call.as_dict()
    out.update(info)
    if look.lens is not None:
        out["canvas_uv"] = [round(u, 1), round(vv, 1)]
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_rejudge(a):
    """click-shots CSV 의 발 위치(원본 화면 u, v)를 지금 코드로 모두 다시 판정 → CSV. 자유투 표시와 사람 라인 판정(line_call)은 지킨다.
    court_points = 코트만으로 낸 점수(사람 판정 전), points = score-check 가 쓰는 점수."""
    from . import scorecheck as SC

    spec = _spec(a.spec)
    v = Video(a.video)
    look = _lookup(a, v)
    rows = SC.read_csv(a.shots)
    out = []
    n_none = 0
    for r in rows:
        r2 = dict(r)
        try:
            t, u, vv = float(r["t_sec"]), float(r["u"]), float(r["v"])
        except (KeyError, TypeError, ValueError):
            out.append(r2)
            continue
        side = r.get("side") if r.get("side") in ("left", "right") else a.side
        call, info, _ = _judge(look, v, t, u, vv, spec, side=side, window=a.window, max_cam_px=a.max_cam_px,
                               off_court_m=a.off_court_m)
        is_ft = str(r.get("zone", "")).upper() == "FT"
        for k in _CALL_KEYS + ("court_points", "h_source", "h_gap_s", "h_agree_m", "h_votes", "h_cam_px", "h_reject"):
            r2.pop(k, None)  # 예전 판정 값이 남지 않게 (좌표 없음이면 빈칸)
        if call is None:
            n_none += 1
            r2.update({k: "" for k in _CALL_KEYS + ("court_points",)})
            if side:
                r2["side"] = side
        else:
            r2.update(call.as_dict())
            r2["court_points"] = call.points
        r2.update(info)
        if r.get("line_call") in ("on", "behind") and call is not None:
            r2["points"] = 2 if r["line_call"] == "on" else 3
        if is_ft:
            r2.update({"points": 1, "zone": "FT", "zone_kr": "자유투"})
        out.append(r2)
        print(f"{r.get('time') or t} {r.get('team', '')} {r.get('result', '')}: "
              + (f"{r2['points']}점 {r2.get('zone_kr', '')} 라인까지 {r2.get('margin_m', '')}m"
                 + (" 라인근처" if str(r2.get("near_line")) == "True" else "") if call is not None else "좌표 없음")
              + f"  [{info['h_source']}" + (f", {info['h_gap_s']}s" if info["h_gap_s"] != "" else "")
              + (f", 두 길 차이 {info['h_agree_m']}m" if info["h_agree_m"] != "" else "")
              + (f", 카메라 {info['h_cam_px']}px" if info.get("h_cam_px", "") != "" else "")
              + (f", 버림: {info['h_reject']}" if info.get("h_reject") else "") + "]")
    keys = list(dict.fromkeys(k for r in out for k in r))
    _write_csv(a.out, [{k: r.get(k, "") for k in keys} for r in out])
    src = {}
    for r in out:
        src[r.get("h_source", "")] = src.get(r.get("h_source", ""), 0) + 1
    print(f"저장: {a.out} ({len(out)}개, 좌표 없음 {n_none}개, 코트 변환 출처 {src})")


def cmd_click_shots(a):
    spec = _spec(a.spec)
    v = Video(a.video)
    look = _lookup(a, v)
    rows = []
    if Path(a.out).exists():
        with open(a.out, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    t = a.start
    playing = False
    team = "home"
    point = None
    call = None
    jersey = ""
    ft = False  # 자유투 모드 (f 로 켜고 끔)
    replay = False  # 리플레이 화면에서 찍은 슛 (r 로 켜고 끔)
    line_call = ""  # 라인 근처 슛을 사람이 확대 화면에서 본 판정: "on"(라인 밟음·안쪽 → 2점) / "behind"(라인 뒤 → 3점)
    clicks: list[tuple] = []  # 같은 슛을 여러 번 찍은 것 [(H, u, v)] — Shift+클릭으로 추가
    state = {"click": None, "add": False}
    h_cache: dict = {}
    win = "click-shots"
    cv2.namedWindow(win)
    s = 1.0

    def on_mouse(ev, x, y, flags, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
            state["add"] = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

    def court_at(t, frame):
        """이 프레임의 코트 변환. 추적 샘플(예: 0.2초 간격) 사이 프레임이면 카메라가 그새 움직였을 수 있어
        가장 가까운 샘플의 H 를 이 프레임의 흰 라인에 다시 맞춘다."""
        key = round(t, 3)
        if key not in h_cache:
            H, dt = look.nearest(t)
            if H is not None and abs(dt) > 0.5 / v.fps:
                Hr, sc = HG.refine_homography(frame, H, spec, ignore=look.ignore)
                if sc >= 0.3:
                    H = Hr
            h_cache.clear()
            h_cache[key] = H
        return h_cache[key]

    cv2.setMouseCallback(win, on_mouse)
    print(__SHOTS_HELP)
    if look.lens is not None:
        print(f"렌즈 휨을 편 화면으로 보여 줍니다 (k1={look.lens.k1:+.3f}). CSV 의 u, v 는 원본 화면 좌표로 저장합니다.")
    frame_at = lambda tt: look.undistort(v.frame_at(tt))  # noqa: E731
    frame = frame_at(t)
    while True:
        if playing:
            t += 1.0 / v.fps * 2
            ok, frame = v.cap.read()
            ok2 = v.cap.grab()
            if not (ok and ok2):
                playing = False
            else:
                frame = look.undistort(frame)
        H = court_at(t, frame)
        if state["click"] is not None:
            u, vv = state["click"][0] / s, state["click"][1] / s
            point = (u, vv)
            if not (state["add"] and clicks):
                clicks = []
                line_call = ""
            state["click"] = None
            playing = False
            if H is not None:
                # 이 프레임에 보이는 3점 라인에 맞춰 보정하도록 화면도 같이 넘긴다
                clicks.append((H.copy(), u, vv, frame.copy()))
                call = (HG.classify_clicks(clicks, spec, side=a.side, ignore=look.ignore) if len(clicks) > 1
                        else HG.classify_click(H, u, vv, spec, side=a.side, frame=frame, ignore=look.ignore))
            else:
                call = None
        view = frame.copy()
        if H is not None:
            HG.draw_court(view, H, spec, (0, 255, 255), 2)
            if point is not None:
                # 3점 라인 띠(바깥·안쪽 가장자리): 발끝이 이 띠에 조금이라도 닿으면 2점
                edges = C.three_line_edges(spec)
                for k in ("L_outer", "L_inner", "R_outer", "R_inner"):
                    q = HG.project(H, HG.densify(edges[k], 0.05)).astype(np.int32)
                    cv2.polylines(view, [q], False, (255, 0, 255), 1, cv2.LINE_AA)
        zoom = None
        if point is not None:
            col = (0, 255, 0) if (call and call.points == 3) else (255, 160, 0)
            # 확대 보기: 찍은 자리 주변을 4배로 키워 발과 라인이 닿았는지 본다 (점 찍기 전 그림에서)
            px, py, r = int(point[0]), int(point[1]), 30
            hh, ww = view.shape[:2]
            if r <= px < ww - r and r <= py < hh - r:
                zoom = cv2.resize(view[py - r:py + r, px - r:px + r], (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(zoom, (4 * r, 4 * r), col, cv2.MARKER_CROSS, 24, 1)
            cv2.circle(view, (px, py), 7, col, -1)
        view, s = _fit_window(view)
        if zoom is not None and view.shape[0] > zoom.shape[0] + 100 and view.shape[1] > zoom.shape[1]:
            view[100:100 + zoom.shape[0], view.shape[1] - zoom.shape[1]:] = zoom
        info = (f"{fmt_time(t)}  team={team}  no={jersey or '-'}  shots={len(rows)}" + ("  [FREE THROW]" if ft else "")
                + ("  [REPLAY]" if replay else "") + (f"  avg of {len(clicks)} clicks" if len(clicks) > 1 else ""))
        if line_call:
            info += "  [LINE: " + ("ON/INSIDE -> 2PT" if line_call == "on" else "BEHIND -> 3PT") + "]"
        if H is None:
            info += "  [NO COURT - recalibrate]"
        _put(view, info, (10, 28), 0.65, (0, 255, 255))
        if call:
            txt = (f"{call.points}PT {call.zone}  dist {call.dist_to_rim:.2f}m  toe-to-line {call.margin:+.2f}m"
                   + ("  ON LINE" if call.on_line else "") + ("  NEAR LINE!" if call.near_line else "")
                   + (f"  [line snap {call.line_snap_px:+.1f}px]" if call.line_snap_px is not None
                      else ("  [LINE BLOCKED by graphics/edge]" if call.line_hidden else "  [line hidden: H only]")))
            _put(view, txt, (10, 58), 0.65, (0, 255, 0) if call.points == 3 else (255, 200, 0))
            if call.ft_spot and not ft:
                _put(view, "FT spot: press f if this is a free throw", (10, 88), 0.6, (0, 200, 255))
            elif call.near_line:
                _put(view, "NEAR LINE: Shift+click again to average, or look at zoom: i = foot on/inside line (2PT), o = behind (3PT)",
                     (10, 88), 0.6, (0, 200, 255))
        cv2.imshow(win, view)
        key = cv2.waitKey(1 if playing else 30) & 0xFF
        if key == ord(" "):
            playing = not playing
            if playing:
                v.cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * v.fps))
        elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
            t = max(0.0, t + {ord("a"): -1, ord("d"): 1, ord("j"): -5, ord("l"): 5}[key])
            frame = frame_at(t)
            point, call, clicks = None, None, []
        elif key == ord(","):
            t = max(0.0, t - 1 / v.fps)
            frame = frame_at(t)
        elif key == ord("."):
            t += 1 / v.fps
            frame = frame_at(t)
        elif key == ord("t"):
            team = "away" if team == "home" else "home"
        elif key == ord("f"):
            ft = not ft
        elif key == ord("r"):
            replay = not replay
        elif key in (ord("i"), ord("o")) and call is not None and not ft:
            # 라인 근처: 확대 화면에서 발이 3점 라인을 밟았는지(또는 안쪽) 사람이 본 판정. 라인을 밟으면 2점 (FIBA)
            line_call = "" if line_call == ("on" if key == ord("i") else "behind") else ("on" if key == ord("i") else "behind")
        elif ord("0") <= key <= ord("9"):
            jersey = (jersey + chr(key))[-3:]
        elif key in (8, 127):  # Backspace
            jersey = ""
        elif key in (ord("m"), ord("x")) and point is not None:
            ur, vr = look.to_raw(point[0], point[1])  # 원본 화면 좌표로 저장 (렌즈 보정 여부와 상관없이 같은 기준)
            row = {"t_sec": round(t, 2), "time": fmt_time(t), "team": team, "no": jersey, "result": "MADE" if key == ord("m") else "MISSED",
                   "u": round(ur, 1), "v": round(vr, 1)}
            row.update(call.as_dict() if call else {"points": "", "zone": "", "zone_kr": "", "near_line": ""})
            row.update({"replay": replay, "clicks": len(clicks), "line_call": line_call})
            if line_call and call is not None:
                row["points"] = 2 if line_call == "on" else 3
            if ft:
                row.update({"points": 1, "zone": "FT", "zone_kr": "자유투"})
            rows.append(row)
            _write_csv(a.out, rows)
            print(f"기록: {row['time']} {team} #{jersey or '?'} {row['result']} {row.get('points')}점 {row.get('zone_kr')}")
            point, call, jersey, clicks, line_call = None, None, "", [], ""
        elif key == ord("z") and rows:
            print(f"삭제: {rows.pop()}")
            _write_csv(a.out, rows)
        elif key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()
    _write_csv(a.out, rows)
    print(f"저장: {a.out} ({len(rows)}개)")


__SHOTS_HELP = """
슛 위치 기록 (슛 라벨링 + 3점 라인 판정 확인)
  space 재생/멈춤   a/d ±1초   j/l ±5초   ,/. 한 프레임
  릴리스 순간 슈터의 두 발 가운데를 클릭 → 화면에 2점/3점과 '림 쪽 발끝 ~ 3점 라인 바깥 가장자리' 거리가 뜹니다
    (발끝은 두 발 가운데에서 0.14m 앞으로 계산. 분홍 두 줄 = 3점 라인 띠, 발끝이 조금이라도 닿으면 2점)
    발 근처에 보이는 칠해진 3점 라인에 맞춰 자동 보정합니다 (line snap). 수비수에 가려 안 보이면 'line hidden' 으로 표시
  숫자키로 슈터 등번호 입력 (Backspace 지우기, 모르면 비워 둠)
  m 성공으로 저장   x 실패로 저장   t 홈/원정(스코어버그 왼쪽/오른쪽 팀) 전환   z 마지막 기록 삭제   q 종료
  f 자유투 모드 켜기/끄기 (켜 있으면 1점·자유투로 저장, 자유투가 끝나면 다시 f 로 끄기)
  NEAR LINE 이 뜨면: , . 로 한두 프레임 옮겨 Shift+클릭으로 같은 발을 더 찍으면 평균해서 다시 판정 (오른쪽 위 확대 보기 참고)
    그래도 라인에 딱 걸쳐 있으면 확대 보기로 직접 판정: i = 발이 3점 라인을 밟았거나 안쪽(2점), o = 두 발 모두 라인 뒤(3점)
    (다시 누르면 취소. 라인 위에 선 발은 위치 계산만으로는 2·3점이 갈리지 않는다)
  r 리플레이 표시 켜기/끄기 (본 화면에서 못 본 슛을 리플레이에서 찍을 때. 점수 변화와 짝지을 때 뒤로 더 넓게 찾음)
"""


# ---------------------------------------------------------------- compare
def cmd_compare(a):
    from . import compare as CP
    from .records import load_records

    rec = load_records(a.records)
    events = CP.read_csv(a.score_events) if a.score_events else None
    shots = CP.read_csv(a.shots) if a.shots else None
    if events is None and shots is None:
        sys.exit("--score-events 나 --shots 중 하나는 있어야 비교할 수 있습니다.")
    games = [g.strip() for g in a.game.split(",") if g.strip()]
    all_zones = []
    for gid in games:
        lih = None
        if a.left_team:
            g = rec.games[gid]
            lih = a.left_team == g.home if a.left_team in (g.home, g.away) else None
        rep = CP.compare_game(rec, gid, events, shots, left_is_home=lih, start=a.start, end=a.end)
        print("\n".join(CP.summary_lines(rep)))
        out = a.out if len(games) == 1 else str(Path(a.out).with_name(f"{Path(a.out).stem}_{gid}{Path(a.out).suffix}"))
        CP.write_report(rep, out)
        print(f"저장: {out}\n")
        all_zones += rep.zones
    if a.apply_zones and all_zones:
        n = CP.apply_zones(a.records, a.apply_zones, all_zones)
        print(f"3점 구역 {n}칸을 채운 기록지 사본 저장: {a.apply_zones} (원본은 그대로)")


def cmd_score_check(a):
    from . import scorecheck as SC

    res = SC.check(SC.read_csv(a.score_events), SC.read_csv(a.shots), max_lag=a.max_lag, start=a.start, end=a.end)
    sm = res.summary()
    print("코트로 계산한 득점 vs 스코어버그 (정답)")
    for k, v in sm.items():
        print(f"  {k}: {v}")
    _write_csv(a.out, res.rows())
    print(f"저장: {a.out}")


def cmd_check_records(a):
    from .records import load_records, sanity_issues

    rec = load_records(a.records)
    issues = sanity_issues(rec)
    print(f"경기 {len(rec.games)}개, 선수-경기 기록 {len(rec.box)}줄, 3점 시도 체크리스트 {len(rec.threes)}줄")
    for i in issues:
        print(f"  {i['경기ID']:<6} {i['팀']:<8} {i['내용']}")
    print(f"확인할 항목 {len(issues)}개")


def cmd_bench(a):
    from .bench import run

    res = run(a.out, a.seconds, a.seed, a.step)
    print(f"[벤치] 어려운 합성 중계 {a.seconds:.0f}초 (seed {a.seed}) → 결과 폴더 {a.out}")
    print(f"  점수 변화: {res['점수 변화']}")
    print(f"  코트 추적: {res['코트 추적']}")
    print("  코트로 계산한 점수 vs 점수판:")
    for k, v in res["코트로 계산한 점수 vs 점수판"].items():
        print(f"    {k}: {v['일치']}/{v['점수판 변화']} 일치, 틀린 조합 {v['점수 비교']}")


def cmd_demo(a):
    from .demo import run

    res = run(a.out, a.seconds)
    print(f"[데모] 합성 영상 {a.seconds:.0f}초로 전체 흐름 점검 → 결과 폴더 {a.out}")
    for k in ("점수 변화", "코트 추적", "슛 판정"):
        print(f"  {k}: {res[k]}")
    for s in res["슛 판정 상세"]:
        print(f"    {'O' if s['맞음'] else 'X'} {s['상황']}: 정답 {s['정답']} / 판정 {s['판정']}")
    print(f"  걸린 시간(초): {res['걸린 시간(초)']}")
    print(f"  40분 경기 예상(분): {res['40분 경기 예상(분)']}")
    print(f"  확인용 그림: {Path(a.out) / 'overlay_grid.png'}")


# ---------------------------------------------------------------- main
SPEC_HELP = "코트 규격: FIBA, NBA 또는 체육관별 규격 JSON 경로 (예: {\"base\": \"FIBA\", \"three_radius\": 6.25})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtt", description="GARBAGE TIME tracker: 스코어버그 인식 + 코트 라인 기반 2점/3점 판정")
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        q = sp.add_parser(name, help=help_, description=help_)
        q.set_defaults(fn=fn)
        return q

    q = add("grab", cmd_grab, "영상의 한 프레임을 이미지로 저장")
    q.add_argument("--video", required=True)
    q.add_argument("--t", type=float, required=True, help="초")
    q.add_argument("--out", required=True)

    q = add("scorebug-setup", cmd_scorebug_setup, "스코어버그 점수 영역(ROI) 지정")
    q.add_argument("--video", required=True)
    q.add_argument("--t", type=float, default=600)
    q.add_argument("--out", default="configs/scorebug.json")
    q.add_argument("--home", default="HOME", help="왼쪽(홈) 팀 이름")
    q.add_argument("--away", default="AWAY", help="오른쪽(원정) 팀 이름")
    q.add_argument("--with-clock", action="store_true")

    q = add("scorebug-learn", cmd_scorebug_learn, "화면을 보며 점수를 입력해 숫자 글꼴 템플릿 등록")
    q.add_argument("--video", required=True)
    q.add_argument("--config", default="configs/scorebug.json")
    q.add_argument("--templates", default="configs/digits.npz")
    q.add_argument("--times", help="쉼표로 구분한 초 (예: 300,900,1500)")
    q.add_argument("--n", type=int, default=12, help="--times 가 없을 때 고르게 뽑을 개수")

    q = add("scorebug-read", cmd_scorebug_read, "영상 전체의 점수 변화 이벤트 추출 → CSV")
    q.add_argument("--video", required=True)
    q.add_argument("--config", default="configs/scorebug.json")
    q.add_argument("--templates", default="configs/digits.npz")
    q.add_argument("--step", type=float, default=0.5, help="몇 초마다 읽을지")
    q.add_argument("--start", type=float, default=0.0)
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--stable-n", type=int, default=3)
    q.add_argument("--out", default="out/score_events.csv")
    q.add_argument("--raw", default=None, help="프레임별 판독값 CSV (디버그용)")

    q = add("calibrate", cmd_calibrate, "코트 랜드마크를 클릭해 화면↔코트 변환 저장")
    q.add_argument("--video")
    q.add_argument("--image")
    q.add_argument("--t", type=float, default=600)
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--out", default="configs/calib.json")
    q.add_argument("--refine", action="store_true", help="흰 라인 픽셀에 자동으로 더 맞춤")
    q.add_argument("--fresh", action="store_true", help="기존 점 무시하고 새로 찍기")

    q = add("calib-edit", cmd_calib_edit, "흰 선 위에 선을 그어(또는 노란 선을 끌어) 캘리브레이션 고치기")
    q.add_argument("--video", required=True)
    q.add_argument("--calib", required=True, help="고칠 캘리브레이션 JSON")
    q.add_argument("--t", type=float, default=None, help="화면 시각 (기본: 캘리브레이션의 t_sec)")
    q.add_argument("--lens", default=None, help="lens-fit 결과 JSON. 주면 렌즈 휨을 편 화면에서 고치고 그 화면 H 로 저장")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--out", default=None, help="저장 파일 (기본: --calib 덮어쓰기, 처음 한 번 .bak.json 백업)")

    q = add("track-court", cmd_track_court, "팬·줌 카메라를 따라 프레임별 코트 변환 계산")
    q.add_argument("--video", required=True)
    q.add_argument("--calib", action="append", default=None,
                   help="캘리브레이션 JSON. 여러 번 주면 첫 번째에서 시작하고 나머지(예: 반대쪽 골대 화면)는 다시 찾기 기준으로 씀")
    q.add_argument("--min-score", type=float, default=0.65, help="라인 일치율이 이보다 낮으면 좌표를 내지 않음")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--start", type=float, default=None)
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--step", type=float, default=0.2)
    q.add_argument("--refine-every", type=int, default=1)
    q.add_argument("--recover-score", type=float, default=0.7, help="놓친 뒤 다시 찾은 화면을 받는 라인 일치율")
    q.add_argument("--reloc-every", type=int, default=5, help="좌표를 못 내는(coast) 샘플이 이만큼 이어질 때마다 기준 화면과 다시 맞춰 봄")
    q.add_argument("--bridge", type=float, default=4.0,
                   help="좌표를 못 낸 구간이 이 초 이하이고 앞뒤가 정상이면, 양쪽에서 화면 이동으로 따라와 서로 맞을 때만 메움 (0 = 끔)")
    q.add_argument("--lens", default=None, help="lens-fit 결과 JSON. 주면 렌즈 휨을 편 화면에서 추적·판정 (화면 가장자리 정확도)")
    q.add_argument("--no-graphics-mask", action="store_true", help="방송 그래픽 자리를 라인 정합에서 빼지 않음 (비교용)")
    q.add_argument("--lost-every", type=int, default=1,
                   help="5샘플 넘게 연달아 놓쳐 있으면 이 샘플마다만 다시 찾기 (1 = 매번). 클로즈업·광고가 잦은 다중 카메라 중계에서 3 정도로 빠르게")
    q.add_argument("--around", default=None,
                   help="슛 CSV(t_sec). 주면 경기 전체 대신 슛마다 앞 --before 초 ~ 뒤 --after 초만 추적해 npz 하나로 합침")
    q.add_argument("--before", type=float, default=20.0, help="--around: 슛 앞 몇 초부터 추적")
    q.add_argument("--after", type=float, default=5.0, help="--around: 슛 뒤 몇 초까지 추적")
    q.add_argument("--max-cam-px", type=float, default=0.0,
                   help="consensus 로 받을 H 가 캘리브레이션 화면들의 카메라 자리에서 돌리기·당기기로 이 px 안으로 흉내 나야 함 "
                        "(0 = 끔, 기본. 실제 영상에서 맞는 H 도 10~28px 이라 아직 진단용)")
    q.add_argument("--consensus-line-frac", type=float, default=0.25,
                   help="코트 안에 흰 선이 보이는데 consensus H 로 그린 줄 가운데 이 비율 미만만 흰 선(8px 안)과 겹치면 버림")
    q.add_argument("--no-consensus", dest="consensus", action="store_false",
                   help="라인으로 확인 못 한 화면을 '서로 다른 시각의 기준 화면 둘 이상이 같은 자리' 로 받지 않음 (비교용)")
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--out", default="out/court_track.npz")

    q = add("lens-fit", cmd_lens_fit, "캘리브레이션 화면들로 렌즈 휨(k1)을 재서 저장 → track-court --lens")
    q.add_argument("--video", required=True)
    q.add_argument("--calib", action="append", required=True, help="t_sec 가 있는 캘리브레이션 JSON (여러 번, 양쪽 골대 화면 권장)")
    q.add_argument("--track", default=None, help="(선택) 추적 결과에서 라인이 잘 맞은 화면을 더 씀")
    q.add_argument("--n-extra", type=int, default=8)
    q.add_argument("--min-score", type=float, default=0.8)
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--out", default="configs/lens.json")

    q = add("cam-check", cmd_cam_check, "추적 결과의 H 가 캘리브레이션 카메라 자리와 맞는지 (비틀린 H 찾기)")
    q.add_argument("--track", required=True)
    q.add_argument("--video", default=None, help="카메라 자리 검사 전에 만든 npz 면 화면 크기를 알기 위해 필요")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--max-cam-px", type=float, default=8.0)
    q.add_argument("--out", default=None, help="샘플별 CSV (t_sec, status, cam_px)")

    q = add("line-check", cmd_line_check, "추적 결과의 H 마다 코트 라인이 화면 흰 선과 얼마나 맞는지 (consensus 기준 진단)")
    q.add_argument("--track", required=True)
    q.add_argument("--video", required=True)
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--status", default="consensus", help="볼 상태 (쉼표로 여러 개, 빈칸 = 모두)")
    q.add_argument("--start", type=float, default=None)
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--tol", type=float, default=8.0)
    q.add_argument("--frac", type=float, default=0.25, help="contradict 열: track-court --consensus-line-frac 와 같은 기준")
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--no-graphics-mask", action="store_true")
    q.add_argument("--out", default="out/line_check.csv")
    q.set_defaults(calib=None)

    q = add("overlay", cmd_overlay, "코트 라인을 영상 프레임에 겹쳐 그려 확인")
    q.add_argument("--video", required=True)
    q.add_argument("--t", type=float, required=True)
    q.add_argument("--calib")
    q.add_argument("--track")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--refine", action="store_true")
    q.add_argument("--pred", action="store_true", help="진단: 추적기가 예측하던 위치(좌표 못 낸 구간 포함)를 빨간 선으로 같이 그림")
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--no-graphics-mask", action="store_true")
    q.add_argument("--out", default="out/overlay.png")

    q = add("shot", cmd_shot, "화면 좌표(슈터 발) 하나로 2점/3점 판정")
    q.add_argument("--video", default=None, help="주면 그 시각 화면의 3점 라인으로 보정 (click-shots 와 같은 판정)")
    q.add_argument("--u", type=float, required=True)
    q.add_argument("--v", type=float, required=True)
    q.add_argument("--t", type=float, default=None)
    q.add_argument("--calib")
    q.add_argument("--track")
    q.add_argument("--side", choices=["left", "right"], default=None, help="공격 방향 림. 모르면 가까운 림")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--window", type=float, default=8.0, help="그 프레임 좌표가 없을 때 앞뒤 몇 초 안의 정상 화면에서 이어 올지")
    q.add_argument("--max-cam-px", type=float, default=0.0, help="rejudge 와 같음 (0 = 끔)")
    q.add_argument("--off-court-m", type=float, default=1.0, help="rejudge 와 같음")
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--no-graphics-mask", action="store_true")

    q = add("click-shots", cmd_click_shots, "영상을 보며 슛 위치를 클릭해 기록 (2점/3점 자동 판정)")
    q.add_argument("--video", required=True)
    q.add_argument("--calib")
    q.add_argument("--track")
    q.add_argument("--start", type=float, default=0.0)
    q.add_argument("--side", choices=["left", "right"], default=None)
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--no-graphics-mask", action="store_true")
    q.add_argument("--out", default="out/shots.csv")

    q = add("rejudge", cmd_rejudge, "click-shots CSV 의 발 위치를 지금 코드로 모두 다시 판정 (좌표 없는 프레임은 앞뒤 화면에서 이어 옴)")
    q.add_argument("--shots", required=True, help="click-shots 결과 CSV (t_sec, u, v 필요)")
    q.add_argument("--video", required=True)
    q.add_argument("--track")
    q.add_argument("--calib")
    q.add_argument("--side", choices=["left", "right"], default=None, help="CSV 에 side 가 없을 때 공격 방향 림")
    q.add_argument("--spec", default="FIBA", help=SPEC_HELP)
    q.add_argument("--window", type=float, default=8.0)
    q.add_argument("--max-cam-px", type=float, default=0.0,
                   help="캘리브레이션 카메라 자리에서 돌리기·당기기로 이 px 안으로 흉내 나지 않는 H 는 버리고 다음 출처로 (0 = 끔, 기본)")
    q.add_argument("--off-court-m", type=float, default=1.0, help="발 자리가 코트 밖으로 이 m 넘게 나가면 그 H 는 버림")
    q.add_argument("--graphics-box", action="append", default=None,
                   help="방송 그래픽 자리 원본 화면 x,y,w,h (여러 번) 또는 {\"boxes\": [[x,y,w,h]...], \"reference_size\": [w,h]} JSON. 라인으로 쓰지 않고 가려진 것으로 봄")
    q.add_argument("--no-graphics-mask", action="store_true")
    q.add_argument("--out", default="out/shots_rejudged.csv")

    q = add("check-records", cmd_check_records, "기록지 엑셀의 이상한 값 점검 (득점 합계, 리바운드, 성공>시도 등)")
    q.add_argument("--records", required=True)

    q = add("compare", cmd_compare, "자동 기록을 D3 기록지 엑셀과 경기 단위로 비교 → 엑셀 보고서")
    q.add_argument("--records", required=True, help="기록지 엑셀 (예: 'D3 기록지_1R-11R.xlsx')")
    q.add_argument("--game", required=True, help="경기ID. 여러 개면 쉼표로 (예: D101,D102)")
    q.add_argument("--score-events", help="scorebug-read 결과 CSV (하루치 여러 경기여도 됨)")
    q.add_argument("--shots", help="click-shots 결과 CSV")
    q.add_argument("--left-team", help="스코어버그 왼쪽 팀 이름. 생략하면 최종 스코어로 자동 판단")
    q.add_argument("--start", type=float, default=None, help="이 시각(초) 이후 기록만 사용")
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--out", default="out/compare.xlsx")
    q.add_argument("--apply-zones", help="3점 구역 제안을 채운 기록지 사본 저장 경로 (원본은 수정 안 함)")

    q = add("score-check", cmd_score_check, "코트(슛 위치)로 정한 자유투/2점/3점이 스코어버그 점수 변화와 맞는지 측정")
    q.add_argument("--score-events", required=True, help="scorebug-read 결과 CSV (정답)")
    q.add_argument("--shots", required=True, help="click-shots 결과 CSV (또는 자동 슛 검출 결과)")
    q.add_argument("--max-lag", type=float, default=20.0, help="슛 뒤 스코어버그가 바뀌기까지 최대 몇 초 (실제 중계에서 16초 지연도 있었음)")
    q.add_argument("--start", type=float, default=None)
    q.add_argument("--end", type=float, default=None)
    q.add_argument("--out", default="out/score_check.csv")

    q = add("demo", cmd_demo, "영상 없이 합성 중계로 전체 흐름 점검 (설치 확인용)")
    q.add_argument("--out", default="out/demo")
    q.add_argument("--seconds", type=float, default=60.0)

    q = add("bench", cmd_bench, "어려운 합성 중계(빠른 팬·리플레이·로고·자유투)로 정확도 측정")
    q.add_argument("--out", default="out/bench")
    q.add_argument("--seconds", type=float, default=90.0)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--step", type=float, default=0.5)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "calib", None) is None and getattr(a, "track", None) is None and a.cmd in ("overlay", "shot", "click-shots", "rejudge"):
        sys.exit("--calib 또는 --track 중 하나가 필요합니다.")
    a.fn(a)


if __name__ == "__main__":
    main()
