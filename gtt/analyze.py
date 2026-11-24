"""`gtt analyze`: 경기 영상 하나를 넣으면 자동으로 분석해 기록지(엑셀)·평가지표·보고서(HTML)를 만든다.

흐름
1. 점수판 읽기 (scorebug-read) → 언제 어느 팀이 +1/+2/+3
2. 코트 추적 (track-court, 캘리브레이션이 있을 때) → 확인용 코트 라인 그림
3. 슛 기록 (--shots, click-shots 결과 또는 나중의 자동 슛 검출 결과)이 있으면 시도·구역·선수별 기록과
   '코트로 정한 점수 vs 점수판' 일치율
4. 기록지 엑셀 (D3 기록지 [기록입력] 형식) + 평가지표(설정 파일 수식) + 보고서 HTML

방송(중계 채널·체육관)마다 한 번 만드는 '프로필' 폴더가 필요하다: scorebug.json, digits.npz(선택),
calib*.json(선택). 기본은 configs/ 이고, 기존 명령(scorebug-setup, scorebug-learn, calibrate)이 만드는 파일 그대로다.

`gtt analyze --demo` 는 합성 중계 영상과 프로필을 만들어 같은 흐름을 영상 없이 끝까지 보여 준다.
"""
from __future__ import annotations

import base64
import contextlib
import csv
import html
import io
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from . import boxscore as BX
from . import metrics as MT
from . import zones as ZN
from .scorecheck import check as score_check, read_csv

AUTO_NOTE = "영상에서 자동"
LATER = "추후 (자동 감지 개발 중)"


# ---------------------------------------------------------------- 파이프라인
def _quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def _profile_files(profile: Path) -> dict:
    sb = profile / "scorebug.json"
    if not sb.exists():
        raise SystemExit(
            f"{sb} 가 없습니다. 이 중계를 처음 분석한다면 한 번만 준비하세요:\n"
            f"  gtt scorebug-setup --video <영상> --out {sb}\n"
            f"  gtt scorebug-learn --video <영상> --config {sb} --templates {profile / 'digits.npz'}\n"
            f"  (선택) gtt calibrate --video <영상> --t <초> --refine --out {profile / 'calib.json'}\n"
            "영상 없이 흐름만 보려면: gtt analyze --demo")
    calibs = sorted(profile.glob("calib*.json"), key=lambda p: (p.name != "calib.json", p.name))
    dg = profile / "digits.npz"
    return {"scorebug": sb, "digits": dg if dg.exists() else None, "calibs": calibs}


def _read_roster(path: str | None, team_names: dict) -> dict:
    """명단 CSV (열: team, no, name). team 은 home/away 또는 팀 이름."""
    if not path:
        return {}
    by_name = {v: k for k, v in team_names.items()}
    out = {}
    for r in read_csv(path):
        t = r.get("team", "").strip()
        t = t if t in ("home", "away") else by_name.get(t)
        if t and r.get("no"):
            out[(t, str(r["no"]).strip())] = r.get("name", "").strip()
    return out


def _overlay_grid(video: Path, track: Path, out: Path, duration: float) -> Path | None:
    import cv2

    from . import cli

    z = np.load(track)
    good = [float(t) for t, s in zip(z["times"], z["status"]) if s in ("ok", "recovered", "calib", "bridged")]
    if not good:
        return None
    picks = [good[int(round(k * (len(good) - 1) / 3))] for k in range(4)]
    tiles = []
    for t in picks:
        p = out / f"overlay_{int(t):04d}s.png"
        try:
            _quiet(cli.main, ["overlay", "--video", str(video), "--t", f"{t:.2f}", "--track", str(track), "--out", str(p)])
        except SystemExit:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.putText(img, f"{int(t // 60)}:{t % 60:04.1f}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360)))
        p.unlink(missing_ok=True)
    if not tiles:
        return None
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = out / "overlay_grid.jpg"
    cv2.imwrite(str(grid), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return grid


def run(video: str | Path, out_dir: str | Path, profile: str | Path = "configs", shots: str | None = None,
        roster: str | None = None, metrics_path: str | None = None, game_id: str = "G001", game_date: str | None = None,
        home: str | None = None, away: str | None = None, step: float = 0.5, track_step: float = 0.2,
        start: float | None = None, end: float | None = None, demo_note: str | None = None, shots_fn=None, hybrid: bool = True) -> dict:
    from . import cli
    from . import scorebug as SB
    from .video import Video

    video, out, profile = Path(video), Path(out_dir), Path(profile)
    out.mkdir(parents=True, exist_ok=True)
    files = _profile_files(profile)
    metric_list, _ = MT.load(metrics_path)  # 수식 오류는 영상 처리 전에 알린다
    cfg = SB.ScorebugConfig.load(files["scorebug"])
    team_names = {"home": home or cfg.home_name, "away": away or cfg.away_name}
    v = Video(video)
    duration = v.duration
    v.close()
    timing: dict[str, float] = {}

    def step_(name, fn):
        t0 = time.perf_counter()
        print(f"[{len(timing) + 1}] {name} ...", file=sys.stderr)
        r = fn()
        timing[name] = round(time.perf_counter() - t0, 1)
        return r

    # 1) 점수판
    ev_csv = out / "score_events.csv"
    args = ["scorebug-read", "--video", str(video), "--config", str(files["scorebug"]), "--step", str(step), "--out", str(ev_csv)]
    args += ["--templates", str(files["digits"]) if files["digits"] else ""]
    if start is not None:
        args += ["--start", str(start)]
    if end is not None:
        args += ["--end", str(end)]
    step_("점수판 읽기", lambda: _quiet(cli.main, args))
    events = read_csv(ev_csv)

    # 2) 코트 추적
    track, grid, track_info = None, None, None
    if files["calibs"]:
        track = out / "court_track.npz"
        args = ["track-court", "--video", str(video), "--step", str(track_step), "--out", str(track)]
        for c in files["calibs"]:
            args += ["--calib", str(c)]
        if end is not None:
            args += ["--end", str(end)]
        step_("코트 추적", lambda: _quiet(cli.main, args))
        z = np.load(track)
        ok = int(sum(s in ("ok", "recovered", "calib", "bridged") for s in z["status"]))
        track_info = {"프레임": int(len(z["times"])), "코트 좌표 있음": ok, "비율": round(ok / max(1, len(z["times"])), 3)}
        grid = step_("코트 라인 확인 그림", lambda: _overlay_grid(video, track, out, duration))

    # 3) 슛 기록
    if shots_fn is not None and track is not None:
        shots = shots_fn(track)
    shot_rows = read_csv(shots) if shots else None
    agree, resolve_stats = None, None
    if shot_rows is not None:
        res = score_check(events, shot_rows, start=start, end=end)  # 코트 단독 지표: 보정 전 슛으로
        agree = res.summary()
        if hybrid:
            from .resolve import resolve
            shot_rows, resolve_stats = resolve(events, shot_rows)
        with open(out / "score_check.csv", "w", encoding="utf-8-sig", newline="") as f:
            rows = res.rows()
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # 4) 기록지·지표
    box = BX.build(events, shot_rows, team_names, _read_roster(roster, team_names))
    met = compute_metrics(box, metric_list)
    ok_ev = [e for e in events if e.get("flag", "ok") == "ok"]
    final = (int(ok_ev[-1]["home"]), int(ok_ev[-1]["away"])) if ok_ev else (0, 0)
    if ok_ev:
        e0 = ok_ev[0]
        h0 = int(e0["home"]) - (int(e0["delta"]) if e0["team"] == "home" else 0)
        a0 = int(e0["away"]) - (int(e0["delta"]) if e0["team"] == "away" else 0)
        if (h0, a0) != (0, 0):
            box.notes.insert(0, f"영상이 {h0}:{a0} 에서 시작해 그 전 득점은 기록에 없습니다. 기록지 득점은 영상 안에서 난 점수만입니다.")
    game = {"경기ID": game_id, "날짜": game_date or date.today().isoformat(), "홈팀": team_names["home"], "원정팀": team_names["away"],
            "홈 득점": final[0], "원정 득점": final[1],
            "영상": video.name, "영상 길이(초)": round(duration, 1)}
    summary = {"경기": game, "걸린 시간(초)": timing, "점수판 변화": len(events),
               "점수판 확인 필요": sum(1 for e in events if e.get("flag", "ok") != "ok"),
               "코트 추적": track_info, "코트 점수 vs 점수판": agree, "점수 출처": resolve_stats, "메모": box.notes,
               "슛 기록": shots or None, "프로필": str(profile), "데모": demo_note}
    if duration:
        per = sum(timing.values()) / duration
        summary["40분 경기 예상(분)"] = round(per * 2400 / 60, 1)
    write_xlsx(out / "기록지_자동.xlsx", game, box, met, events, shot_rows, summary, metric_list)
    write_html(out / "report.html", game, box, met, events, shot_rows, summary, metric_list, grid)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return summary


def compute_metrics(box: BX.Box, metric_list: list[MT.Metric]) -> dict:
    out = {"players": [], "teams": []}
    for L in box.players:
        if L.name == "(미배정)":
            continue
        out["players"].append((L, {m.key: MT.evaluate(m, L.s, box.available) for m in metric_list}))
    for t in ("home", "away"):
        L = box.teams[t]
        out["teams"].append((L, {m.key: MT.evaluate(m, L.s, box.team_available) for m in metric_list}))
    return out


# ---------------------------------------------------------------- 엑셀 (D3 기록지 형식)
D3_HEAD3 = [("기본 정보", 4), ("2점슛", 2), ("3점슛", 2), ("3점 코너", 2), ("3점 45도", 2), ("3점 탑", 2), ("자유투", 2),
            ("기타", 4), ("리바운드", 2), ("자동 계산", 10)]
D3_HEAD4 = ["경기ID", "날짜", "팀", "선수명", "시도", "성공", "시도", "성공", "시도", "성공", "시도", "성공", "시도", "성공", "시도", "성공",
            "어시스트", "블락", "스틸", "턴오버", "수비", "공격", "3P 시도", "3P 성공", "야투 시도", "야투 성공", "득점", "총 리바운드",
            "2P%", "3P%", "FT%", "검증"]
D3_KEYS = ["P2A", "P2M", "P3A", "P3M", "corner_A", "corner_M", "wing_A", "wing_M", "top_A", "top_M", "FTA", "FTM",
           "AST", "BLK", "STL", "TOV", "DRB", "ORB", "P3A", "P3M", "FGA", "FGM", "PTS", "REB"]


def _pct(m, a):
    if m is None or a is None:
        return None
    return "-" if a == 0 else m / a


def d3_row(game: dict, team_name: str, name: str, s: dict) -> list:
    vals = [game["경기ID"], game["날짜"], team_name, name] + [s.get(k) for k in D3_KEYS]
    vals += [_pct(s.get("P2M"), s.get("P2A")), _pct(s.get("P3M"), s.get("P3A")), _pct(s.get("FTM"), s.get("FTA"))]
    vals.append("자동")
    return vals


def write_xlsx(path: Path, game, box: BX.Box, met, events, shots, summary, metric_list) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    head = PatternFill("solid", fgColor="FFD9E1F2")
    later = PatternFill("solid", fgColor="FFE7E6E6")
    auto = PatternFill("solid", fgColor="FFE2EFDA")
    bold = Font(bold=True)
    thin = Side(style="thin", color="FFBFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    wb = Workbook()
    ws = wb.active
    ws.title = "기록입력"
    ws["A1"] = "경기별 선수 기록 (영상 자동 분석)"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "초록 칸 = 영상에서 자동으로 채움 · 회색 빈칸 = 아직 자동 감지 안 됨(추후) · D3 기록지 [기록입력]과 같은 열 순서라 그대로 복사해 붙일 수 있습니다"
    c = 1
    for title, n in D3_HEAD3:
        ws.cell(3, c, title).font = bold
        if n > 1:
            ws.merge_cells(start_row=3, start_column=c, end_row=3, end_column=c + n - 1)
        c += n
    for i, h in enumerate(D3_HEAD4, 1):
        for r in (3, 4):
            ws.cell(r, i).fill = head
            ws.cell(r, i).alignment = center
            ws.cell(r, i).border = border
        ws.cell(4, i, h).font = bold
    rows = [(L, box.available) for L in box.players if L.name != "(미배정)"]
    if not rows:  # 선수 구분이 없으면 팀 합계 줄
        rows = [(box.teams[t], box.team_available) for t in ("home", "away")]
    r = 5
    for L, avail in rows:
        name = L.name if rows[0][0] not in box.teams.values() else "(팀 합계)"
        vals = d3_row(game, box.team_names[L.team], name, L.s)
        for i, v in enumerate(vals, 1):
            cell = ws.cell(r, i, v)
            cell.border = border
            if i > 4:
                cell.fill = later if v is None else auto
            if isinstance(v, float):
                cell.number_format = "0.0%"
        r += 1
    for i in range(1, len(D3_HEAD4) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 12 if i <= 4 else 8
    ws.freeze_panes = "E5"

    # 팀 기록
    t = wb.create_sheet("팀 기록")
    keys = [("PTS", "득점"), ("FGM", "야투 성공"), ("FGA", "야투 시도"), ("P2M", "2점 성공"), ("P2A", "2점 시도"), ("P3M", "3점 성공"),
            ("P3A", "3점 시도"), ("corner_M", "코너3 성공"), ("corner_A", "코너3 시도"), ("wing_M", "45도3 성공"), ("wing_A", "45도3 시도"),
            ("top_M", "탑3 성공"), ("top_A", "탑3 시도"), ("FTM", "자유투 성공"), ("FTA", "자유투 시도"), ("REB", "리바운드"),
            ("AST", "어시스트"), ("STL", "스틸"), ("BLK", "블락"), ("TOV", "턴오버")]
    t.append(["항목", box.team_names["home"], box.team_names["away"], "출처"])
    for k, lab in keys:
        hv, av = box.teams["home"].s.get(k), box.teams["away"].s.get(k)
        src = "점수판" if k == "PTS" else (AUTO_NOTE if hv is not None else LATER)
        t.append([lab, hv, av, src])
    for row in t.iter_rows(min_row=2):
        for cell in row[1:3]:
            cell.fill = later if cell.value is None else auto
    for cell in t[1]:
        cell.font, cell.fill = bold, head
    t.column_dimensions["A"].width, t.column_dimensions["D"].width = 14, 26

    # 경기목록 (D3 [경기목록] 열 순서)
    g = wb.create_sheet("경기목록")
    g.append(["경기ID", "날짜", "홈팀", "원정팀", "홈 득점", "원정 득점", "대회/시즌", "비고"])
    g.append([game["경기ID"], game["날짜"], game["홈팀"], game["원정팀"], game["홈 득점"], game["원정 득점"], "", f"영상 자동 분석: {game['영상']}"])
    for cell in g[1]:
        cell.font, cell.fill = bold, head

    # 평가지표
    m = wb.create_sheet("평가지표")
    hdr = ["팀", "선수"]
    for mm in metric_list:
        hdr += [mm.name, f"{mm.key} 부분값(없는 기록=0)", f"{mm.key} 빠진 기록"]
    m.append(hdr)
    for L, vals in met["players"] + [(L, v) for L, v in met["teams"]]:
        who = L.name if L.name else "(팀 합계)"
        if L in box.teams.values():
            who = "(팀 합계)"
        row = [box.team_names[L.team], who]
        for mm in metric_list:
            e = vals[mm.key]
            row += [e["value"] if e["value"] is not None else ("계산 불가" if e["missing"] else "-"), e["partial"], ", ".join(e["missing_kr"])]
        m.append(row)
    for cell in m[1]:
        cell.font, cell.fill = bold, head
    for j, mm in enumerate(metric_list):
        if "%" in mm.fmt:
            for col in (3 + 3 * j, 4 + 3 * j):
                for row in m.iter_rows(min_row=2, min_col=col, max_col=col):
                    row[0].number_format = "0.0%"
    m.append([])
    m.append(["수식 (설정 파일에서 바꿀 수 있음)"])
    for mm in metric_list:
        p = ", ".join(f"{k}={v}" for k, v in mm.params.items())
        m.append([mm.name, mm.formula + (f"   [{p}]" if p else ""), mm.note])

    # 점수 흐름, 슛 기록
    e = wb.create_sheet("점수 흐름")
    e.append(["시각", "초", "팀", "점수", box.team_names["home"], box.team_names["away"], "판독"])
    for ev in events:
        e.append([ev.get("time"), float(ev["t_sec"]), box.team_names.get(ev["team"], ev["team"]), int(ev["delta"]),
                  int(ev["home"]), int(ev["away"]), ev.get("flag")])
    for cell in e[1]:
        cell.font, cell.fill = bold, head
    if shots:
        s = wb.create_sheet("슛 기록")
        cols = ["time", "team", "no", "result", "points", "points_source", "zone_kr", "margin_m", "near_line", "court_x", "court_y"]
        s.append(["시각", "팀", "등번호", "결과", "점수", "점수 출처", "구역", "3점 라인까지(m)", "라인 근처", "코트 x(m)", "코트 y(m)",
                  "세부 구역", "거리대", "림까지(m)", "각도(°)"])
        for r_ in shots:
            row = [r_.get(k) if k != "team" else box.team_names.get(r_.get("team"), r_.get("team")) for k in cols]
            z = _zone_row(r_)
            row += [z["zone"], z["band"], z["dist_m"], z["angle_deg"]] if z else ["", "", "", ""]
            s.append(row)
        for cell in s[1]:
            cell.font, cell.fill = bold, head

        zs = wb.create_sheet("구역별 슛")
        tal = ZN.tally(shots)
        r0 = 1
        for kind, title, order in (("zone", "세부 구역 (코트 좌표 기준)", ZN.ZONES_2 + ZN.ZONES_3), ("band", "거리대 (림 중심에서)", ZN.BANDS)):
            zs.cell(r0, 1, title).font = bold
            hdr = ["구역", "점수"]
            for t in ("home", "away"):
                hdr += [f"{box.team_names[t]} 시도", "성공", "성공률"]
            for j, h in enumerate(hdr, 1):
                c_ = zs.cell(r0 + 1, j, h)
                c_.font, c_.fill = bold, head
            r = r0 + 2
            for zname in order:
                pts = 3 if (zname in ZN.ZONES_3 or zname in ZN.BANDS[4:]) else 2
                row = [zname, pts]
                for t in ("home", "away"):
                    a, m_ = tal[kind].get((t, zname), [0, 0])
                    row += [a, m_, (m_ / a) if a else None]
                for j, v in enumerate(row, 1):
                    c_ = zs.cell(r, j, v)
                    if j in (5, 8) and v is not None:
                        c_.number_format = "0%"
                r += 1
            r0 = r + 1
        zs.column_dimensions["A"].width = 20

    # 설명
    d = wb.create_sheet("설명")
    lines = [
        "영상 자동 분석 기록지",
        f"영상: {game['영상']} ({game['영상 길이(초)']}초)",
        f"자동으로 채운 것: 팀 득점(점수판), " + ("슛 시도·성공·3점 구역·선수별(등번호) 기록" if shots else "3점·자유투 성공 횟수(점수판 +3/+1)"),
        "아직 비어 있는 것: 리바운드, 어시스트, 스틸, 블락, 턴오버, 파울 — 자동 감지 개발 중",
    ] + ([] if shots else ["슛 시도와 2점 성공은 슛 기록(click-shots 또는 자동 슛 검출)이 있어야 채워집니다."]) + list(box.notes)
    if summary.get("데모"):
        lines.insert(1, summary["데모"])
    for ln in lines:
        d.append([ln])
    d["A1"].font = Font(bold=True, size=13)
    d.column_dimensions["A"].width = 110
    wb.move_sheet("설명", offset=-(len(wb.sheetnames) - 1))
    wb.active = 1
    wb.save(path)


# ---------------------------------------------------------------- 보고서 HTML
CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e3e1dc;--home:#d4572a;--away:#2c6fbb;--ok:#2f8f4e;--bad:#c0392b;
--later:#eeece8;--auto:#e7f3ea;--court:#f3e6cf;--courtline:#9b8466}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171715;--card:#22221f;--ink:#ecebe7;--mute:#a19f99;--line:#3a3935;
--home:#f08a5d;--away:#6aa8f0;--ok:#5cc07c;--bad:#f07a6d;--later:#2c2b28;--auto:#1f3326;--court:#3a3226;--courtline:#a08c6c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI","Malgun Gothic","Apple SD Gothic Neo",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px}
.sub{color:var(--mute);margin:0 0 18px}.banner{background:var(--later);border-radius:10px;padding:10px 14px;margin:0 0 18px;font-size:14px}
.score{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:8px 0 18px}.score .t{font-size:18px;font-weight:600}
.score .n{font-size:44px;font-weight:700;font-variant-numeric:tabular-nums}.home{color:var(--home)}.away{color:var(--away)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.card b{display:block;font-size:22px}
.card span{color:var(--mute);font-size:13px}.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:600}td:first-child,th:first-child,td.l,th.l{text-align:left}td.na{background:var(--later);color:var(--mute);text-align:center}
td.auto{background:var(--auto)}.pill{display:inline-block;border-radius:99px;padding:1px 8px;font-size:12px;background:var(--later)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin:8px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}
svg{max-width:100%;height:auto;display:block}img{max-width:100%;border-radius:8px;display:block}
ul{margin:6px 0;padding-left:20px}code{background:var(--later);padding:1px 5px;border-radius:4px;font-size:12.5px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:760px){.two{grid-template-columns:1fr}}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(v, k=None):
    if v is None:
        return None
    if isinstance(v, float):
        return f"{v:.1%}" if k == "pct" else f"{v:g}"
    return str(v)


def _score_svg(events, names, duration) -> str:
    ok = [e for e in events if e.get("flag", "ok") == "ok"]
    if not ok:
        return "<p class=sub>점수판 변화가 없습니다.</p>"
    W, H, P = 900, 260, 40
    tmax = max(duration, max(float(e["t_sec"]) for e in ok)) or 1
    h0 = int(ok[0]["home"]) - (int(ok[0]["delta"]) if ok[0]["team"] == "home" else 0)
    a0 = int(ok[0]["away"]) - (int(ok[0]["delta"]) if ok[0]["team"] == "away" else 0)
    vmax = max(max(int(e["home"]), int(e["away"])) for e in ok) or 1
    vmin = min(h0, a0)
    X = lambda t: P + (W - 2 * P) * t / tmax
    Y = lambda v: H - P + -(H - 2 * P) * (v - vmin) / max(1, vmax - vmin)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="점수 흐름">']
    for k in range(5):
        v = vmin + (vmax - vmin) * k / 4
        out.append(f'<line x1="{P}" x2="{W - P}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
                   f'<text x="{P - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="var(--mute)">{v:.0f}</text>')
    for k in range(5):
        t = tmax * k / 4
        out.append(f'<text x="{X(t):.1f}" y="{H - P + 18}" font-size="11" text-anchor="middle" fill="var(--mute)">{int(t // 60)}:{int(t % 60):02d}</text>')
    for team, col, v0 in (("home", "var(--home)", h0), ("away", "var(--away)", a0)):
        pts, cur = [(0.0, v0)], v0
        for e in ok:
            nv = int(e[team])
            if nv != cur:
                pts += [(float(e["t_sec"]), cur), (float(e["t_sec"]), nv)]
                cur = nv
        pts.append((tmax, cur))
        d = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for e in ok:
            if e["team"] == team:
                out.append(f'<circle cx="{X(float(e["t_sec"])):.1f}" cy="{Y(int(e[team])):.1f}" r="4" fill="{col}"><title>'
                           f'{_e(e.get("time"))} {_e(names[team])} +{e["delta"]}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def _court_svg(shots, names) -> str:
    from .court import FIBA, court_polylines

    if not shots:
        return ""
    s = 30
    W, H = FIBA.length * s, FIBA.width * s
    X = lambda x: x * s
    Y = lambda y: (FIBA.width - y) * s
    out = [f'<svg viewBox="-10 -10 {W + 20:.0f} {H + 20:.0f}" role="img" aria-label="슛 위치"><rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="var(--court)"/>']
    for line in court_polylines(FIBA).values():
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in line)
        out.append(f'<polyline points="{d}" fill="none" stroke="var(--courtline)" stroke-width="2"/>')
    for r in shots:
        try:
            x, y = float(r["court_x"]), float(r["court_y"])
        except (KeyError, TypeError, ValueError):
            continue
        col = "var(--home)" if r.get("team") == "home" else "var(--away)"
        made = str(r.get("result", "")).upper() in ("MADE", "성공", "1", "TRUE")
        tip = f'{r.get("time", "")} {names.get(r.get("team"), "")} #{r.get("no", "")} {r.get("zone_kr", "")} {"성공" if made else "실패"}'
        if made:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="8" fill="{col}"><title>{_e(tip)}</title></circle>')
        else:
            a, b = X(x), Y(y)
            out.append(f'<g stroke="{col}" stroke-width="3"><title>{_e(tip)}</title><line x1="{a - 7:.1f}" y1="{b - 7:.1f}" x2="{a + 7:.1f}" y2="{b + 7:.1f}"/>'
                       f'<line x1="{a - 7:.1f}" y1="{b + 7:.1f}" x2="{a + 7:.1f}" y2="{b - 7:.1f}"/></g>')
    out.append("</svg>")
    return "".join(out)


def _zone_row(r: dict) -> dict | None:
    if str(r.get("zone", "")).upper() == "FT":
        return None
    try:
        x, y = float(r["court_x"]), float(r["court_y"])
    except (KeyError, TypeError, ValueError):
        return None
    side = r.get("side") if r.get("side") in ("left", "right") else None
    return ZN.zone_of(x, y, side=side)


def _zone_tables(shots, names) -> str:
    tal = ZN.tally(shots)
    P = ["<h2>구역별 슛 (코트 좌표 기준)</h2><p class=sub>카메라 화면이 아니라 코트 규격 좌표(m)로 나눈 구역입니다. "
         "좌/우는 베이스라인에서 코트를 바라본 방향, 칸은 성공/시도.</p><div class=two>"]
    for kind, title, order in (("zone", "세부 구역", ZN.ZONES_2 + ZN.ZONES_3), ("band", "거리대", ZN.BANDS)):
        h = [f"<div class=panel><table><thead><tr><th class=l>{title}</th><th>점수</th>"
             f"<th class=home>{_e(names['home'])}</th><th class=away>{_e(names['away'])}</th></tr></thead><tbody>"]
        for z in order:
            pts = 3 if (z in ZN.ZONES_3 or z in ZN.BANDS[4:]) else 2
            cells = []
            for t in ("home", "away"):
                a, m = tal[kind].get((t, z), [0, 0])
                cells.append(f"<td>{m}/{a} <small>{m / a:.0%}</small></td>" if a else "<td style='color:var(--mute)'>0/0</td>")
            h.append(f"<tr><td class=l>{_e(z)}</td><td>{pts}</td>{''.join(cells)}</tr>")
        h.append("</tbody></table></div>")
        P.append("".join(h))
    P.append("</div>")
    if tal["skipped"]:
        P.append(f"<p class=sub>코트 좌표가 없는 슛 {tal['skipped']}개는 구역에서 뺐습니다.</p>")
    return "".join(P)


def _cell(v, k=None, avail=True):
    f = _fmt(v, k)
    if f is None:
        return '<td class=na title="아직 자동 감지 안 됨">–</td>'
    return f"<td class=auto>{_e(f)}</td>"


BOX_COLS = [("PTS", "득점"), ("P2M", "2점 성공"), ("P2A", "2점 시도"), ("P3M", "3점 성공"), ("P3A", "3점 시도"),
            ("corner_M", "코너"), ("wing_M", "45도"), ("top_M", "탑"), ("FTM", "자유투 성공"), ("FTA", "자유투 시도"),
            ("REB", "리바"), ("AST", "어시"), ("STL", "스틸"), ("BLK", "블락"), ("TOV", "턴오버")]


def _box_table(lines, names, team_line=False) -> str:
    h = ['<table><thead><tr><th class=l>팀</th><th class=l>선수</th>']
    h += [f"<th>{_e(l)}</th>" for _, l in BOX_COLS] + ["</tr></thead><tbody>"]
    for L in lines:
        who = "(팀 합계)" if team_line else L.name
        h.append(f'<tr><td class="l {L.team}">{_e(names[L.team])}</td><td class=l>{_e(who)}</td>')
        for k, _ in BOX_COLS:
            v = L.s.get(k)
            if k in ("corner_M", "wing_M", "top_M") and v is not None:
                a = L.s.get(k.replace("_M", "_A"))
                h.append(f"<td class=auto>{v}/{a}</td>" if a is not None else f"<td class=auto>{v}</td>")
            else:
                h.append(_cell(v))
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def _metric_table(rows, metric_list, names, team=False) -> str:
    h = ['<table><thead><tr><th class=l>팀</th><th class=l>선수</th>'] + [f"<th>{_e(m.name)}</th>" for m in metric_list] + ["</tr></thead><tbody>"]
    for L, vals in rows:
        h.append(f'<tr><td class="l {L.team}">{_e(names[L.team])}</td><td class=l>{_e("(팀 합계)" if team else L.name)}</td>')
        for m in metric_list:
            e = vals[m.key]
            if e["value"] is None and not e["missing"]:
                h.append("<td title='분모가 0 (시도 없음)'>-</td>")
            elif e["value"] is None:
                miss = ", ".join(e["missing_kr"])
                h.append(f'<td class=na title="빠진 기록: {_e(miss)} · 없는 기록을 0으로 둔 부분값 {_e(e["partial_text"])}">'
                         f'계산 불가<br><small>부분값 {_e(e["partial_text"])}</small></td>')
            else:
                h.append(f"<td class=auto>{_e(e['text'])}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def write_html(path: Path, game, box: BX.Box, met, events, shots, summary, metric_list, grid: Path | None) -> None:
    names = box.team_names
    hp, ap = game["홈 득점"], game["원정 득점"]
    tr = summary.get("코트 추적")
    ag = summary.get("코트 점수 vs 점수판")
    cards = [(f"{len(events)}", "점수판 변화 감지" + (f" (확인 필요 {summary['점수판 확인 필요']})" if summary["점수판 확인 필요"] else ""))]
    if tr:
        cards.append((f"{tr['비율']:.0%}", f"코트 추적 성공 프레임 ({tr['코트 좌표 있음']}/{tr['프레임']})"))
    if ag and ag.get("짝지은 변화"):
        n_ok, n_all = ag["일치"], ag["점수판 변화"]
        cls = "ok" if n_ok == n_all else "bad"
        cards.append((f'<span class="{cls}">{n_ok}/{n_all}</span>', "코트로 정한 점수 = 점수판 (목표 100%)"))
    if shots:
        cards.append((f"{len(shots)}", "슛 기록 (성공·실패)"))
    total = sum(summary["걸린 시간(초)"].values())
    cards.append((f"{total:.0f}초", f"분석 시간 · 40분 경기 예상 {summary.get('40분 경기 예상(분)', '-')}분"))

    auto_list = ["팀 득점과 득점 시각 (점수판)"]
    if shots:
        auto_list += ["슛 시도·성공, 2점/3점 판정 (코트 라인 기준)", "3점 구역 (코너·45도·탑)", "선수별 기록 (등번호)"]
    else:
        auto_list += ["3점 성공 (+3), 자유투 성공 (+1) 횟수"]
    if tr:
        auto_list.append("카메라 팬·줌을 따라가는 코트 좌표")
    later = ["리바운드 (공격/수비)", "어시스트", "스틸", "블락", "턴오버", "파울"]
    if not shots:
        later = ["슛 시도·실패, 2점 성공 (자동 슛 검출 개발 중 · 지금은 click-shots 로 기록)"] + later

    P = ["<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<title>경기 분석 리포트</title><style>{CSS}</style></head><body><main>"]
    P.append(f"<h1>경기 자동 분석 리포트</h1><p class=sub>{_e(game['경기ID'])} · {_e(game['날짜'])} · 영상 {_e(game['영상'])} "
             f"({int(game['영상 길이(초)'] // 60)}분 {int(game['영상 길이(초)'] % 60)}초)</p>")
    if summary.get("데모"):
        P.append(f"<div class=banner>{_e(summary['데모'])}</div>")
    P.append(f"<div class=score><span class='t home'>{_e(names['home'])}</span><span class='n home'>{hp}</span>"
             f"<span class=n>:</span><span class='n away'>{ap}</span><span class='t away'>{_e(names['away'])}</span></div>")
    P.append("<div class=cards>" + "".join(f"<div class=card><b>{v}</b><span>{_e(l)}</span></div>" for v, l in cards) + "</div>")

    P.append("<h2>자동으로 된 것 / 아직인 것</h2><div class=two><div class=panel><b class=ok>영상에서 자동</b><ul>"
             + "".join(f"<li>{_e(x)}</li>" for x in auto_list) + "</ul></div><div class=panel><b>추후 (빈칸으로 둠)</b><ul>"
             + "".join(f"<li>{_e(x)}</li>" for x in later) + "</ul></div></div>")
    if box.notes:
        P.append("<div class=panel style='margin-top:12px'><b>확인할 점</b><ul>" + "".join(f"<li>{_e(n)}</li>" for n in box.notes) + "</ul></div>")

    P.append("<h2>점수 흐름</h2><div class=panel>" + _score_svg(events, names, game["영상 길이(초)"])
             + f"<div class=legend><span><i class=sw style='background:var(--home)'></i>{_e(names['home'])}</span>"
               f"<span><i class=sw style='background:var(--away)'></i>{_e(names['away'])}</span></div></div>")

    legend = ("<div class=legend><span><i class=sw style='background:var(--auto)'></i>영상에서 자동</span>"
              "<span><i class=sw style='background:var(--later)'></i>– 추후 (아직 자동 감지 안 됨)</span></div>")
    P.append("<h2>기록지</h2>" + legend + "<div class=panel>" + _box_table([box.teams['home'], box.teams['away']], names, True) + "</div>")
    if box.players:
        P.append("<h3>선수별</h3><div class=panel>" + _box_table([L for L in box.players if L.name != "(미배정)"], names) + "</div>")

    P.append("<h2>평가지표</h2><p class=sub>수식은 설정 파일에서 바꿀 수 있습니다. 수식에 아직 못 재는 기록이 있으면 '계산 불가'로 두고, "
             "참고용으로 그 기록을 0으로 둔 부분값을 작게 보여 줍니다.</p>")
    if met["players"]:
        P.append("<div class=panel>" + _metric_table(met["players"], metric_list, names) + "</div><div style='height:12px'></div>")
    P.append("<div class=panel>" + _metric_table(met["teams"], metric_list, names, team=True) + "</div>")
    P.append("<div class=panel style='margin-top:12px'><table><thead><tr><th class=l>지표</th><th class=l>수식</th><th class=l>메모</th></tr></thead><tbody>")
    for m in metric_list:
        p = ", ".join(f"{k}={v:g}" for k, v in m.params.items())
        P.append(f"<tr><td class=l>{_e(m.name)}</td><td class=l style='white-space:normal;min-width:240px'><code>{_e(m.formula)}</code>{('<br><small>' + _e(p) + '</small>') if p else ''}</td>"
                 f"<td class=l style='white-space:normal'>{_e(m.note)}</td></tr>")
    P.append("</tbody></table></div>")

    if shots:
        P.append("<h2>슛 위치</h2><div class=panel>" + _court_svg(shots, names)
                 + f"<div class=legend><span>● 성공</span><span>✕ 실패</span><span><i class=sw style='background:var(--home)'></i>{_e(names['home'])}</span>"
                   f"<span><i class=sw style='background:var(--away)'></i>{_e(names['away'])}</span></div></div>")
    rs = summary.get("점수 출처")
    if rs:
        P.append("<h2>2점/3점은 어떻게 정했나</h2><p class=sub>코트 좌표로 확실한 슛은 코트로, 라인 근처이거나 발이 가려져 "
                 "코트로 못 정한 <b>성공</b> 슛만 그 순간 점수판(+2/+3)으로 정했습니다. 실패 슛은 점수판이 안 바뀌어 보정할 수 없습니다. "
                 "위의 '코트로 정한 점수 = 점수판' 일치율은 보정 전 코트 판정만으로 잰 값입니다.</p><div class=panel><table><tbody>")
        for k, lab in (("코트", "코트 좌표로 확실히 판정"), ("점수판 보정", "점수판으로 보정 (코트로 못 정한 성공 슛)"),
                       ("학습 기준", "이 경기 성공 슛으로 학습한 발끝 기준으로 판정 (실패 슛)"),
                       ("판정 불가", "판정 불가 (사람 확인)"), ("코트≠점수판", "코트 판정과 점수판이 다름 (사람 확인)")):
            v = rs.get(k, 0)
            cls = " class=bad" if k in ("판정 불가", "코트≠점수판") and v else ""
            P.append(f"<tr><td class=l>{_e(lab)}</td><td{cls}>{v}</td></tr>")
        lm = rs.get("라인 기준")
        if lm and lm["n2"] + lm["n3"]:
            P.append(f"<tr><td class=l style='white-space:normal'>학습된 발끝 기준: 찍은 발 가운데가 3점 라인 바깥에서 "
                     f"{lm['lo']:+.2f}~{lm['hi']:+.2f} m 사이면 판정 불가 (점수판 정답 성공 슛 2점 {lm['n2']}개·3점 {lm['n3']}개로 학습"
                     f"{', 설명 안 되는 기록 ' + str(lm['errors']) + '개' if lm['errors'] else ''})</td><td></td></tr>")
        P.append("</tbody></table></div>")
    if shots:
        P.append(_zone_tables(shots, names))
    if ag and ag.get("짝지은 변화"):
        P.append("<h2>코트로 정한 점수 vs 점수판</h2><div class=panel><table><tbody>")
        for k in ("점수판 변화", "짝지은 변화", "일치", "슛 없이 바뀐 점수", "점수 변화 없는 성공 슛", "라인 근처 슛이 낀 변화", "라인 근처에서 틀림"):
            P.append(f"<tr><td class=l>{_e(k)}</td><td>{_e(ag.get(k))}</td></tr>")
        P.append("</tbody></table></div>")

    if grid and grid.exists():
        b64 = base64.b64encode(grid.read_bytes()).decode()
        P.append("<h2>코트 라인 추적 확인</h2><p class=sub>노란 선이 실제 코트 라인에 겹쳐 있으면 카메라 추적이 맞은 것입니다.</p>"
                 f"<div class=panel><img alt='코트 라인 추적 확인' src='data:image/jpeg;base64,{b64}'></div>")

    P.append("<h2>득점 기록</h2><div class=panel><table><thead><tr><th class=l>시각</th><th class=l>팀</th><th>점수</th><th>스코어</th><th class=l>판독</th></tr></thead><tbody>")
    for e in events:
        fl = e.get("flag", "ok")
        P.append(f"<tr><td class=l>{_e(e.get('time'))}</td><td class='l {e['team']}'>{_e(names.get(e['team'], e['team']))}</td>"
                 f"<td>+{_e(e['delta'])}</td><td>{_e(e['home'])} : {_e(e['away'])}</td>"
                 f"<td class=l>{'<span class=ok>정상</span>' if fl == 'ok' else '<span class=bad>확인 필요</span>'}</td></tr>")
    P.append("</tbody></table></div>")
    P.append("<p class=sub style='margin-top:28px'>분석 단계별 시간(초): " + _e(", ".join(f"{k} {v}" for k, v in summary["걸린 시간(초)"].items()))
             + " · 만든 도구: <code>gtt analyze</code></p></main></body></html>")
    path.write_text("".join(P), encoding="utf-8")


# ---------------------------------------------------------------- 데모
DEMO_SHOTS = [
    # (초, 팀, 등번호, 결과, (x, y) 코트 m) — 득점 슛은 demo.SCORE_PLAN 보다 1.5초 앞
    (2.5, "away", "9", "MISSED", (8.6, 9.6)),
    (4.5, "home", "7", "MADE", (2.6, 7.9)),
    (11.0, "home", "23", "MISSED", (1.575 + 7.25, 7.5)),
    (12.5, "away", "5", "MADE", (0.9, 0.6), "hidden"),  # 수비수에 발이 가려 좌표 없음 → 점수판 +3 으로 보정
    (17.0, "away", "9", "MISSED", (1.575 + 5.3, 11.6)),
    (19.5, "home", "11", "MADE", "FT"),
    (25.5, "home", "11", "MADE", "FT"),
    (30.0, "home", "7", "MISSED", (1.575 + 3.8, 3.4)),
    (31.5, "away", "13", "MADE", (5.4, 9.2)),
    (36.0, "away", "5", "MISSED", (1.575 + 7.0 * 0.7, 7.5 + 7.0 * 0.7)),
    (38.5, "home", "23", "MADE", (1.575 + 6.95, 7.3)),  # 발끝이 라인 바로 뒤 (라인 근처) → 점수판 +3 으로 보정
    (44.0, "home", "7", "MISSED", (1.2, 14.3)),
    (47.5, "away", "9", "MADE", (3.4, 6.2)),
]


def _demo_shots(track: Path, truth: list, fps: int) -> list[dict]:
    """합성 정답 슛 위치를 화면 좌표로 옮긴 뒤, 실제 click-shots 와 똑같이 추적된 코트 변환으로 판정한다."""
    from . import homography as HG
    from .court import FIBA
    from .video import fmt_time

    z = np.load(track)
    out = []
    for t, team, no, res, pos, *flag in DEMO_SHOTS:
        row = {"t_sec": t, "time": fmt_time(t), "team": team, "no": no, "result": res}
        i = int(np.argmin(np.abs(z["times"] - t)))
        Ht = z["H"][i]
        if pos == "FT":
            x, y = FIBA.rim("left")[0] + 4.6, 7.5
        else:
            x, y = pos
        u, v = HG.court_to_image(truth[min(int(round(t * fps)), len(truth) - 1)], x, y)
        row.update({"u": round(float(u), 1), "v": round(float(v), 1)})
        if np.all(np.isfinite(Ht)):
            row.update(HG.classify_click(Ht, float(u), float(v), FIBA, side="left").as_dict())
        if pos == "FT":
            row.update({"points": 1, "zone": "FT", "zone_kr": "자유투"})
        if flag and flag[0] == "hidden":  # 발이 안 보여 찍지 못함
            row.update({k: "" for k in ("u", "v", "points", "zone", "zone_kr", "margin_m", "near_line", "court_x", "court_y")})
        row.update({"replay": False, "clicks": 1})
        out.append(row)
    return out


def run_demo(out_dir: str | Path, seconds: float = 60.0, metrics_path: str | None = None) -> dict:
    from . import demo as D

    out = Path(out_dir)
    prof = out / "profile"
    prof.mkdir(parents=True, exist_ok=True)
    video = out / "demo_game.avi"
    print("[0] 합성 중계 영상과 방송 프로필 만들기 ...", file=sys.stderr)
    truth = D.make_video(video, seconds, start=(0, 0))
    D.build_scorebug_profile(prof)
    D.build_calib(video, truth, prof / "calib.json")
    # 슛 위치: 실제 영상에서는 click-shots 로 찍는 단계. 데모에서는 합성 정답 위치를 추적된 코트 변환으로 판정한다.
    def shots_fn(track):
        shots = _demo_shots(track, truth, D.FPS)
        shots_csv = out / "shots.csv"
        with open(shots_csv, "w", encoding="utf-8-sig", newline="") as f:
            keys = list(dict.fromkeys(k for r in shots for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(shots)
        return str(shots_csv)

    roster = out / "roster.csv"
    roster.write_text("team,no,name\nhome,7,레드 7번\nhome,11,레드 11번\nhome,23,레드 23번\n"
                      "away,5,블루 5번\naway,9,블루 9번\naway,13,블루 13번\n", encoding="utf-8")
    note = ("데모: 합성 중계 영상입니다. 점수판 읽기·코트 추적·점수 판정은 실제 명령 그대로 돌렸고, "
            "사람이 한 번 하는 준비(점수판 영역·숫자 등록, 코트 점 찍기)와 슛 위치 클릭은 합성 정답으로 대신했습니다.")
    return run(video, out, prof, shots_fn=shots_fn, roster=str(roster), metrics_path=metrics_path, game_id="DEMO",
               home="레드", away="블루", step=0.5, track_step=0.5, demo_note=note)
