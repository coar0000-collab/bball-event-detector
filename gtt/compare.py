"""자동 기록(스코어버그 점수 이벤트, 슛 기록)과 D3 기록지를 경기 단위로 비교한다.

- 하루치 중계(여러 경기)의 점수 이벤트에서 해당 경기 구간을 최종 스코어로 자동으로 찾는다.
- 스코어버그 왼쪽/오른쪽이 기록지의 홈/원정 중 어느 쪽인지도 스코어로 정한다.
- 3점 시도 슛 기록이 있으면 [3점슛_구역체크]의 각 줄에 코너/45도/탑 구역을 제안한다.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .records import STAT_LABELS, Records, ThreeAttempt, derive, sanity_issues

ZONE_KEYS = {"코너": "c3", "45도": "w3", "탑": "t3"}


# ---------------------------------------------------------------- 입력
def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Segment:
    """점수 이벤트 중 한 경기 구간."""

    events: list[dict]
    t0: float
    t1: float
    left: int
    right: int


def segment_games(events: list[dict], reset_max: int = 3, min_total_before: int = 20) -> list[Segment]:
    """하루 중계의 점수 이벤트를 경기별로 나눈다. 점수가 0 근처로 돌아가면 새 경기."""
    segs: list[list[dict]] = [[]]
    prev_total = 0
    for e in events:
        h, a = int(e["home"]), int(e["away"])
        if h <= reset_max and a <= reset_max and prev_total >= min_total_before and segs[-1]:
            segs.append([])
        if e.get("flag") == "correction" and h <= reset_max and a <= reset_max:
            prev_total = h + a
            continue  # 리셋 이벤트 자체는 경기 기록이 아님
        segs[-1].append(e)
        prev_total = h + a
    out = []
    for s in segs:
        if not s:
            continue
        out.append(Segment(s, float(s[0]["t_sec"]), float(s[-1]["t_sec"]), int(s[-1]["home"]), int(s[-1]["away"])))
    return out


@dataclass
class Match:
    segment: Segment | None
    left_is_home: bool
    score_error: int | None  # |자동 최종 스코어 − 기록지 스코어| 합


def match_game(segments: list[Segment], home_pts: int | None, away_pts: int | None, left_is_home: bool | None = None) -> Match:
    """기록지 최종 스코어와 가장 가까운 구간과 좌/우 방향을 고른다."""
    if not segments:
        return Match(None, True if left_is_home is None else left_is_home, None)
    if home_pts is None or away_pts is None:
        seg = max(segments, key=lambda s: s.left + s.right)
        return Match(seg, True if left_is_home is None else left_is_home, None)
    best = None
    for s in segments:
        for lih in ([left_is_home] if left_is_home is not None else [True, False]):
            hp, ap = (s.left, s.right) if lih else (s.right, s.left)
            err = abs(hp - home_pts) + abs(ap - away_pts)
            if best is None or err < best.score_error:
                best = Match(s, lih, err)
    return best


# ---------------------------------------------------------------- 자동 기록 집계
def scorebug_totals(seg: Segment) -> dict[str, dict[str, int]]:
    """왼쪽/오른쪽 팀별 최종 득점과 +1/+2/+3 횟수 (flag=ok 만)."""
    out = {side: {"pts": 0, "plus1": 0, "plus2": 0, "plus3": 0, "check": 0} for side in ("home", "away")}
    out["home"]["pts"], out["away"]["pts"] = seg.left, seg.right
    for e in seg.events:
        d = int(e["delta"])
        if e.get("flag") == "ok" and d in (1, 2, 3):
            out[e["team"]][f"plus{d}"] += 1
        else:
            out[e["team"]]["check"] += 1
    return out


def shot_totals(shots: list[dict], by_player: bool = False) -> dict:
    """click-shots 기록을 팀(또는 팀·등번호)별로 집계."""
    out: dict = {}
    for s in shots:
        key = (s["team"], s.get("no") or "") if by_player else s["team"]
        t = out.setdefault(key, {k: 0 for k in ("p2a", "p2m", "p3a", "p3m", "c3a", "c3m", "w3a", "w3m", "t3a", "t3m")})
        made = s["result"] == "MADE"
        pts = str(s.get("points") or "")
        if pts == "3":
            t["p3a"] += 1
            t["p3m"] += made
            zk = ZONE_KEYS.get(s.get("zone_kr") or "")
            if zk:
                t[zk + "a"] += 1
                t[zk + "m"] += made
        elif pts == "2":
            t["p2a"] += 1
            t["p2m"] += made
    for t in out.values():
        t["fga"] = t["p2a"] + t["p3a"]
        t["fgm"] = t["p2m"] + t["p3m"]
    return out


# ---------------------------------------------------------------- 3점 구역 제안
@dataclass
class ZoneSuggestion:
    attempt: ThreeAttempt
    shot: dict | None
    status: str  # 일치 / 결과 다름 / 영상 기록 없음

    def as_row(self) -> dict:
        a, s = self.attempt, self.shot or {}
        return {
            "엑셀 행": a.row, "경기ID": a.game, "쿼터": a.quarter, "3점 순번": a.seq, "직전 스코어": a.prev_score,
            "팀": a.team, "등번호": a.number, "선수명": a.player, "결과(기록지)": "성공" if a.made else "실패",
            "기존 구역": a.zone or "", "제안 구역": s.get("zone_kr", ""), "영상 시각": s.get("time", ""),
            "결과(영상)": {"MADE": "성공", "MISSED": "실패"}.get(s.get("result", ""), ""),
            "라인까지(m)": s.get("margin_m", ""), "상태": self.status,
        }


def score_at(seg: Segment | None, t: float, left_is_home: bool) -> str | None:
    """t 초에 스코어버그가 보여 주던 점수 (기록지 방향 '홈 : 원정')."""
    if seg is None:
        return None
    h = a = 0
    for e in seg.events:
        if float(e["t_sec"]) > t:
            break
        h, a = int(e["home"]), int(e["away"])
    if not left_is_home:
        h, a = a, h
    return f"{h} : {a}"


def _norm_score(s: str) -> str:
    return ":".join(p.strip() for p in str(s).split(":"))


def align_threes(attempts: list[ThreeAttempt], shots: list[dict], seg: Segment | None = None, left_is_home: bool = True) -> list[ZoneSuggestion]:
    """기록지 3점 시도(순서)와 영상 3점 시도(시각 순)를 팀별로 정렬해 짝짓는다.

    결과(성공/실패)가 같으면 +2, 스코어버그의 그 순간 점수가 '직전 스코어'와 같으면 +3,
    등번호가 같으면 +2 (다르면 -1) 를 주는
    서열 정렬(Needleman-Wunsch). 짝이 없는 기록지 줄은 '영상 기록 없음'.
    """
    out: list[ZoneSuggestion] = []
    teams = list(dict.fromkeys(a.team for a in attempts))
    for team in teams:
        A = sorted([a for a in attempts if a.team == team], key=lambda a: a.seq)
        B = sorted([s for s in shots if s.get("team_name") == team and str(s.get("points")) == "3"], key=lambda s: float(s["t_sec"]))
        sb = [score_at(seg, float(s["t_sec"]), left_is_home) for s in B]

        def sim(i, j):
            v = 2 if A[i].made == (B[j]["result"] == "MADE") else -1
            if sb[j] is not None and _norm_score(sb[j]) == _norm_score(A[i].prev_score):
                v += 3
            no = str(B[j].get("no") or "").strip()
            if no and A[i].number is not None:
                v += 2 if no == str(A[i].number) else -1
            return v

        n, m, gap = len(A), len(B), -1
        D = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            D[i][0] = i * gap
        for j in range(1, m + 1):
            D[0][j] = j * gap
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i][j] = max(D[i - 1][j - 1] + sim(i - 1, j - 1), D[i - 1][j] + gap, D[i][j - 1] + gap)
        i, j, pairs = n, m, {}
        while i > 0 and j > 0:
            if D[i][j] == D[i - 1][j - 1] + sim(i - 1, j - 1):
                pairs[i - 1] = j - 1
                i, j = i - 1, j - 1
            elif D[i][j] == D[i - 1][j] + gap:
                i -= 1
            else:
                j -= 1
        for k, a in enumerate(A):
            if k in pairs:
                s = B[pairs[k]]
                same = a.made == (s["result"] == "MADE")
                out.append(ZoneSuggestion(a, s, "일치" if same else "결과 다름"))
            else:
                out.append(ZoneSuggestion(a, None, "영상 기록 없음"))
    out.sort(key=lambda z: z.attempt.row)
    return out


# ---------------------------------------------------------------- 비교
@dataclass
class Report:
    game: str
    home: str
    away: str
    match: Match | None
    team_rows: list[dict] = field(default_factory=list)
    player_rows: list[dict] = field(default_factory=list)
    zones: list[ZoneSuggestion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _row(level, team, player, key, ref, auto, note=""):
    diff = None if ref is None or auto is None else auto - ref
    return {"구분": level, "팀": team, "선수": player, "항목": STAT_LABELS.get(key, key), "기록지": ref, "자동": auto, "차이": diff, "비고": note}


def compare_game(
    rec: Records,
    game: str,
    score_events: list[dict] | None = None,
    shots: list[dict] | None = None,
    left_is_home: bool | None = None,
    start: float | None = None,
    end: float | None = None,
) -> Report:
    if game not in rec.games:
        raise KeyError(f"기록지 [경기목록]에 {game} 이 없습니다.")
    g = rec.games[game]
    rep = Report(game, g.home, g.away, None)
    ref = rec.team_totals(game)
    if not ref:
        rep.notes.append("기록지에 이 경기의 선수 기록이 없습니다 (몰수 경기일 수 있음).")
    # 기록지 자체가 의심스러운 값 (예: 1R 리바운드가 슛한 선수에게 붙어 있음) → 비교할 때 믿지 말 것
    for issue in sanity_issues(rec, games=[game]):
        rep.notes.append(f"기록지 점검 ({issue['팀']}): {issue['내용']} → 이 항목은 기준값으로 믿지 마세요")

    def in_range(t: float) -> bool:
        return (start is None or t >= start) and (end is None or t <= end)

    # 1) 스코어버그
    seg = None
    if score_events:
        evs = [e for e in score_events if in_range(float(e["t_sec"]))]
        segs = segment_games(evs)
        m = match_game(segs, g.home_pts, g.away_pts, left_is_home)
        rep.match = m
        seg = m.segment
        if seg is not None:
            lih = m.left_is_home
            side_team = {"home": g.home if lih else g.away, "away": g.away if lih else g.home}
            rep.notes.append(
                f"스코어버그 구간 {seg.t0:.0f}~{seg.t1:.0f}초 (전체 {len(segs)}개 구간 중 선택), 왼쪽 팀 = {side_team['home']}, "
                f"최종 {seg.left}:{seg.right} → 기록지와 차이 {m.score_error}점")
            sb = scorebug_totals(seg)
            for side, team in side_team.items():
                r = ref.get(team, {})
                s = sb[side]
                rep.team_rows.append(_row("팀", team, "", "pts", r.get("pts"), s["pts"], "스코어버그 최종 점수"))
                rep.team_rows.append(_row("팀", team, "", "p3m", r.get("p3m"), s["plus3"], "스코어버그 +3 횟수"))
                rep.team_rows.append(_row("팀", team, "", "p2m", r.get("p2m"), s["plus2"], "스코어버그 +2 횟수 (자유투 2개가 한 번에 바뀌면 여기 섞임)"))
                rep.team_rows.append(_row("팀", team, "", "ftm", r.get("ftm"), s["plus1"], "스코어버그 +1 횟수"))
                if s["check"]:
                    rep.notes.append(f"{team}: 확인이 필요한 점수 변화 {s['check']}건 (check/correction)")
        else:
            rep.notes.append("점수 이벤트에서 경기 구간을 찾지 못했습니다.")

    lih = rep.match.left_is_home if rep.match else (True if left_is_home is None else left_is_home)
    side_team = {"home": g.home if lih else g.away, "away": g.away if lih else g.home}

    # 2) 슛 기록
    if shots:
        t0 = seg.t0 - 120 if seg else None
        t1 = seg.t1 + 30 if seg else None
        sel = [dict(s) for s in shots if in_range(float(s["t_sec"])) and (t0 is None or t0 <= float(s["t_sec"]) <= t1)]
        for s in sel:
            s["team_name"] = side_team.get(s["team"], s["team"])
        tot = shot_totals(sel)
        for side, team in side_team.items():
            r, s = ref.get(team, {}), tot.get(side)
            if s is None:
                continue
            for k in ("fga", "fgm", "p2a", "p2m", "p3a", "p3m", "c3a", "c3m", "w3a", "w3m", "t3a", "t3m"):
                rep.team_rows.append(_row("팀", team, "", k, r.get(k), s[k], "슛 기록(click-shots)"))
        # 선수별 (등번호가 기록된 슛만)
        if any(s.get("no") for s in sel):
            ptot = shot_totals(sel, by_player=True)
            for (side, no), s in sorted(ptot.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
                if not no:
                    continue
                team = side_team.get(side, side)
                name = next((p for (gg, tt, p), n in rec.numbers.items() if gg == game and tt == team and str(n) == str(no)), None)
                pr = next((derive(dict(b)) for b in rec.game_box(game) if b["team"] == team and b["player"] == name), {}) if name else {}
                label = f"#{no} {name or '(명단에 없음)'}"
                for k in ("fga", "fgm", "p3a", "p3m"):
                    rep.player_rows.append(_row("선수", team, label, k, pr.get(k), s[k]))
        # 3점 구역 제안
        attempts = [a for a in rec.threes if a.game == game]
        if attempts:
            rep.zones = align_threes(attempts, sel, seg, lih)
            ok = sum(z.status == "일치" for z in rep.zones)
            rep.notes.append(f"3점 구역 제안: 기록지 3점 시도 {len(attempts)}개 중 {ok}개를 영상 기록과 짝지음")
    return rep


# ---------------------------------------------------------------- 출력
def summary_lines(rep: Report) -> list[str]:
    lines = [f"[{rep.game}] {rep.home} vs {rep.away}"]
    lines += ["  " + n for n in rep.notes]
    rows = [r for r in rep.team_rows if r["기록지"] is not None and r["자동"] is not None]
    if rows:
        lines.append("  팀            항목            기록지   자동   차이")
        for r in rows:
            lines.append(f"  {r['팀']:<12} {r['항목']:<12} {r['기록지']:>6} {r['자동']:>6} {r['차이']:>+6}")
    return lines


def write_report(rep: Report, path: str | Path) -> None:
    """엑셀 보고서: 요약 / 팀 비교 / 선수 비교 / 3점 구역 제안 시트."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "요약"
    for line in summary_lines(rep):
        ws.append([line])
    red = PatternFill("solid", fgColor="FFC7CE")

    def sheet(name, rows):
        if not rows:
            return
        s = wb.create_sheet(name)
        keys = list(rows[0].keys())
        s.append(keys)
        for c in s[1]:
            c.font = Font(bold=True)
        for r in rows:
            s.append([r[k] for k in keys])
            d = r.get("차이")
            if isinstance(d, int) and d != 0:
                for c in s[s.max_row]:
                    c.fill = red
            if r.get("상태") in ("결과 다름", "영상 기록 없음"):
                for c in s[s.max_row]:
                    c.fill = red
        for col in s.columns:
            s.column_dimensions[col[0].column_letter].width = max(8, min(40, max(len(str(c.value or "")) for c in col) * 1.6))

    sheet("팀 비교", rep.team_rows)
    sheet("선수 비교", rep.player_rows)
    sheet("3점 구역 제안", [z.as_row() for z in rep.zones])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def apply_zones(workbook_in: str | Path, workbook_out: str | Path, zones: list[ZoneSuggestion], only_matched: bool = True) -> int:
    """제안 구역을 기록지 사본의 [3점슛_구역체크] '구역' 칸에 써 넣는다. 원본은 건드리지 않는다.

    엑셀 파일(zip) 안의 해당 시트 XML 에서 그 칸만 고친다. 그래서 드롭다운·수식·서식이 그대로 남는다
    (openpyxl 로 다시 저장하면 일부 드롭다운이 사라진다). 이미 구역이 들어 있는 칸은 덮어쓰지 않는다.
    쓴 칸 수를 돌려준다.
    """
    import re
    import zipfile
    from xml.sax.saxutils import escape

    import openpyxl
    from openpyxl.utils import column_index_from_string, get_column_letter

    if Path(workbook_in).resolve() == Path(workbook_out).resolve():
        raise ValueError("원본과 다른 파일 이름으로 저장하세요.")
    sheet_name = "3점슛_구역체크"
    wb = openpyxl.load_workbook(workbook_in, read_only=True)
    ws = wb[sheet_name]
    hdr_row = next(r for r in ws.iter_rows(min_row=1, max_row=15, values_only=True) if r and r[0] == "경기ID")
    col = get_column_letter(1 + list(hdr_row).index("구역"))
    wb.close()

    with zipfile.ZipFile(workbook_in) as zin:
        wbxml = zin.read("xl/workbook.xml").decode("utf-8")
        rels = zin.read("xl/_rels/workbook.xml.rels").decode("utf-8")
        rid = re.search(r'<sheet [^>]*name="%s"[^>]*r:id="([^"]+)"' % re.escape(sheet_name), wbxml).group(1)
        target = re.search(r'<Relationship [^>]*Id="%s"[^>]*Target="([^"]+)"' % re.escape(rid), rels)
        if target is None:  # 속성 순서가 다른 경우
            target = re.search(r'<Relationship [^>]*Target="([^"]+)"[^>]*Id="%s"' % re.escape(rid), rels)
        path = target.group(1).lstrip("/")
        path = path if path.startswith("xl/") else "xl/" + path
        xml = zin.read(path).decode("utf-8")

        n = 0
        for z in zones:
            if not z.shot or not z.shot.get("zone_kr") or (only_matched and z.status != "일치"):
                continue
            ref = f"{col}{z.attempt.row}"
            text = f'<is><t>{escape(z.shot["zone_kr"])}</t></is>'
            m = re.search(r'<c r="%s"(?P<attrs>[^>]*?)(?:/>|>(?P<body>.*?)</c>)' % ref, xml, re.S)
            if m is not None:
                if m.group("body") and ("<v>" in m.group("body") or "<is>" in m.group("body")):
                    continue  # 이미 값이 있음
                style = re.search(r'\ss="(\d+)"', m.group("attrs"))
                s_attr = f' s="{style.group(1)}"' if style else ""
                xml = xml[:m.start()] + f'<c r="{ref}"{s_attr} t="inlineStr">{text}</c>' + xml[m.end():]
                n += 1
                continue
            # 칸 자체가 없으면 그 행 안에 열 순서에 맞춰 끼워 넣는다
            rm = re.search(r'<row r="%d"[^>]*?>(?P<cells>.*?)</row>' % z.attempt.row, xml, re.S)
            if rm is None:
                continue
            new_cell = f'<c r="{ref}" t="inlineStr">{text}</c>'
            pos = rm.end("cells")
            for cm in re.finditer(r'<c r="([A-Z]+)\d+"', rm.group("cells")):
                if column_index_from_string(cm.group(1)) > column_index_from_string(col):
                    pos = rm.start("cells") + cm.start()
                    break
            xml = xml[:pos] + new_cell + xml[pos:]
            n += 1
        # 엑셀이 열 때 수식(구역별 합계)을 다시 계산하도록
        if "fullCalcOnLoad" not in wbxml:
            wbxml = re.sub(r"<calcPr([^>]*?)/>", r'<calcPr\1 fullCalcOnLoad="1"/>', wbxml, count=1)

        Path(workbook_out).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(workbook_out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == path:
                    zout.writestr(item, xml.encode("utf-8"))
                elif item.filename == "xl/workbook.xml":
                    zout.writestr(item, wbxml.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))
    return n
