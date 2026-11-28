"""D3 기록지 엑셀(기록입력·경기목록·공식기록_대조·3점슛_구역체크 시트) 읽기.

열 위치가 아니라 머리글 글자로 열을 찾으므로, 열 순서가 조금 바뀌어도 동작한다.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

# (그룹 머리글, 세부 머리글) → 표준 키
BOX_COLUMNS = {
    ("기본 정보", "경기ID"): "game",
    ("기본 정보", "날짜"): "date",
    ("기본 정보", "팀"): "team",
    ("기본 정보", "선수명"): "player",
    ("2점슛", "시도"): "p2a",
    ("2점슛", "성공"): "p2m",
    ("3점슛", "시도"): "p3a",
    ("3점슛", "성공"): "p3m",
    ("3점 코너", "시도"): "c3a",
    ("3점 코너", "성공"): "c3m",
    ("3점 45도", "시도"): "w3a",
    ("3점 45도", "성공"): "w3m",
    ("3점 탑", "시도"): "t3a",
    ("3점 탑", "성공"): "t3m",
    ("자유투", "시도"): "fta",
    ("자유투", "성공"): "ftm",
    ("기타", "어시스트"): "ast",
    ("기타", "블락"): "blk",
    ("기타", "스틸"): "stl",
    ("기타", "턴오버"): "tov",
    ("리바운드", "수비"): "drb",
    ("리바운드", "공격"): "orb",
}
STAT_KEYS = [v for v in BOX_COLUMNS.values() if v not in ("game", "date", "team", "player")]

# 비교 보고서에 쓰는 한글 이름
STAT_LABELS = {
    "pts": "득점", "p2a": "2점 시도", "p2m": "2점 성공", "p3a": "3점 시도", "p3m": "3점 성공",
    "c3a": "코너3 시도", "c3m": "코너3 성공", "w3a": "45도3 시도", "w3m": "45도3 성공", "t3a": "탑3 시도", "t3m": "탑3 성공",
    "fta": "자유투 시도", "ftm": "자유투 성공", "ast": "어시스트", "blk": "블락", "stl": "스틸", "tov": "턴오버",
    "drb": "수비 리바운드", "orb": "공격 리바운드", "fga": "야투 시도", "fgm": "야투 성공",
}


@dataclass
class Game:
    game: str
    date: dt.date | None
    home: str
    away: str
    home_pts: int | None
    away_pts: int | None
    note: str = ""


@dataclass
class ThreeAttempt:
    """[3점슛_구역체크] 한 줄: 공식 문자중계 순서의 3점 시도 하나."""

    row: int  # 엑셀 행 번호 (나중에 구역을 채워 넣을 때 사용)
    game: str
    quarter: str
    seq: int
    prev_score: str
    team: str
    number: int | None
    player: str
    made: bool
    zone: str | None


@dataclass
class Records:
    games: dict[str, Game] = field(default_factory=dict)
    box: list[dict] = field(default_factory=list)  # 기록입력 한 줄 = 경기 × 선수
    numbers: dict[tuple[str, str, str], int] = field(default_factory=dict)  # (경기, 팀, 선수) → 등번호
    threes: list[ThreeAttempt] = field(default_factory=list)

    def game_box(self, game: str) -> list[dict]:
        return [r for r in self.box if r["game"] == game]

    def team_totals(self, game: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.game_box(game):
            t = out.setdefault(r["team"], {k: 0 for k in STAT_KEYS})
            for k in STAT_KEYS:
                t[k] += r[k] or 0
        for t in out.values():
            derive(t)
        return out

    def player_number(self, game: str, team: str, player: str) -> int | None:
        return self.numbers.get((game, team, player))


def derive(s: dict) -> dict:
    """3점 전체 칸이 비어 있으면 구역 합계로 채우고, 득점·야투를 계산한다 (제자리 수정)."""
    zone_a = (s.get("c3a") or 0) + (s.get("w3a") or 0) + (s.get("t3a") or 0)
    zone_m = (s.get("c3m") or 0) + (s.get("w3m") or 0) + (s.get("t3m") or 0)
    if not s.get("p3a"):
        s["p3a"] = zone_a
    if not s.get("p3m"):
        s["p3m"] = zone_m
    s["fga"] = (s.get("p2a") or 0) + s["p3a"]
    s["fgm"] = (s.get("p2m") or 0) + s["p3m"]
    s["pts"] = 2 * (s.get("p2m") or 0) + 3 * s["p3m"] + (s.get("ftm") or 0)
    return s


def _find_header(ws, first: str, max_rows: int = 15) -> int:
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=max_rows, values_only=True), start=1):
        if row and row[0] == first:
            return i
    raise ValueError(f"[{ws.title}] 시트에서 '{first}' 머리글을 찾지 못했습니다.")


def _int(v) -> int | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _date(v):
    if isinstance(v, dt.datetime):
        return v.date()
    return v


def load_records(path: str | Path) -> Records:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rec = Records()

    # 경기목록
    ws = wb["경기목록"]
    h = _find_header(ws, "경기ID")
    for row in ws.iter_rows(min_row=h + 1, values_only=True):
        if not row or not row[0]:
            continue
        gid = str(row[0]).strip()
        rec.games[gid] = Game(gid, _date(row[1]), str(row[2] or ""), str(row[3] or ""), _int(row[4]), _int(row[5]), str(row[7] or ""))

    # 기록입력: 그룹 머리글(윗줄, 병합 셀이라 왼쪽 값으로 채움) + 세부 머리글
    ws = wb["기록입력"]
    h = _find_header(ws, "경기ID")
    rows = list(ws.iter_rows(min_row=h - 1, max_row=h, values_only=True))
    groups, subs = rows[0], rows[1]
    cur, cols = None, {}
    for j, (g, s) in enumerate(zip(groups, subs)):
        if g:
            cur = str(g).strip()
        key = BOX_COLUMNS.get((cur, str(s).strip() if s else None))
        if key and key not in cols:
            cols[key] = j
    missing = [k for k in BOX_COLUMNS.values() if k not in cols]
    if missing:
        raise ValueError(f"[기록입력] 시트에서 찾지 못한 열: {missing}")
    for row in ws.iter_rows(min_row=h + 1, values_only=True):
        if not row or not row[cols["game"]] or not row[cols["player"]]:
            continue
        r = {
            "game": str(row[cols["game"]]).strip(),
            "date": _date(row[cols["date"]]),
            "team": str(row[cols["team"]]).strip(),
            "player": str(row[cols["player"]]).strip(),
        }
        for k in STAT_KEYS:
            r[k] = _int(row[cols[k]])
        rec.box.append(r)

    # 공식기록_대조: 경기별 등번호
    if "공식기록_대조" in wb.sheetnames:
        ws = wb["공식기록_대조"]
        h = _find_header(ws, "경기ID")
        for row in ws.iter_rows(min_row=h + 1, values_only=True):
            if not row or not row[0] or not row[2]:
                continue
            n = _int(row[3])
            if n is not None:
                rec.numbers[(str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip())] = n

    # 3점슛_구역체크
    if "3점슛_구역체크" in wb.sheetnames:
        ws = wb["3점슛_구역체크"]
        h = _find_header(ws, "경기ID")
        for i, row in enumerate(ws.iter_rows(min_row=h + 1, values_only=True), start=h + 1):
            if not row or not row[0] or row[2] is None:
                continue
            rec.threes.append(ThreeAttempt(
                row=i, game=str(row[0]).strip(), quarter=str(row[1] or ""), seq=_int(row[2]) or 0,
                prev_score=str(row[3] or ""), team=str(row[4] or "").strip(), number=_int(row[5]),
                player=str(row[6] or "").strip(), made=str(row[7]).strip() == "성공",
                zone=(str(row[8]).strip() if row[8] else None),
            ))
    wb.close()
    return rec


def _fg_miss(r: dict) -> int:
    return (r["p2a"] or 0) - (r["p2m"] or 0) + (r["p3a"] or 0) - (r["p3m"] or 0)


def rebound_issue(players: list[dict]) -> str | None:
    """한 팀 한 경기 선수 기록의 리바운드 이상. 없으면 None.

    KBA 2026 D3 1R 문자중계는 슛 실패 바로 뒤에 슛한 선수 본인의 '공격 리바운드'를 붙여 기록했다.
    그래서 선수 대부분의 리바운드가 본인 야투 실패 수와 똑같아진다 (정상 경기에서는 10% 안팎).
    """
    if players and all(r["drb"] is None and r["orb"] is None for r in players):
        return "리바운드 칸이 모두 비어 있음 (영상으로 입력 필요)"
    shooters = [r for r in players if _fg_miss(r) >= 2]
    same = [r for r in shooters if _fg_miss(r) in ((r["orb"] or 0), (r["orb"] or 0) + (r["drb"] or 0))]
    ratio = len(same) / len(shooters) if shooters else 0.0
    orb = sum(r["orb"] or 0 for r in players)
    drb = sum(r["drb"] or 0 for r in players)
    lopsided = orb >= 20 and orb > 3 * drb
    if len(shooters) >= 4 and (ratio >= 0.7 or (lopsided and ratio >= 0.5)):
        return (f"리바운드가 슛한 선수에게 붙어 있음: 선수 {len(same)}/{len(shooters)}명이 리바운드 = 본인 야투 실패 수 "
                "(원본 문자중계 기록 오류, 실제 리바운드는 영상 확인 필요)")
    if lopsided:
        return f"공격 리바운드 {orb}가 수비 리바운드 {drb}보다 훨씬 많음 (열이 뒤바뀌었을 수 있음)"
    return None


def sanity_issues(rec: Records, games: list[str] | None = None) -> list[dict]:
    """기록지에서 사람이 한 번 볼 만한 이상한 값을 찾는다 (자동 비교 전에 기준 데이터 점검용).

    games 를 주면 그 경기만 본다.
    """
    out = []
    for gid, g in rec.games.items():
        if games is not None and gid not in games:
            continue
        tt = rec.team_totals(gid)
        if not tt:
            continue
        for team, pts in ((g.home, g.home_pts), (g.away, g.away_pts)):
            s = tt.get(team)
            if s is None:
                out.append({"경기ID": gid, "팀": team, "내용": "선수 기록 없음"})
                continue
            if pts is not None and s["pts"] != pts:
                out.append({"경기ID": gid, "팀": team, "내용": f"선수 득점 합계 {s['pts']} ≠ 경기 스코어 {pts}"})
            msg = rebound_issue([r for r in rec.game_box(gid) if r["team"] == team])
            if msg:
                out.append({"경기ID": gid, "팀": team, "내용": msg})
            if s["p3a"] >= 5 and s["p3a"] == s["p3m"]:
                out.append({"경기ID": gid, "팀": team, "내용": f"3점 {s['p3m']}/{s['p3a']} 전부 성공 (실패 시도가 빠졌을 수 있음)"})
    for r in rec.box:
        if games is not None and r["game"] not in games:
            continue
        for a, m, name in (("p2a", "p2m", "2점"), ("p3a", "p3m", "3점"), ("fta", "ftm", "자유투")):
            if r[a] is not None and r[m] is not None and r[m] > r[a]:
                out.append({"경기ID": r["game"], "팀": r["team"], "내용": f"{r['player']} {name} 성공 {r[m]} > 시도 {r[a]}"})
    return out
