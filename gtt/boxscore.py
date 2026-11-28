"""점수판 이벤트 + 슛 기록 → 팀·선수 기록(박스스코어).

어떤 기록이 '영상으로 잰 것'인지(available)를 함께 돌려준다. 재지 못한 기록은 0이 아니라 빈칸(None)이다.

- 점수판(scorebug-read)만 있을 때: 팀 득점과 성공 횟수(+1 자유투, +2 2점, +3 3점). 시도·선수는 모름.
  +2 는 자유투 2개가 한꺼번에 반영된 것일 수도 있어 '2점 성공'은 확인 필요로 표시한다.
- 슛 기록(click-shots, 또는 나중의 자동 슛 검출)이 있으면: 시도·성공·3점 구역, 등번호가 있으면 선수별.
  이때도 팀 득점은 점수판 값을 정답으로 쓰고, 선수 합계와 다르면 차이를 '(미배정)' 줄로 남긴다.
- 리바운드·어시스트·스틸·블락·턴오버·파울은 아직 자동 감지가 없어 비워 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .scorecheck import shot_points

ZONES = {"코너": "corner", "45도": "wing", "탑": "top"}
NOT_YET = ["ORB", "DRB", "REB", "AST", "STL", "BLK", "TOV", "PF"]


@dataclass
class Line:
    team: str  # home / away
    name: str
    no: str = ""
    s: dict = field(default_factory=dict)  # 기록 이름 → 값 (None = 모름)

    def add(self, k: str, v: int = 1) -> None:
        self.s[k] = (self.s.get(k) or 0) + v


def _made(row: dict) -> bool:
    return str(row.get("result", "")).upper() in ("MADE", "성공", "1", "TRUE")


def _finish(s: dict, have_attempts: bool) -> None:
    for k in ("P2M", "P3M", "FTM", "P2A", "P3A", "FTA"):
        s.setdefault(k, 0)
    s["FGM"] = s["P2M"] + s["P3M"]
    s["FGA"] = s["P2A"] + s["P3A"]
    if not have_attempts:
        for k in ("P2A", "P3A", "FTA", "FGA"):
            s[k] = None
    for z in ZONES.values():
        if not have_attempts:
            s[f"{z}_A"] = s[f"{z}_M"] = None
        else:
            s.setdefault(f"{z}_A", 0)
            s.setdefault(f"{z}_M", 0)
    for k in NOT_YET:
        s[k] = None


@dataclass
class Box:
    team_names: dict
    teams: dict  # home/away → Line (팀 합계)
    players: list  # Line (선수별, 슛 기록에 등번호가 있을 때)
    available: set  # 영상으로 잰 기록 이름
    team_available: set
    notes: list = field(default_factory=list)
    ambiguous_plus2: int = 0


def build(events: list[dict], shots: list[dict] | None, team_names: dict, roster: dict | None = None) -> Box:
    """events: scorebug-read CSV 행들, shots: click-shots CSV 행들(없으면 None), roster: (team, 등번호) → 이름."""
    roster = roster or {}
    ok = [e for e in events if e.get("flag", "ok") == "ok"]
    teams = {t: Line(t, team_names.get(t, t)) for t in ("home", "away")}
    notes = []
    skipped = len(events) - len(ok)
    if skipped:
        notes.append(f"점수판 변화 중 {skipped}건은 판독 확인 필요(check/correction)라 득점에서 뺐습니다.")
    amb = 0
    for e in ok:
        d = int(e["delta"])
        L = teams[e["team"]]
        L.add("PTS", d)
        if d == 1:
            L.add("FTM")
        elif d == 2:
            amb += 1
        elif d == 3:
            L.add("P3M")

    players: dict[tuple[str, str], Line] = {}
    if shots:
        team_attempts = {t: Line(t, "") for t in teams}
        for r in shots:
            team = r.get("team", "home")
            pts = shot_points(r)
            no = str(r.get("no") or "").strip()
            key = (team, no)
            if key not in players:
                name = roster.get(key) or (f"#{no}" if no else "(등번호 미상)")
                players[key] = Line(team, name, no)
            made = _made(r)
            for L in (players[key], team_attempts[team]):
                if pts == 1:
                    L.add("FTA")
                    L.add("FTM", int(made))
                elif pts in (2, 3):
                    L.add(f"P{pts}A")
                    L.add(f"P{pts}M", int(made))
                    z = ZONES.get(str(r.get("zone_kr", "")))
                    if pts == 3 and z:
                        L.add(f"{z}_A")
                        L.add(f"{z}_M", int(made))
                if made and pts:
                    L.add("PTS", pts)
        has_ft = any(shot_points(r) == 1 for r in shots)
        # 팀 줄: 득점은 점수판(정답), 시도·성공·구역은 슛 기록
        for t, L in teams.items():
            A = team_attempts[t].s
            bug_pts, bug_ftm = L.s.get("PTS") or 0, L.s.get("FTM") or 0
            L.s = {k: v for k, v in A.items() if k != "PTS"} | {"PTS": bug_pts}
            _finish(L.s, True)
            if not has_ft:  # 자유투는 클릭하지 않았다 → 성공은 점수판 +1 횟수, 시도는 모름
                L.s["FTM"], L.s["FTA"] = bug_ftm, None
            shot_pts = A.get("PTS") or 0
            if not has_ft:
                shot_pts += bug_ftm
            if shot_pts != bug_pts:
                diff = bug_pts - shot_pts
                notes.append(f"{L.name}: 점수판 득점 {bug_pts} / 슛 기록 득점 {shot_pts} → 차이 {diff:+d}점은 선수에게 배정하지 못했습니다.")
                players[(t, "__unassigned")] = Line(t, "(미배정)", "", {"PTS": diff})
        for L in players.values():
            if L.name != "(미배정)":
                _finish(L.s, True)
                if not has_ft:
                    L.s["FTM"] = L.s["FTA"] = None
        avail = {"FGM", "FGA", "P2M", "P2A", "P3M", "P3A"} | ({"PTS", "FTM", "FTA"} if has_ft else set())
        team_avail = avail | {"PTS", "FTM"}
        if not has_ft:
            notes.append("슛 기록에 자유투가 없어 선수별 자유투와 득점의 자유투 몫은 비어 있습니다. 팀 자유투 성공은 점수판 +1 횟수입니다.")
    else:
        for L in teams.values():
            _finish(L.s, False)
            L.s["P2M"] = L.s["FGM"] = None  # +2 는 2점 1개인지 자유투 2개인지 모른다
        avail, team_avail = set(), {"PTS", "P3M", "FTM"}
        if amb:
            notes.append(f"점수판 +2 {amb}건은 2점 성공인지 자유투 2개인지 몰라 '2점 성공'은 비워 두고 득점에만 넣었습니다.")
    pl = sorted(players.values(), key=lambda L: (L.team, L.name == "(미배정)", L.name == "(등번호 미상)",
                                                  int(L.no) if L.no.isdigit() else 999))
    return Box(team_names, teams, pl, avail, team_avail, notes, amb)
