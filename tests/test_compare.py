import datetime as dt

import openpyxl
import pytest

from gtt import cli
from gtt import compare as CP
from gtt.records import load_records

GROUPS = ["기본 정보", None, None, None, "2점슛", None, "3점슛", None, "3점 코너", None, "3점 45도", None, "3점 탑", None,
          "자유투", None, "기타", None, None, None, "리바운드", None, "자동 계산"]
SUBS = ["경기ID", "날짜", "팀", "선수명", "시도", "성공", "시도", "성공", "시도", "성공", "시도", "성공", "시도", "성공",
        "시도", "성공", "어시스트", "블락", "스틸", "턴오버", "수비", "공격", "득점"]


def box(game, team, name, p2=(0, 0), p3=(0, 0), ft=(0, 0)):
    return [game, dt.datetime(2026, 5, 16), team, name, p2[0], p2[1], p3[0], p3[1], None, None, None, None, None, None,
            ft[0], ft[1], 1, 0, 1, None, 3, 1, None]


@pytest.fixture
def workbook(tmp_path):
    """실제 D3 기록지와 같은 시트·머리글 구조의 작은 엑셀."""
    wb = openpyxl.Workbook()
    wb.active.title = "사용법"
    ws = wb.create_sheet("경기목록")
    ws.append(["경기 목록"]); ws.append([])
    ws.append(["경기ID", "날짜", "홈팀", "원정팀", "홈 득점", "원정 득점", "대회/시즌", "비고"])
    ws.append(["D101", dt.datetime(2026, 5, 16), "레드팀", "MI", 12, 9, "2026 시즌", "1R"])
    ws.append(["D102", dt.datetime(2026, 5, 16), "SA", "블루팀", 7, 5, "2026 시즌", "1R"])
    ws = wb.create_sheet("기록입력")
    ws.append(["경기별 선수 기록 입력"]); ws.append(["노란 칸 입력"])
    ws.append(GROUPS); ws.append(SUBS)
    # D101 레드팀 12 = 2점 3개 + 3점 1개 + 자유투 3개 / MI 9 = 3점 2개 + 2점 1개 + 자유투 1개
    ws.append(box("D101", "레드팀", "가선수", p2=(5, 3), p3=(2, 1), ft=(4, 3)))
    ws.append(box("D101", "MI", "나선수", p2=(2, 1), p3=(4, 2), ft=(2, 1)))
    ws.append(box("D102", "SA", "다선수", p2=(3, 2), p3=(1, 1)))
    ws.append(box("D102", "블루팀", "라선수", p2=(4, 1), p3=(1, 1)))
    ws = wb.create_sheet("공식기록_대조")
    ws.append(["공식기록 대조"]); ws.append(["설명"])
    ws.append(["경기ID", "팀", "선수명", "등번호", "공식 표기", "득점"]); ws.append([None] * 6)
    ws.append(["D101", "레드팀", "가선수", 5, None, 12])
    ws.append(["D101", "MI", "나선수", 1, None, 9])
    ws = wb.create_sheet("3점슛_구역체크")
    ws.append(["3점슛 구역 체크리스트"]); ws.append(["설명"])
    ws.append(["경기ID", "쿼터", "3점 순번", "직전 스코어(홈:원정)", "팀", "등번호", "선수명", "결과", "구역", "메모"])
    ws.append(["D101", "1Q", 1, "0 : 0", "MI", 1, "나선수", "실패", None, None])
    ws.append(["D101", "1Q", 2, "2 : 0", "MI", 1, "나선수", "성공", None, None])
    ws.append(["D101", "1Q", 3, "2 : 3", "레드팀", 5, "가선수", "성공", None, None])
    ws.append(["D101", "2Q", 4, "7 : 3", "MI", 1, "나선수", "실패", None, None])
    ws.append(["D101", "2Q", 5, "7 : 3", "MI", 1, "나선수", "성공", None, None])
    ws.append(["D101", "2Q", 6, "7 : 6", "레드팀", 5, "가선수", "실패", "탑", None])
    p = tmp_path / "기록지.xlsx"
    wb.save(p)
    return p


def ev(t, team, d, h, a, flag="ok"):
    return {"t_sec": str(t), "team": team, "delta": str(d), "home": str(h), "away": str(a), "flag": flag}


# 하루 중계: 첫 경기 D101 (스코어버그 왼쪽 = MI, 오른쪽 = 레드팀), 리셋, 둘째 경기 D102
DAY = [
    ev(30, "away", 2, 0, 2), ev(60, "home", 3, 3, 2), ev(80, "away", 3, 3, 5), ev(100, "away", 2, 3, 7),
    ev(130, "home", 3, 6, 7), ev(150, "away", 1, 6, 8), ev(160, "away", 1, 6, 9), ev(170, "away", 1, 6, 10),
    ev(200, "home", 2, 8, 10), ev(230, "home", 1, 9, 10), ev(260, "away", 2, 9, 12),
    ev(900, "home", -9, 0, 0, "correction"), ev(900, "away", -12, 0, 0, "correction"),
    ev(950, "home", 2, 2, 0), ev(990, "away", 3, 2, 3), ev(1000, "home", 3, 5, 3), ev(1030, "away", 2, 5, 5), ev(1100, "home", 2, 7, 5),
]


def shot(t, team, result, points, zone_kr="", no=""):
    return {"t_sec": str(t), "time": "", "team": team, "no": no, "result": result, "points": str(points), "zone_kr": zone_kr, "margin_m": "0.5"}


SHOTS = [
    shot(20, "home", "MISSED", 3, "코너", "1"), shot(25, "away", "MADE", 2, "", "5"),  # 0:0 에서 MI 3점 실패
    shot(57, "home", "MADE", 3, "45도", "1"),  # 2:0 (홈:원정 = 레드팀:MI) 에서 MI 3점 성공
    shot(78, "away", "MADE", 3, "탑", "5"),
    shot(120, "home", "MISSED", 3, "탑", "1"), shot(127, "home", "MADE", 3, "코너", "1"),
    shot(140, "away", "MISSED", 3, "45도", "5"),
]


def test_load_records(workbook):
    r = load_records(workbook)
    assert set(r.games) == {"D101", "D102"}
    tt = r.team_totals("D101")
    assert tt["레드팀"]["pts"] == 12 and tt["MI"]["pts"] == 9
    assert tt["MI"]["fga"] == 6
    assert r.player_number("D101", "MI", "나선수") == 1
    assert len([a for a in r.threes if a.game == "D101"]) == 6


def test_segment_and_match():
    segs = CP.segment_games(DAY)
    assert [(s.left, s.right) for s in segs] == [(9, 12), (7, 5)]
    m = CP.match_game(segs, 12, 9)  # D101 레드팀 12 : MI 9
    assert m.segment is segs[0] and m.left_is_home is False and m.score_error == 0
    m = CP.match_game(segs, 7, 5)
    assert m.segment is segs[1] and m.left_is_home is True


def test_compare_game(workbook):
    rec = load_records(workbook)
    rep = CP.compare_game(rec, "D101", DAY, SHOTS)
    assert rep.match.left_is_home is False
    rows = {(r["팀"], r["항목"], r["비고"][:6]): r for r in rep.team_rows}
    assert rows[("레드팀", "득점", "스코어버그 ")]["차이"] == 0
    assert rows[("MI", "3점 성공", "스코어버그 ")]["자동"] == 2
    assert rows[("레드팀", "자유투 성공", "스코어버그 ")]["자동"] == 3
    shots_mi = {r["항목"]: r for r in rep.team_rows if r["팀"] == "MI" and r["비고"].startswith("슛")}
    assert shots_mi["3점 시도"]["자동"] == 4 and shots_mi["3점 시도"]["차이"] == 0
    assert any(r["선수"] == "#1 나선수" for r in rep.player_rows)
    z = {zz.attempt.seq: zz for zz in rep.zones}
    assert [z[i].status for i in range(1, 6)] == ["일치"] * 5
    assert [z[i].shot["zone_kr"] for i in range(1, 6)] == ["코너", "45도", "탑", "탑", "코너"]
    assert z[6].shot["zone_kr"] == "45도"  # 기존 구역(탑)은 apply_zones 가 덮어쓰지 않음


def test_compare_cli_and_apply(workbook, tmp_path):
    import csv

    evp, shp = tmp_path / "ev.csv", tmp_path / "shots.csv"
    for path, rows in ((evp, DAY), (shp, SHOTS)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    out, filled = tmp_path / "rep.xlsx", tmp_path / "filled.xlsx"
    cli.main(["compare", "--records", str(workbook), "--game", "D101", "--score-events", str(evp), "--shots", str(shp),
              "--out", str(out), "--apply-zones", str(filled)])
    wb = openpyxl.load_workbook(out)
    assert {"요약", "팀 비교", "선수 비교", "3점 구역 제안"} <= set(wb.sheetnames)
    ws = openpyxl.load_workbook(filled)["3점슛_구역체크"]
    zones = [ws.cell(row=r, column=9).value for r in range(4, 10)]
    assert zones == ["코너", "45도", "탑", "탑", "코너", "탑"]
    # 원본은 그대로
    assert openpyxl.load_workbook(workbook)["3점슛_구역체크"].cell(row=4, column=9).value is None


def test_check_records(workbook, capsys):
    cli.main(["check-records", "--records", str(workbook)])
    out = capsys.readouterr().out
    assert "확인할 항목 0개" in out


def _player(miss, orb, drb, made=1):
    return {"p2a": miss + made, "p2m": made, "p3a": 0, "p3m": 0, "orb": orb, "drb": drb}


def test_rebound_issue():
    from gtt.records import rebound_issue
    # 1R처럼 리바운드가 슛한 선수 본인에게 붙은 경우 (선수마다 리바운드 = 본인 실패 수)
    msg = rebound_issue([_player(10, 10, 0), _player(4, 4, 0), _player(7, 6, 1), _player(3, 3, 0), _player(2, 0, 0)])
    assert msg and "슛한 선수" in msg and "4/5" in msg
    # 정상 경기: 리바운드와 실패 수가 따로 논다
    assert rebound_issue([_player(10, 2, 5), _player(4, 0, 3), _player(7, 3, 6), _player(3, 1, 0)]) is None
    # 칸이 모두 비어 있으면 입력 필요로 알려 준다
    assert "비어 있음" in rebound_issue([_player(5, None, None), _player(2, None, None)])
    # 공격이 수비보다 훨씬 많으면 열 뒤바뀜 의심
    assert "열이 뒤바뀌었을" in rebound_issue([_player(0, 12, 1), _player(0, 10, 2)])


def test_compare_notes_unreliable_rebounds():
    from gtt.records import Game, Records
    rec = Records()
    rec.games["D101"] = Game("D101", None, "레드팀", "MI", None, None)
    base = {"game": "D101", "date": None, "c3a": None, "c3m": None, "w3a": None, "w3m": None, "t3a": None, "t3m": None,
            "fta": 0, "ftm": 0, "ast": 0, "blk": 0, "stl": 0, "tov": None}
    for i, miss in enumerate((10, 4, 7, 3)):
        rec.box.append(dict(base, team="레드팀", player=f"a{i}", **_player(miss, miss, 0)))
        rec.box.append(dict(base, team="MI", player=f"m{i}", **_player(miss, 1, 2)))
    rep = CP.compare_game(rec, "D101")
    notes = [n for n in rep.notes if n.startswith("기록지 점검")]
    assert len(notes) == 1 and "레드팀" in notes[0] and "슛한 선수" in notes[0]
