import json

import openpyxl

from gtt import analyze as AN
from gtt import boxscore as BX
from gtt import metrics as MT

NAMES = {"home": "레드", "away": "블루"}
EVENTS = [
    {"t_sec": "5", "team": "home", "delta": "2", "home": "2", "away": "0", "flag": "ok"},
    {"t_sec": "9", "team": "away", "delta": "3", "home": "2", "away": "3", "flag": "ok"},
    {"t_sec": "15", "team": "home", "delta": "1", "home": "3", "away": "3", "flag": "ok"},
    {"t_sec": "20", "team": "away", "delta": "5", "home": "3", "away": "8", "flag": "check"},
]
SHOTS = [
    {"t_sec": "4", "team": "home", "no": "7", "result": "MADE", "points": "2", "zone_kr": "2점"},
    {"t_sec": "8", "team": "away", "no": "5", "result": "MADE", "points": "3", "zone_kr": "코너"},
    {"t_sec": "11", "team": "away", "no": "5", "result": "MISSED", "points": "3", "zone_kr": "탑"},
    {"t_sec": "14", "team": "home", "no": "7", "result": "MADE", "points": "1", "zone": "FT", "zone_kr": "자유투"},
    {"t_sec": "14.5", "team": "home", "no": "7", "result": "MISSED", "points": "1", "zone": "FT", "zone_kr": "자유투"},
]


def test_default_metrics_load_and_missing_stats_are_not_computed():
    ms, _ = MT.load()
    keys = [m.key for m in ms]
    assert "ABEi" in keys and "TS" in keys
    abei = next(m for m in ms if m.key == "ABEi")
    r = MT.evaluate(abei, {"PTS": 10, "REB": None}, {"PTS"})
    assert r["value"] is None and "REB" in r["missing"] and r["partial"] == 10
    ts = next(m for m in ms if m.key == "TS")
    r = MT.evaluate(ts, {"PTS": 10, "FGA": 8, "FTA": 0}, {"PTS", "FGA", "FTA"})
    assert abs(r["value"] - 0.625) < 1e-9


def test_custom_formula_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"metrics": [{"key": "X", "formula": "w*PTS + max(P3M, 1)", "params": {"w": 2}}]}), encoding="utf-8")
    (m,), _ = MT.load(p)
    assert MT.evaluate(m, {"PTS": 3, "P3M": 0}, {"PTS", "P3M"})["value"] == 7


def test_bad_formula_is_rejected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"metrics": [{"key": "X", "formula": "__import__('os')"}]}), encoding="utf-8")
    try:
        MT.load(p)
    except ValueError:
        return
    raise AssertionError("should reject")


def test_box_from_scoreboard_only():
    box = BX.build(EVENTS, None, NAMES)
    h, a = box.teams["home"].s, box.teams["away"].s
    assert h["PTS"] == 3 and h["FTM"] == 1 and h["P2M"] is None and h["P2A"] is None
    assert a["PTS"] == 3 and a["P3M"] == 1  # check 이벤트(+5)는 빠짐
    assert h["REB"] is None and box.players == []
    assert any("확인 필요" in n for n in box.notes)


def test_box_with_shots_players_and_zones():
    box = BX.build(EVENTS, SHOTS, NAMES, roster={("home", "7"): "레드 7번"})
    p7 = next(L for L in box.players if L.no == "7")
    assert p7.name == "레드 7번" and p7.s["PTS"] == 3 and p7.s["FTA"] == 2 and p7.s["FTM"] == 1 and p7.s["P2M"] == 1
    p5 = next(L for L in box.players if L.no == "5")
    assert p5.s["P3A"] == 2 and p5.s["corner_M"] == 1 and p5.s["top_A"] == 1 and p5.s["top_M"] == 0
    assert box.teams["away"].s["PTS"] == 3 and not any("미배정" in n for n in box.notes)


def test_unassigned_points_reported():
    box = BX.build(EVENTS, SHOTS[:1], NAMES)  # 원정 3점 슛 기록이 없음
    assert any(L.name == "(미배정)" and L.team == "away" and L.s["PTS"] == 3 for L in box.players)


def test_demo_end_to_end(tmp_path):
    res = AN.run_demo(tmp_path, seconds=30)
    assert res["코트 점수 vs 점수판"]["일치"] == res["코트 점수 vs 점수판"]["점수판 변화"] >= 3
    assert (tmp_path / "report.html").read_text(encoding="utf-8").count("계산 불가") > 0
    wb = openpyxl.load_workbook(tmp_path / "기록지_자동.xlsx")
    ws = wb["기록입력"]
    assert ws.cell(4, 1).value == "경기ID" and ws.cell(4, 27).value == "득점"
    names = [ws.cell(r, 4).value for r in range(5, ws.max_row + 1)]
    assert "레드 7번" in names


def test_zones_from_court_coordinates():
    from gtt import zones as ZN

    rim = 1.575
    assert ZN.zone_of(rim + 0.5, 7.5)["zone"] == "림 근처"
    assert ZN.zone_of(rim + 2.5, 7.5)["zone"] == "페인트"
    assert ZN.zone_of(rim + 7.3, 7.5)["zone"] == "탑"
    assert ZN.zone_of(0.8, 0.5)["zone"].startswith("코너")
    assert ZN.zone_of(rim + 7.3, 7.5)["band"] == "3점 라인~8m"
    # 오른쪽 골대도 같은 규칙 (좌우 대칭)
    a, b = ZN.zone_of(0.8, 0.5), ZN.zone_of(28 - 0.8, 15 - 0.5)
    assert a["zone"] == b["zone"] and a["band"] == b["band"]
    t = ZN.tally([{"team": "home", "result": "MADE", "court_x": rim + 7.3, "court_y": 7.5},
                  {"team": "home", "result": "MISSED", "court_x": rim + 7.2, "court_y": 7.4},
                  {"team": "home", "result": "MADE", "zone": "FT", "court_x": 6, "court_y": 7.5}])
    assert t["zone"][("home", "탑")] == [2, 1]
