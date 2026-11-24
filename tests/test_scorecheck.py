from gtt import scorecheck as SC
from gtt.court import FIBA, classify_shot, is_ft_spot


def ev(t, team, d, flag="ok"):
    return {"t_sec": str(t), "team": team, "delta": str(d), "flag": flag}


def shot(t, team, pts, result="MADE", zone="", near=False):
    return {"t_sec": str(t), "team": team, "points": str(pts), "zone": zone, "result": result, "near_line": str(near)}


def test_ft_spot():
    assert is_ft_spot(6.2, 7.5, FIBA)            # 자유투 라인 바로 뒤
    assert is_ft_spot(28 - 6.2, 7.0, FIBA)       # 오른쪽 골대
    assert not is_ft_spot(8.0, 7.5, FIBA)        # 3점 라인 근처
    assert not is_ft_spot(6.0, 10.0, FIBA)       # 레인 밖 옆
    assert classify_shot(6.2, 7.5, FIBA).ft_spot and classify_shot(6.2, 7.5, FIBA).points == 2


def test_check_matches_and_scores():
    events = [ev(6.5, "home", 2), ev(13, "away", 3), ev(30, "home", 2), ev(41, "away", 1), ev(50, "home", 2, "check")]
    shots = [
        shot(5.0, "home", 2),
        shot(11.5, "away", 2, near=True),            # 코트는 2점, 점수판은 3점 → 불일치
        shot(20.0, "home", 3, result="MISSED"),       # 실패는 무시
        shot(27.0, "home", 1, zone="FT"), shot(29.0, "home", 1, zone="FT"),  # 자유투 2개가 한 번에 +2
        shot(44.0, "away", 2),                        # 점수 변화보다 3초 뒤 → 짝 없음
    ]
    res = SC.check(events, shots)
    s = res.summary()
    assert s["점수판 변화"] == 4  # check 플래그는 정답에서 뺀다
    assert s["짝지은 변화"] == 3 and s["일치"] == 2
    assert s["라인 근처 슛이 낀 변화"] == 1 and s["라인 근처에서 틀림"] == 1
    assert s["점수 변화 없는 성공 슛"] == 1
    assert s["점수 비교"]["코트 2 / 점수판 3"] == 1
    ft_pair = next(p for p in res.pairs if p.event["t_sec"] == "30")
    assert len(ft_pair.shots) == 2 and ft_pair.agree
    assert len(res.rows()) == 5


def test_score_check_cli(tmp_path, capsys):
    import csv

    from gtt import cli

    def write(p, rows):
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(tmp_path / "ev.csv", [ev(6.5, "home", 2), ev(13, "away", 3)])
    write(tmp_path / "shots.csv", [shot(5.0, "home", 2), shot(11.5, "away", 3)])
    cli.main(["score-check", "--score-events", str(tmp_path / "ev.csv"), "--shots", str(tmp_path / "shots.csv"),
              "--out", str(tmp_path / "sc.csv")])
    out = capsys.readouterr().out
    assert "일치율(전체 변화 중): 1.0" in out
    assert (tmp_path / "sc.csv").exists()


def test_classify_click_uncertainty():
    """먼 코너는 1px 이 코트에서 더 크므로, 같은 거리라도 '라인 근처'로 더 넓게 잡는다."""
    from gtt import demo
    from gtt import homography as HG
    from gtt.court import FIBA

    top, T = demo._court_texture()
    H = demo._camera(0.0, top.shape[1::-1]) @ T
    # 코너 3점 라인(사이드라인에서 0.9m) 바깥 0.25m: 먼 쪽(y 작음)과 가까운 쪽(y 큼)
    far = HG.classify_click(H, *HG.court_to_image(H, 1.0, 0.65), FIBA, side="left")
    near = HG.classify_click(H, *HG.court_to_image(H, 1.0, 15 - 0.65), FIBA, side="left")
    assert far.points == 3 and near.points == 3
    assert far.margin_sigma > 1.5 * near.margin_sigma
    # 라인에서 멀면 확실한 판정
    mid = HG.classify_click(H, *HG.court_to_image(H, 5.0, 7.5), FIBA, side="left")
    assert mid.points == 2 and not mid.near_line


def test_check_late_scorebug_keeps_order():
    """실제 중계처럼 점수판이 16초 늦게 바뀌어도, 늦은 변화가 다음 슛을 가로채지 않는다."""
    events = [ev(325.5, "home", 2), ev(329.0, "home", 3)]
    shots = [shot(309.5, "home", 2), shot(327.0, "home", 3)]
    res = SC.check(events, shots)
    got = {p.event["t_sec"]: [x["t_sec"] for x in p.shots] for p in res.pairs}
    assert got == {"325.5": ["309.5"], "329.0": ["327.0"]}
    assert res.summary()["일치"] == 2


def test_check_counts_shots_without_court_separately():
    events = [ev(10, "home", 2), ev(20, "home", 3)]
    shots = [shot(8, "home", ""), shot(18, "home", 3)]
    s = SC.check(events, shots).summary()
    assert s["짝지은 변화"] == 2 and s["코트 좌표 없음"] == 1
    assert s["확실한 판정"] == 1 and s["확실한 판정 일치"] == 1 and s["일치율(짝지은 것 중)"] == 1.0


def test_check_replay_shot_after_event():
    """본 화면에서 놓친 득점을 리플레이(득점보다 늦게 나옴)에서 찍은 슛과 짝짓는다."""
    events = [ev(20.0, "away", 2)]
    rep = dict(shot(22.5, "away", 2), replay="True")
    res = SC.check(events, [rep])
    assert res.pairs[0].shots == [rep] and res.summary()["일치"] == 1


def test_correction_revert_is_not_a_score():
    """기록원이 잠깐 1점 내렸다가 되돌리면(실제 영상 87.5s) 그 +1 은 자유투가 아니다."""
    events = [ev(60, "home", 2), ev(85, "home", -1, "correction"), ev(87.5, "home", 1), ev(120, "home", 1)]
    fixed = SC.resolve_corrections(events)
    assert [e["flag"] for e in fixed] == ["ok", "correction", "revert", "ok"]
    assert events[2]["flag"] == "ok"  # 원본은 그대로
    assert SC.resolve_corrections(fixed) == fixed  # 이미 정리된 CSV 에 다시 해도 같다
    s = SC.check(events, [shot(59, "home", 2), shot(119, "home", 1, zone="FT")]).summary()
    assert s["점수판 변화"] == 2 and s["일치"] == 2


def test_correction_fixes_previous_score():
    """+3 을 넣었다가 -1 로 고치면 +2 득점, +2 를 넣었다가 -2 로 지우면 득점 없음."""
    events = [ev(10, "away", 3), ev(18, "away", -1, "correction"), ev(40, "home", 2), ev(45, "home", -2, "correction")]
    fixed = SC.resolve_corrections(events)
    assert [(e["delta"], e["flag"]) for e in fixed] == [(2, "ok"), ("-1", "merged"), ("2", "reverted"), ("-2", "merged")]
    assert SC.resolve_corrections(fixed) == fixed
    s = SC.check(events, [shot(9, "away", 2)]).summary()
    assert s["점수판 변화"] == 1 and s["일치"] == 1


def test_line_call_counts_as_human_judgment():
    """라인에 딱 걸친 슛을 사람이 확대 화면에서 '밟음(2점)'으로 판정하면 그 점수로 비교한다."""
    s = dict(shot(10, "home", 2, near=True), line_call="on")
    summ = SC.check([ev(12, "home", 2)], [s]).summary()
    assert summ["일치"] == 1 and summ["라인 근처 사람 판정"] == 1


def test_correction_late_batch_cancels_recent_scores():
    """슛 없이 +1 이 두 번 올랐다가 몇 분 뒤 -2 로 한꺼번에 지워지면 (실제 처음 보는 구간 147s·173s → 341s) 둘 다 득점 아님."""
    events = [ev(19.5, "home", 1), ev(56.5, "home", 2), ev(147, "home", 1), ev(173, "home", 1), ev(200, "away", 3),
              ev(341, "home", -2, "correction"), ev(474, "home", 2)]
    fixed = SC.resolve_corrections(events)
    assert [e["flag"] for e in fixed] == ["ok", "ok", "reverted", "reverted", "ok", "merged", "ok"]
    assert "늦은 정정" in fixed[2]["note"]
    assert SC.resolve_corrections(fixed) == fixed
    # 감소량과 딱 맞는 묶음이 없으면 그대로 둔다 (사람이 볼 수 있게 correction 으로 남김)
    odd = SC.resolve_corrections([ev(10, "home", 3), ev(100, "home", 3), ev(300, "home", -2, "correction")])
    assert [e["flag"] for e in odd] == ["ok", "ok", "correction"]


def test_ft_spot_whole_semicircle():
    """자유투는 자유투 라인 뒤 반원 안 어디서나 쏠 수 있다 (실제 영상: 라인 1.1m 뒤)."""
    assert is_ft_spot(5.79 + 1.1, 7.5, FIBA)
    assert is_ft_spot(5.79 + 1.6, 7.9, FIBA)
    assert not is_ft_spot(5.79 + 2.3, 7.5, FIBA)  # 반원 밖
    assert not is_ft_spot(5.79 + 1.75, 7.5 + 1.15, FIBA)  # 반원(+찍기 오차 0.2m) 바깥 대각선
