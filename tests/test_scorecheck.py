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
