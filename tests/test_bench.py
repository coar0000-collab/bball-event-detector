from gtt import bench


def test_bench_short(tmp_path):
    """첫 리플레이(17~21.5초)를 지나는 짧은 벤치: 리플레이 중 틀린 코트 좌표를 내지 않고, 끝나면 다시 맞춘다."""
    res = bench.run(tmp_path, seconds=26, seed=0)
    sc = res["점수 변화"]
    assert sc["잘못 읽음"] == 0 and sc["맞음"] == sc["정답"]
    tr = res["코트 추적"]
    assert tr["리플레이 중 틀린 좌표"] == 0
    assert tr["리플레이 뒤 복구(초)"][0] is not None and tr["리플레이 뒤 복구(초)"][0] <= 3
    assert tr["정확(<3px)"] >= 0.95 * tr["본 카메라 샘플"]
    court = res["코트로 계산한 점수 vs 점수판"]["발 위치 오차 0px, 자유투 라벨"]
    assert court["점수판 변화"] > 0 and court["일치"] >= court["점수판 변화"] - 1
    assert court["확실한 판정 일치"] == court["확실한 판정"]  # 라인에서 먼 슛은 틀리면 안 된다
