import cv2
import numpy as np

from gtt import scorebug as SB


def bug(text, font=cv2.FONT_HERSHEY_DUPLEX, h=40):
    """중계 스코어버그처럼 어두운 바탕에 흰 숫자."""
    w = 30 * len(text) + 20
    img = np.full((h, w, 3), (40, 20, 10), np.uint8)
    cv2.putText(img, text, (8, h - 8), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def test_segment_counts_digits():
    assert len(SB.segment_digits(bug("47"))) == 2
    assert len(SB.segment_digits(bug("105"))) == 3


def test_segment_light_background():
    img = 255 - bug("38")
    assert len(SB.segment_digits(img)) == 2


def test_learned_templates_read_new_numbers():
    t = SB.DigitTemplates()
    for v in ["47", "52", "36", "18", "90"]:
        assert SB.learn(t, bug(v), v)
    assert t.missing() == []
    for v in ["105", "63", "7", "88", "41", "29"]:
        r = SB.read_number(bug(v), t)
        assert r.value == int(v), (v, r)


def test_font_templates_fallback():
    t = SB.DigitTemplates.from_fonts()
    assert t.missing() == []
    ok = sum(SB.read_number(bug(v, cv2.FONT_HERSHEY_SIMPLEX), t).value == int(v) for v in ["12", "34", "56", "78", "90"])
    assert ok >= 4


def test_templates_roundtrip(tmp_path):
    t = SB.DigitTemplates()
    SB.learn(t, bug("0123456789"), "0123456789")
    p = tmp_path / "d.npz"
    t.save(p)
    t2 = SB.DigitTemplates.load(p)
    assert t2.missing() == []
    assert SB.read_number(bug("64"), t2).value == 64


def test_config_crop_scales(tmp_path):
    cfg = SB.ScorebugConfig(rois={"home": [100, 50, 40, 20], "away": [200, 50, 40, 20]}, reference_size=(1280, 720))
    big = np.zeros((1080, 1920, 3), np.uint8)
    assert cfg.crop(big, "home").shape[:2] == (30, 60)
    cfg.save(tmp_path / "c.json")
    assert SB.ScorebugConfig.load(tmp_path / "c.json").rois == cfg.rois


def run(det, seq, dt=0.5):
    out = []
    for i, (h, a) in enumerate(seq):
        out += det.push(i * dt, h, a)
    return out


def test_detector_basic_and_noise():
    seq = [(10, 8)] * 4 + [(None, None), (12, 8), (12, 8), (17, 8), (12, 8), (12, 8)] + [(12, 11)] * 4 + [(13, 11)] * 3
    ev = run(SB.ScoreChangeDetector(stable_n=3), seq)
    assert [(e.team, e.delta, e.flag) for e in ev] == [("home", 2, "ok"), ("away", 3, "ok"), ("home", 1, "ok")]
    assert ev[0].t == 2.5  # 처음 12:8 이 읽힌 시각


def test_detector_flags_odd_changes():
    seq = [(10, 8)] * 3 + [(20, 8)] * 6 + [(18, 8)] * 6
    ev = run(SB.ScoreChangeDetector(stable_n=3, stable_n_big=6), seq)
    assert [(e.delta, e.flag) for e in ev] == [(10, "check"), (-2, "correction")]


def test_short_glitch_ignored():
    seq = [(10, 8)] * 3 + [(70, 8)] * 2 + [(10, 8)] * 5
    assert run(SB.ScoreChangeDetector(), seq) == []


def test_split_digit_read_near_previous_score():
    """중계 글꼴의 '4' 가 사선과 세로획 두 조각으로 끊겨 '241'·'221' 처럼 읽히던 경우 (실제 처음 보는 구간 533.5s: 22→24 를
    114 로 읽어 +92). 직전 점수 근처 값 중 템플릿과 닮은 값으로 읽는다. 실제 화면에서 '4' 를 등록하지 않아 글꼴 대체
    템플릿만 있는 경우."""
    t = SB.DigitTemplates()
    for v in ["10", "23", "56", "78", "90", "33", "15"]:
        assert SB.learn(t, bug(v), v)
    assert t.missing() == ["4"]
    for g in SB.DigitTemplates.from_fonts().samples["4"]:
        t.add("4", g)
    img = bug("24")
    img[:, 39:40] = (40, 20, 10)  # '4' 를 두 조각으로 끊는다
    assert SB.read_number(img, t).value != 24
    assert SB.read_number_near(img, t, 22).value == 24
    # 정상 판독은 그대로, 크게 바뀐 값(다음 경기 0:0)은 닮은 값이 없으면 그냥 읽은 값
    assert SB.read_number_near(bug("23"), t, 22).value == 23
    assert SB.read_number_near(bug("0"), t, 87).value == 0
    # 끊긴 조각도 자릿수를 알면 합쳐서 등록할 수 있다
    t2 = SB.DigitTemplates()
    assert SB.learn(t2, img, "24") and "4" in t2.samples
