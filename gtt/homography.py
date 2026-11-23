"""화면 좌표 ↔ 코트 좌표 변환(호모그래피), 코트 라인 정합, 팬·줌 카메라 추적.

H 는 항상 "코트(m) → 화면(px)" 방향으로 저장한다. 화면 → 코트는 역행렬.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .court import FIBA, CourtSpec, court_polylines, landmarks


# ---------------------------------------------------------------- 기본 변환
def fit_homography(
    image_points: dict[str, tuple[float, float]], spec: CourtSpec = FIBA
) -> tuple[np.ndarray, float]:
    """랜드마크 이름 → 화면 좌표 로 H(코트→화면)를 구한다. (H, 평균 재투영 오차 px) 반환."""
    lm = landmarks(spec)
    names = [n for n in image_points if n in lm]
    unknown = [n for n in image_points if n not in lm]
    if unknown:
        raise ValueError(f"모르는 랜드마크 이름: {unknown}")
    if len(names) < 4:
        raise ValueError("랜드마크가 4개 이상 필요합니다.")
    src = np.array([lm[n] for n in names], dtype=np.float64)
    dst = np.array([image_points[n] for n in names], dtype=np.float64)
    method = cv2.RANSAC if len(names) >= 6 else 0
    H, _ = cv2.findHomography(src, dst, method, 5.0)
    if H is None:
        raise ValueError("호모그래피를 구하지 못했습니다. 점들이 한 직선 위에 몰려 있지 않은지 확인하세요.")
    err = float(np.mean(np.linalg.norm(project(H, src) - dst, axis=1)))
    return H, err


def project(H: np.ndarray, pts) -> np.ndarray:
    """점 배열(N,2)에 H 를 적용한다."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def image_to_court(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    x, y = project(np.linalg.inv(H), [(u, v)])[0]
    return float(x), float(y)


def court_to_image(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    u, v = project(H, [(x, y)])[0]
    return float(u), float(v)


def classify_click(H: np.ndarray, u: float, v: float, spec: CourtSpec = FIBA, side: str | None = None,
                   click_px: float = 3.0, k: float = 2.0, min_near_m: float = 0.3):
    """화면에서 찍은 발 위치(u, v)로 2점/3점을 판정하고, 찍기 오차를 반영해 '라인 근처'를 정한다.

    같은 1px 이라도 카메라에서 먼 쪽(특히 먼 코너)은 코트에서 수 cm~10cm 가 된다. 그래서 고정 30cm 대신
    click_px 만큼 잘못 찍었을 때 3점 라인까지 거리가 흔들리는 정도(margin_sigma)의 k 배와 30cm 중 큰 값 안쪽이면 near_line.
    """
    from .court import classify_shot, three_point_margin

    x, y = image_to_court(H, u, v)
    call = classify_shot(x, y, spec, side=side)
    m0 = call.margin
    mu = three_point_margin(*image_to_court(H, u + 1, v), spec, call.side) - m0
    mv = three_point_margin(*image_to_court(H, u, v + 1), spec, call.side) - m0
    sigma = click_px * float(np.hypot(mu, mv))
    call.margin_sigma = sigma
    call.near_line = abs(m0) < max(min_near_m, k * sigma)
    return call


def classify_clicks(clicks, spec: CourtSpec = FIBA, side: str | None = None,
                    click_px: float = 3.0, k: float = 2.0, min_near_m: float = 0.3):
    """같은 슛의 발 위치를 여러 프레임(또는 확대해서 여러 번) 찍은 것을 코트에서 평균해 판정한다.

    clicks: [(H, u, v), ...] — 프레임마다 카메라가 움직여도 각자의 H 로 코트 좌표를 구한 뒤 평균한다.
    찍기 오차는 서로 독립이라 n 번 찍으면 흔들림이 1/√n 로 준다. 라인 근처 슛을 확실히 가를 때 쓴다.
    """
    from .court import classify_shot

    calls = [classify_click(H, u, v, spec, side=side, click_px=click_px, k=k, min_near_m=min_near_m) for H, u, v in clicks]
    side = side or calls[0].side
    x = float(np.mean([c.court_xy[0] for c in calls]))
    y = float(np.mean([c.court_xy[1] for c in calls]))
    call = classify_shot(x, y, spec, side=side)
    sigma = float(np.sqrt(np.sum([c.margin_sigma ** 2 for c in calls]))) / len(calls)
    call.margin_sigma = sigma
    call.near_line = abs(call.margin) < max(min_near_m, k * sigma)
    return call


# ---------------------------------------------------------------- 그리기
def render_topdown(spec: CourtSpec = FIBA, px_per_m: float = 40.0, margin_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """위에서 본 코트 그림과, 코트(m)→그림(px) 변환 행렬을 돌려준다. 테스트·디버그용."""
    s = px_per_m
    w = int((spec.length + 2 * margin_m) * s)
    h = int((spec.width + 2 * margin_m) * s)
    img = np.full((h, w, 3), (70, 140, 200), np.uint8)  # 나무 바닥색(BGR)
    T = np.array([[s, 0, margin_m * s], [0, s, margin_m * s], [0, 0, 1]], dtype=np.float64)
    draw_court(img, T, spec, color=(255, 255, 255), thickness=max(2, int(0.05 * s)))
    return img, T


def draw_court(img: np.ndarray, H: np.ndarray, spec: CourtSpec = FIBA, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """코트 라인을 H(코트→화면)로 투영해 img 위에 그린다 (제자리 수정)."""
    for _, poly in court_polylines(spec).items():
        pts = densify(poly, 0.25)
        uv = project(H, pts)
        if not np.all(np.isfinite(uv)):
            continue
        cv2.polylines(img, [np.round(uv).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def densify(poly, step: float) -> np.ndarray:
    """폴리라인을 step(m) 간격 점으로 쪼갠다. 원근 투영 후에도 곡선이 매끄럽게 보이도록."""
    out = []
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n):
            t = i / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    out.append(poly[-1])
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------- 라인 정합
def line_mask(frame: np.ndarray) -> np.ndarray:
    """흰색 코트 라인 후보 픽셀 마스크 (0/255).

    흰색은 R·G·B 가 모두 높으므로 세 채널 중 가장 낮은 값(min 채널)으로 본다. 나무 바닥(주황·노랑)은 파랑이 낮아
    min 채널에서 어둡게 빠지고, 조명이 밝아 바닥이 하얗게 떠도 라인과의 차이가 남는다.
    (예전에는 HLS 채도를 썼는데, 아주 밝은 픽셀은 색이 조금만 섞여도 채도가 크게 튀어 라인이 사라졌다.)
    min 채널에서 주변보다 확실히 밝은 가는 구조(top-hat)만 남기고, 색이 뚜렷한 픽셀(max-min 이 큼)은 뺀다.
    """
    f = frame.astype(np.int16)
    mn = f.min(axis=2)
    chroma = f.max(axis=2) - mn
    mn8 = mn.astype(np.uint8)
    k = max(9, (min(frame.shape[:2]) // 40) | 1)
    tophat = cv2.morphologyEx(mn8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = ((tophat > 30) & (mn > 110) & (chroma < 70)).astype(np.uint8) * 255
    return mask


def _line_elements(spec: CourtSpec = FIBA) -> list[tuple[str, np.ndarray]]:
    """코트 라인을 한 줄씩(베이스라인, 사이드라인, 키 세 변, 코너 직선, 아크 …) 나눈 것. 줄마다 맞았는지 따로 본다."""
    L, W = spec.length, spec.width
    m, kw, c, ce, kl = W / 2, spec.key_width / 2, spec.corner_offset, spec.corner_straight_end, spec.key_length
    el = {"baseline_L": [(0, 0), (0, W)], "baseline_R": [(L, 0), (L, W)], "side_far": [(0, 0), (L, 0)],
          "side_near": [(0, W), (L, W)], "halfcourt": [(L / 2, 0), (L / 2, W)]}
    polys = court_polylines(spec)
    for sd in ("L", "R"):
        fx = (lambda x: x) if sd == "L" else (lambda x: L - x)
        el[f"{sd}_key_a"] = [(fx(0), m - kw), (fx(kl), m - kw)]
        el[f"{sd}_key_b"] = [(fx(0), m + kw), (fx(kl), m + kw)]
        el[f"{sd}_ft"] = [(fx(kl), m - kw), (fx(kl), m + kw)]
        el[f"{sd}_corner_a"] = [(fx(0), m - c), (fx(ce), m - c)]
        el[f"{sd}_corner_b"] = [(fx(0), m + c), (fx(ce), m + c)]
        el[f"{sd}_arc"] = polys[f"{sd}_three"][2:-2]
    return [(k, densify(v, 0.2)) for k, v in el.items()]


def key_area(H: np.ndarray, spec: CourtSpec = FIBA) -> float:
    """양쪽 키(페인트존)를 화면에 투영한 넓이(부호 있음, px²). 코트 전체가 선 하나로 쪼그라든 H 는 0 에 가깝다."""
    tot = 0.0
    for sd, sgn in (("L", 1.0), ("R", -1.0)):  # 오른쪽 키는 좌우가 뒤집힌 순서라 부호를 맞춘다
        q = project(H, court_polylines(spec)[f"{sd}_key"])
        x, y = q[:, 0], q[:, 1]
        tot += sgn * 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return tot


def _area_ok(H: np.ndarray, A0: float, spec: CourtSpec, lo: float, hi: float) -> bool:
    A = key_area(H, spec)
    return A0 != 0 and A * A0 > 0 and lo <= A / A0 <= hi


def line_support(dist: np.ndarray, H: np.ndarray, spec: CourtSpec = FIBA, tol: float = 4.0,
                 min_pts: int = 10, good: float = 0.4) -> tuple[int, int]:
    """화면에 보이는 코트 라인 줄 수와, 그중 실제 흰 선과 good 비율 이상 겹치는 줄 수.

    전체 일치율 하나로는 틀린 정합(예: 키를 하프코트 근처 선에 겹침)도 0.5~0.65 가 나온다. 맞는 H 라면 보이는 줄
    대부분이 각자 선과 겹치고, 틀린 H 는 몇 줄만 우연히 겹친다."""
    h, w = dist.shape
    n_vis = n_good = 0
    for _, pts in _line_elements(spec):
        uv = project(H, pts)
        ok = (uv[:, 0] >= 5) & (uv[:, 0] < w - 6) & (uv[:, 1] >= 5) & (uv[:, 1] < h - 6)
        if ok.sum() < min_pts:
            continue
        n_vis += 1
        ui = np.round(uv[ok]).astype(int)
        if float(np.mean(dist[ui[:, 1], ui[:, 0]] <= tol)) >= good:
            n_good += 1
    return n_vis, n_good


def refine_homography(
    frame: np.ndarray,
    H: np.ndarray,
    spec: CourtSpec = FIBA,
    search_px: int = 12,
    iterations: int = 10,
    min_matches: int = 30,
    return_support: bool = False,
):
    """현재 H 로 투영한 코트 라인을 실제 흰 라인 픽셀에 맞춰 다듬는다.

    투영된 라인 점마다 주변 search_px 안의 라인 픽셀 무게중심(=라인 중심)으로 끌어당긴 뒤
    H 를 다시 맞추는 과정을 반복한다 (mean-shift 방식).
    반환: (다듬은 H, 라인 일치율 0~1). 일치율 = 투영한 라인 점 중 실제 라인 위(±4px)에 있는 비율.
    return_support=True 면 (H, 일치율, (보이는 줄 수, 맞은 줄 수)).

    실제 중계에서 끌려온 점이 모두 흰 선 하나 위에 있으면 코트 전체가 그 선으로 쪼그라든 H 가 나와 일치율 0.9~1.0 으로
    통과한 적이 있다. 그래서 끌려온 점이 한 직선 위에만 있으면 그 단계를 버리고, 키 투영 넓이가 입력의 0.6~1.67배를
    벗어나는 결과는 받지 않는다.
    """
    mask = (line_mask(frame) > 0).astype(np.float32)
    h, w = mask.shape
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    def out(Hx, sc):
        return (Hx, sc, line_support(dist, Hx, spec)) if return_support else (Hx, sc)

    court_pts = np.concatenate([densify(p, 0.2) for p in court_polylines(spec).values()])
    if mask.sum() < min_matches:
        return out(H, 0.0)
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    A0 = key_area(H, spec)
    best_H, best_score = H, _line_score(dist, project(H, court_pts), w, h)
    cur = H
    # 넓은 창으로 크게 당긴 뒤 창을 줄여 가며 정밀하게 맞춘다 (coarse-to-fine)
    radii = [r for r in (search_px, max(3, search_px // 2), 3) if r <= search_px]
    for r in dict.fromkeys(radii):
        k = (2 * r + 1, 2 * r + 1)
        cnt = cv2.boxFilter(mask, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
        sx = cv2.boxFilter(mask * xs, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
        sy = cv2.boxFilter(mask * ys, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
        for _ in range(iterations):
            uv = project(cur, court_pts)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
            if ok.sum() < min_matches:
                break
            ui = np.round(uv[ok]).astype(int)
            c = cnt[ui[:, 1], ui[:, 0]]
            near = c >= 2
            if near.sum() < min_matches:
                break
            src = court_pts[ok][near]
            # 끌려온 코트 점이 거의 한 직선 위에만 있으면 H 가 정해지지 않는다(쪼그라듦) → 멈춘다
            sv = np.linalg.svd(src - src.mean(axis=0), compute_uv=False)
            if sv[1] < 0.08 * sv[0]:
                break
            target = np.stack([sx[ui[near, 1], ui[near, 0]], sy[ui[near, 1], ui[near, 0]]], axis=1) / c[near, None]
            newH, _ = cv2.findHomography(src, target.astype(np.float64), 0)
            if newH is None:
                break
            # 선수·관중 때문에 엉뚱한 곳으로 끌린 점은 빼고 한 번 더 맞춘다
            res = np.linalg.norm(project(newH, src) - target, axis=1)
            keep = res <= max(2.0, 2.5 * float(np.median(res)))
            if keep.sum() >= min_matches:
                newH, _ = cv2.findHomography(src[keep], target[keep].astype(np.float64), 0)
            if newH is None or not _area_ok(newH, A0, spec, 0.6, 1.67):
                break
            move = float(np.mean(np.linalg.norm(project(newH, src) - project(cur, src), axis=1)))
            cur = newH
            if move < 0.2:
                break
    score = _line_score(dist, project(cur, court_pts), w, h)
    if score >= best_score - 0.02 and _area_ok(cur, A0, spec, 0.6, 1.67):  # 일치율이 눈에 띄게 나빠지지 않았으면 채택
        best_H, best_score = cur, score
    return out(best_H, best_score)


def _line_score(dist: np.ndarray, uv: np.ndarray, w: int, h: int, tol: float = 4.0) -> float:
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if ok.sum() == 0:
        return 0.0
    ui = np.round(uv[ok]).astype(int)
    return float(np.mean(dist[ui[:, 1], ui[:, 0]] <= tol))


# ---------------------------------------------------------------- 카메라 추적
# 화면 간 변환을 맞출 때 쓰는 강건 추정: MAGSAC 이 있으면 쓰고(잘못 짝지은 점에 덜 끌림), 없으면 RANSAC
ROBUST = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def make_orb() -> "cv2.ORB":
    """중계 화면용 ORB. 바닥 결이 약하고 압축·잡음이 있으면 기본 문턱(20)으로는 특징점이 몇백 개뿐이라
    선수 몇 명에 끌려 변환이 수십 px 틀어진다. 문턱을 낮추고 개수를 늘린다."""
    return cv2.ORB_create(3000, fastThreshold=5)


@dataclass
class TrackResult:
    H: np.ndarray | None
    line_score: float
    status: str  # ok / lost / cut


class StaticMask:
    """카메라가 움직여도 화면에 그대로 있는 부분(스코어버그·방송사 로고·자막)을 찾는다.

    이런 그래픽의 특징점은 '안 움직임'에 투표해서, 바닥 무늬가 약한 실제 중계에서는 팬을 따라가지 못하게 만든다.
    최근 몇 프레임 사이 밝기 변화의 중앙값이 작은 곳을 정지 영역으로 보고 특징점 검출에서 뺀다.
    카메라가 계속 멈춰 있으면(화면 대부분이 정지) 마스크를 쓰지 않는다.
    """

    def __init__(self, history: int = 10, scale: float = 0.25, thresh: float = 2.5, max_static: float = 0.5):
        from collections import deque

        self.buf = deque(maxlen=history)
        self.scale = scale
        self.thresh = thresh
        self.max_static = max_static

    def update(self, gray: np.ndarray) -> None:
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.buf.append(small)

    @classmethod
    def from_frames(cls, frames, **kw) -> "StaticMask":
        """영상 전체에서 띄엄띄엄 뽑은 프레임들로 만든다 (그래픽은 그대로, 코트는 카메라 이동으로 달라짐)."""
        m = cls(history=max(4, len(frames)), **kw)
        for f in frames:
            m.update(f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        return m

    def mask(self, shape) -> np.ndarray | None:
        """특징점을 찾을 곳 = 255, 정지 영역 = 0. 판단할 근거가 부족하면 None."""
        if len(self.buf) < 4:
            return None
        stack = np.stack(self.buf)
        med = np.median(np.abs(np.diff(stack, axis=0)), axis=0)
        static = (med < self.thresh).astype(np.uint8)
        # 넓은 정지 영역만 (작은 점은 잡음)
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        static = cv2.dilate(static, np.ones((5, 5), np.uint8))
        # 무늬가 있는 곳(특징점이 생길 곳) 중 정지 비율로 판단
        tex = cv2.Laplacian(stack[-1], cv2.CV_32F)
        textured = np.abs(tex) > 4
        if textured.sum() == 0 or static[textured].mean() > self.max_static:
            return None
        m = cv2.resize(static, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return np.where(m > 0, 0, 255).astype(np.uint8)


class CourtTracker:
    """팬·줌하는 중계 카메라에서 프레임마다 H 를 이어 붙인다.

    1) 이전 프레임 ↔ 현재 프레임 ORB 특징점으로 화면 간 변환을 구해 H 를 전파하고 (방송 그래픽은 StaticMask 로 제외)
    2) 코트 라인 정합으로 누적 오차를 지운다.
    3) 라인 일치율이 min_score 보다 낮으면 H 를 내지 않고(status="coast") 화면 이동만 따라간다.
    4) 화면 전환(리플레이·클로즈업)으로 놓치면, 잘 맞았던 순간들(키프레임)과 현재 화면을 맞춰 다시 찾는다 (relocalize).
    """

    def __init__(self, H0: np.ndarray, frame0: np.ndarray, spec: CourtSpec = FIBA, refine_every: int = 1,
                 min_score: float = 0.5, use_static_mask: bool = True, overlay_mask: np.ndarray | None = None,
                 keyframe_score: float = 0.6, max_jump_px: float = 15.0, weak_inliers: int = 100,
                 min_good_lines: int = 2, min_line_frac: float = 0.6):
        """min_score: 라인 일치율이 이보다 낮으면 좌표를 내지 않는다(coast). 실제 중계에서 반대쪽 골대에 잘못 붙은 화면이
        0.36~0.47 이었으므로 0.5. keyframe_score: 이 이상일 때만 키프레임으로 저장(틀린 화면이 기준이 되지 않게).
        max_jump_px: 라인 정합이 화면 이동으로 예측한 위치에서 이보다 많이 끌고 가면 다른 선(다른 종목 선, 반대쪽 키)에
        붙은 것으로 보고 받지 않는다. weak_inliers: 화면 간 변환의 맞은 점이 이보다 적으면 빠른 패닝으로 보고
        샘플 사이 프레임들로 이어서 다시 구한다.
        min_good_lines, min_line_frac: 화면에 보이는 코트 라인 줄 가운데 실제 흰 선과 겹치는 줄이 min_good_lines 줄 이상이고
        min_line_frac 비율 이상이어야 좌표를 낸다. 실제 중계에서 일치율 0.5~0.65 화면의 절반이 1~4 m 틀렸는데, 틀린 화면은
        몇 줄만 우연히 겹친다."""
        self.spec = spec
        self.H = H0.copy()
        self.prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        self.orb = make_orb()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.refine_every = refine_every
        self.min_score = min_score
        self.keyframe_score = keyframe_score
        self.max_jump_px = max_jump_px
        self.weak_inliers = weak_inliers
        self.sure_score = 0.8
        self.min_good_lines = min_good_lines
        self.min_line_frac = min_line_frac
        self.last_support = (0, 0)  # (보이는 줄, 맞은 줄)
        self._fast = False
        self.n = 0
        self.lost = False
        self.coast_n = 0  # 연속 coast 횟수
        self.chained = 0  # 샘플 사이 프레임으로 이어 붙인 횟수 (통계)
        self.last_inliers = 0
        # overlay_mask: 미리 찾은 방송 그래픽 마스크(특징점 찾을 곳 = 255). 없으면 최근 프레임으로 그때그때 찾는다.
        self.overlay_mask = overlay_mask
        self.static = StaticMask() if (use_static_mask and overlay_mask is None) else None
        if self.static is not None:
            self.static.update(self.prev)
        self.keyframes: list[tuple] = []  # (keypoints, descriptors, H)
        self.anchors: list[tuple] = []  # 사람이 맞춘 캘리브레이션 화면들 — 지워지지 않는 키프레임
        self._court_pts = np.concatenate([densify(p, 0.5) for p in court_polylines(spec).values()])
        self.add_anchor(frame0, H0)

    def add_anchor(self, frame: np.ndarray, H: np.ndarray) -> None:
        """캘리브레이션한 화면(예: 왼쪽 골대, 오른쪽 골대)을 다시 찾기 기준으로 넣는다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, self._mask(gray))
        if des is not None and len(kp) >= 50:
            self.anchors.append((kp, des, H.copy()))

    def _mask(self, gray):
        if self.overlay_mask is not None:
            return self.overlay_mask
        return self.static.mask(gray.shape) if self.static is not None else None

    def _add_keyframe(self, gray, H, max_keep: int = 80) -> None:
        kp, des = self.orb.detectAndCompute(gray, self._mask(gray))
        if des is not None and len(kp) >= 50:
            self.keyframes.append((kp, des, H.copy()))
            self.keyframes = self.keyframes[-max_keep:]

    def _jump(self, Ha: np.ndarray, Hb: np.ndarray, shape) -> float:
        """두 H 로 투영한 코트 점들이 화면 안에서 평균 몇 px 떨어지는지."""
        h, w = shape[:2]
        a = project(Ha, self._court_pts)
        b = project(Hb, self._court_pts)
        vis = (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
        return float(np.linalg.norm(a[vis] - b[vis], axis=1).mean()) if vis.any() else 0.0

    def _motion(self, gray, between) -> np.ndarray | None:
        """이전 샘플 → 지금 화면 변환. 직접 구한 변환이 약하면(빠른 패닝·흐림) 샘플 사이 프레임들을 차례로 이어 붙인다."""
        mask = self._mask(gray)
        F, n = frame_to_frame(self.prev, gray, self.orb, self.matcher, mask=mask, return_inliers=True)
        self.last_inliers = n
        self._fast = False
        if (F is None or n < self.weak_inliers) and between:
            G = np.eye(3)
            g0 = self.prev
            for b in list(between) + [gray]:
                g1 = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                Fi = frame_to_frame(g0, g1, self.orb, self.matcher, mask=mask)
                if Fi is None:
                    G = None
                    break
                G = Fi @ G
                g0 = g1
            if G is not None:
                self.chained += 1
                self._fast = True
                return G / G[2, 2]
        return F

    def update(self, frame: np.ndarray, between=None) -> TrackResult:
        """한 샘플 진행. between: 이전 샘플과 이 샘플 사이의 프레임들(빠른 패닝일 때만 쓴다).

        라인 일치율이 낮으면 H 를 내지 않되(coast) 화면 이동은 계속 따라가다가, 라인이 다시 잘 보이면 그대로 이어 간다.
        화면 전환으로 이동을 못 구하면(cut) 다시 찾기 전까지 멈춘다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.static is not None:
            self.static.update(gray)
        F = None if self.lost else self._motion(gray, between)
        self.prev = gray
        if F is None:
            was_lost = self.lost
            self.lost = True
            return TrackResult(None, 0.0, "lost" if was_lost else "cut")
        self.H = F @ self.H
        self.H /= self.H[2, 2]
        self.n += 1
        score = float("nan")
        if self.refine_every and self.n % self.refine_every == 0:
            Hr, score, sup = refine_homography(frame, self.H, self.spec, return_support=True)
            self.last_support = sup
            sure = score >= self.sure_score and self.supported(sup)
            if self._jump(self.H, Hr, frame.shape) > self.max_jump_px and not sure:
                # 화면 이동으로 예측한 자리에서 멀리 끌려갔는데 라인도 확실히 맞지 않는다 → 다른 선(반대쪽 키, 다른 종목 선)에
                # 붙었을 수 있다. 예측을 믿고 좌표는 내지 않는다. (라인이 거의 다 맞으면 흐린 패닝 뒤의 정상 보정으로 본다)
                self.coast_n += 1
                return TrackResult(None, score, "coast")
            self.H = Hr
            if score < self.min_score or not self.supported(sup) or (self._fast and not sure):
                # 빠른 패닝(흐림) 중에는 라인이 확실히 맞을 때만 좌표를 낸다
                self.coast_n += 1
                return TrackResult(None, score, "coast")
            if score >= self.keyframe_score and self.n % 2 == 0:
                self._add_keyframe(gray, self.H)
        self.coast_n = 0
        return TrackResult(self.H.copy(), score, "ok")

    def supported(self, sup: tuple[int, int]) -> bool:
        """보이는 코트 라인 줄 대부분이 실제 선과 겹치는가 (line_support 결과)."""
        n_vis, n_good = sup
        return n_good >= self.min_good_lines and n_good >= self.min_line_frac * n_vis

    def relocalize(self, frame: np.ndarray, min_inliers: int = 40) -> tuple[np.ndarray | None, float]:
        """놓친 뒤 현재 화면을 키프레임들과 맞춰 H 를 다시 찾는다. (H, 라인 일치율) 또는 (None, 0).

        키프레임과 특징점이 아주 많이 맞으면(strong_inliers 이상) 라인이 흐려 일치율이 낮아도 믿을 만하므로
        self.last_inliers 에 맞은 수를 남긴다.
        """
        self.last_inliers = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, d2 = self.orb.detectAndCompute(gray, self._mask(gray))
        if d2 is None or len(kp2) < min_inliers:
            return None, 0.0
        best = (None, 0)
        # 앵커 전부 + 최근 키프레임 10개 + 그 이전은 띄엄띄엄 (놓친 동안 매 샘플 호출되므로 비용을 묶어 둔다)
        recent = list(reversed(self.keyframes))
        older = recent[10:]
        cands = list(self.anchors) + recent[:10] + older[:: max(1, len(older) // 14 + 1)]
        for kp1, d1, Hk in cands:
            m = self.matcher.match(d1, d2)
            if len(m) < min_inliers:
                continue
            p1 = np.float32([kp1[x.queryIdx].pt for x in m])
            p2 = np.float32([kp2[x.trainIdx].pt for x in m])
            F, inl = cv2.findHomography(p1, p2, ROBUST, 3.0)
            if F is None or inl is None:
                continue
            n = int(inl.sum())
            if n > best[1]:
                best = (F @ Hk, n)
        if best[0] is None or best[1] < min_inliers:
            return None, 0.0
        self.last_inliers = best[1]
        H, score, sup = refine_homography(frame, best[0] / best[0][2, 2], self.spec, return_support=True)
        self.last_support = sup
        if not self.supported(sup):
            # 다시 찾은 자리의 라인이 대부분 안 겹친다 → 믿지 않는다 (일치율을 0 으로 낮춰 받는 쪽이 버리게)
            score = min(score, 0.0)
        return H, score

    def reset(self, H: np.ndarray, frame: np.ndarray) -> None:
        self.H = H.copy()
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lost = False
        self.coast_n = 0


def frame_to_frame(prev_gray, gray, orb=None, matcher=None, min_inliers: int = 25, mask=None,
                   return_inliers: bool = False):
    """두 프레임 사이 화면 변환(3x3). 실패하면 None (화면 전환 가능성). mask: 특징점을 찾을 곳(255).
    return_inliers=True 면 (F, 맞은 점 수)."""
    orb = orb or make_orb()
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    fail = (None, 0) if return_inliers else None
    k1, d1 = orb.detectAndCompute(prev_gray, mask)
    k2, d2 = orb.detectAndCompute(gray, mask)
    if d1 is None or d2 is None or len(k1) < min_inliers or len(k2) < min_inliers:
        return fail
    matches = matcher.match(d1, d2)
    if len(matches) < min_inliers:
        return fail
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    F, inl = cv2.findHomography(p1, p2, ROBUST, 3.0)
    if F is None or inl is None or int(inl.sum()) < min_inliers:
        return fail
    return (F, int(inl.sum())) if return_inliers else F
