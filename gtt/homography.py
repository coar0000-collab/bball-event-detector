"""화면 좌표 ↔ 코트 좌표 변환(호모그래피), 코트 라인 정합, 팬·줌 카메라 추적.

H 는 항상 "코트(m) → 화면(px)" 방향으로 저장한다. 화면 → 코트는 역행렬.
"""
from __future__ import annotations

import math

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


def _three_center_samples(H: np.ndarray, spec: CourtSpec, side: str, step_m: float = 0.05):
    """3점 라인 띠 가운데 선의 점들을 화면에 투영한 것과, 각 점에서 라인 바깥쪽(3점 쪽)을 가리키는 화면 단위 벡터."""
    from .court import three_line_edges

    pts = densify(three_line_edges(spec)[("L" if side == "left" else "R") + "_center"], step_m)
    rx, ry = spec.rim(side)
    d_base = pts[:, 0] if side == "left" else spec.length - pts[:, 0]
    dirs = np.zeros_like(pts)
    corner = d_base <= spec.corner_straight_end + 1e-6
    dirs[corner, 1] = np.sign(pts[corner, 1] - ry)
    r = np.hypot(pts[~corner, 0] - rx, pts[~corner, 1] - ry)
    dirs[~corner, 0] = (pts[~corner, 0] - rx) / r
    dirs[~corner, 1] = (pts[~corner, 1] - ry) / r
    a = project(H, pts)
    b = project(H, pts + 0.1 * dirs)
    n = b - a
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return a, n


def snap_three_line(frame: np.ndarray, H: np.ndarray, u: float, v: float, spec: CourtSpec = FIBA, side: str = "left",
                    exclude_px: float = 22.0, reach_px: float = 90.0, search_px: int = 12, min_samples: int = 6,
                    mask: np.ndarray | None = None, ignore: np.ndarray | None = None, far_reach_px: float = 150.0,
                    wide_px: int = 28):
    """찍은 발 근처에서 실제로 보이는 3점 라인이 H 로 계산한 라인보다 화면에서 몇 px 바깥(+)/안쪽(-)에 있는지.

    코트 전체로 맞춘 H 는 군데군데 몇 px(화면 가장자리는 렌즈 휨 때문에 20px 넘게도) 어긋날 수 있다. 2점/3점은 발끝과
    '칠해진 라인' 사이 거리로 정해지므로, 발 주변(신발이 가리는 exclude_px 안쪽은 빼고 reach_px 까지)에서 라인 가운데를
    직접 찾아 그 차이를 잰다. 가까이에 쓸 만한 조각이 모자라면 far_reach_px 까지 넓힌다.
    계산 라인에서 ±search_px 안의 가는 흰 줄(폭 8px 이하)을 모두 후보로 두고, 여러 조각이 같은 차이를 가리키는 값(투표)을
    고른다. 가장 가까운 줄만 쓰면 신발·그래픽 테두리 같은 엉뚱한 줄에 붙는다(실제 영상 383s: +2.8px 로 잘못 보정).
    ±search_px 에서 못 찾으면 ±wide_px 로 한 번 더 찾는다(실제 영상: 화면 아래 라인이 20px 어긋남). 처음부터 넓게 찾으면 먼 코너에서
    0.9m 옆 사이드라인까지 창에 들어와 애매해진다.
    두 값이 비슷하게 많이 나오면(다른 선이 나란히 있음) 믿지 않는다. 그래픽 자리(ignore)에 걸친 조각은 쓰지 않는다.
    반환: (차이 px, 쓴 조각 수, 발에서 가장 가까운 라인 점의 바깥 방향 단위 벡터) 또는 None(가려짐·애매함).
    """
    m = (line_mask(frame) > 0) if mask is None else mask
    r = _snap_three(m, H, u, v, spec, side, exclude_px, reach_px, search_px, min_samples, ignore, far_reach_px)
    if r is None and wide_px > search_px:
        r = _snap_three(m, H, u, v, spec, side, exclude_px, reach_px, wide_px, min_samples, ignore, far_reach_px)
    return r


def _snap_three(m, H, u, v, spec, side, exclude_px, reach_px, search_px, min_samples, ignore, far_reach_px):
    h, w = m.shape
    a, n = _three_center_samples(H, spec, side)
    d = np.hypot(a[:, 0] - u, a[:, 1] - v)
    if not len(d):
        return None
    n_near = n[int(np.argmin(d))]
    ts = np.arange(-search_px, search_px + 1)
    bad = None
    if ignore is not None:
        bad = cv2.dilate(ignore.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def runs_at(i):
        q = a[i][None, :] + ts[:, None] * n[i][None, :]
        qi = np.round(q).astype(int)
        inside = (qi[:, 0] >= 0) & (qi[:, 0] < w) & (qi[:, 1] >= 0) & (qi[:, 1] < h)
        if not inside.all():
            return None
        if bad is not None and bad[qi[:, 1], qi[:, 0]].any():
            return None
        on = m[qi[:, 1], qi[:, 0]]
        out, k = [], 0
        while k < len(on):
            if on[k]:
                j = k
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                # 너무 두꺼운 덩어리(신발·유니폼·큰 흰 면)나 창 끝에 걸린 줄은 버린다
                if j - k + 1 <= 8 and k > 0 and j < len(on) - 1:
                    out.append((ts[k] + ts[j]) / 2.0)
                k = j + 1
            else:
                k += 1
        return out

    # 다른 코트 라인(사이드라인·베이스라인 등)이 각 조각의 바깥 방향 직선과 만나는 자리(계산 라인 기준 px).
    # 먼 코너에서는 0.9m 옆 사이드라인이 10~20px 안에 들어와 두 줄이 같이 보인다. H 가 어긋나면 두 줄이 같이 밀리므로
    # '3점 라인이 s 에 있다'는 가정이 맞으면 다른 줄도 s + (그 줄 자리) 에 보여야 한다 → 그 가정에 점수를 더 준다.
    from .court import court_polylines as _cp

    other_pts = np.concatenate([densify(pl, 0.05) for k, pl in _cp(spec).items() if not k.endswith("_three")])
    other_uv = project(H, other_pts)

    def others_at(i):
        rel = other_uv - a[i]
        along = rel @ n[i]
        perp = np.abs(rel @ np.array([-n[i][1], n[i][0]]))
        o = np.sort(along[(perp <= 1.0) & (np.abs(along) <= search_px + 30)])
        out = []
        for x in o:
            if not out or x - out[-1] > 2.0:
                out.append(float(x))
        return out

    def collect(lo, hi):
        idx = np.flatnonzero((d >= lo) & (d <= hi))
        if len(idx) > 60:
            idx = idx[np.linspace(0, len(idx) - 1, 60).astype(int)]
        res = []
        for i in idx:
            r = runs_at(i)
            if r:
                res.append((r, n[i], others_at(i)))
        return res

    samples = collect(exclude_px, reach_px)
    if len(samples) < min_samples and far_reach_px > reach_px:
        samples += collect(reach_px + 1e-6, far_reach_px)
    if len(samples) < min_samples:
        return None

    def near(r, x):
        return any(abs(q - x) <= 2.0 for q in r)

    # 투표: '3점 라인이 c 에 있다' → 조각마다 c 에 줄이 보이면 1점, 창 안에 들어오는 다른 코트 라인도 제자리(c + o)에 보이면
    # 그 비율만큼 더한다.
    cands = sorted({round(x * 2) / 2 for r, _, _ in samples for x in r})
    seen, score = [], []
    for c in cands:
        n3 = sc = 0.0
        for r, _, os_ in samples:
            if near(r, c):
                n3 += 1
                sc += 1
                inw = [o for o in os_ if abs(c + o) <= search_px - 1]
                if inw:
                    sc += sum(near(r, c + o) for o in inw) / len(inw)
        seen.append(n3)
        score.append(sc)
    seen, score = np.array(seen), np.array(score)
    best = int(np.argmax(score - 0.01 * np.abs(np.array(cands))))  # 같으면 계산 위치에 가까운 쪽
    c0, v0 = cands[best], int(seen[best])
    if v0 < max(min_samples, 0.6 * len(samples)):
        return None
    # 4px 넘게 떨어진 다른 값이 비슷하게 그럴듯하면 애매하다 (설명 안 되는 나란한 선)
    other = [sv for c, sv in zip(cands, score) if abs(c - c0) > 4.0]
    if other and max(other) >= 0.6 * score[best]:
        return None
    samples = [(r, ni) for r, ni, _ in samples]
    # 라인이 화면에서 (du, dv) 만큼 통째로 밀렸다고 보고 맞춘다: 조각마다 차이 = 그 자리 바깥 방향 · (du, dv).
    # 아크를 따라 바깥 방향이 돌아가므로, 발에서 먼 조각까지 쓸 때 차이를 그냥 평균하면 발 자리 값과 달라진다.
    pred = np.full(len(samples), c0)
    t = np.zeros(2)
    for it in range(3):
        tol = 3.0 if it == 0 else 2.0
        rows, offs = [], []
        for (r, ni), p in zip(samples, pred):
            x = min(r, key=lambda q: abs(q - p))
            if abs(x - p) <= tol:
                rows.append(ni)
                offs.append(x)
        if len(offs) < min_samples:
            return None
        A, b = np.array(rows), np.array(offs)
        lam = 1e-3 * len(b)
        t = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ b)
        pred = np.array([float(ni @ t) for _, ni in samples])
    res = b - A @ t
    if float(np.median(np.abs(res))) > 1.5:
        return None
    off = float(n_near @ t)
    if abs(off) > search_px - 2:
        return None
    return off, len(b), n_near


def classify_click(H: np.ndarray, u: float, v: float, spec: CourtSpec = FIBA, side: str | None = None,
                   click_px: float = 3.0, k: float = 2.0, min_near_m: float = 0.3, frame: np.ndarray | None = None,
                   foot: str = "mid", line_mask_img: np.ndarray | None = None, ignore: np.ndarray | None = None,
                   hidden_near_m: float = 1.2, wide_flip_px: float = 12.0):
    """화면에서 찍은 발 위치(u, v)로 2점/3점을 판정하고, 찍기 오차를 반영해 '라인 근처'를 정한다.

    같은 1px 이라도 카메라에서 먼 쪽(특히 먼 코너)은 코트에서 수 cm~10cm 가 된다. 그래서 고정 30cm 대신
    click_px 만큼 잘못 찍었을 때 3점 라인까지 거리가 흔들리는 정도(margin_sigma)의 k 배와 30cm 중 큰 값 안쪽이면 near_line.
    frame 을 주면 발 근처에 실제로 보이는 3점 라인에 맞춰 위치를 보정한다(snap_three_line). 라인이 가려져 못 찾으면
    H 로 계산한 라인을 그대로 쓰고 line_snap_px=None.
    그때 발 근처 라인이 방송 그래픽(ignore)이나 화면 가장자리에 걸려 있고 라인까지 hidden_near_m 안이면, 계산만으로는
    믿을 수 없으므로 line_hidden=True, near_line=True (사람 판정). 실제 영상: 스코어버그 바로 위 윙 3점 2개를 계산 라인이
    20px 어긋나 2점으로 냈다.
    라인 보정이 wide_flip_px 보다 크게 옮겼는데(넓힌 창에서 찾음) 그 때문에 2점↔3점이 바뀌면, 다른 선을 잡았을 수도 있으므로
    near_line=True (실제 영상: 골밑 슛에서 17~23px 보정이 나옴 — 다른 선을 잡은 것으로 보임, 판정은 그대로였음).
    foot: "mid" = 두 발 가운데를 찍음 / "toe" = 림 쪽 발끝을 찍음 (court.classify_shot 참고).
    """
    from .court import classify_shot, three_point_margin

    x, y = image_to_court(H, u, v)
    call = classify_shot(x, y, spec, side=side, foot=foot)
    pts0 = call.points
    snap = None
    if frame is not None:
        snap = snap_three_line(frame, H, u, v, spec, call.side, mask=line_mask_img, ignore=ignore)
        if snap is not None:
            off, _, n_near = snap
            # 칠해진 라인이 off px 바깥에 있으면, 라인에 대한 발 위치는 그만큼 안쪽 → 찍은 점을 반대로 옮긴 것과 같다
            u, v = u - off * float(n_near[0]), v - off * float(n_near[1])
            x, y = image_to_court(H, u, v)
            call = classify_shot(x, y, spec, side=call.side, foot=foot)
            call.line_snap_px = off
    m0 = call.margin
    reach = spec.foot_reach if foot == "mid" else 0.0
    mu = three_point_margin(*image_to_court(H, u + 1, v), spec, call.side) - reach - m0
    mv = three_point_margin(*image_to_court(H, u, v + 1), spec, call.side) - reach - m0
    sigma = click_px * float(np.hypot(mu, mv))
    call.margin_sigma = sigma
    call.near_line = abs(m0) < max(min_near_m, k * sigma)
    if snap is not None and abs(snap[0]) > wide_flip_px and call.points != pts0:
        call.near_line = True
    if frame is not None and snap is None and abs(m0) < hidden_near_m:
        call.line_hidden = _line_blocked(H, u, v, spec, call.side, frame.shape, ignore)
        call.near_line = call.near_line or call.line_hidden
    return call


def _line_blocked(H: np.ndarray, u: float, v: float, spec: CourtSpec, side: str, shape, ignore: np.ndarray | None,
                  reach_px: float = 90.0, edge_px: int = 12) -> bool:
    """발 근처(reach_px 안) 3점 라인 조각의 절반 이상이 그래픽(ignore)이나 화면 가장자리(edge_px)에 걸려 있는가."""
    a, _ = _three_center_samples(H, spec, side)
    d = np.hypot(a[:, 0] - u, a[:, 1] - v)
    near = a[d <= reach_px]
    if not len(near):
        return True  # 발 근처에 라인 자체가 화면 안에 없다
    h, w = shape[:2]
    ui = np.round(near).astype(int)
    out = (ui[:, 0] < edge_px) | (ui[:, 0] >= w - edge_px) | (ui[:, 1] < edge_px) | (ui[:, 1] >= h - edge_px)
    blk = out.copy()
    if ignore is not None:
        inside = ~out
        g = cv2.dilate(ignore.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        blk[inside] = g[ui[inside, 1], ui[inside, 0]]
    return float(blk.mean()) >= 0.5


def classify_clicks(clicks, spec: CourtSpec = FIBA, side: str | None = None,
                    click_px: float = 3.0, k: float = 2.0, min_near_m: float = 0.3, foot: str = "mid",
                    ignore: np.ndarray | None = None):
    """같은 슛의 발 위치를 여러 프레임(또는 확대해서 여러 번) 찍은 것을 코트에서 평균해 판정한다.

    clicks: [(H, u, v), ...] 또는 [(H, u, v, frame), ...] — 프레임마다 카메라가 움직여도 각자의 H 로 코트 좌표를 구한 뒤
    평균한다. frame 이 있으면 프레임마다 보이는 3점 라인에 맞춰 보정한 뒤 평균한다.
    찍기 오차는 서로 독립이라 n 번 찍으면 흔들림이 1/√n 로 준다. 라인 근처 슛을 확실히 가를 때 쓴다.
    """
    from .court import classify_shot

    calls = [classify_click(c[0], c[1], c[2], spec, side=side, click_px=click_px, k=k, min_near_m=min_near_m,
                            frame=c[3] if len(c) > 3 else None, foot=foot, ignore=ignore) for c in clicks]
    side = side or calls[0].side
    x = float(np.mean([c.court_xy[0] for c in calls]))
    y = float(np.mean([c.court_xy[1] for c in calls]))
    call = classify_shot(x, y, spec, side=side, foot=foot)
    snaps = [c.line_snap_px for c in calls if c.line_snap_px is not None]
    call.line_snap_px = float(np.mean(snaps)) if snaps else None
    sigma = float(np.sqrt(np.sum([c.margin_sigma ** 2 for c in calls]))) / len(calls)
    call.margin_sigma = sigma
    # 라인을 직접 본(보정한) 화면이 하나도 없고 가려진 화면이 있으면 계산만으로 믿을 수 없다
    call.line_hidden = not snaps and any(c.line_hidden for c in calls)
    call.near_line = abs(call.margin) < max(min_near_m, k * sigma) or call.line_hidden
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


def view_aspect(H: np.ndarray, shape) -> float:
    """화면 가운데에 보이는 코트 지점에서, 코트 1m×1m 가 화면에서 얼마나 납작한가 (짧은 변/긴 변, 0~1).

    카메라가 같은 자리에서 팬·줌만 하면 거의 그대로다(줌은 크기만 바꾼다). 코트가 선 하나로 찌그러진 H 는 0 에 가깝다.
    화면 가운데가 코트 뒤(지평선 위)로 넘어가는 H 는 0."""
    h, w = shape[:2]
    try:
        p = np.linalg.solve(H, np.array([w / 2.0, h / 2.0, 1.0]))
    except np.linalg.LinAlgError:
        return 0.0
    if not np.all(np.isfinite(p)) or abs(p[2]) < 1e-12:
        return 0.0
    x, y = p[0] / p[2], p[1] / p[2]
    q = H @ np.array([x, y, 1.0])
    if q[2] * (H @ np.array([14.0, 7.5, 1.0]))[2] <= 0:  # 코트 가운데와 반대편(카메라 뒤)
        return 0.0
    u, v = q[0] / q[2], q[1] / q[2]
    J = np.array([[H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]], [H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]]]) / q[2]
    sv = np.linalg.svd(J, compute_uv=False)
    return float(sv[1] / sv[0]) if sv[0] > 0 else 0.0


def line_support(dist: np.ndarray, H: np.ndarray, spec: CourtSpec = FIBA, tol: float = 4.0,
                 min_pts: int = 10, good: float = 0.4, cross_deg: float = 20.0,
                 ignore: np.ndarray | None = None) -> tuple[int, int, int]:
    """(화면에 보이는 코트 라인 줄 수, 그중 실제 흰 선과 good 비율 이상 겹치는 줄 수, 맞은 줄의 방향이 엇갈리는가 0/1).

    전체 일치율 하나로는 틀린 정합(예: 키를 하프코트 근처 선에 겹침)도 0.5~0.65 가 나온다. 맞는 H 라면 보이는 줄
    대부분이 각자 선과 겹치고, 틀린 H 는 몇 줄만 우연히 겹친다.
    실제 중계에서 맞은 줄이 거의 평행한 먼쪽 가로선들뿐이면 일치율 0.8·줄 9/9 여도 틀렸다(선 방향으로 미끄러져도
    겹침은 그대로라 위치가 정해지지 않는다). 코트 전체가 흰 선 하나로 찌그러진 H 도 모든 줄이 같은 선 위에 겹친다.
    그래서 맞은 줄 중 화면에서 cross_deg 이상 방향이 다른 두 줄(또는 휘어 보이는 아크)이 있어야 엇갈림=1."""
    h, w = dist.shape
    n_vis = n_good = 0
    dirs = []
    curved = False
    for _, pts in _line_elements(spec):
        uv = project(H, pts)
        ok = _visible(uv, w, h, ignore, border=5)
        if ok.sum() < min_pts:
            continue
        n_vis += 1
        ui = np.round(uv[ok]).astype(int)
        if float(np.mean(dist[ui[:, 1], ui[:, 0]] <= tol)) >= good:
            n_good += 1
            q = uv[ok] - uv[ok].mean(axis=0)
            _, sv, vt = np.linalg.svd(q, full_matrices=False)
            if sv[0] < 20:  # 화면에서 20px 도 안 되는 줄은 방향을 못 믿는다
                continue
            if sv[1] > 0.12 * sv[0]:
                curved = True  # 휘어 보이는 아크는 혼자서도 두 방향을 잡는다
            dirs.append(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180.0)
    cross = curved
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            d = abs(dirs[i] - dirs[j])
            if min(d, 180.0 - d) >= cross_deg:
                cross = True
    return n_vis, n_good, int(cross)


def refine_homography(
    frame: np.ndarray,
    H: np.ndarray,
    spec: CourtSpec = FIBA,
    search_px: int = 12,
    iterations: int = 10,
    min_matches: int = 30,
    return_support: bool = False,
    ignore: np.ndarray | None = None,
):
    """현재 H 로 투영한 코트 라인을 실제 흰 라인 픽셀에 맞춰 다듬는다.

    투영된 라인 점마다 주변 search_px 안의 라인 픽셀 무게중심(=라인 중심)으로 끌어당긴 뒤
    H 를 다시 맞추는 과정을 반복한다 (mean-shift 방식).
    반환: (다듬은 H, 라인 일치율 0~1). 일치율 = 투영한 라인 점 중 실제 라인 위(±4px)에 있는 비율.
    return_support=True 면 (H, 일치율, line_support 결과).

    실제 중계에서 끌려온 점이 모두 흰 선 하나 위에 있으면 코트 전체가 그 선으로 쪼그라든 H 가 나와 일치율 0.9~1.0 으로
    통과한 적이 있다. 그래서 끌려온 점이 한 직선 위에만 있으면 그 단계를 버리고, 키 투영 넓이가 입력의 0.8~1.25배를
    벗어나는 결과는 받지 않는다 (줌은 화면 이동이 따라가므로 라인 정합이 크기를 크게 바꿀 일이 없다. 0.6 까지 허용했더니
    실제 영상에서 샘플마다 조금씩 줄어 결국 찌그러졌다 — 절대 기준은 CourtTracker.sane).
    """
    mask = line_mask(frame) > 0
    if ignore is not None:
        mask &= ~ignore  # 그래픽의 흰 글자·테두리는 코트 라인이 아니다
    mask = mask.astype(np.float32)
    h, w = mask.shape
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    def out(Hx, sc):
        return (Hx, sc, line_support(dist, Hx, spec, ignore=ignore)) if return_support else (Hx, sc)

    court_pts = np.concatenate([densify(p, 0.2) for p in court_polylines(spec).values()])
    if mask.sum() < min_matches:
        return out(H, 0.0)
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    A0 = key_area(H, spec)
    best_H, best_score = H, _line_score(dist, project(H, court_pts), w, h, ignore=ignore)
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
            ok = _visible(uv, w, h, ignore)
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
            if newH is None or not _area_ok(newH, A0, spec, 0.8, 1.25):
                break
            move = float(np.mean(np.linalg.norm(project(newH, src) - project(cur, src), axis=1)))
            cur = newH
            if move < 0.2:
                break
    score = _line_score(dist, project(cur, court_pts), w, h, ignore=ignore)
    if score >= best_score - 0.02 and _area_ok(cur, A0, spec, 0.8, 1.25):  # 일치율이 눈에 띄게 나빠지지 않았으면 채택
        best_H, best_score = cur, score
    return out(best_H, best_score)


def _visible(uv: np.ndarray, w: int, h: int, ignore: np.ndarray | None = None, border: int = 0) -> np.ndarray:
    """화면 안(가장자리 border px 제외)에 있고 ignore(방송 그래픽·편 화면의 검정 여백)에 안 걸린 점."""
    ok = (uv[:, 0] >= border) & (uv[:, 0] < w - 1 - border) & (uv[:, 1] >= border) & (uv[:, 1] < h - 1 - border)
    if ignore is not None and ok.any():
        ui = np.round(uv[ok]).astype(int)
        sub = ~ignore[ui[:, 1], ui[:, 0]]
        ok[np.flatnonzero(ok)] = sub
    return ok


def _line_score(dist: np.ndarray, uv: np.ndarray, w: int, h: int, tol: float = 4.0,
                ignore: np.ndarray | None = None) -> float:
    """투영한 코트 라인 점 중 실제 라인 위(±tol px)에 있는 비율. 그래픽에 가려 볼 수 없는 점(ignore)은 세지 않는다."""
    ok = _visible(uv, w, h, ignore)
    if ok.sum() == 0:
        return 0.0
    ui = np.round(uv[ok]).astype(int)
    return float(np.mean(dist[ui[:, 1], ui[:, 0]] <= tol))


def line_residual(frame: np.ndarray, H: np.ndarray, spec: CourtSpec = FIBA, cap: float = 6.0,
                  ignore: np.ndarray | None = None, mask: np.ndarray | None = None) -> tuple[float, int]:
    """투영한 코트 라인 점에서 주변(±cap px) 흰 라인 픽셀의 무게중심(=라인 가운데)까지 평균 거리(px, cap 에서 자름)와
    쓴 점 수. 일치율(±4px 안 비율)보다 세밀해서, 렌즈 휨처럼 몇 px 차이를 가를 때 쓴다. 주변에 라인이 없으면 cap."""
    m = (line_mask(frame) > 0) if mask is None else mask.astype(bool)
    if ignore is not None:
        m = m & ~ignore
    h, w = m.shape
    mf = m.astype(np.float32)
    r = int(math.ceil(cap))
    k = (2 * r + 1, 2 * r + 1)
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    cnt = cv2.boxFilter(mf, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sx = cv2.boxFilter(mf * xs, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    sy = cv2.boxFilter(mf * ys, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    pts = np.concatenate([densify(p, 0.1) for p in court_polylines(spec).values()])
    uv = project(H, pts)
    ok = _visible(uv, w, h, ignore, border=r + 1)
    if ok.sum() == 0:
        return cap, 0
    ui = np.round(uv[ok]).astype(int)
    c = cnt[ui[:, 1], ui[:, 0]]
    has = c >= 2
    d = np.full(len(ui), cap)
    cx = sx[ui[has, 1], ui[has, 0]] / c[has]
    cy = sy[ui[has, 1], ui[has, 0]] / c[has]
    d[has] = np.minimum(np.hypot(cx - uv[ok][has, 0], cy - uv[ok][has, 1]), cap)
    return float(np.mean(d)), int(ok.sum())


def graphics_mask(frames, still: int = 14, close_px: int = 31, grow_px: int = 6, max_frac: float = 0.35) -> np.ndarray | None:
    """방송 그래픽(스코어버그·로고·자막) 자리 = True. 라인 정합·라인 보정에서 이 자리의 흰 픽셀(숫자·테두리)을 코트 라인으로
    잘못 쓰지 않고, 그래픽에 가려 안 보이는 코트 라인 점을 '틀림'으로 세지 않게 한다.

    frames: 구간 전체에서 띄엄띄엄 뽑은 화면들 (카메라가 움직여야 코트와 그래픽이 갈린다).
    모든 화면에서 밝기가 거의 같고(still 안) 그 자리에 글자·테두리 같은 경계가 있는 곳을 씨앗으로, 상자 모양으로 메운다.
    거의 전부가 그대로면(카메라가 안 움직임) 판단할 수 없어 None."""
    gs = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames if f is not None]
    if len(gs) < 4:
        return None
    # 압축·센서 잡음은 살짝 흐려서 줄이고, 리플레이·화면 전환으로 그래픽이 한두 장 빠져도 되게 위아래 15% 는 빼고 범위를 본다
    g = np.stack([cv2.GaussianBlur(x, (5, 5), 0) for x in gs]).astype(np.int16)
    srt = np.sort(g, axis=0)
    lo_i = int(0.15 * (len(gs) - 1))
    hi_i = len(gs) - 1 - lo_i
    rng = (srt[hi_i] - srt[lo_i]) < still
    med = np.median(np.stack(gs), axis=0).astype(np.uint8)
    edges = cv2.Canny(med, 60, 150) > 0
    # 무늬가 있는 곳(경계)이 대부분 그대로면 카메라가 안 움직인 것 → 그래픽과 코트를 가를 수 없다.
    # (무늬 없는 검정 여백·단색 벽은 원래 안 변하므로 세지 않는다)
    tex = np.zeros_like(edges)
    for x in gs[:: max(1, len(gs) // 4)]:
        tex |= cv2.Canny(x, 60, 150) > 0
    if tex.sum() == 0 or (rng & tex).sum() / tex.sum() > max_frac:
        return None
    seed = (rng & edges).astype(np.uint8)
    m = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    # 테두리로 둘러싸인 그래픽 상자는 안쪽(단색이라 경계가 없음)까지 채운다. 화면 대부분을 감싸는 테두리는 채우지 않는다
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_max = 0.08 * m.shape[0] * m.shape[1]
    for c in cnts:
        if cv2.contourArea(c) <= area_max:
            cv2.drawContours(m, [c], -1, 1, thickness=cv2.FILLED)
    # 작은 점(우연히 같은 밝기였던 곳)은 버린다
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 300:
            keep[lab == i] = 1
    if grow_px > 0:
        keep = cv2.dilate(keep, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    return keep.astype(bool) if keep.any() else None


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
                 min_score: float = 0.65, use_static_mask: bool = True, overlay_mask: np.ndarray | None = None,
                 keyframe_score: float = 0.7, max_jump_px: float = 15.0, weak_inliers: int = 100,
                 min_good_lines: int = 2, min_line_frac: float = 0.6, min_area_frac: float = 0.1,
                 min_aspect_frac: float = 0.35, ignore: np.ndarray | None = None):
        """min_score: 라인 일치율이 이보다 낮으면 좌표를 내지 않는다(coast). 실제 중계에서 반대쪽 골대에 잘못 붙은 화면이
        0.36~0.47, 키가 옆으로 밀린 틀린 화면이 0.55~0.60, 맞은 화면은 0.71 이상이었으므로 0.65.
        keyframe_score: 이 이상일 때만 키프레임으로 저장(틀린 화면이 기준이 되지 않게).
        max_jump_px: 라인 정합이 화면 이동으로 예측한 위치에서 이보다 많이 끌고 가면 다른 선(다른 종목 선, 반대쪽 키)에
        붙은 것으로 보고 받지 않는다. weak_inliers: 화면 간 변환의 맞은 점이 이보다 적으면 빠른 패닝으로 보고
        샘플 사이 프레임들로 이어서 다시 구한다.
        min_good_lines, min_line_frac: 화면에 보이는 코트 라인 줄 가운데 실제 흰 선과 겹치는 줄이 min_good_lines 줄 이상이고
        min_line_frac 비율 이상이어야 좌표를 낸다. 실제 중계에서 일치율 0.5~0.65 화면의 절반이 1~4 m 틀렸는데, 틀린 화면은
        몇 줄만 우연히 겹친다. 맞은 줄 중 방향이 엇갈리는 줄도 있어야 한다(line_support 참고).
        min_area_frac, min_aspect_frac: 사람이 맞춘 캘리브레이션 화면들과 비교한 절대 기준 (sane). 키 투영 넓이가 그중 가장
        작은 것의 min_area_frac 배보다 작거나, 화면 가운데의 코트 납작한 정도(view_aspect)가 min_aspect_frac 배보다 작으면
        코트가 찌그러진 H 로 보고 쓰지 않는다. (실제 영상에서 좌표 1322개 중 1091개가 키 넓이 0~60px² 인 찌그러진 H 였다)
        ignore: 라인 정합에서 뺄 자리(방송 그래픽, 렌즈 휨을 편 화면의 검정 여백) = True (graphics_mask)."""
        self.spec = spec
        self.ignore = ignore
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
        self.last_support = (0, 0, 0)  # (보이는 줄, 맞은 줄, 방향 엇갈림)
        self.min_area_frac = min_area_frac
        self.min_aspect_frac = min_aspect_frac
        self._ref_geom: list[tuple[float, float]] = []  # 캘리브레이션 화면들의 (키 넓이, view_aspect)
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
        self.keyframes: list[tuple] = []  # (keypoints, descriptors, H, 저장한 샘플 번호)
        self.anchors: list[tuple] = []  # 사람이 맞춘 캘리브레이션 화면들 — 지워지지 않는 키프레임 (번호는 음수)
        self.last_found: list[tuple] = []  # 마지막 relocalize 에서 특징점으로 옮겨 온 후보들 (맞은 점, H, 앵커인가, 번호)
        self._court_pts = np.concatenate([densify(p, 0.5) for p in court_polylines(spec).values()])
        self.add_anchor(frame0, H0)

    def add_anchor(self, frame: np.ndarray, H: np.ndarray) -> None:
        """캘리브레이션한 화면(예: 왼쪽 골대, 오른쪽 골대)을 다시 찾기 기준으로 넣는다."""
        self._ref_geom.append((key_area(H, self.spec), view_aspect(H, frame.shape)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, self._mask(gray))
        if des is not None and len(kp) >= 50:
            self.anchors.append((kp, des, H.copy(), -1000 * (len(self.anchors) + 1)))

    def _mask(self, gray):
        if self.overlay_mask is not None:
            return self.overlay_mask
        return self.static.mask(gray.shape) if self.static is not None else None

    def _add_keyframe(self, gray, H, max_keep: int = 80) -> None:
        kp, des = self.orb.detectAndCompute(gray, self._mask(gray))
        if des is not None and len(kp) >= 50:
            self.keyframes.append((kp, des, H.copy(), self.n))
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
        Hp = F @ self.H
        Hp = Hp / Hp[2, 2]
        if not self.sane(Hp, frame.shape):
            # 화면 이동을 이어 붙인 결과가 찌그러졌다 → 놓친 것으로 보고 기준 화면으로 다시 찾게 한다
            self.lost = True
            return TrackResult(None, 0.0, "cut")
        self.H = Hp
        self.n += 1
        score = float("nan")
        if self.refine_every and self.n % self.refine_every == 0:
            Hr, score, sup = refine_homography(frame, Hp, self.spec, return_support=True, ignore=self.ignore)
            self.last_support = sup
            if not self.sane(Hr, frame.shape) or not self.supported(sup):
                # 라인 정합이 찌그러졌거나 보이는 라인과 안 맞는다 → 화면 이동으로 예측한 H 를 그대로 들고 간다
                self.coast_n += 1
                return TrackResult(None, score, "coast")
            sure = score >= self.sure_score
            if self._jump(Hp, Hr, frame.shape) > self.max_jump_px and not sure:
                # 화면 이동으로 예측한 자리에서 멀리 끌려갔는데 라인도 확실히 맞지 않는다 → 다른 선(반대쪽 키, 다른 종목 선)에
                # 붙었을 수 있다. 예측을 믿고 좌표는 내지 않는다. (라인이 거의 다 맞으면 흐린 패닝 뒤의 정상 보정으로 본다)
                self.coast_n += 1
                return TrackResult(None, score, "coast")
            self.H = Hr
            if score < self.min_score or (self._fast and not sure):
                # 빠른 패닝(흐림) 중에는 라인이 확실히 맞을 때만 좌표를 낸다
                self.coast_n += 1
                return TrackResult(None, score, "coast")
            if score >= self.keyframe_score and self.n % 2 == 0:
                self._add_keyframe(gray, self.H)
        self.coast_n = 0
        return TrackResult(self.H.copy(), score, "ok")

    def relocalize(self, frame: np.ndarray, min_inliers: int = 40, top_k: int = 3,
                   good_enough: float = 0.8) -> tuple[np.ndarray | None, float]:
        """놓친 뒤 현재 화면을 기준 화면(앵커)·키프레임들과 맞춰 H 를 다시 찾는다. (H, 라인 일치율) 또는 (None, 0).

        특징점이 가장 많이 맞은 후보 하나만 라인에 맞추면, 관중·선수에 많이 맞은 엉뚱한 키프레임이 뽑혀 놓친 상태가 길게
        이어졌다 (실제 영상 195~206s: 왼쪽 기준 화면에서 옮겨 와 맞추면 0.81~0.83 인데 계속 lost). 그래서 앵커마다 하나씩과
        키프레임 중 많이 맞은 top_k 개를 (서로 다른 자리만) 라인에 맞춰 보고, 라인이 가장 잘 맞고 support·sane 을 통과한 것을
        고른다. good_enough 이상이 나오면 더 보지 않는다.
        self.last_inliers 에 고른 후보의 특징점 수를 남긴다.
        """
        self.last_inliers = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, d2 = self.orb.detectAndCompute(gray, self._mask(gray))
        if d2 is None or len(kp2) < min_inliers:
            return None, 0.0
        # 앵커 전부 + 최근 키프레임 10개 + 그 이전은 띄엄띄엄 (놓친 동안 매 샘플 호출되므로 비용을 묶어 둔다)
        recent = list(reversed(self.keyframes))
        older = recent[10:]
        kfs = recent[:10] + older[:: max(1, len(older) // 14 + 1)]
        tagged = [(True, c) for c in self.anchors] + [(False, c) for c in kfs]
        found = []  # (inliers, H, 앵커인가)
        self.last_found = []
        for is_anchor, (kp1, d1, Hk, nk) in tagged:
            m = self.matcher.match(d1, d2)
            if len(m) < min_inliers:
                continue
            p1 = np.float32([kp1[x.queryIdx].pt for x in m])
            p2 = np.float32([kp2[x.trainIdx].pt for x in m])
            F, inl = cv2.findHomography(p1, p2, ROBUST, 3.0)
            if F is None or inl is None:
                continue
            n = int(inl.sum())
            Hc = F @ Hk
            Hc = Hc / Hc[2, 2]
            if n >= min_inliers and self.sane(Hc, frame.shape):
                found.append((n, Hc, is_anchor))
                self.last_found.append((n, Hc, is_anchor, nk))
        if not found:
            return None, 0.0
        found.sort(key=lambda r: -r[0])
        # 라인에 맞춰 볼 후보: 특징점이 많은 순으로 top_k 개 + 각 앵커에서 온 것 (화면에서 거의 같은 자리면 하나만)
        picks: list[tuple] = []
        for n, Hc, is_anchor in found:
            if len(picks) >= top_k and not is_anchor:
                continue
            if any(h_distance(Hc, P[1], frame.shape, self.spec) < 15.0 for P in picks):
                continue
            picks.append((n, Hc))
        best = None  # (받을 수 있는가, 일치율, H, sup, inliers)
        for n, Hc in picks:
            H, score, sup = refine_homography(frame, Hc, self.spec, return_support=True, ignore=self.ignore)
            # 기준 화면에서 옮겨 온 H 는 멀리서 시작하는 경우가 많다. 한 번에 크기를 0.8~1.25배까지만 바꾸므로
            # 일치율이 오르는 동안 몇 번 더 맞춘다 (실제 영상: 4번 반복해 0.86~0.88 로 잡힌 슛 장면들)
            for _ in range(3):
                H2, s2, sup2 = refine_homography(frame, H, self.spec, return_support=True, ignore=self.ignore)
                if s2 <= score + 0.005:
                    break
                H, score, sup = H2, s2, sup2
            okay = self.supported(sup) and self.sane(H, frame.shape)
            cand = (okay, score if okay else min(score, 0.0), H, sup, n)
            if best is None or cand[:2] > best[:2]:
                best = cand
            if okay and score >= good_enough:
                break
        okay, score, H, sup, n = best
        self.last_inliers = n
        self.last_support = sup
        # 다시 찾은 자리의 라인이 대부분 안 겹치거나 찌그러졌으면 일치율 0 이하 → 받는 쪽이 버린다
        return H, score

    def consensus(self, shape, min_inliers: int = 80, agree_px: float = 4.0, min_sep: int = 10,
                  min_votes: int = 2) -> np.ndarray | None:
        """마지막 relocalize 의 후보 중, 서로 다른 시각(min_sep 샘플 이상 떨어진)의 기준 화면 min_votes 개 이상이 특징점만으로
        옮겨 온 H 가 화면에서 agree_px 안으로 같으면 그 평균 H. 아니면 None.

        라인이 적게 보이는 화면(스코어버그 근처 윙·코너, 반코트 확대)은 라인 정합이 support 기준을 못 넘어 몇십 초씩 좌표를
        못 냈다 (실제 B-LAB: 놓침 31%, coast 28%). 중계 카메라는 한 자리에서 돌고 당기기만 하므로 기준 화면과 지금 화면 사이는
        관중석까지 통째로 화면 변환 하나다. 따로 맞춘 둘 이상이 같은 답을 내면 라인을 못 봐도 믿을 수 있다."""
        c = [f for f in self.last_found if f[0] >= min_inliers]
        if len(c) < min_votes:
            return None
        best = None
        for f in c:
            grp = [g for g in c if h_distance(g[1], f[1], shape, self.spec) <= agree_px]
            ns = sorted(g[3] for g in grp)
            votes = sum(1 for k, x in enumerate(ns) if k == 0 or x - ns[k - 1] >= min_sep)
            key = (votes, sum(g[0] for g in grp))
            if best is None or key > best[0]:
                best = (key, grp)
        (votes, _), grp = best
        if votes < min_votes:
            return None
        w = np.array([g[0] for g in grp], float)
        H = sum(wi * g[1] for wi, g in zip(w, grp)) / w.sum()
        H = H / H[2, 2]
        return H if self.sane(H, shape) else None

    def supported(self, sup: tuple) -> bool:
        """보이는 코트 라인 줄 대부분이 실제 선과 겹치고, 맞은 줄의 방향이 엇갈리는가 (line_support 결과)."""
        return support_ok(sup, self.min_good_lines, self.min_line_frac)

    def sane(self, H: np.ndarray | None, shape) -> bool:
        """캘리브레이션 화면들과 비교해 코트가 찌그러지지 않았는가 (절대 기준)."""
        if H is None or not np.all(np.isfinite(H)) or not self._ref_geom:
            return H is not None and bool(np.all(np.isfinite(H)))
        A = key_area(H, self.spec)
        refA = [a for a, _ in self._ref_geom]
        sgn = 1.0 if refA[0] >= 0 else -1.0
        if sgn * A < self.min_area_frac * min(abs(a) for a in refA):
            return False
        return view_aspect(H, shape) >= self.min_aspect_frac * min(r for _, r in self._ref_geom)

    def reset(self, H: np.ndarray, frame: np.ndarray) -> None:
        self.H = H.copy()
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lost = False
        self.coast_n = 0


def support_ok(sup: tuple, min_good_lines: int = 2, min_line_frac: float = 0.6) -> bool:
    """line_support 결과 (보이는 줄, 맞은 줄, 방향 엇갈림) 로 좌표를 믿을 수 있는가 (CourtTracker 와 같은 기준)."""
    n_vis, n_good = sup[0], sup[1]
    cross = sup[2] if len(sup) > 2 else 1
    return n_good >= min_good_lines and n_good >= min_line_frac * n_vis and bool(cross)


def pair_motion(g0: np.ndarray, g1: np.ndarray, between=(), orb=None, matcher=None, mask=None, weak: int = 100,
                min_ok: int = 40) -> np.ndarray | None:
    """회색 화면 g0 → g1 화면 변환(3x3). 바로 구한 변환의 맞은 점이 weak 보다 적으면(빠른 패닝·흐림) between(사이 프레임들)을
    차례로 이어 붙인다. 그것도 안 되면 맞은 점 min_ok 이상일 때만 바로 구한 것을 쓰고, 아니면 None(화면 전환일 수 있음)."""
    orb = orb or make_orb()
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    F, n = frame_to_frame(g0, g1, orb, matcher, mask=mask, return_inliers=True)
    if F is not None and n >= weak:
        return F / F[2, 2]
    if between:
        G, ga = np.eye(3), g0
        for b in list(between) + [g1]:
            Fi = frame_to_frame(ga, b, orb, matcher, mask=mask)
            if Fi is None:
                G = None
                break
            G = Fi @ G
            ga = b
        if G is not None:
            return G / G[2, 2]
    return F / F[2, 2] if (F is not None and n >= min_ok) else None


def h_distance(Ha: np.ndarray, Hb: np.ndarray, shape, spec: CourtSpec = FIBA) -> float:
    """두 H 로 투영한 코트 라인 점들이 화면 안에서 평균 몇 px 떨어지는지. 화면 안에 점이 없으면 inf."""
    h, w = shape[:2]
    pts = np.concatenate([densify(p, 0.5) for p in court_polylines(spec).values()])
    a, b = project(Ha, pts), project(Hb, pts)
    vis = (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
    return float(np.linalg.norm(a[vis] - b[vis], axis=1).mean()) if vis.sum() >= 10 else float("inf")


def bridge_gap(grays: list, H_start: np.ndarray, H_end: np.ndarray, spec: CourtSpec = FIBA, orb=None, matcher=None,
               mask=None, min_inliers: int = 100, agree_px: float = 3.0, ok=None) -> list:
    """좌표를 못 낸 짧은 구간 메우기. grays[0]·grays[-1] 은 앞뒤 정상 화면(H_start·H_end), 그 사이가 빈 샘플들.

    앞 정상 화면에서 화면 이동만으로 끝까지 따라간 H 가 뒤 정상 화면의 H 와 agree_px 안에서 만나고(이동 계산이 정확했다는
    뜻), 뒤에서 거꾸로 따라온 H 와도 각 샘플에서 agree_px 안에서 겹칠 때만 그 샘플을 채운다. 서로 따로 구한 두 길이 같은
    곳을 가리켜야 하므로 한쪽만 틀린 경우는 걸러진다. 반환: 사이 샘플마다 H 또는 None (len(grays)-2 개).
    ok: 추가 검사 함수(H, shape) → bool (예: CourtTracker.sane)."""
    n = len(grays)
    none = [None] * max(0, n - 2)
    if n < 3:
        return none
    orb = orb or make_orb()
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shape = grays[0].shape
    fwd = [H_start / H_start[2, 2]]
    for k in range(1, n):
        F, m = frame_to_frame(grays[k - 1], grays[k], orb, matcher, mask=mask, return_inliers=True)
        if F is None or m < min_inliers:
            return none
        Hk = F @ fwd[-1]
        fwd.append(Hk / Hk[2, 2])
    if h_distance(fwd[-1], H_end, shape, spec) > agree_px:
        return none  # 화면 이동을 이어 붙인 길이 뒤 정상 화면과 안 만난다 → 사이도 믿을 수 없다
    bwd = [None] * n
    bwd[-1] = H_end / H_end[2, 2]
    for k in range(n - 2, 0, -1):
        F, m = frame_to_frame(grays[k + 1], grays[k], orb, matcher, mask=mask, return_inliers=True)
        if F is None or m < min_inliers:
            return none
        Hk = F @ bwd[k + 1]
        bwd[k] = Hk / Hk[2, 2]
    out = []
    for k in range(1, n - 1):
        if h_distance(fwd[k], bwd[k], shape, spec) > agree_px:
            out.append(None)
            continue
        w = k / (n - 1)
        Hk = (1 - w) * fwd[k] + w * bwd[k]
        Hk = Hk / Hk[2, 2]
        out.append(Hk if ok is None or ok(Hk, shape) else None)
    return out


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
