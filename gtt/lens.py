"""렌즈 휨(방사 왜곡) 보정.

넓게 찍는 중계 카메라는 화면 가장자리로 갈수록 곧은 코트 라인이 휘어 보인다(배럴 왜곡). 코트→화면 변환(H)은
곧은 선을 곧게만 옮기므로, 휜 화면에 H 하나로 맞추면 가운데는 맞아도 가장자리(특히 화면 아래 가까운 코트)가
수십 px 어긋난다. 실제 중계에서 화면 아래 스코어버그 근처의 3점 라인이 계산보다 20px 안팎 어긋나 3점 2개가 2점으로
나왔다.

여기서는 휨을 k1 하나(화면 가운데 기준 방사 왜곡, division model)로 보고, 화면을 '휨을 편 화면(canvas)'으로 바꿔서
추적·라인 정합·판정을 모두 그 위에서 한다. 휨을 펴면 화면 가장자리가 바깥으로 늘어나므로 canvas 는 원본보다 pad 만큼 크다.
    원본 화면 점 p_d → 편 화면 점 p_u:   p_u = c + (p_d - c) / (1 + k1·r_d²) + pad,   r_d = |p_d - c| / (화면 대각선/2)
k1 < 0 이 배럴(가장자리가 안쪽으로 오그라듦). k1 = 0 이면 원본과 같다. (p_d = p_u(1+k1 r²) 꼴은 k1 이 크면 화면 구석에서
식이 접혀 풀리지 않아 division model 을 쓴다)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Lens:
    k1: float
    width: int
    height: int
    pad: int = -1  # -1: k1 에 맞게 자동
    _maps: tuple | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.pad < 0:
            self.pad = self.auto_pad()

    # ------------------------------------------------------------ 기본 값
    @property
    def center(self) -> np.ndarray:
        return np.array([(self.width - 1) / 2.0, (self.height - 1) / 2.0])

    @property
    def norm(self) -> float:
        return 0.5 * math.hypot(self.width, self.height)

    @property
    def canvas_size(self) -> tuple[int, int]:
        """(폭, 높이)"""
        return self.width + 2 * self.pad, self.height + 2 * self.pad

    @property
    def is_identity(self) -> bool:
        return abs(self.k1) < 1e-9 and self.pad == 0

    def auto_pad(self) -> int:
        """원본 화면 가장자리를 편 자리가 canvas 안에 들어오도록 하는 여백 (최대 화면 짧은 변의 30%)."""
        if self.k1 >= 0:
            return 0
        w, h = self.width, self.height
        edge = np.concatenate([
            np.stack([np.linspace(0, w - 1, 64), np.zeros(64)], 1), np.stack([np.linspace(0, w - 1, 64), np.full(64, h - 1.0)], 1),
            np.stack([np.zeros(64), np.linspace(0, h - 1, 64)], 1), np.stack([np.full(64, w - 1.0), np.linspace(0, h - 1, 64)], 1)])
        u = self._undistort_centered(edge)
        over = max(float(-u[:, 0].min()), float(u[:, 0].max() - (w - 1)), float(-u[:, 1].min()), float(u[:, 1].max() - (h - 1)), 0.0)
        return int(min(math.ceil(over) + 2, 0.3 * min(w, h)))

    # ------------------------------------------------------------ 점 변환
    def _undistort_centered(self, pd: np.ndarray) -> np.ndarray:
        c, n = self.center, self.norm
        d = np.asarray(pd, np.float64).reshape(-1, 2) - c
        r2 = (d ** 2).sum(1, keepdims=True) / n ** 2
        return c + d / (1.0 + self.k1 * r2)

    def undistort_pts(self, pd) -> np.ndarray:
        """원본 화면 점(들) → 편 화면(canvas) 점(들). (N, 2)"""
        return self._undistort_centered(pd) + self.pad

    def distort_pts(self, pu) -> np.ndarray:
        """편 화면(canvas) 점(들) → 원본 화면 점(들). (N, 2). 원본 화면에 대응하는 점이 없으면 NaN."""
        c, n = self.center, self.norm
        d = np.asarray(pu, np.float64).reshape(-1, 2) - self.pad - c
        if abs(self.k1) < 1e-12:
            return c + d
        ru = np.sqrt((d ** 2).sum(1, keepdims=True)) / n
        disc = 1.0 - 4.0 * self.k1 * ru ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            rd = (1.0 - np.sqrt(disc)) / (2.0 * self.k1 * ru)
            scale = np.where(ru > 1e-12, rd / ru, 1.0)
        scale = np.where(disc >= 0, scale, np.nan)
        return c + d * scale

    # ------------------------------------------------------------ 화면 변환
    def maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """cv2.remap 용 (map_x, map_y)와 canvas 중 원본 화면이 있는 곳(valid) 마스크."""
        if self._maps is None:
            cw, ch = self.canvas_size
            xs, ys = np.meshgrid(np.arange(cw, dtype=np.float64), np.arange(ch, dtype=np.float64))
            p = np.stack([xs.ravel(), ys.ravel()], 1)
            d = self.distort_pts(p)
            mx = d[:, 0].reshape(ch, cw).astype(np.float32)
            my = d[:, 1].reshape(ch, cw).astype(np.float32)
            valid = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx <= self.width - 1) & (my >= 0) & (my <= self.height - 1)
            mx[~valid] = -1
            my[~valid] = -1
            self._maps = (mx, my, valid)
        return self._maps

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """원본 화면 → 편 화면(canvas). 원본 밖은 검정."""
        if self.is_identity:
            return frame
        mx, my, _ = self.maps()
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def invalid_mask(self) -> np.ndarray | None:
        """canvas 중 원본 화면이 없는 곳(검정 여백) = True. 휨이 없으면 None."""
        if self.is_identity:
            return None
        return ~self.maps()[2]

    # ------------------------------------------------------------ 저장
    def as_dict(self) -> dict:
        return {"k1": float(self.k1), "width": int(self.width), "height": int(self.height), "pad": int(self.pad)}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Lens":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(d["k1"]), int(d["width"]), int(d["height"]), int(d.get("pad", -1)))

    @classmethod
    def identity(cls, width: int, height: int) -> "Lens":
        return cls(0.0, width, height, 0)


def convert_h(H_raw: np.ndarray, lens: Lens, spec=None) -> np.ndarray:
    """원본 화면에서 맞춘 H(코트 → 원본 화면)를 편 화면(canvas)용으로 옮긴다.

    원본 화면 안에 보이는 코트 점들을 H 로 원본 화면에 찍고, 그 자리를 편 화면으로 옮긴 뒤 다시 H 를 맞춘다.
    (휜 화면에 억지로 맞춘 H 라 정확하지 않다 → 편 화면에서 refine_homography 로 다시 맞춰야 한다)"""
    from .court import FIBA, court_polylines
    from .homography import densify, project

    spec = spec or FIBA
    if lens.is_identity:
        return H_raw.copy()
    pts = np.concatenate([densify(p, 0.25) for p in court_polylines(spec).values()])
    uv = project(H_raw, pts)
    m = (uv[:, 0] >= 0) & (uv[:, 0] < lens.width) & (uv[:, 1] >= 0) & (uv[:, 1] < lens.height)
    if m.sum() < 8:
        T = np.array([[1, 0, lens.pad], [0, 1, lens.pad], [0, 0, 1]], float)
        return T @ H_raw
    H, _ = cv2.findHomography(pts[m], lens.undistort_pts(uv[m]), 0)
    return H / H[2, 2]


def convert_h_between(H: np.ndarray, lens_from: "Lens | None", lens_to: "Lens | None", spec=None) -> np.ndarray:
    """lens_from 의 편 화면(None 이면 원본 화면)에서 맞춘 H 를 lens_to 의 편 화면(None 이면 원본)용으로 옮긴다.
    두 화면 사이는 호모그래피가 아니므로 근사다 (옮긴 뒤 라인에 다시 맞춰야 한다)."""
    from .court import FIBA, court_polylines
    from .homography import densify, project

    spec = spec or FIBA
    pts = np.concatenate([densify(p, 0.25) for p in court_polylines(spec).values()])
    uv = project(H, pts)
    raw = lens_from.distort_pts(uv) if lens_from is not None and not lens_from.is_identity else uv
    ref = lens_from or lens_to
    w, h = (ref.width, ref.height) if ref is not None else (np.inf, np.inf)
    m = np.all(np.isfinite(raw), axis=1) & (raw[:, 0] >= 0) & (raw[:, 0] < w) & (raw[:, 1] >= 0) & (raw[:, 1] < h)
    if m.sum() < 8:
        raise ValueError("원본 화면 안에 코트 점이 너무 적습니다")
    dst = lens_to.undistort_pts(raw[m]) if lens_to is not None and not lens_to.is_identity else raw[m]
    Hc, _ = cv2.findHomography(pts[m], dst, 0)
    return Hc / Hc[2, 2]


class LensVideo:
    """Video 와 같은 모양으로 쓰되 화면을 편 화면(canvas)으로 돌려준다."""

    def __init__(self, video, lens: Lens):
        self.v = video
        self.lens = lens

    def __getattr__(self, name):
        return getattr(self.v, name)

    def frame_at(self, t: float) -> np.ndarray:
        return self.lens.undistort(self.v.frame_at(t))

    def iterate(self, *args, **kw):
        for t, f in self.v.iterate(*args, **kw):
            yield t, self.lens.undistort(f)

    def iterate_between(self, *args, **kw):
        for t, f, between in self.v.iterate_between(*args, **kw):
            yield t, self.lens.undistort(f), [self.lens.undistort(b) for b in between]


def remap_mask(mask: np.ndarray | None, lens: Lens) -> np.ndarray | None:
    """원본 화면 기준 마스크(True/False)를 편 화면(canvas) 기준으로 옮긴다."""
    if mask is None or lens.is_identity:
        return mask
    mx, my, _ = lens.maps()
    out = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 0


def canvas_ignore(lens: Lens, graphics_raw: np.ndarray | None) -> np.ndarray | None:
    """편 화면에서 라인 정합에 쓰지 않을 곳: 원본 화면 밖(검정 여백) + 방송 그래픽."""
    inv = lens.invalid_mask()
    g = remap_mask(graphics_raw, lens)
    if inv is None:
        return g
    return inv if g is None else (inv | g)


def fit_on_canvas(frame_raw: np.ndarray, H_raw: np.ndarray, lens: Lens, spec=None, graphics_raw: np.ndarray | None = None,
                  rounds: int = 6, wide_below: float = 0.8):
    """원본 화면에서 맞춘 H 를 편 화면으로 옮겨 라인에 다시 맞춘다. (편 화면, H, 일치율, 라인까지 평균 거리 px, ignore)"""
    from .court import FIBA
    from .homography import line_residual, refine_best

    spec = spec or FIBA
    canvas = lens.undistort(frame_raw)
    ign = canvas_ignore(lens, graphics_raw)
    # 펴면서 가장자리 라인이 12px 넘게 옮겨지면 12px 창 정합이 멈추므로 넓은 창도 본다
    # (합성 MJPG 영상: 맞는 k1=-0.12 에서 일치율 0.63 에 멈춰 엉뚱한 -0.06 이 뽑혔다)
    H, sc, _ = refine_best(canvas, convert_h(H_raw, lens, spec), spec, ignore=ign, rounds=rounds, min_gain=0.002,
                           wide_below=wide_below)
    res, _ = line_residual(canvas, H, spec, ignore=ign)
    return canvas, H, sc, res, ign


def fit_k1(samples, spec=None, ks=None, graphics_raw: np.ndarray | None = None, fine: bool = True, log=None):
    """여러 화면 [(원본 화면, 그 화면의 H)] 에서 렌즈 휨 k1 을 찾는다.

    k1 마다 화면을 펴고 H 를 다시 맞춘 뒤, 코트 라인 점에서 실제 흰 라인까지 평균 거리(px)를 잰다. 휨을 맞게 펴면
    가장자리 라인까지 곧은 선으로 맞으므로 거리가 가장 작아진다. 반환: (가장 좋은 k1, [(k1, 평균 거리, 평균 일치율)])."""
    if not samples:
        raise ValueError("화면이 없습니다.")
    h, w = samples[0][0].shape[:2]
    ks = list(ks) if ks is not None else [round(x, 3) for x in np.arange(-0.30, 0.1001, 0.02)]
    table = {}

    def eval_k(k):
        if k in table:
            return table[k]
        lens = Lens(float(k), w, h)
        rs, ss = [], []
        for fr, H in samples:
            _, _, sc, res, _ = fit_on_canvas(fr, H, lens, spec, graphics_raw)
            rs.append(res)
            ss.append(sc)
        table[k] = (float(np.mean(rs)), float(np.mean(ss)))
        if log:
            log(f"k1={k:+.3f}  라인까지 평균 {table[k][0]:.3f}px  일치율 {table[k][1]:.3f}")
        return table[k]

    for k in ks:
        eval_k(k)
    best = min(table, key=lambda k: table[k][0])
    if fine:
        for k in np.arange(best - 0.015, best + 0.0151, 0.005):
            eval_k(round(float(k), 4))
        best = min(table, key=lambda k: table[k][0])
    return float(best), sorted((k, r, s) for k, (r, s) in table.items())
