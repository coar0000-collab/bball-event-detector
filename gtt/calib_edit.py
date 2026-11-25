"""흰 선 위에 직접 선을 그어(또는 노란 선을 끌어) 캘리브레이션을 고치는 편집 창 (gtt calib-edit).

점 이름을 하나씩 찍는 calibrate 와 달리, 화면에 그려진 노란 코트 선 아무 곳이나 잡아 실제 흰 선 위로 끌면
코트 전체(호모그래피)가 다시 맞춰진다.
- 곧은 선(베이스라인·사이드라인·키·코너 직선·하프라인)의 중간을 잡으면 '이 선이 이 자리를 지나게' (선 방향으로는 자유)
- 선이 만나는 모서리(랜드마크)나 3점 아크를 잡으면 '이 점이 이 자리에' 오게 한다.
끌어 놓은 것(고정)이 적을 때는 처음 화면 위치를 약하게 붙잡아 두어 나머지가 크게 흔들리지 않게 하고,
고정이 충분히 쌓이면(자유도 8 이상) 고정만으로 맞춘다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import homography as HG
from .court import FIBA, CourtSpec, court_polylines, landmarks


@dataclass
class Pin:
    kind: str  # "pt" (점이 그 자리에) / "line" (점이 그 직선 위에)
    X: np.ndarray  # 코트 좌표 (m)
    target: np.ndarray  # 화면 좌표 (끈 마우스 자리)
    direction: np.ndarray | None = None  # kind == "line": 화면에서 선 방향 (단위 벡터)
    name: str = ""
    weight: float = 1.0

    def line(self) -> np.ndarray:
        """kind == "line": 화면 직선 (a, b, c), a²+b²=1, a·u + b·v + c = 0."""
        d = self.direction
        n = np.array([-d[1], d[0]])
        return np.array([n[0], n[1], -float(n @ self.target)])

    def residual(self, H: np.ndarray) -> float:
        p = HG.project(H, [tuple(self.X)])[0]
        if self.kind == "pt":
            return float(np.linalg.norm(p - self.target))
        a, b, c = self.line()
        return float(abs(a * p[0] + b * p[1] + c))


def _norm_T(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    s = np.sqrt(2) / max(1e-9, float(np.mean(np.linalg.norm(pts - c, axis=1))))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def solve_h(pins: list[Pin], soft: list[tuple[np.ndarray, np.ndarray]] = (), w_pin: float = 30.0,
            w_soft: float = 1.0) -> np.ndarray | None:
    """점·직선 조건(pins)과 약한 점 조건(soft: (코트점, 화면점))으로 H(코트→화면)를 가중 DLT 로 구한다."""
    court = [p.X for p in pins] + [s[0] for s in soft]
    img = [p.target for p in pins] + [s[1] for s in soft]
    if len(court) < 4:
        return None
    Tc, Ti = _norm_T(np.array(court, float)), _norm_T(np.array(img, float))
    Ti_invT = np.linalg.inv(Ti).T
    rows = []

    def pt_rows(X, u, w):
        x, y, _ = Tc @ np.array([X[0], X[1], 1.0])
        un, vn, _ = Ti @ np.array([u[0], u[1], 1.0])
        rows.append(w * np.array([x, y, 1, 0, 0, 0, -un * x, -un * y, -un]))
        rows.append(w * np.array([0, 0, 0, x, y, 1, -vn * x, -vn * y, -vn]))

    for p in pins:
        if p.kind == "pt":
            pt_rows(p.X, p.target, w_pin * p.weight)
        else:
            x, y, _ = Tc @ np.array([p.X[0], p.X[1], 1.0])
            ln = Ti_invT @ p.line()
            ln = ln / max(1e-12, np.hypot(ln[0], ln[1]))
            a, b, c = ln
            rows.append(w_pin * p.weight * np.array([a * x, a * y, a, b * x, b * y, b, c * x, c * y, c]))
    for X, u in soft:
        pt_rows(X, u, w_soft)
    A = np.array(rows)
    if A.shape[0] < 8:
        return None
    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tc
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def straight_lines(spec: CourtSpec = FIBA) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """코트의 곧은 선들: 이름 → (코트 끝점 A, 끝점 B)."""
    L, W, m = spec.length, spec.width, spec.width / 2
    kw, kl, c, ce = spec.key_width / 2, spec.key_length, spec.corner_offset, spec.corner_straight_end
    out = {"sideline_bot": ((0, 0), (L, 0)), "sideline_top": ((0, W), (L, W)), "halfcourt": ((L / 2, 0), (L / 2, W)),
           "baseline_L": ((0, 0), (0, W)), "baseline_R": ((L, 0), (L, W))}
    for sd in ("L", "R"):
        fx = (lambda x: x) if sd == "L" else (lambda x: L - x)
        out[f"{sd}_key_bot"] = ((fx(0), m - kw), (fx(kl), m - kw))
        out[f"{sd}_key_top"] = ((fx(0), m + kw), (fx(kl), m + kw))
        out[f"{sd}_ft_line"] = ((fx(kl), m - kw), (fx(kl), m + kw))
        out[f"{sd}_corner_bot"] = ((fx(0), m - c), (fx(ce), m - c))
        out[f"{sd}_corner_top"] = ((fx(0), m + c), (fx(ce), m + c))
    return {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in out.items()}


def arcs(spec: CourtSpec = FIBA) -> dict[str, np.ndarray]:
    """곡선: 3점 아크 (코너 직선 빼고) — 이름 → 코트 점들 (촘촘히)."""
    out = {}
    for sd in ("L", "R"):
        pts = court_polylines(spec, arc_steps=480)[f"{sd}_three"]
        out[f"{sd}_arc"] = np.array(pts[2:-2], float)
    return out


@dataclass
class Stroke:
    """사용자가 흰 선 위에 그은 선. kind "line": 곧은 선(두 끝점), "arc": 곡선(따라 그은 점들)."""
    kind: str
    pts: np.ndarray  # 화면 점들
    name: str = ""  # 맞춘 코트 선 이름


def _dof(pins: list[Pin]) -> int:
    return sum(2 if p.kind == "pt" else 1 for p in pins)


@dataclass
class Editor:
    """편집 상태 (창 없이도 쓸 수 있게 분리: 테스트·스크립트용).

    두 가지 방법을 함께 쓸 수 있다.
    - 그리기(add_stroke): 흰 선 위에 선을 그으면 가장 가까운 코트 선으로 이름을 붙이고, 그은 선들에 코트를 맞춘다.
    - 끌기(start/drag/release): 노란 선을 잡아 흰 선 위로 끈다."""

    H0: np.ndarray
    shape: tuple[int, int]  # (높이, 폭)
    spec: CourtSpec = FIBA
    grab_px: float = 14.0
    pins: list[Pin] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)
    history: list[tuple] = field(default_factory=list)
    active: Pin | None = None

    def __post_init__(self):
        self.H0 = np.asarray(self.H0, float) / self.H0[2, 2]
        self.H = self.H0.copy()
        h, w = self.shape
        lm = landmarks(self.spec)
        names = list(lm)
        uv = HG.project(self.H0, [lm[n] for n in names])
        inside = [(np.array(lm[n], float), uv[i]) for i, n in enumerate(names)
                  if -0.1 * w <= uv[i][0] <= 1.1 * w and -0.1 * h <= uv[i][1] <= 1.1 * h]
        if len(inside) < 4:
            inside = [(np.array(lm[n], float), uv[i]) for i, n in enumerate(names)]
        self.soft = inside  # 처음 화면 위치 (약하게 붙잡아 둠)
        self._lines = {k: HG.densify(v, 0.05) for k, v in court_polylines(self.spec).items()}
        self._lm = {n: np.array(p, float) for n, p in lm.items()}
        self._straight = straight_lines(self.spec)
        self._arcs = arcs(self.spec)
        self.candidates: list[str] = []  # 마지막으로 그은 선의 후보 이름 (Tab 으로 바꿈)

    # ------------------------------------------------------------ 되돌리기
    def _snapshot(self) -> tuple:
        pins = [Pin(q.kind, q.X.copy(), q.target.copy(), None if q.direction is None else q.direction.copy(), q.name, q.weight)
                for q in self.pins]
        return pins, [Stroke(t.kind, t.pts.copy(), t.name) for t in self.strokes], list(self.candidates)

    def undo(self) -> None:
        if self.history:
            self.pins, self.strokes, self.candidates = self.history.pop()
            self.active = None
            self._refit()

    def reset(self) -> None:
        self.history.append(self._snapshot())
        self.pins, self.strokes, self.active, self.candidates = [], [], None, []
        self.H = self.H0.copy()

    # ------------------------------------------------------------ 그리기
    def classify(self, pts) -> str:
        """그은 자취가 곧으면 "line", 휘었으면 "arc"."""
        pts = np.asarray(pts, float)
        e1, e2 = pts[0], pts[-1]
        L = float(np.linalg.norm(e2 - e1))
        if L < 1e-6:
            return "line"
        n = np.array([-(e2 - e1)[1], (e2 - e1)[0]]) / L
        dev = float(np.max(np.abs((pts - e1) @ n)))
        return "arc" if dev > max(8.0, 0.06 * L) else "line"

    def rank_lines(self, e1, e2, H: np.ndarray | None = None) -> list[tuple[float, str]]:
        """곧게 그은 선(e1→e2)과 가까운 코트 곧은 선 순위 [(점수, 이름)] (작을수록 가까움)."""
        H = self.H if H is None else H
        e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
        d = e2 - e1
        Ld = float(np.linalg.norm(d))
        if Ld < 1e-6:
            return []
        out = []
        for name, (A, B) in self._straight.items():
            a, b = HG.project(H, [tuple(A), tuple(B)])
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            t = b - a
            Lt = float(np.linalg.norm(t))
            if Lt < 1e-6:
                continue
            t = t / Lt
            n = np.array([-t[1], t[0]])
            dist = float(np.mean(np.abs([(e1 - a) @ n, (e2 - a) @ n, ((e1 + e2) / 2 - a) @ n])))
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(d @ t) / Ld)))))
            # 그은 선이 그 코트 선의 길이 밖으로 나간 만큼 (코너 직선 vs 사이드라인처럼 나란한 선 구분)
            s1, s2 = sorted([float((e1 - a) @ t), float((e2 - a) @ t)])
            outside = max(0.0, -s2) + max(0.0, s1 - Lt) + max(0.0, -s1) * 0.5 + max(0.0, s2 - Lt) * 0.5
            out.append((dist + 2.0 * ang + 0.2 * outside, name))
        out.sort()
        return out

    def rank_arcs(self, pts, H: np.ndarray | None = None) -> list[tuple[float, str]]:
        H = self.H if H is None else H
        pts = np.asarray(pts, float)
        out = []
        for name, X in self._arcs.items():
            uv = HG.project(H, X)
            ok = np.all(np.isfinite(uv), axis=1)
            if ok.sum() < 2:
                continue
            dd = np.min(np.linalg.norm(pts[:, None] - uv[ok][None], axis=2), axis=1)
            out.append((float(np.mean(dd)), name))
        out.sort()
        return out

    def add_stroke(self, pts, max_dist: float = 120.0) -> Stroke | None:
        """흰 선 위에 그은 자취를 더한다. 가장 가까운 코트 선 이름을 붙이고 다시 맞춘다. 너무 짧거나 맞는 선이 없으면 None."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) < 2 or float(np.linalg.norm(pts[-1] - pts[0])) < 15:
            return None
        kind = self.classify(pts)
        if kind == "line":
            ranked = self.rank_lines(pts[0], pts[-1])
            st = Stroke("line", np.array([pts[0], pts[-1]]))
        else:
            ranked = self.rank_arcs(pts)
            st = Stroke("arc", _resample(pts, 8.0))
        ranked = [r for r in ranked if r[0] <= max_dist]
        if not ranked:
            return None
        self.history.append(self._snapshot())
        st.name = ranked[0][1]
        self.candidates = [n for _, n in ranked[:6]]
        self.strokes.append(st)
        self._refit()
        return st

    def cycle_last(self) -> str | None:
        """마지막으로 그은 선의 이름을 다음 후보로 바꾼다 (자동으로 붙인 이름이 틀렸을 때)."""
        if not self.strokes or len(self.candidates) < 2:
            return None
        st = self.strokes[-1]
        k = self.candidates.index(st.name) if st.name in self.candidates else -1
        st.name = self.candidates[(k + 1) % len(self.candidates)]
        self._refit()
        return st.name

    def remove_stroke_near(self, u: float, v: float, tol: float = 12.0) -> bool:
        q = np.array([u, v], float)
        for k, st in enumerate(self.strokes):
            P = st.pts
            for a, b in zip(P[:-1], P[1:]):
                ab = b - a
                t = float(np.clip((q - a) @ ab / max(1e-9, ab @ ab), 0, 1))
                if np.linalg.norm(a + t * ab - q) <= tol:
                    self.history.append(self._snapshot())
                    self.strokes = self.strokes[:k] + self.strokes[k + 1:]
                    self._refit()
                    return True
        return False

    def _stroke_pins(self, H: np.ndarray) -> list[Pin]:
        out = []
        for st in self.strokes:
            if st.kind == "line" and st.name in self._straight:
                A, B = self._straight[st.name]
                e1, e2 = st.pts[0], st.pts[-1]
                d = (e2 - e1) / max(1e-9, float(np.linalg.norm(e2 - e1)))
                out.append(Pin("line", A, e1.copy(), d, st.name))
                out.append(Pin("line", B, e1.copy(), d, st.name))
            elif st.kind == "arc" and st.name in self._arcs:
                X = self._arcs[st.name]
                uv = HG.project(H, X)
                w = float(np.sqrt(3.0 / max(1, len(st.pts))))  # 아크 하나가 곧은 선 1.5개쯤의 무게
                for q in st.pts:
                    i = int(np.argmin(np.linalg.norm(uv - q, axis=1)))
                    # 촘촘한 아크 점 사이를 이어 가장 가까운 자리 (점 간격 만큼의 계단 오차를 없앰)
                    best = (float(np.linalg.norm(uv[i] - q)), X[i])
                    for j in (i - 1, i):
                        if 0 <= j < len(X) - 1:
                            ab = uv[j + 1] - uv[j]
                            t = float(np.clip((q - uv[j]) @ ab / max(1e-12, ab @ ab), 0.0, 1.0))
                            d = float(np.linalg.norm(uv[j] + t * ab - q))
                            if d < best[0]:
                                best = (d, X[j] + t * (X[j + 1] - X[j]))
                    out.append(Pin("pt", np.array(best[1], float), q.copy(), None, st.name, w))
        return out

    # ------------------------------------------------------------ 끌기
    def pick(self, u: float, v: float, H: np.ndarray | None = None) -> Pin | None:
        """화면 (u, v) 근처의 노란 선을 잡는다. 모서리(랜드마크)·아크는 점, 곧은 선은 직선 조건."""
        H = self.H if H is None else H
        q = np.array([u, v], float)
        best = None
        for name, X in self._lm.items():
            p = HG.project(H, [tuple(X)])[0]
            d = float(np.linalg.norm(p - q))
            if d <= self.grab_px and (best is None or d < best[0]):
                best = (d, Pin("pt", X.copy(), p.copy(), None, name))
        if best is not None:
            return best[1]
        for name, pts in self._lines.items():
            uv = HG.project(H, pts)
            dd = np.linalg.norm(uv - q, axis=1)
            i = int(np.argmin(dd))
            if dd[i] > self.grab_px or (best is not None and dd[i] >= best[0]):
                continue
            X = pts[i]
            j0, j1 = max(0, i - 4), min(len(pts) - 1, i + 4)
            seg = uv[j0:j1 + 1]
            if self._straight_at(name, X) and len(seg) >= 2:
                t = seg[-1] - seg[0]
                t = t / max(1e-9, np.linalg.norm(t))
                best = (float(dd[i]), Pin("line", X.copy(), uv[i].copy(), t, name))
            else:
                best = (float(dd[i]), Pin("pt", X.copy(), uv[i].copy(), None, name))
        return None if best is None else best[1]

    def _straight_at(self, name: str, X: np.ndarray) -> bool:
        if not name.endswith("_three"):
            return True
        m, c = self.spec.width / 2, self.spec.corner_offset
        return bool(abs(abs(X[1] - m) - c) < 0.02)  # 코너 직선 위

    def start(self, u: float, v: float) -> bool:
        p = self.pick(u, v)
        if p is None:
            return False
        self.history.append(self._snapshot())
        # 같은 자리를 다시 잡으면 앞의 고정을 바꾼다
        self.pins = [q for q in self.pins if not (q.kind == p.kind and np.linalg.norm(q.X - p.X) < 0.5)]
        self.pins.append(p)
        self.active = p
        return True

    def drag(self, u: float, v: float) -> None:
        if self.active is None:
            return
        self.active.target = np.array([u, v], float)
        self._refit(iters=6)  # 끄는 동안은 빠르게, 놓을 때 끝까지

    def release(self) -> None:
        if self.active is not None:
            self._refit()
        self.active = None

    def remove_near(self, u: float, v: float) -> bool:
        """오른쪽 클릭: 가까운 고정(끌기) 또는 그은 선을 뺀다."""
        q = np.array([u, v], float)
        for k, p in enumerate(self.pins):
            if np.linalg.norm(HG.project(self.H, [tuple(p.X)])[0] - q) <= self.grab_px:
                self.history.append(self._snapshot())
                self.pins = self.pins[:k] + self.pins[k + 1:]
                self._refit()
                return True
        return self.remove_stroke_near(u, v)

    # ------------------------------------------------------------ 맞추기
    def _refit(self, iters: int = 60) -> None:
        if not self.pins and not self.strokes:
            self.H = self.H0.copy()
            return
        n_dof = _dof(self.pins) + sum(2 if st.kind == "line" else 3 for st in self.strokes)
        soft = [] if n_dof >= 8 else self.soft
        H = self.H
        for _ in range(iters if any(st.kind == "arc" for st in self.strokes) else 1):
            H2 = solve_h(self.pins + self._stroke_pins(H), soft)
            if H2 is None or HG.key_area(H2, self.spec) * HG.key_area(self.H0, self.spec) <= 0:
                return
            done = HG.h_distance(H2, H, self.shape, self.spec) < 0.005
            H = H2
            if done:
                break
        self.H = H

    def residuals(self) -> list[tuple[str, float]]:
        """그은 선마다 노란 선까지 평균 거리(px)."""
        out = []
        for st in self.strokes:
            ps = [p for p in self._stroke_pins(self.H) if p.name == st.name]
            if st.kind == "line" and st.name in self._straight:
                A, B = self._straight[st.name]
                a, b = HG.project(self.H, [tuple(A), tuple(B)])
                t = (b - a) / max(1e-9, float(np.linalg.norm(b - a)))
                n = np.array([-t[1], t[0]])
                out.append((st.name, float(np.mean([abs((e - a) @ n) for e in st.pts]))))
            elif ps:
                out.append((st.name, float(np.mean([p.residual(self.H) for p in ps]))))
        return out

    # ------------------------------------------------------------ 그리기
    def render(self, frame: np.ndarray, cursor: tuple[float, float] | None = None, mode: str = "draw",
               drawing=None) -> np.ndarray:
        img = frame.copy()
        for pts in court_polylines(self.spec).values():
            uv = HG.project(self.H, HG.densify(pts, 0.1))
            ok = np.all(np.isfinite(uv), axis=1) & (np.abs(uv) < 1e5).all(axis=1)
            if ok.sum() >= 2:
                cv2.polylines(img, [np.round(uv[ok]).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
        for k, st in enumerate(self.strokes):
            P = np.round(st.pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [P], False, (255, 0, 255), 2, cv2.LINE_AA)
            mid = st.pts[len(st.pts) // 2]
            col = (255, 255, 255) if k == len(self.strokes) - 1 else (255, 0, 255)
            cv2.putText(img, st.name, (int(mid[0]) + 6, int(mid[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, st.name, (int(mid[0]) + 6, int(mid[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        for p in self.pins:
            q = HG.project(self.H, [tuple(p.X)])[0]
            c = (0, 0, 255)
            if p.kind == "pt":
                cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 4, c, -1, cv2.LINE_AA)
            else:
                n = np.array([-p.direction[1], p.direction[0]])
                a, b = q - 7 * n, q + 7 * n
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 2, cv2.LINE_AA)
        if drawing is not None and len(drawing) >= 2:
            P = np.asarray(drawing, float)
            if self.classify(P) == "line":
                cv2.line(img, (int(P[0][0]), int(P[0][1])), (int(P[-1][0]), int(P[-1][1])), (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.polylines(img, [np.round(P).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 2, cv2.LINE_AA)
        if mode == "move" and cursor is not None and self.active is None:
            p = self.pick(*cursor)
            if p is not None:
                q = HG.project(self.H, [tuple(p.X)])[0]
                cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 6, (255, 255, 0), 1, cv2.LINE_AA)
        return img


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    """자취를 step px 간격으로 다시 뽑는다."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step:
        return pts[[0, -1]]
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def magnifier(img: np.ndarray, cx: float, cy: float, half: int = 40, zoom: int = 4) -> np.ndarray:
    """(cx, cy) 둘레를 zoom 배 키운 정사각형 (가운데 십자선)."""
    h, w = img.shape[:2]
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x, y = int(round(cx)) + half, int(round(cy)) + half
    crop = pad[max(0, y - half):y + half, max(0, x - half):x + half]
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    m = big.shape[0] // 2
    cv2.line(big, (m - 12, m), (m + 12, m), (255, 0, 255), 1)
    cv2.line(big, (m, m - 12), (m, m + 12), (255, 0, 255), 1)
    cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (255, 255, 255), 1)
    return big


HELP1 = "DRAW: drag along a white line (straight or 3pt arc) | TAB: change last line name | right-click: delete"
HELP2 = "M: switch draw/move | Z undo | R reset | S save | Q quit"


def run(frame: np.ndarray, H0: np.ndarray, spec: CourtSpec, save, win_max_w: int = 1600, title: str = "calib-edit") -> bool:
    """편집 창. 기본은 그리기: 흰 선 위에 선을 그으면 그 선에 코트를 맞춘다. M 으로 끌기 모드. save(H) 로 저장."""
    ed = Editor(H0, frame.shape[:2], spec)
    s = min(1.0, win_max_w / frame.shape[1])
    state = {"cursor": None, "down": False, "saved": False, "dirty": False, "quit_warn": False, "msg": "",
             "mode": "draw", "path": None}

    def on_mouse(ev, x, y, flags, _):
        u, v = x / s, y / s
        state["cursor"] = (u, v)
        if ev == cv2.EVENT_LBUTTONDOWN:
            if state["mode"] == "draw":
                state["path"] = [(u, v)]
            else:
                state["down"] = ed.start(u, v)
        elif ev == cv2.EVENT_MOUSEMOVE:
            if state["mode"] == "draw" and state["path"] is not None:
                state["path"].append((u, v))
            elif state["down"]:
                ed.drag(u, v)
                state["dirty"], state["quit_warn"] = True, False
        elif ev == cv2.EVENT_LBUTTONUP:
            if state["mode"] == "draw" and state["path"] is not None:
                st = ed.add_stroke(state["path"])
                state["path"] = None
                if st is None:
                    state["msg"] = "too short or no court line near it"
                else:
                    state.update(dirty=True, quit_warn=False, msg=f"{st.name} (TAB to change)")
            state["down"] = False
            ed.release()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if ed.remove_near(u, v):
                state.update(dirty=True, msg="removed")

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        img = ed.render(frame, state["cursor"], state["mode"], state["path"])
        if state["cursor"] is not None:
            mg = magnifier(img, *state["cursor"])
            cu = state["cursor"][0]
            x0 = 10 if cu > frame.shape[1] / 2 else img.shape[1] - mg.shape[1] - 10
            img[10:10 + mg.shape[0], x0:x0 + mg.shape[1]] = mg
        view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        bar = np.zeros((70, view.shape[1], 3), np.uint8)
        cv2.putText(bar, HELP1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, HELP2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        res = ed.residuals()
        worst = f"  worst {max(r for _, r in res):.1f}px" if res else ""
        status = (f"[{state['mode'].upper()}] lines: {len(ed.strokes)} fixed: {len(ed.pins)}{worst}  "
                  + ("SAVED" if state["saved"] and not state["dirty"] else ("unsaved" if state["dirty"] else ""))
                  + "  " + state["msg"])
        cv2.putText(bar, status, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, np.vstack([view, bar]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), ord("S")):
            save(ed.H, [{"name": st.name, "kind": st.kind, "pts": np.round(st.pts, 1).tolist()} for st in ed.strokes])
            state.update(saved=True, dirty=False, quit_warn=False, msg="saved")
        elif key in (ord("z"), ord("Z")):
            ed.undo()
            state.update(dirty=True, msg="undo")
        elif key in (ord("r"), ord("R")):
            ed.reset()
            state.update(dirty=True, msg="reset")
        elif key in (ord("m"), ord("M")):
            state["mode"] = "move" if state["mode"] == "draw" else "draw"
            state["msg"] = "move: drag a yellow line" if state["mode"] == "move" else "draw: drag along a white line"
        elif key == 9:  # Tab
            n = ed.cycle_last()
            if n:
                state.update(dirty=True, msg=f"{n} (TAB to change)")
        elif key in (ord("q"), ord("Q"), 27):
            if state["dirty"] and not state["quit_warn"]:
                state.update(quit_warn=True, msg="NOT SAVED - press S to save, or Q again to quit without saving")
                continue
            break
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return state["saved"]
