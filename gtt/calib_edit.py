"""노란 코트 선을 마우스로 끌어 캘리브레이션을 고치는 편집 창 (gtt calib-edit).

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
            pt_rows(p.X, p.target, w_pin)
        else:
            x, y, _ = Tc @ np.array([p.X[0], p.X[1], 1.0])
            ln = Ti_invT @ p.line()
            ln = ln / max(1e-12, np.hypot(ln[0], ln[1]))
            a, b, c = ln
            rows.append(w_pin * np.array([a * x, a * y, a, b * x, b * y, b, c * x, c * y, c]))
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


def _dof(pins: list[Pin]) -> int:
    return sum(2 if p.kind == "pt" else 1 for p in pins)


@dataclass
class Editor:
    """편집 상태 (창 없이도 쓸 수 있게 분리: 테스트·스크립트용)."""

    H0: np.ndarray
    shape: tuple[int, int]  # (높이, 폭)
    spec: CourtSpec = FIBA
    grab_px: float = 14.0
    pins: list[Pin] = field(default_factory=list)
    history: list[list[Pin]] = field(default_factory=list)
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

    # ------------------------------------------------------------ 잡기
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
            straight = self._straight_at(name, X)
            if straight and len(seg) >= 2:
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

    # ------------------------------------------------------------ 끌기
    def start(self, u: float, v: float) -> bool:
        p = self.pick(u, v)
        if p is None:
            return False
        self.history.append([Pin(q.kind, q.X.copy(), q.target.copy(), None if q.direction is None else q.direction.copy(), q.name)
                             for q in self.pins])
        # 같은 자리를 다시 잡으면 앞의 고정을 바꾼다
        self.pins = [q for q in self.pins if not (q.kind == p.kind and np.linalg.norm(q.X - p.X) < 0.5)]
        self.pins.append(p)
        self.active = p
        return True

    def drag(self, u: float, v: float) -> None:
        if self.active is None:
            return
        self.active.target = np.array([u, v], float)
        self._refit()

    def release(self) -> None:
        self.active = None

    def remove_near(self, u: float, v: float) -> bool:
        q = np.array([u, v], float)
        for k, p in enumerate(self.pins):
            if np.linalg.norm(HG.project(self.H, [tuple(p.X)])[0] - q) <= self.grab_px:
                self.history.append(list(self.pins))
                self.pins = self.pins[:k] + self.pins[k + 1:]
                self._refit()
                return True
        return False

    def undo(self) -> None:
        if self.history:
            self.pins = self.history.pop()
            self.active = None
            self._refit()

    def reset(self) -> None:
        self.history.append(list(self.pins))
        self.pins, self.active = [], None
        self.H = self.H0.copy()

    def _refit(self) -> None:
        if not self.pins:
            self.H = self.H0.copy()
            return
        soft = [] if _dof(self.pins) >= 8 else self.soft
        H = solve_h(self.pins, soft)
        if H is not None and HG.key_area(H, self.spec) * HG.key_area(self.H0, self.spec) > 0:
            self.H = H

    # ------------------------------------------------------------ 그리기
    def render(self, frame: np.ndarray, cursor: tuple[float, float] | None = None) -> np.ndarray:
        img = frame.copy()
        for pts in court_polylines(self.spec).values():
            uv = HG.project(self.H, HG.densify(pts, 0.1))
            ok = np.all(np.isfinite(uv), axis=1) & (np.abs(uv) < 1e5).all(axis=1)
            if ok.sum() >= 2:
                cv2.polylines(img, [np.round(uv[ok]).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
        for p in self.pins:
            q = HG.project(self.H, [tuple(p.X)])[0]
            c = (0, 0, 255)
            if p.kind == "pt":
                cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 4, c, -1, cv2.LINE_AA)
            else:
                n = np.array([-p.direction[1], p.direction[0]])
                a, b = q - 7 * n, q + 7 * n
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 2, cv2.LINE_AA)
        if cursor is not None and self.active is None:
            p = self.pick(*cursor)
            if p is not None:
                q = HG.project(self.H, [tuple(p.X)])[0]
                cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 6, (255, 255, 0), 1, cv2.LINE_AA)
        return img


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


HELP = "Drag a YELLOW line onto the white line | right-click: remove | Z undo | R reset | S save | Q quit"


def run(frame: np.ndarray, H0: np.ndarray, spec: CourtSpec, save, win_max_w: int = 1600, title: str = "calib-edit") -> bool:
    """편집 창. save(H) 로 저장. 저장했으면 True."""
    ed = Editor(H0, frame.shape[:2], spec)
    s = min(1.0, win_max_w / frame.shape[1])
    state = {"cursor": None, "down": False, "saved": False, "dirty": False, "quit_warn": False, "msg": ""}

    def on_mouse(ev, x, y, flags, _):
        u, v = x / s, y / s
        state["cursor"] = (u, v)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["down"] = ed.start(u, v)
        elif ev == cv2.EVENT_MOUSEMOVE and state["down"]:
            ed.drag(u, v)
            state["dirty"], state["quit_warn"] = True, False
        elif ev == cv2.EVENT_LBUTTONUP:
            state["down"] = False
            ed.release()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if ed.remove_near(u, v):
                state["dirty"] = True

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        img = ed.render(frame, state["cursor"])
        if state["cursor"] is not None:
            mg = magnifier(img, *state["cursor"])
            cu = state["cursor"][0]
            x0 = 10 if cu > frame.shape[1] / 2 else img.shape[1] - mg.shape[1] - 10
            img[10:10 + mg.shape[0], x0:x0 + mg.shape[1]] = mg
        view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        bar = np.zeros((48, view.shape[1], 3), np.uint8)
        cv2.putText(bar, HELP, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        status = f"fixed: {len(ed.pins)}  " + ("SAVED" if state["saved"] and not state["dirty"] else
                                               ("unsaved changes" if state["dirty"] else "")) + "  " + state["msg"]
        cv2.putText(bar, status, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, np.vstack([view, bar]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), ord("S")):
            save(ed.H)
            state.update(saved=True, dirty=False, quit_warn=False, msg="saved")
        elif key in (ord("z"), ord("Z")):
            ed.undo()
            state["dirty"] = True
        elif key in (ord("r"), ord("R")):
            ed.reset()
            state["dirty"] = True
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
