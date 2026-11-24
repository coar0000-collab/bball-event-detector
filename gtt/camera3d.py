"""코트 라인(호모그래피)으로 카메라를 3D로 복원해 '가상 코트'를 만든다.

코트 규격을 알면 바닥 평면의 호모그래피 H(코트 m → 화면 px)에서 카메라 초점거리·자세·위치를 구할 수 있다
(정사각 픽셀, 주점 = 화면 가운데 가정). 그러면 바닥이 아닌 높이의 점도 다룰 수 있다.

쓰임: 수비수에 슈터의 발이 가려졌을 때, 보이는 머리(또는 골반)를 찍고 그 높이(키 ± 편차)를 주면
그 바로 아래 바닥 위치(= 발 위치)를 계산한다. 키 편차가 바닥 위치에 주는 흔들림도 함께 돌려줘
'라인 근처라 판정 불가'를 정직하게 낼 수 있다. 점프 중에는 머리가 키보다 높으므로 점프 직전 디딤 프레임을 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3, 코트 → 카메라
    t: np.ndarray  # 3
    up: float  # 코트 z 축에서 '위'의 부호 (+1 또는 -1)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def height_m(self) -> float:
        return float(self.center[2] * self.up)

    def project(self, X) -> np.ndarray:
        """코트 3D 점 (x, y, 높이 m) 들 → 화면 px."""
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, 2] *= self.up
        p = (self.K @ (self.R @ X.T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:3]

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        d = self.R.T @ np.linalg.solve(self.K, np.array([u, v, 1.0]))
        return self.center, d

    def point_at_height(self, u: float, v: float, h: float) -> tuple[float, float]:
        """화면 (u, v) 를 지나는 시선이 높이 h(m) 평면과 만나는 코트 (x, y)."""
        C, d = self.ray(u, v)
        z = self.up * h
        if abs(d[2]) < 1e-12:
            raise ValueError("시선이 바닥과 평행합니다")
        s = (z - C[2]) / d[2]
        X = C + s * d
        return float(X[0]), float(X[1])


def camera_from_homography(H: np.ndarray, image_size: tuple[int, int]) -> Camera:
    """H(코트 m → 화면 px)와 화면 크기 (w, h) → Camera."""
    w, h = image_size
    T = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], float)
    Hc = np.linalg.solve(T, H)  # 주점을 원점으로
    h1, h2 = Hc[:, 0], Hc[:, 1]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-15:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            cands.append(f2)
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-15:
        f2 = (h2[0] ** 2 + h2[1] ** 2 - h1[0] ** 2 - h1[1] ** 2) / den
        if f2 > 0:
            cands.append(f2)
    if not cands:
        raise ValueError("이 호모그래피로는 초점거리를 구할 수 없습니다 (원근이 거의 없는 화면)")
    f = float(np.sqrt(cands[0]))
    K0 = np.diag([f, f, 1.0])
    r1 = np.linalg.solve(K0, h1)
    r2 = np.linalg.solve(K0, h2)
    lam = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    r1, r2, tt = r1 * lam, r2 * lam, np.linalg.solve(K0, Hc[:, 2]) * lam
    if tt[2] < 0:  # 코트가 카메라 앞에 있도록
        r1, r2, tt = -r1, -r2, -tt
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    K = T @ K0
    C = -R.T @ tt
    return Camera(K=K, R=R, t=tt, up=1.0 if C[2] > 0 else -1.0)


def ground_from_body(cam: Camera, u: float, v: float, part: str = "head", height: float = 1.85,
                     height_sd: float = 0.08) -> dict:
    """보이는 신체 부위(머리/골반)의 화면 위치와 선수 키 → 바로 아래 바닥(발) 위치와 오차.

    part: head(정수리, 키와 같은 높이), hip(골반, 키의 0.53배). 반환: x, y, sd_m(키 ±1 편차로 바닥 위치가 움직이는 거리)
    """
    ratio = {"head": 1.0, "hip": 0.53}[part]
    x, y = cam.point_at_height(u, v, ratio * height)
    xa, ya = cam.point_at_height(u, v, ratio * (height + height_sd))
    xb, yb = cam.point_at_height(u, v, ratio * (height - height_sd))
    sd = 0.5 * float(np.hypot(xa - xb, ya - yb))
    return {"x": x, "y": y, "sd_m": sd, "part": part, "height": height}
