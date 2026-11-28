import numpy as np

from gtt.camera3d import Camera, camera_from_homography, ground_from_body


def _cam(f=1400.0, w=1280, h=720, C=(14.0, -12.0, 8.0), look=(8.0, 7.5, 0.0)):
    C, look = np.array(C, float), np.array(look, float)
    z = look - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return Camera(K=K, R=R, t=-R @ C, up=1.0)


def test_recover_camera_from_floor_homography():
    cam = _cam()
    H = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
    H = H / H[2, 2] * -3.7  # 배율·부호는 아무래도 된다
    got = camera_from_homography(H, (1280, 720))
    assert abs(got.K[0, 0] - 1400) < 1
    assert np.allclose(got.center, cam.center, atol=0.02)
    assert abs(got.height_m - 8.0) < 0.02
    X = np.array([[9.0, 3.0, 1.9], [2.0, 12.0, 0.0]])
    assert np.allclose(got.project(X), cam.project(X), atol=0.5)


def test_feet_from_head_when_feet_hidden():
    cam = _cam()
    H = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
    got = camera_from_homography(H, (1280, 720))
    foot = (8.6, 11.0)
    for part, z in (("head", 1.92), ("hip", 0.53 * 1.92)):
        u, v = cam.project([[*foot, z]])[0]
        r = ground_from_body(got, u, v, part=part, height=1.92, height_sd=0.06)
        assert abs(r["x"] - foot[0]) < 0.01 and abs(r["y"] - foot[1]) < 0.01
        assert 0 < r["sd_m"] < 0.5
    # 키를 틀리게 넣으면 그만큼 어긋나고, 그 크기가 sd_m 로 표현된다
    u, v = cam.project([[*foot, 1.92]])[0]
    r = ground_from_body(got, u, v, height=1.92 + 0.06, height_sd=0.06)
    assert abs(np.hypot(r["x"] - foot[0], r["y"] - foot[1]) - r["sd_m"]) < 0.02
