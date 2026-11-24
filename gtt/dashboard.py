"""`gtt dashboard`: 브라우저에서 영상을 올리면 `gtt analyze` 를 돌리고 결과를 보여 주는 로컬 대시보드.

    gtt dashboard            # 이 컴퓨터에서만: http://127.0.0.1:8765
    gtt dashboard --lan      # 같은 와이파이의 폰·노트북에서도 (비밀 키가 든 주소를 출력)

표준 라이브러리만 쓴다 (설치할 것 없음). 영상은 videos/uploads/, 결과는 out/ 아래에 쌓이고 깃에는 올라가지 않는다.
"""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_OUT = Path("out")
UPLOADS = Path("videos") / "uploads"
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    ext = Path(name).suffix.lower()
    stem = re.sub(r"[^\w가-힣.-]+", "_", stem).strip("._")[:60] or "video"
    return stem + ext


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.items: dict[str, dict] = {}

    def start(self, label: str, args: list[str], out_dir: Path) -> str:
        jid = secrets.token_hex(4)
        job = {"id": jid, "label": label, "status": "running", "steps": [], "log": [], "out": out_dir.name,
               "started": time.time(), "ended": None}
        with self.lock:
            self.items[jid] = job
        threading.Thread(target=self._run, args=(job, args), daemon=True).start()
        return jid

    def _run(self, job: dict, args: list[str]) -> None:
        cmd = [sys.executable, "-u", "-m", "gtt", "analyze", *args]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                 env={**__import__("os").environ, "PYTHONUTF8": "1"})
            for line in p.stdout:
                line = line.rstrip()
                with self.lock:
                    job["log"] = (job["log"] + [line])[-40:]
                    m = re.match(r"\[(\d+)\] (.+?) \.\.\.", line)
                    if m:
                        job["steps"].append(m.group(2))
            rc = p.wait()
            job["status"] = "done" if rc == 0 else "failed"
        except Exception as e:  # noqa: BLE001 — 화면에 보여 주기
            job["log"].append(f"오류: {e}")
            job["status"] = "failed"
        job["ended"] = time.time()

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((dict(j) for j in self.items.values()), key=lambda j: -j["started"])


def list_analyses() -> list[dict]:
    out = []
    if not ROOT_OUT.exists():
        return out
    for d in ROOT_OUT.iterdir():
        sj = d / "summary.json"
        if not (d.is_dir() and sj.exists() and (d / "report.html").exists()):
            continue
        try:
            s = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        g = s.get("경기", {})
        c = s.get("코트 점수 vs 점수판") or {}
        src = s.get("점수 출처") or {}
        out.append({
            "dir": d.name, "mtime": sj.stat().st_mtime, "game": g, "demo": bool(s.get("데모")),
            "events": s.get("점수판 변화"), "agree": c.get("일치"), "agree_of": c.get("점수판 변화"), "no_coord": c.get("코트 좌표 없음") or 0,
            "track": (s.get("코트 추적") or {}).get("비율"), "sources": {k: v for k, v in src.items() if isinstance(v, int)},
            "notes": s.get("메모") or [], "xlsx": (d / "기록지_자동.xlsx").exists(),
            "seconds": sum((s.get("걸린 시간(초)") or {}).values()),
        })
    return sorted(out, key=lambda a: -a["mtime"])


def make_handler(jobs: Jobs, token: str | None, profile: str):
    page = (Path(__file__).with_name("dashboard.html")).read_text(encoding="utf-8")

    class H(BaseHTTPRequestHandler):
        server_version = "gtt-dashboard"

        def log_message(self, *a):  # 조용히
            pass

        def _authed(self, q: dict) -> bool:
            if token is None:
                return True
            got = (q.get("k") or [""])[0] or self.headers.get("X-Key", "")
            if secrets.compare_digest(got, token):
                return True
            cookie = self.headers.get("Cookie", "")
            return f"gttk={token}" in cookie

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if not self._authed(q):
                return self._send(403, "비밀 키가 필요합니다. 대시보드를 켠 창에 나온 주소로 들어오세요.".encode(), "text/plain; charset=utf-8")
            extra = {"Set-Cookie": f"gttk={token}; Path=/; SameSite=Strict"} if token else None
            if u.path == "/":
                return self._send(200, page.encode(), "text/html; charset=utf-8", extra)
            if u.path == "/api/state":
                return self._json({"jobs": jobs.list(), "analyses": list_analyses(), "profile": profile,
                                   "profile_ok": (Path(profile) / "scorebug.json").exists()})
            m = re.match(r"^/r/([^/]+)/(.+)$", u.path)
            if m:
                d = ROOT_OUT / urllib.parse.unquote(m.group(1))
                f = (d / urllib.parse.unquote(m.group(2))).resolve()
                if ROOT_OUT.resolve() not in f.parents or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                if f.suffix == ".html":
                    ctype = "text/html; charset=utf-8"
                hdr = {"Content-Disposition": "attachment; filename*=UTF-8''" + urllib.parse.quote(f.name)} if f.suffix == ".xlsx" else None
                return self._send(200, f.read_bytes(), ctype, hdr)
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if not self._authed(q):
                return self._json({"error": "비밀 키가 필요합니다"}, 403)
            g = lambda k: (q.get(k) or [""])[0].strip()
            if u.path == "/api/demo":
                out = ROOT_OUT / f"analysis_demo_{time.strftime('%m%d_%H%M%S')}"
                jid = jobs.start("데모 (합성 중계)", ["--demo", "--out", str(out)], out)
                return self._json({"job": jid})
            if u.path == "/api/upload":
                name = _safe_name(g("name"))
                if Path(name).suffix not in VIDEO_EXT:
                    return self._json({"error": f"영상 파일만 올릴 수 있어요 ({', '.join(sorted(VIDEO_EXT))})"}, 400)
                n = int(self.headers.get("Content-Length", "0"))
                UPLOADS.mkdir(parents=True, exist_ok=True)
                dst = UPLOADS / name
                with open(dst, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                out = ROOT_OUT / f"analysis_{Path(name).stem}_{time.strftime('%m%d_%H%M%S')}"
                args = ["--video", str(dst), "--profile", profile, "--out", str(out)]
                for k in ("home", "away", "game-id", "date"):
                    if g(k):
                        args += [f"--{k}", g(k)]
                shots = g("shots")
                if shots and Path(shots).exists():
                    args += ["--shots", shots]
                jid = jobs.start(name, args, out)
                return self._json({"job": jid})
            return self._json({"error": "not found"}, 404)

    return H


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def serve(port: int = 8765, lan: bool = False, profile: str = "configs", open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(9) if lan else None
    host = "0.0.0.0" if lan else "127.0.0.1"
    httpd = ThreadingHTTPServer((host, port), make_handler(Jobs(), token, profile))
    local = f"http://127.0.0.1:{port}/" + (f"?k={token}" if token else "")
    print(f"대시보드: {local}")
    if lan:
        print(f"폰·노트북(같은 와이파이): http://{_lan_ip()}:{port}/?k={token}")
        print("  이 주소를 아는 사람만 들어올 수 있습니다. 끝나면 Ctrl+C 로 끄세요.")
    print("끄기: Ctrl+C")
    if open_browser:
        import webbrowser
        webbrowser.open(local)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
