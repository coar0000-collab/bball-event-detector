import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from gtt import dashboard as DB


def _serve(tmp_path, monkeypatch, token=None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "analysis_x").mkdir(parents=True)
    d = tmp_path / "out" / "analysis_x"
    (d / "summary.json").write_text(json.dumps({"경기": {"홈팀": "레드", "홈 득점": 3}, "점수판 변화": 2,
                                                 "코트 점수 vs 점수판": {"일치": 2, "점수판 변화": 2}, "걸린 시간(초)": {"a": 1}}), encoding="utf-8")
    (d / "report.html").write_text("<p>ok</p>", encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), DB.make_handler(DB.Jobs(), token, "configs"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_state_and_files(tmp_path, monkeypatch):
    srv, base = _serve(tmp_path, monkeypatch)
    try:
        st = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert st["analyses"][0]["dir"] == "analysis_x" and st["analyses"][0]["agree"] == 2
        assert st["profile_ok"] is False
        assert urllib.request.urlopen(base + "/r/analysis_x/report.html").read() == b"<p>ok</p>"
        assert "대시보드" in urllib.request.urlopen(base + "/").read().decode()
        try:
            urllib.request.urlopen(base + "/r/..%2F..%2Fsecret/x")
            raise AssertionError
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()


def test_lan_token_required(tmp_path, monkeypatch):
    srv, base = _serve(tmp_path, monkeypatch, token="abc")
    try:
        try:
            urllib.request.urlopen(base + "/api/state")
            raise AssertionError
        except urllib.error.HTTPError as e:
            assert e.code == 403
        assert urllib.request.urlopen(base + "/api/state?k=abc").status == 200
    finally:
        srv.shutdown()


def test_upload_rejects_non_video(tmp_path, monkeypatch):
    srv, base = _serve(tmp_path, monkeypatch)
    try:
        req = urllib.request.Request(base + "/api/upload?name=a.exe", data=b"x", method="POST")
        try:
            urllib.request.urlopen(req)
            raise AssertionError
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()
