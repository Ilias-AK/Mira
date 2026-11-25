from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mira.ui import MiraUIApp, create_server


STATIC_DIR = Path(__file__).resolve().parents[1] / "mira" / "ui_static"


@dataclass
class FakeAnswer:
    text: str
    route: str = "bank:test"
    seconds: float = 0.123


class FakeRuntime:
    mode = "hall"
    intent_status = "on (3 темы, порог 0.50)"
    max_characters = 300

    def __init__(self):
        self.questions = []
        self.resets = 0
        self.closed = False

    def answer(self, question):
        self.questions.append(question)
        return FakeAnswer(f"Ответ на: {question}")

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def call(url, path, *, data=None):
    body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=4) as response:
        payload = response.read()
        if response.headers.get_content_type() == "application/json":
            return response.status, json.loads(payload.decode("utf-8"))
        return response.status, payload.decode("utf-8")


def test_ui_status_message_reset_and_static_page():
    runtime = FakeRuntime()
    server = create_server("127.0.0.1", 0, MiraUIApp(runtime, log_misses=False))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, page = call(url, "/")
        assert status == 200
        assert "Mira 1.0" in page
        assert "Ну что, поговорим?" in page
        assert "Локальная модель" not in page
        assert "mira-mark-cobalt.png" in page
        assert "Астана или Кокшетау?</button>" not in page

        assert (STATIC_DIR / "assets" / "mira-mark-cobalt.png").read_bytes().startswith(b"\x89PNG")
        assert (STATIC_DIR / "assets" / "fonts" / "Rubik-Variable.ttf").stat().st_size > 300_000
        assert (STATIC_DIR / "assets" / "fonts" / "GolosText-Variable.ttf").stat().st_size > 150_000
        with urllib.request.urlopen(url + "/assets/fonts/Rubik-Variable.ttf", timeout=4) as font_response:
            assert font_response.status == 200
            assert font_response.headers.get_content_type() == "font/ttf"
            assert font_response.read(4) == b"\x00\x01\x00\x00"
        with urllib.request.urlopen(url + "/assets/fonts/GolosText-Variable.ttf", timeout=4) as font_response:
            assert font_response.status == 200
            assert font_response.headers.get_content_type() == "font/ttf"

        status, payload = call(url, "/api/status")
        assert status == 200
        assert payload["offline"] is True
        assert payload["mode"] == "hall"

        status, payload = call(url, "/api/message", data={"message": "  Привет   Mira  "})
        assert status == 200
        assert payload["text"] == "Ответ на: Привет Mira"
        assert runtime.questions == ["Привет Mira"]

        old_session = server.app.session_id
        status, payload = call(url, "/api/reset", data={})
        assert status == 200
        assert payload["sessionId"] != old_session
        assert runtime.resets == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert runtime.closed is True


def test_ui_rejects_empty_and_long_messages():
    runtime = FakeRuntime()
    server = create_server("127.0.0.1", 0, MiraUIApp(runtime, log_misses=False))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for value in ("  ", "x" * 501):
            try:
                call(url, "/api/message", data={"message": value})
            except urllib.error.HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("invalid message was accepted")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_ui_keeps_page_fixed_and_scrolls_only_conversation():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="chatScroll"' in page
    assert 'id="scrollBottomButton"' in page
    assert 'id="addButton"' in page
    assert 'id="conversationTitle"' in page
    assert 'id="threadPanelButton"' in page
    assert 'id="sidebarCollapseButton"' in page
    assert "grid-template-columns: var(--rail-width) var(--sidebar-width) minmax(0, 1fr);" in styles
    assert ".app.sidebar-collapsed" in styles
    assert "cubic-bezier(.22, 1, .36, 1)" in styles
    assert "position: fixed;" in styles
    assert "overscroll-behavior: none;" in styles
    assert "overscroll-behavior-y: contain;" in styles
    assert "overflow-y: auto;" in styles
    assert 'chatScroll.addEventListener("scroll", updateScrollButton' in script
    assert 'scrollBottomButton.addEventListener("click"' in script
    assert 'threadPanelButton.addEventListener("click", toggleThreadPanel)' in script
