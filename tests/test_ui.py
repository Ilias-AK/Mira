from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from mira.ui import MiraUIApp, create_server


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
        assert "С чего начнём?" in page

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
