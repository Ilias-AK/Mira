from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mira.ui import MiraUIApp, create_server, memory_items


STATIC_DIR = Path(__file__).resolve().parents[1] / "mira" / "ui_static"


@dataclass
class FakeAnswer:
    text: str
    route: str = "bank:test"
    seconds: float = 0.123


class FakeFacts:
    def __init__(self):
        self.facts = {}


class FakeRuntime:
    mode = "hall"
    intent_status = "on (3 темы, порог 0.50)"
    max_characters = 300

    def __init__(self):
        self.questions = []
        self.resets = 0
        self.closed = False
        self.facts = FakeFacts()

    def answer(self, question):
        self.questions.append(question)
        if question.lower().startswith("меня зовут"):
            self.facts.facts["name"] = question.split()[-1]
        return FakeAnswer(f"Ответ на: {question}")

    def reset(self):
        self.resets += 1
        self.facts = FakeFacts()

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


def serve(runtime):
    server = create_server("127.0.0.1", 0, MiraUIApp(runtime, log_misses=False))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_ui_status_message_reset_and_static_page():
    runtime = FakeRuntime()
    server, thread, url = serve(runtime)
    try:
        status, page = call(url, "/")
        assert status == 200
        assert "Mira 1.0" in page
        assert "Чем займёмся?" in page
        assert "Демо-режим" not in page
        assert "mira-mark-cobalt.png" not in page

        status, payload = call(url, "/api/status")
        assert status == 200
        assert payload["offline"] is True
        assert payload["mode"] == "hall"
        assert payload["maxQuestionCharacters"] == 500

        status, payload = call(url, "/api/message", data={"message": "  Привет   Mira  ", "options": {"search": True, "think": True}})
        assert status == 200
        assert payload["text"] == "Ответ на: Привет Mira"
        assert payload["route"] == "bank:test"
        assert payload["memoryUpdated"] is False
        assert runtime.questions == ["Привет Mira"]

        old_session = server.app.session_id
        status, payload = call(url, "/api/reset", data={})
        assert status == 200
        assert payload["sessionId"] != old_session
        assert runtime.resets == 1
    finally:
        stop(server, thread)
    assert runtime.closed is True


def test_ui_reports_and_lists_memory():
    runtime = FakeRuntime()
    server, thread, url = serve(runtime)
    try:
        status, payload = call(url, "/api/memory")
        assert status == 200
        assert payload == {"ok": True, "items": []}

        status, payload = call(url, "/api/message", data={"message": "Меня зовут Аня"})
        assert payload["memoryUpdated"] is True
        status, payload = call(url, "/api/message", data={"message": "Как дела?"})
        assert payload["memoryUpdated"] is False

        status, payload = call(url, "/api/memory")
        assert payload["items"] == [{"label": "Имя", "value": "Аня"}]
    finally:
        stop(server, thread)


def test_memory_items_are_readable():
    items = memory_items({
        "name": "Аня",
        "pets": {"кош": "Мурка"},
        "likes": ["чай", "книги"],
        "memos": ["я учу Python"],
        "definitions": {"кринж": "неловко"},
        "_private": "x",
        "age": "",
    })
    assert {"label": "Имя", "value": "Аня"} in items
    assert {"label": "Питомцы", "value": "кошка Мурка"} in items
    assert {"label": "Любит", "value": "чай, книги"} in items
    assert {"label": "Просили запомнить", "value": "я учу Python"} in items
    assert {"label": "Слово «кринж»", "value": "неловко"} in items
    assert all(item["label"] != "Возраст" for item in items)


def test_ui_serves_design_assets_and_hides_old_ones():
    server, thread, url = serve(FakeRuntime())
    try:
        for name in ("inter-cyrillic.woff2", "inter-latin.woff2"):
            with urllib.request.urlopen(f"{url}/assets/fonts/{name}", timeout=4) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "font/woff2"
                assert response.read(4) == b"wOF2"
        with urllib.request.urlopen(url + "/assets/favicon.svg", timeout=4) as response:
            assert response.headers.get_content_type() == "image/svg+xml"
            assert b"prefers-color-scheme:dark" in response.read()
        for name in ("favicon-32.png", "apple-touch-icon.png", "icon-192.png", "icon-512.png"):
            with urllib.request.urlopen(f"{url}/assets/{name}", timeout=4) as response:
                assert response.read(8) == b"\x89PNG\r\n\x1a\n"
        with urllib.request.urlopen(url + "/manifest.webmanifest", timeout=4) as response:
            assert response.headers.get_content_type() == "application/manifest+json"
            manifest = json.loads(response.read().decode("utf-8"))
            assert {icon["src"] for icon in manifest["icons"]} >= {"/assets/icon-192.png", "/assets/icon-512.png"}

        for old in ("/assets/mira-mark-cobalt.png", "/assets/fonts/Rubik-Variable.ttf", "/assets/fonts/InterVariable.ttf"):
            try:
                urllib.request.urlopen(url + old, timeout=4)
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError(f"{old} should not be served")
    finally:
        stop(server, thread)


def test_ui_rejects_empty_and_long_messages():
    server, thread, url = serve(FakeRuntime())
    try:
        for value in ("  ", "x" * 501):
            try:
                call(url, "/api/message", data={"message": value})
            except urllib.error.HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("invalid message was accepted")
    finally:
        stop(server, thread)


def test_ui_has_every_control_from_the_design():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "sidebar", "sidebarClose", "sidebarOpen", "newChatButton", "historySearch", "projectsButton", "history",
        "rowMenu", "settingsButton", "settingsMenu", "memoryButton", "clearHistory", "modelButton", "modelMenu",
        "chatTitle", "shareButton", "moreButton", "moreMenu", "newChatTop", "notice", "scroller", "welcome",
        "greeting", "heroSlot", "messages", "toBottom", "dock", "archiveBar", "composer", "composerInput",
        "attachButton", "fileInput", "searchToggle", "thinkToggle", "charCount", "voiceButton", "voiceModeButton",
        "sendButton", "renameDialog", "memoryDialog",
    ):
        assert f'id="{element_id}"' in page, element_id

    # Кнопки из макета не скрываются, даже если бэкенд пока их не поддерживает
    assert "features: { attach: true, search: true, think: true, voice: true }" in script
    assert "Демо-режим" not in script
    for text in ("Поделиться", "Проекты", "Подумать глубже", "Память", "Переименовать", "Удалить чат"):
        assert text in page, text
    for text in ("Память обновлена", "Продолжить", "Озвучить", "Размышляла"):
        assert text in script, text
    for name in ("speechSynthesis", "SpeechRecognition", "tableBlock", "highlight(code, lang)",
                 "JSON.stringify({ message, options })", "api.memory()", "memoryUpdated"):
        assert name in script, name
    assert 'els.scroller.addEventListener("scroll", onScroll' in script
    assert 'els.toBottom.addEventListener("click"' in script


def test_ui_styles_follow_the_design_system():
    styles = (STATIC_DIR / "app.css").read_text(encoding="utf-8")

    assert 'url("/assets/fonts/inter-cyrillic.woff2") format("woff2")' in styles
    assert "Rubik" not in styles and "Golos" not in styles and "InterVariable" not in styles
    assert "--accent: #1f3bd6;" in styles and "--coral: #ff5a4e;" in styles
    assert ':root[data-theme="dark"]' in styles and "@media (prefers-color-scheme: dark)" in styles
    assert "position: fixed;" in styles
    assert "overscroll-behavior: none;" in styles
    assert "overscroll-behavior-y: contain;" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ".app.sidebar-collapsed" in styles

    # Все цвета — только токены: после блока токенов не должно быть ни одного HEX
    body = styles.split("/* ---------- База ---------- */", 1)[1]
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", body)
