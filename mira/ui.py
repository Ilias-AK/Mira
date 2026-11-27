"""Local web interface for Mira.

The server uses only Python's standard library and talks directly to
``MiraRuntime``. It never makes a network request outside localhost.
"""
from __future__ import annotations

import argparse
import json
import random
import secrets
import threading
import time
import webbrowser
from types import SimpleNamespace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).with_name("ui_static")
MODEL_NAME = "Mira 1.0"
MAX_REQUEST_BYTES = 64 * 1024
MAX_QUESTION_CHARACTERS = 500

MEMORY_LABELS = {
    "name": "Имя",
    "age": "Возраст",
    "origin": "Родом из",
    "lives": "Живёт в",
    "place": "Учится в городе",
    "studies": "Учёба",
    "works": "Работа",
    "color": "Любимый цвет",
    "likes": "Любит",
    "dislikes": "Не любит",
    "pets": "Питомцы",
    "definitions": "Слова",
    "memos": "Просили запомнить",
}
PET_KINDS = (
    ("кош", "кошка"), ("кот", "кот"), ("пс", "пёс"), ("пёс", "пёс"), ("пес", "пёс"), ("соб", "собака"),
    ("щен", "щенок"), ("хом", "хомяк"), ("поп", "попугай"), ("чер", "черепаха"), ("кры", "крыса"), ("рыб", "рыбка"),
)


def pet_kind(key: str) -> str:
    """UserFacts stores a pet under the first letters of its kind ('кош', 'соб', …)."""
    return next((name for prefix, name in PET_KINDS if str(key).startswith(prefix)), str(key))


def memory_items(facts: dict) -> list[dict]:
    """Turn ``UserFacts.facts`` into readable rows for the «Память» window."""
    items: list[dict] = []
    for key, value in facts.items():
        if key.startswith("_") or value in (None, "", [], {}):
            continue
        label = MEMORY_LABELS.get(key, key.replace("_", " ").capitalize())
        if key == "pets" and isinstance(value, dict):
            text = ", ".join(f"{pet_kind(kind)} {name}" for kind, name in value.items())
        elif key == "definitions" and isinstance(value, dict):
            for word, meaning in value.items():
                items.append({"label": f"Слово «{word}»", "value": str(meaning)})
            continue
        elif key == "memos" and isinstance(value, list):
            for memo in value:
                items.append({"label": label, "value": str(memo)})
            continue
        elif isinstance(value, (list, tuple, set)):
            text = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            text = ", ".join(f"{k}: {v}" for k, v in value.items())
        else:
            text = str(value)
        items.append({"label": label, "value": text})
    return items


def build_runtime(args: argparse.Namespace):
    from mira.runtime import MiraRuntime

    return MiraRuntime(
        args.persona,
        args.bank,
        args.fallbacks,
        args.index if args.retrieval else None,
        use_retrieval=args.retrieval,
        mode="chat" if args.chat else "hall",
        seed=args.seed if args.seed is not None else random.randrange(10**6),
        typo_rate=0.0 if args.no_typos else args.typo_rate,
        intent_path=None if args.no_intent else args.intent,
    )


class MiraUIApp:
    """Small synchronized adapter between HTTP requests and MiraRuntime."""

    def __init__(self, runtime, *, typing_delay: bool = False, log_misses: bool = True):
        self.runtime = runtime
        self.typing_delay = typing_delay
        self.log_misses = log_misses
        self.session_id = secrets.token_urlsafe(12)
        self.lock = threading.RLock()

    def status(self) -> dict:
        return {
            "ok": True,
            "model": MODEL_NAME,
            "mode": self.runtime.mode,
            "modeLabel": "Режим зала" if self.runtime.mode == "hall" else "Обычный чат",
            "intentStatus": self.runtime.intent_status,
            "offline": True,
            "maxAnswerCharacters": self.runtime.max_characters,
            "maxQuestionCharacters": MAX_QUESTION_CHARACTERS,
            "sessionId": self.session_id,
        }

    def answer(self, message: str) -> dict:
        message = " ".join(str(message).split())
        if not message:
            raise ValueError("Введите сообщение")
        if len(message) > MAX_QUESTION_CHARACTERS:
            raise ValueError(f"Сообщение должно быть короче {MAX_QUESTION_CHARACTERS + 1} символа")

        started = time.monotonic()
        with self.lock:
            memory_before = self._memory_snapshot()
            try:
                result = self.runtime.answer(message)
            except Exception as error:   # на презентации лучше живая отговорка, чем ошибка в окне
                self._log_error(message, error)
                result = SimpleNamespace(text="Ой, отвлеклась. Давайте следующий вопрос", route="error_fallback", seconds=0.0)
            memory_updated = self._memory_snapshot() != memory_before
            if result.route.startswith("fallback") and self.log_misses:
                self._log_miss(message, result.text, result.route)
            if self.typing_delay:
                target = min(35.0, 6.0 + len(result.text) / 5.0)
                time.sleep(max(0.0, target - result.seconds))

        return {
            "ok": True,
            "model": MODEL_NAME,
            "text": result.text,
            "route": result.route,
            "seconds": result.seconds,
            "totalSeconds": round(time.monotonic() - started, 3),
            "characters": len(result.text),
            "memoryUpdated": memory_updated,
        }

    def _log_error(self, message, error) -> None:
        try:
            import traceback
            log = ROOT / "runs" / "ui_errors.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{message}\n{traceback.format_exc()}\n")
        except OSError:
            pass

    def memory(self) -> dict:
        with self.lock:
            facts = self._facts()
            items = memory_items(facts) if facts else []
        return {"ok": True, "items": items}

    def _facts(self) -> dict | None:
        facts = getattr(getattr(self.runtime, "facts", None), "facts", None)
        return facts if isinstance(facts, dict) else None

    def _memory_snapshot(self) -> str:
        facts = self._facts()
        if not facts:
            return ""
        try:
            return json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return ""

    def reset(self) -> dict:
        with self.lock:
            self.runtime.reset()
        self.session_id = secrets.token_urlsafe(12)
        return {"ok": True, "sessionId": self.session_id}

    def close(self) -> None:
        with self.lock:
            self.runtime.close()

    @staticmethod
    def _log_miss(question: str, answer: str, route: str) -> None:
        try:
            path = ROOT / "runs" / "mira" / "misses.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {"question": question, "answer": answer, "route": route, "source": "ui"}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass


class MiraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, app: MiraUIApp):
        self.app = app
        super().__init__(address, MiraRequestHandler)

    def server_close(self) -> None:
        try:
            self.app.close()
        finally:
            super().server_close()


class MiraRequestHandler(BaseHTTPRequestHandler):
    server: MiraHTTPServer

    STATIC = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/app.css": ("app.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json; charset=utf-8"),
        "/favicon.ico": ("assets/favicon-32.png", "image/png"),
        "/assets/favicon.svg": ("assets/favicon.svg", "image/svg+xml"),
        "/assets/favicon-32.png": ("assets/favicon-32.png", "image/png"),
        "/assets/apple-touch-icon.png": ("assets/apple-touch-icon.png", "image/png"),
        "/assets/icon-192.png": ("assets/icon-192.png", "image/png"),
        "/assets/icon-512.png": ("assets/icon-512.png", "image/png"),
        "/assets/mira-mark.svg": ("assets/mira-mark.svg", "image/svg+xml"),
        "/assets/fonts/inter-cyrillic.woff2": ("assets/fonts/inter-cyrillic.woff2", "font/woff2"),
        "/assets/fonts/inter-latin.woff2": ("assets/fonts/inter-latin.woff2", "font/woff2"),
        "/assets/fonts/LICENSE-Inter-OFL.txt": ("assets/fonts/LICENSE-Inter-OFL.txt", "text/plain; charset=utf-8"),
    }

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._send_json(HTTPStatus.OK, self.server.app.status())
            return
        if path == "/api/memory":
            self._send_json(HTTPStatus.OK, self.server.app.memory())
            return
        item = self.STATIC.get(path)
        if item is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Страница не найдена"})
            return
        relative, content_type = item
        file_path = STATIC_ROOT / relative
        try:
            payload = file_path.read_bytes()
        except OSError:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "Файл интерфейса недоступен"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store" if path in ("/", "/index.html") else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/message":
                response = self.server.app.answer(payload.get("message", ""))
            elif path == "/api/reset":
                response = self.server.app.reset()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Метод не найден"})
                return
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        except Exception as error:  # UI should fail clearly while keeping the server alive
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"Mira не смогла ответить: {type(error).__name__}"},
            )
            return
        self._send_json(HTTPStatus.OK, response)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Некорректный размер запроса") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Некорректный размер запроса")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Ожидался JSON в UTF-8") from error
        if not isinstance(value, dict):
            raise ValueError("Ожидался JSON-объект")
        return value

    def _send_json(self, status: HTTPStatus, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        if urlsplit(self.path).path.startswith("/api/") and int(args[1]) >= 400:
            super().log_message(format, *args)


def create_server(host: str, port: int, app: MiraUIApp) -> MiraHTTPServer:
    return MiraHTTPServer((host, port), app)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    result.add_argument("--typing-delay", action="store_true", help="выдавать ответ с человеческой паузой")
    result.add_argument("--chat", action="store_true", help="обычный чат вместо режима зала")
    result.add_argument("--retrieval", action="store_true")
    result.add_argument("--no-intent", action="store_true")
    result.add_argument("--no-typos", action="store_true")
    result.add_argument("--no-log-misses", action="store_true")
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--typo-rate", type=float, default=0.04)
    result.add_argument("--index", default="artifacts/retrieval")
    result.add_argument("--intent", default="runs/mira_intent/intent.pt")
    result.add_argument("--persona", default=str(ROOT / "config" / "persona.json"))
    result.add_argument("--bank", default=str(ROOT / "config" / "bank.json"))
    result.add_argument("--fallbacks", default=str(ROOT / "config" / "fallbacks.json"))
    return result


def main() -> None:
    args = parser().parse_args()
    runtime = build_runtime(args)
    app = MiraUIApp(runtime, typing_delay=args.typing_delay, log_misses=not args.no_log_misses)
    server = None
    for port in range(args.port, args.port + 10):   # порт занят (вторая копия, другая программа) — берём следующий
        try:
            server = create_server(args.host, port, app)
            break
        except OSError:
            print(f"Порт {port} занят, пробую {port + 1}", flush=True)
    if server is None:
        raise SystemExit(f"Не удалось открыть порты {args.port}–{args.port + 9}: закройте другие окна Миры")
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Mira UI: {url}", flush=True)
    print(f"{MODEL_NAME} · {'обычный чат' if args.chat else 'режим зала'} · {runtime.intent_status}", flush=True)
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
