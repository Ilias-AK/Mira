"""Download and prepare reproducible local datasets without adding text labels.

Examples:
  python -m arena.data fetch --sources toloka oasst wikipedia --wiki-shards 1
  python -m arena.data prepare --include-toloka --wiki-shards 1
  python -m arena.data fetch --sources wikipedia --wiki-shards all

Downloaded README/LICENSE files are preserved separately. Source metadata never
becomes part of a dialogue utterance. All text selection is deterministic.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import html
import io
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests


TOLOKA_URL = "https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip"
TOLOKA_SHA256 = "50ac87b445bd53d0a90451b76991a5de32f80fb1c9e0efb6092e57a096f20a47"
OASST_BASE = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
OASST_FILE = "2023-04-12_oasst_ready.messages.jsonl.gz"
OASST_SHA256 = "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c"
WIKI_BASE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
WIKI_CONFIG = "20231101.ru"
SOURCES = ("toloka", "oasst", "wikipedia")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_file(
    url: str, destination: str | Path, expected_sha256: str | None = None,
    expected_size: int | None = None, retries: int = 5,
) -> dict[str, Any]:
    """Stream to a resumable .part file and atomically install a verified download."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial_meta = destination.with_name(destination.name + ".part.json")
    if destination.exists():
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if (expected_size is None or size == expected_size) and (
            expected_sha256 is None or digest == expected_sha256
        ):
            return {"url": url, "path": str(destination), "bytes": size, "sha256": digest}
        raise ValueError(f"Existing file fails verification: {destination}; keep or rename it before retrying")
    if partial.exists():
        try:
            previous = json.loads(partial_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("url") != url:
            partial.unlink()
    _atomic_json(partial_meta, {"url": url, "expected_sha256": expected_sha256, "expected_size": expected_size})
    failure: Exception | None = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                if response.status_code == 416 and expected_size and offset == expected_size:
                    break
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0  # Server ignored Range; replace partial, never append duplicate bytes.
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise ValueError(f"Invalid resume Content-Range for {destination.name}")
                    total = int(match.group(3)) if match.group(3) != "*" else expected_size
                else:
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else expected_size
                written = offset
                last_log = time.monotonic()
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_log > 30:
                            print(f"Downloading {destination.name}: {written:,}/{total or '?'} bytes", flush=True)
                            last_log = time.monotonic()
                if total is not None and written != total:
                    raise IOError(f"Incomplete download {destination.name}: {written} != {total}")
                break
        except (requests.RequestException, OSError) as exc:
            failure = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if 400 <= exc.response.status_code < 500 and exc.response.status_code not in (408, 429):
                    raise RuntimeError(f"HTTP {exc.response.status_code} downloading {destination.name}") from exc
            if attempt + 1 == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {destination.name}") from exc
            time.sleep(min(2 ** attempt, 8))
    else:
        raise RuntimeError(f"Unable to download {destination.name}") from failure
    size = partial.stat().st_size
    digest = sha256_file(partial)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Size mismatch for {destination.name}: {size} != {expected_size}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {destination.name}; partial preserved for inspection")
    os.replace(partial, destination)
    partial_meta.unlink(missing_ok=True)
    print(f"Ready {destination.name}: {size:,} bytes, SHA256 {digest[:12]}", flush=True)
    return {"url": url, "path": str(destination), "bytes": size, "sha256": digest}


def _shard_count(value: str, available: int) -> int:
    if value == "all":
        return available
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError("wiki-shards must be a positive count or 'all'") from exc
    if count < 1 or count > available:
        raise ValueError(f"wiki-shards must be between 1 and {available}, or 'all'")
    return count


def fetch(raw_dir: str | Path, sources: Iterable[str], wiki_shards: str = "1") -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"format_version": 1, "files": {}}

    def get(key: str, url: str, path: Path, digest: str | None = None, size: int | None = None) -> None:
        manifest["files"][key] = download_file(url, path, digest, size)
        _atomic_json(manifest_path, manifest)

    for source in sources:
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        folder = raw_dir / source
        if source == "toloka":
            archive_path = folder / "TlkPersonaChatRus.zip"
            get("toloka/archive", TOLOKA_URL, archive_path, TOLOKA_SHA256, 6139432)
            with zipfile.ZipFile(archive_path) as archive:
                readme_name = "TlkPersonaChatRus/readme_TlkPersonaChatRus.txt"
                readme_path = folder / "readme_TlkPersonaChatRus.txt"
                readme_path.write_bytes(archive.read(readme_name))
                manifest["files"]["toloka/readme"] = {
                    "path": str(readme_path), "archive_member": readme_name,
                    "bytes": readme_path.stat().st_size, "sha256": sha256_file(readme_path),
                }
                _atomic_json(manifest_path, manifest)
        elif source == "oasst":
            get("oasst/messages", OASST_BASE + OASST_FILE, folder / OASST_FILE, OASST_SHA256, 34196309)
            for name in ("README.md", "LICENSE"):
                get("oasst/" + name, OASST_BASE + name, folder / name)
        elif source == "wikipedia":
            api = f"https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/{WIKI_CONFIG}?limit=1000"
            response = requests.get(api, timeout=30)
            response.raise_for_status()
            shards = sorted((item for item in response.json() if item["path"].endswith(".parquet")), key=lambda x: x["path"])
            count = _shard_count(wiki_shards, len(shards))
            _atomic_json(folder / "shards.json", shards)
            for item in shards[:count]:
                name = Path(item["path"]).name
                get("wikipedia/" + name, WIKI_BASE + item["path"], folder / name,
                    item.get("lfs", {}).get("oid"), item["size"])
            get("wikipedia/README.md", WIKI_BASE + "README.md", folder / "README.md")
            # The dataset's original README contains its license information.
            # A supplementary live website page may be unavailable independently
            # of the dataset files and must not invalidate a verified download.
            try:
                get("wikipedia/legal.html", "https://dumps.wikimedia.org/legal.html", folder / "legal.html")
                manifest.get("unavailable_files", {}).pop("wikipedia/legal.html", None)
            except RuntimeError as exc:
                manifest.setdefault("unavailable_files", {})["wikipedia/legal.html"] = {
                    "url": "https://dumps.wikimedia.org/legal.html", "error": str(exc),
                }
                print(f"Supplementary legal.html unavailable; original dataset README retained: {exc}", flush=True)
    manifest["config"] = {"sources": list(sources), "wiki_shards": wiki_shards, "wiki_config": WIKI_CONFIG}
    _atomic_json(manifest_path, manifest)
    return manifest


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def normalized_fingerprint_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def split_group(group_id: str, seed: int = 20261120, val_fraction: float = .05, test_fraction: float = .05) -> str:
    if not 0 <= val_fraction + test_fraction < 1 or min(val_fraction, test_fraction) < 0:
        raise ValueError("Invalid split fractions")
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


class _TolokaHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.speaker: str | None = None
        self.parts: list[str] = []
        self.turns: list[tuple[str, str]] = []
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "span":
            match = re.fullmatch(r"participant_([12])", dict(attrs).get("class") or "")
            if match and self.speaker is None:
                self.speaker = match.group(1)
                self.parts = []
                self.depth = 1
            elif self.speaker is not None:
                self.depth += 1
        elif tag == "br" and self.speaker is not None:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self.speaker is not None:
            self.depth -= 1
            if self.depth == 0:
                text = clean_text(re.sub(r"^\s*(?:Участник|Пользователь)\s*[12]\s*:\s*", "", "".join(self.parts), flags=re.I))
                if text:
                    if self.turns and self.turns[-1][0] == self.speaker:
                        speaker, previous = self.turns[-1]
                        self.turns[-1] = (speaker, previous + " " + text)
                    else:
                        self.turns.append((self.speaker, text))
                self.speaker = None
                self.parts = []

    def handle_data(self, data: str) -> None:
        if self.speaker is not None:
            self.parts.append(data)


def parse_toloka_dialogue(markup: str) -> list[tuple[str, str]]:
    parser = _TolokaHTML()
    parser.feed(markup)
    parser.close()
    return parser.turns


def _persona(markup: str) -> list[str]:
    text = re.sub(r"<br\s*/?>|</(?:p|div)>", "\n", markup, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return [value for line in text.splitlines() if (value := clean_text(line))]


def _window(context: list[dict[str, str]], max_turns: int, max_chars: int) -> list[dict[str, str]]:
    result = context[-max_turns:] if max_turns else list(context)
    while len(result) > 1 and (sum(len(t["text"]) for t in result) > max_chars or result[0]["role"] != "user"):
        result = result[1:]
    return result


def iter_toloka(
    archive_path: str | Path, max_response_chars: int = 512, max_history_turns: int = 8,
    max_history_chars: int = 4096, counters: collections.Counter | None = None,
) -> Iterator[dict[str, Any]]:
    counters = counters if counters is not None else collections.Counter()
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("TlkPersonaChatRus/dialogues.tsv") as binary:
            rows = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8"), delimiter="\t")
            for row_number, row in enumerate(rows):
                turns = parse_toloka_dialogue(row["dialogue"])
                counters["toloka_input_dialogues"] += 1
                counters["toloka_input_turns"] += len(turns)
                canonical = "\n".join(f"{speaker}:{normalized_fingerprint_text(text)}" for speaker, text in turns)
                group_id = "toloka:" + hashlib.sha256(canonical.encode()).hexdigest()[:24]
                for index, (target_speaker, text) in enumerate(turns):
                    if index == 0 or not text or (max_response_chars and len(text) > max_response_chars):
                        continue
                    context = [{"role": "assistant" if speaker == target_speaker else "user", "text": t}
                               for speaker, t in turns[:index]]
                    context = _window(context, max_history_turns, max_history_chars)
                    if not context or context[-1]["role"] != "user":
                        continue
                    yield {
                        "id": f"toloka:{row_number}:{index}", "group_id": group_id,
                        "source": "toloka", "kind": "dialogue", "context": context,
                        "response": text, "persona": _persona(row[f"persona_{target_speaker}_profile"]),
                    }


def iter_oasst(
    gzip_path: str | Path, database_path: str | Path, max_response_chars: int = 512,
    max_history_turns: int = 8, max_history_chars: int = 4096,
    counters: collections.Counter | None = None,
) -> Iterator[dict[str, Any]]:
    """Disk-backed parent lookup; reject an example if any ancestor fails checks."""
    counters = counters if counters is not None else collections.Counter()
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE IF NOT EXISTS oasst_nodes (id TEXT PRIMARY KEY, parent TEXT, tree TEXT, role TEXT, text TEXT, allowed INTEGER)")
    connection.execute("DELETE FROM oasst_nodes")
    try:
        with gzip.open(gzip_path, "rt", encoding="utf-8") as handle:
            batch = []
            for line in handle:
                node = json.loads(line)
                counters["oasst_input_messages"] += 1
                allowed = (node.get("lang") == "ru" and node.get("synthetic") is False
                           and not node.get("model_name") and not node.get("deleted", False)
                           and node.get("review_result") is not False)
                if not allowed:
                    counters["oasst_filtered_input_messages"] += 1
                batch.append((node["message_id"], node.get("parent_id"), node.get("message_tree_id"),
                              node["role"], clean_text(node.get("text", "")), int(allowed)))
                if len(batch) >= 2000:
                    connection.executemany("INSERT OR REPLACE INTO oasst_nodes VALUES (?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT OR REPLACE INTO oasst_nodes VALUES (?,?,?,?,?,?)", batch)
        connection.commit()
        candidates = connection.execute("SELECT id,parent,tree,role,text,allowed FROM oasst_nodes WHERE role='assistant' AND allowed=1 ORDER BY id")
        for node in candidates:
            if not node[4] or (max_response_chars and len(node[4]) > max_response_chars):
                continue
            chain = [node]
            seen = {node[0]}
            parent = node[1]
            valid = True
            while parent:
                previous = connection.execute("SELECT id,parent,tree,role,text,allowed FROM oasst_nodes WHERE id=?", (parent,)).fetchone()
                if (previous is None or previous[0] in seen or not previous[5] or not previous[4]
                        or previous[3] == chain[-1][3] or (previous[2] and node[2] and previous[2] != node[2])):
                    valid = False
                    break
                seen.add(previous[0])
                chain.append(previous)
                parent = previous[1]
                if len(chain) > 1000:
                    valid = False
                    break
            if not valid or len(chain) < 2 or chain[-1][3] != "prompter":
                counters["oasst_rejected_parent_chains"] += 1
                continue
            context = [{"role": "user" if n[3] == "prompter" else "assistant", "text": n[4]}
                       for n in reversed(chain[1:])]
            context = _window(context, max_history_turns, max_history_chars)
            yield {"id": "oasst:" + node[0], "group_id": "oasst:" + (node[2] or chain[-1][0]),
                   "source": "oasst", "kind": "dialogue", "context": context, "response": node[4], "persona": []}
    finally:
        connection.close()


def chunk_text(text: str, max_chars: int = 2400, min_chars: int = 100) -> Iterator[str]:
    if max_chars < min_chars or min_chars < 1:
        raise ValueError("Require max_chars >= min_chars >= 1")
    pending = ""
    for paragraph in clean_text(text).split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            boundary = paragraph.rfind(" ", 0, max_chars + 1)
            boundary = boundary if boundary >= max_chars // 2 else max_chars
            if pending:
                if len(pending) >= min_chars:
                    yield pending
                pending = ""
            yield paragraph[:boundary].strip()
            paragraph = paragraph[boundary:].strip()
        if pending and len(pending) + len(paragraph) + 1 > max_chars:
            if len(pending) >= min_chars:
                yield pending
            pending = paragraph
        else:
            pending = (pending + "\n" + paragraph).strip()
    if len(pending) >= min_chars:
        yield pending


def iter_wikipedia(
    paths: Iterable[str | Path], max_chars: int = 100_000_000, chunk_chars: int = 2400,
    counters: collections.Counter | None = None,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    counters = counters if counters is not None else collections.Counter()
    total = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=["id", "text"]):
            for article in batch.to_pylist():
                counters["wikipedia_input_articles"] += 1
                text = article["text"] or ""
                letters = sum(c.isalpha() for c in text)
                cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
                if letters < 100 or cyrillic / max(letters, 1) < .5:
                    counters["wikipedia_filtered_articles"] += 1
                    continue
                group_id = f"wikipedia:{WIKI_CONFIG}:{article['id']}"
                for index, chunk in enumerate(chunk_text(text, max_chars=chunk_chars)):
                    if max_chars and total + len(chunk) > max_chars:
                        counters["wikipedia_character_limit_reached"] = 1
                        return
                    total += len(chunk)
                    yield {"id": group_id + f":{index}", "group_id": group_id, "source": "wikipedia",
                           "kind": "text", "text": chunk}


def record_fingerprint(record: dict[str, Any]) -> str:
    if record["kind"] == "text":
        value = "text\0" + normalized_fingerprint_text(record["text"])
    else:
        # A matching question + target cannot survive in two splits even if its
        # longer history differs. Do not deduplicate all short answers alone.
        value = "dialogue\0" + normalized_fingerprint_text(record["context"][-1]["text"])
        value += "\0" + normalized_fingerprint_text(record["response"])
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(
    raw_dir: str | Path = "data/raw", output_dir: str | Path = "data/processed",
    sources: Iterable[str] = SOURCES, include_toloka: bool = False,
    wiki_shards: str = "1", max_wiki_chars: int = 100_000_000,
    max_response_chars: int = 512, max_history_turns: int = 8,
    max_history_chars: int = 4096, wiki_chunk_chars: int = 2400,
    seed: int = 20261120, val_fraction: float = .05, test_fraction: float = .05,
) -> dict[str, Any]:
    from . import quality

    quality_sha256 = sha256_file(Path(quality.__file__))
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    sources = list(sources)
    if not sources or any(source not in SOURCES for source in sources):
        raise ValueError(f"sources must be a nonempty selection from {SOURCES}")
    if "toloka" in sources and not include_toloka:
        raise ValueError("Toloka README limits use to non-commercial purposes. Pass --include-toloka to explicitly select this source; its original README is preserved in data/raw/toloka.")
    if min(max_wiki_chars, max_response_chars, max_history_turns) < 0 or max_history_chars < 1:
        raise ValueError("Invalid preparation size limits")
    split_group("validate", seed, val_fraction, test_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    database = output_dir / f".prepare-{run_id}.sqlite"
    counters: collections.Counter = collections.Counter()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE seen (fingerprint TEXT PRIMARY KEY, split TEXT NOT NULL)")
    paths = {split: output_dir / f".{split}-{run_id}.jsonl" for split in ("train", "val", "test")}
    handles = {split: path.open("w", encoding="utf-8", newline="\n") for split, path in paths.items()}
    inputs: list[Path] = []
    generators: list[Iterator[dict[str, Any]]] = []
    started = time.monotonic()
    try:
        if "toloka" in sources:
            path = raw_dir / "toloka" / "TlkPersonaChatRus.zip"
            inputs.append(path)
            generators.append(iter_toloka(path, max_response_chars, max_history_turns, max_history_chars, counters))
        if "oasst" in sources:
            path = raw_dir / "oasst" / OASST_FILE
            inputs.append(path)
            generators.append(iter_oasst(path, database, max_response_chars, max_history_turns, max_history_chars, counters))
        if "wikipedia" in sources:
            available = sorted((raw_dir / "wikipedia").glob("*.parquet"))
            if not available:
                raise FileNotFoundError("No Wikipedia parquet found; run fetch first")
            selected = available[:_shard_count(wiki_shards, len(available))]
            inputs.extend(selected)
            generators.append(iter_wikipedia(selected, max_wiki_chars, wiki_chunk_chars, counters))
        for records in generators:
            for record in records:
                if record["kind"] == "dialogue":
                    issues = quality.response_issues(record["response"])
                    if issues:
                        counters["dialogue_targets_filtered"] += 1
                        counters[f"dialogue_targets_filtered_{record['source']}"] += 1
                        for code in set(issues):
                            counters[f"response_filter_{code}"] += 1
                        continue
                split = split_group(record["group_id"], seed, val_fraction, test_fraction)
                fingerprint = record_fingerprint(record)
                try:
                    connection.execute("INSERT INTO seen VALUES (?,?)", (fingerprint, split))
                except sqlite3.IntegrityError:
                    counters["duplicate_records_removed"] += 1
                    continue
                record["split"] = split
                handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counters["records"] += 1
                counters[f"records_{split}"] += 1
                counters[f"records_{record['source']}"] += 1
                counters[f"records_{record['source']}_{split}"] += 1
                counters[f"records_{record['kind']}"] += 1
                counters[f"characters_{record['kind']}"] += len(record.get("text", record.get("response", "")))
                if counters["records"] % 10000 == 0:
                    connection.commit()
                    print(f"Prepared {counters['records']:,} records ({record['source']})", flush=True)
            connection.commit()
        for handle in handles.values():
            handle.close()
        outputs = {}
        for split, path in paths.items():
            final = output_dir / f"{split}.jsonl"
            os.replace(path, final)
            outputs[split] = {"path": str(final), "bytes": final.stat().st_size,
                              "sha256": sha256_file(final), "records": counters[f"records_{split}"]}
        manifest = {
            "format_version": 1,
            "config": {"sources": sources, "include_toloka": include_toloka, "wiki_shards": wiki_shards,
                       "wiki_config": WIKI_CONFIG, "max_wiki_chars": max_wiki_chars,
                       "max_response_chars": max_response_chars, "max_history_turns": max_history_turns,
                       "max_history_chars": max_history_chars, "wiki_chunk_chars": wiki_chunk_chars,
                       "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
                       "response_filter_sha256": quality_sha256,
                       "duplicate_key": "normalized_last_context_and_response_or_text"},
            "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs],
            "outputs": outputs, "counts": dict(counters), "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        _atomic_json(output_dir / "manifest.json", manifest)
        return manifest
    finally:
        for generator in generators:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        connection.close()
        database.unlink(missing_ok=True)
        Path(str(database) + "-wal").unlink(missing_ok=True)
        Path(str(database) + "-shm").unlink(missing_ok=True)


def validate(output_dir: str | Path = "data/processed", verify_hashes: bool = True) -> dict[str, Any]:
    """Check whole-group separation, exact duplicates, schema, and file hashes."""
    from .quality import response_issues
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    database = output_dir / f".validate-{uuid.uuid4().hex[:12]}.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE groups (id TEXT PRIMARY KEY, split TEXT)")
    connection.execute("CREATE TABLE fingerprints (id TEXT PRIMARY KEY)")
    counts: collections.Counter = collections.Counter()
    allowed = {"id", "group_id", "source", "kind", "split", "context", "response", "persona", "text"}
    try:
        for split in ("train", "val", "test"):
            path = output_dir / f"{split}.jsonl"
            for line_number, line in enumerate(path.open("r", encoding="utf-8"), 1):
                record = json.loads(line)
                location = f"{path}:{line_number}"
                if set(record) - allowed or record.get("split") != split or not record.get("id") or not record.get("group_id"):
                    raise ValueError(f"Invalid record schema or split at {location}")
                if record.get("kind") == "dialogue":
                    context = record.get("context")
                    if not context or context[-1].get("role") != "user" or not record.get("response"):
                        raise ValueError(f"Invalid dialogue at {location}")
                    if any(set(t) != {"role", "text"} or t["role"] not in ("user", "assistant") or not t["text"] for t in context):
                        raise ValueError(f"Invalid context turn at {location}")
                    if any(a["role"] == b["role"] for a, b in zip(context, context[1:])):
                        raise ValueError(f"Nonalternating dialogue at {location}")
                    if record.get("source") == "toloka" and any(
                        re.match(r"^\s*(?:Участник|Пользователь)\s*[12]\s*:", text, flags=re.I)
                        for text in [record["response"]] + [turn["text"] for turn in context]
                    ):
                        raise ValueError(f"Unremoved source speaker prefix at {location}")
                    if response_issues(record["response"]):
                        raise ValueError(f"Dialogue target fails the shared response filter at {location}")
                elif record.get("kind") != "text" or not record.get("text"):
                    raise ValueError(f"Invalid text at {location}")
                existing = connection.execute("SELECT split FROM groups WHERE id=?", (record["group_id"],)).fetchone()
                if existing and existing[0] != split:
                    raise ValueError(f"Group crosses splits at {location}")
                connection.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (record["group_id"], split))
                try:
                    connection.execute("INSERT INTO fingerprints VALUES (?)", (record_fingerprint(record),))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate record at {location}") from exc
                counts[split] += 1
                if counts[split] % 10000 == 0:
                    connection.commit()
                if sum(counts.values()) % 50000 == 0:
                    print(f"Validated {sum(counts.values()):,} records", flush=True)
            if counts[split] != manifest["outputs"][split]["records"]:
                raise ValueError(f"Manifest record count mismatch for {split}")
            if verify_hashes and sha256_file(path) != manifest["outputs"][split]["sha256"]:
                raise ValueError(f"Manifest checksum mismatch for {split}")
        return {"records": dict(counts), "groups": connection.execute("SELECT count(*) FROM groups").fetchone()[0], "valid": True}
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    from arena.console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch")
    prepare_parser = sub.add_parser("prepare")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--output-dir", default="data/processed")
    for item in (fetch_parser, prepare_parser):
        item.add_argument("--raw-dir", default="data/raw")
        item.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
        item.add_argument("--wiki-shards", default="1", help="Number of shards, or all")
    prepare_parser.add_argument("--output-dir", default="data/processed")
    prepare_parser.add_argument("--include-toloka", action="store_true", help="Explicitly select Toloka, subject to its original README")
    prepare_parser.add_argument("--max-wiki-chars", type=int, default=100_000_000, help="0 processes all selected articles")
    prepare_parser.add_argument("--max-response-chars", type=int, default=512, help="0 disables response length selection")
    prepare_parser.add_argument("--max-history-turns", type=int, default=8)
    prepare_parser.add_argument("--max-history-chars", type=int, default=4096)
    prepare_parser.add_argument("--wiki-chunk-chars", type=int, default=2400)
    prepare_parser.add_argument("--seed", type=int, default=20261120)
    prepare_parser.add_argument("--val-fraction", type=float, default=.05)
    prepare_parser.add_argument("--test-fraction", type=float, default=.05)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "fetch":
        result = fetch(**args)
    elif command == "prepare":
        result = prepare(**args)
    else:
        result = validate(**args)
    print(json.dumps({"command": command, **result} if command == "validate" else {
        "command": command, "files": len(result.get("files", {})), "counts": result.get("counts", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
