"""Explicit conversational memory: raw turns plus narrowly extracted facts.

No pretrained model, invented biography or semantic completion is used. Facts
retain their speaker and turn number. A later explicit correction supersedes an
earlier fact only within the same speaker/scope and key.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
import unicodedata

WORD = r"[^\W_]+(?:[-'][^\W_]+)*"
WORD_RE = re.compile(WORD, re.UNICODE)
QUOTES = '«»"“”„'


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).casefold().replace("ё", "е")


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(normalize(text)))


def extract_explicit_facts(text: str, *, role: str, source_turn: int, scope: str | None = None) -> list[dict]:
    """Extract only named definitions and literal first-person assertions.

    Preference extraction is deliberately conservative: hypothetical, quoted,
    interrogative and conditional sentences are not factual commitments.
    """
    facts = []
    default_scope = scope or role

    def add(kind, key, value, polarity=1, confidence=1.0, fact_scope=None, surface_key=None, aliases=None):
        fact = {"text": text, "kind": kind, "key": key, "value": value, "polarity": polarity, "role": role, "source_turn": source_turn, "scope": fact_scope or default_scope, "confidence": confidence, "surface_key": surface_key or key}
        if aliases:
            fact["aliases"] = aliases
        facts.append(fact)

    # The exact spelling is retained. The unquoted construction "назовём X-ом"
    # additionally admits the narrow instrumental -> stem lookup alias; quoted
    # coined names are literal and never have their suffix stripped.
    definition_patterns = (
        rf"(?:назов[её]м|будем\s+называть)\s+[«\"“]?({WORD})[»\"”]?\s*(?:[—–:=]\s*)?(.+?)(?=[.!?](?:\s|$)|$)",
        rf"(?:пусть\s+)?(?:слово\s+)?[«\"“]({WORD})[»\"”]\s+(?:означает|значит|[—–-]\s*это)\s+(.+?)(?=[.!?](?:\s|$)|$)",
        rf"(?:пусть\s+)?({WORD})\s+(?:означает|значит)\s+(.+?)(?=[.!?](?:\s|$)|$)",
    )
    seen_definitions = set()
    for pattern_number, pattern in enumerate(definition_patterns):
        for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE):
            name, meaning = match.group(1), match.group(2).strip(" ;:—–-")
            if normalize(name) in {"это", "что", "он", "она", "оно", "они", "я", "ты", "мы", "вы"}:
                continue
            if meaning and normalize(name) not in seen_definitions:
                seen_definitions.add(normalize(name))
                aliases = []
                preceding = text[match.start(1) - 1:match.start(1)]
                if pattern_number == 0 and preceding not in '«"“' and len(name) >= 5 and normalize(name).endswith(("ом", "ем")):
                    aliases.append(normalize(name[:-2]))
                add("definition", normalize(name), meaning, confidence=.98, fact_scope="conversation", surface_key=name, aliases=aliases)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        clean = sentence.strip()
        if not clean or "?" in clean or re.search(r"\b(?:если|бы|допустим|представь|предположим)\b", clean, re.IGNORECASE):
            continue
        # Quoted statements may describe somebody else; only bare first-person
        # statements are committed as the speaker's own facts.
        if clean.startswith(tuple(QUOTES)):
            continue
        name = re.search(rf"(?:^|[.!]\s*)меня\s+зовут\s+({WORD}(?:\s+(?!(?:и|но|а)\b){WORD})?)", clean, re.IGNORECASE)
        if name:
            value = re.split(r"\s+(?:и|но|а)\s+", name.group(1), flags=re.IGNORECASE)[0]
            add("identity", "name", value.rstrip(".!"))
        age = re.search(r"\bмне\s+(\d{1,3})\s+(?:лет|год|года)\b", clean, re.IGNORECASE)
        if age:
            add("identity", "age", int(age.group(1)))
        preference = re.search(r"(?:^|[.!]\s*|\bно\s+)(?:(?:да|нет|ну|конечно),?\s+)?(?:я\s+)?(никогда\s+не\s+|не\s+)?(пью|люблю|ем|предпочитаю|смотрю|слушаю)\s+([^,.!?;]+)", clean, re.IGNORECASE)
        if preference:
            negative, verb, raw_object = preference.groups()
            obj = re.split(r"\s+(?:но|а|зато|потому|так\s+как)\s+", raw_object, flags=re.IGNORECASE)[0].strip()
            # Conjoined objects or conditional qualification require deeper
            # parsing; retain the raw turn but avoid an unsafe normalized fact.
            if not obj or re.search(r"\b(?:если|кроме|только|иногда|обычно|сегодня)\b", obj, re.IGNORECASE):
                continue
            operation = {"пью": "drink", "люблю": "like", "ем": "eat", "предпочитаю": "prefer", "смотрю": "watch", "слушаю": "listen"}[verb.casefold()]
            add("preference", operation + ":" + normalize(obj), obj, polarity=-1 if negative else 1, confidence=.98)
    return facts


class ConversationMemory:
    def __init__(self, persona: dict | list[str] | None = None):
        if isinstance(persona, list):
            persona = {"facts": list(persona)}
        if persona is not None and not isinstance(persona, dict):
            raise TypeError("persona must be a dict, a list of explicit fact strings or None")
        self._persona = deepcopy(persona or {})
        self._history: list[dict] = []
        self._facts: list[dict] = []
        for fact_text in self._persona.get("facts", []):
            if not isinstance(fact_text, str):
                raise TypeError("persona.facts entries must be strings")
            extracted = extract_explicit_facts(fact_text, role="assistant", source_turn=-1, scope="assistant")
            if extracted:
                self._facts.extend(extracted)
            else:
                self._facts.append({"text": fact_text, "kind": "persona", "key": normalize(fact_text), "surface_key": fact_text, "value": fact_text, "polarity": 1, "role": "assistant", "source_turn": -1, "scope": "assistant", "confidence": 1.0})
        for key in ("name", "age"):
            if key in self._persona:
                self._facts.append({"text": str(self._persona[key]), "kind": "identity", "key": key, "surface_key": key, "value": self._persona[key], "polarity": 1, "role": "assistant", "source_turn": -1, "scope": "assistant", "confidence": 1.0})

    @property
    def history(self) -> list[dict]:
        return deepcopy(self._history)

    @property
    def persona(self) -> dict:
        return deepcopy(self._persona)

    @property
    def facts(self) -> list[dict]:
        return deepcopy(self._facts)

    def _add(self, role: str, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("A memory turn must be text")
        turn = {"role": role, "text": text, "turn_id": len(self._history)}
        self._history.append(turn)
        self._facts.extend(extract_explicit_facts(text, role=role, source_turn=turn["turn_id"]))
        return deepcopy(turn)

    def add_user(self, text: str) -> dict:
        return self._add("user", text)

    def add_assistant(self, text: str) -> dict:
        return self._add("assistant", text)

    def active_facts(self, *, before_turn: int | None = None, scope: str | None = None) -> list[dict]:
        """Latest literal fact per key/speaker, optionally as of a past turn."""
        latest = {}
        for fact in self._facts:
            if before_turn is not None and fact["source_turn"] >= before_turn:
                continue
            if scope is not None and fact["scope"] != scope:
                continue
            key = (fact["scope"], fact["kind"], fact["key"])
            latest[key] = fact
        return deepcopy(list(latest.values()))

    def retrieve(self, query: str, limit: int = 6, *, before_turn: int | None = None) -> list[dict]:
        if limit <= 0:
            return []
        query_tokens = _tokens(query)
        scored = []
        for fact in self.active_facts(before_turn=before_turn):
            fact_tokens = _tokens(fact["text"] + " " + str(fact["surface_key"]) + " " + str(fact["value"]) + " " + " ".join(fact.get("aliases", [])))
            overlap = len(query_tokens & fact_tokens)
            score = float(overlap)
            if fact["kind"] == "definition" and ({normalize(fact["surface_key"]), *fact.get("aliases", [])} & query_tokens):
                score += 6
            if fact["kind"] == "identity" and ((fact["key"] == "name" and query_tokens & {"зовут", "имя", "представься"}) or (fact["key"] == "age" and query_tokens & {"лет", "возраст"})):
                score += 3
            if score > 0:
                scored.append((score, fact["source_turn"], fact))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]["key"]))
        return [fact for _, _, fact in scored[:limit]]

    def snapshot(self) -> dict:
        return {"version": 1, "persona": self.persona, "history": self.history}

    @classmethod
    def from_snapshot(cls, snapshot: dict):
        if snapshot.get("version") != 1:
            raise ValueError("Unsupported memory snapshot version")
        memory = cls(snapshot.get("persona"))
        for turn in snapshot.get("history", []):
            if turn.get("role") not in ("user", "assistant"):
                raise ValueError("Invalid memory turn role")
            memory._add(turn["role"], turn["text"])
        return memory

