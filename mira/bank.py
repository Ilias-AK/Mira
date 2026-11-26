"""Team-written reply bank: regex topics -> several human variants."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


GENDER_SLOT = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")


def fill_gender(text: str, gender: str | None) -> str:
    """{male|female} -> form for the user's gender; unknown defaults to male."""
    return GENDER_SLOT.sub(lambda m: m.group(2) if gender == "f" else m.group(1), text)


FRAME_WORDS = set("""как тебе ты тебя твой твоя твое твои что скажешь о об теме относишься к если коротко это или нет быстрый
выбор первая мысль представь будешь делать бы сделала отреагировала согласилась на такое насчет такого идея а и ну
слушай кстати вообще когда-нибудь у есть ли мне меня я вы вас вам ваш ваша скажи подскажи можно ли хочу спросить
интересно вот же да не в во с со по за из для про от до какой какая какое какие""".split())


def grams(text: str) -> Counter:
    """Character trigrams of the content words (question frame words left out)."""
    from mira.hall import to_ty
    out = Counter()
    words = re.findall(r"[а-яёa-z0-9]+", to_ty(text))
    content = [w for w in words if w not in FRAME_WORDS] or words
    for w in content:
        w = f"^{w}$"
        out.update(w[i:i + 3] for i in range(max(1, len(w) - 2)))
    return out


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


class Bank:
    def __init__(self, path, extra_paths=None):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        items = list(data.get("intents", []))
        if extra_paths is None:
            # team topics, then generated ones (config/team_qa_gen.json; delete the file to drop them)
            extra_paths = [Path(path).with_name("team_qa.json"), Path(path).with_name("team_qa_gen.json")]
        for extra in extra_paths:
            if Path(extra).is_file():
                items += json.loads(Path(extra).read_text(encoding="utf-8-sig")).get("intents", [])
        self.intents = []
        for item in items:
            answers = [a for a in item.get("answers", []) if str(a).strip()]
            if not answers:
                continue
            self.intents.append({
                **item,
                "answers": answers,
                "_patterns": [re.compile(p) for p in item.get("patterns", [])],
                "_not": [re.compile(p) for p in item.get("not", [])],
                "_examples": [grams(x) for x in item.get("examples", [])],
            })
        self.by_id = {item["id"]: item for item in self.intents}
        # inverted trigram index over all example questions: similarity to thousands of topics in milliseconds
        self._postings, self._ex_item, self._ex_norm = {}, [], []
        for index, item in enumerate(self.intents):
            for g in item["_examples"]:
                ex = len(self._ex_item)
                self._ex_item.append(index)
                self._ex_norm.append(math.sqrt(sum(v * v for v in g.values())) or 1.0)
                for gram, count in g.items():
                    self._postings.setdefault(gram, []).append((ex, count))

    def example_scores(self, q: Counter):
        """{intent index: best cosine of the question against that intent's examples}."""
        if not q:
            return {}
        dots = {}
        for gram, qc in q.items():
            for ex, count in self._postings.get(gram, ()):
                dots[ex] = dots.get(ex, 0) + qc * count
        qn = math.sqrt(sum(v * v for v in q.values()))
        best = {}
        for ex, dot in dots.items():
            value = dot / (qn * self._ex_norm[ex])
            index = self._ex_item[ex]
            if value > best.get(index, 0.0):
                best[index] = value
        return best

    def nearest(self, question: str, recent_topics=(), mode: str = "hall"):
        """Topic whose team-written example question is most similar: (item, similarity) or (None, 0)."""
        best, score = None, 0.0
        recent = set(recent_topics)
        for index, value in self.example_scores(grams(question)).items():
            item = self.intents[index]
            if value <= score or (item.get("mode") and item["mode"] != mode):
                continue
            if item.get("after") and not set(item["after"]) & recent:
                continue
            best, score = item, value
        return best, score

    def match(self, question, recent_topics=(), turn_index: int = 0, mode: str = "hall"):
        """All matching intents (higher priority, then longer match first).

        question may be a list of variants (e.g. the text as written and its "ты" form);
        an intent matches if any variant matches."""
        variants = [norm(x) for x in (question if isinstance(question, (list, tuple)) else [question])]
        found = []
        for item in self.intents:
            if item.get("mode") and item["mode"] != mode:
                continue
            if "turn" in item and item["turn"] != turn_index:
                continue
            if item.get("after") and not set(item["after"]) & set(recent_topics):
                continue
            spans = []
            for q in variants:
                if any(p.search(q) for p in item["_not"]):
                    continue
                spans += [m.end() - m.start() for p in item["_patterns"] for m in [p.search(q)] if m]
            if spans:
                found.append((item.get("priority", 0), max(spans), item))
        # Higher priority first, then the most specific (longest) match; stable for ties.
        found.sort(key=lambda entry: (-entry[0], -min(entry[1], 40)))
        return [entry[2] for entry in found]
