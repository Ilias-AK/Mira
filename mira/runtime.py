"""Mira: a human-like dialogue runtime on top of the team's offline retrieval index.

Order of routes for every message:
  1. local tools (arithmetic, letters) and a tiny counting solver
  2. memory of what the user said about themselves (pet, city, name, likes ...)
  3. the team-written reply bank (regex topics, several variants each)
  4. retrieval from the team's dialogue index, strictly filtered for Mira's
     character (no foreign biography, male forms, farewells, other names ...)
  5. short human fallbacks chosen by the message type

No pretrained models, no network. The own neural generator is not used here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mira.bank import Bank, cosine, fill_gender, grams, norm
from mira.filters import clean, mentioned, persona_problem, score as fit_score, turn_problem, content_words
from mira.user_facts import UserFacts
from mira.word_problems import CountingState, spoken_arithmetic
from mira.generic import Generic
from mira.frames import frame_answer, frame_key
from mira.dialog_memory import DialogMemory
from mira.hall import to_ty, hall_style, clock_answer, constraint, apply_constraint
import copy

try:  # the team's arena package (present in D:\turing-arena)
    from arena.runtime import bounded_text, output_problem
    from arena.constraints import evaluate_candidate, execute_local_tool
    from arena.quality import acceptable_response
    from arena.memory import ConversationMemory
    HAVE_ARENA = True
except Exception:  # pragma: no cover - lets the module run in a bare test environment
    HAVE_ARENA = False

    def bounded_text(text, limit=280):
        text = re.sub(r"\s+", " ", str(text)).strip()
        return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]

    def output_problem(text):
        return None if text.strip() else "empty"

    def evaluate_candidate(query, response, memory=None):
        return {"accepted": True, "issues": []}

    def execute_local_tool(query):
        return None

    def acceptable_response(text, **_):
        return True

    ConversationMemory = None


FILLERS = re.compile(r"^\W*(?:(?:мира|слушай|кстати|ну|окей|ок|ладно|хорошо|да|так|блин|короче|слушай-ка|ну\s+хорошо|ну\s+ладно)\W+)+", re.IGNORECASE)
QUESTION_WORD = re.compile(r"(?:^|[,;:]\s*|\s)(?:а\s+)?(?:что|как|какой|какая|какое|какие|какую|какого|каком|где|когда|куда|откуда|почему|зачем|сколько|кто|кем|чем|кого|кому|ч[её]|каким|зачем)\b(?!-)")
QUESTION_WORD_START = re.compile(r"^\W*(?:а\s+)?(?:что|как|какой|какая|какое|какие|где|когда|куда|откуда|почему|зачем|сколько|кто|чем|чего|кого|кому|ч[её]|каким)\b")
ADVICE = re.compile(r"\b(?:посоветуй\w*|подскажи\w*|что\s+(?:мне\s+)?делать|как\s+(?:мне\s+)?(?:лучше|быть|поступить)|стоит\s+ли|"
                    r"что\s+бы\s+ты\s+(?:сделала|выбрала)|на\s+моем\s+месте|как\s+думаешь,?\s+(?:стоит|надо|нужно))\b")
NEGATIVE = re.compile(r"\b(?:плохо|грустно|обидно|устал\w*|болит|заболел\w*|страшно|ужасн\w*|не\s+помогает|не\s+получается|бесит|"
                      r"расстро\w*|скукот\w*|скучно|жуть|кошмар|поссорил\w*|потерял\w*|сломал\w*|опоздал\w*|завалил\w*|уволили|бросил\w*|надоел\w*|достал\w*)\b")
POSITIVE = re.compile(r"\b(?:ура|круто|класс\w*|рад|рада|наконец|получилось|сдал\w*|выиграл\w*|победил\w*|предложили|купил\w*|"
                      r"отлично|здорово|счастлив\w*|повезло)\b|!\s*$")
BACK_TO_YOU = re.compile(r"(?:^|[,.!?]\s*)а\s+(?:ты|у\s+тебя|тебе)(?:\s+(?:как|что|тоже|сама))?\s*\?\s*$")
SENTENCE = re.compile(r"[^.!?]+[.!?]*")
KAZAKH = re.compile(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]")
PROPER = {"мира", "гульмира", "гульмайра", "ену", "кокшетау", "астана", "астане", "дильназ", "алихан", "мурка", "абай", "друзей"}
LAST_Q = re.compile(r"\bкакой\s+(?:был\s+)?(?:мой\s+|прошлый\s+|предыдущий\s+|последний\s+){1,2}вопрос\b|\bчто\s+(?:я|тебя)\s+(?:только\s+что|сейчас|до\s+этого|в\s+прошлый\s+раз)\s+спрашивал?\w*|\bповтори\s+(?:мой\s+)?(?:последний\s+|прошлый\s+|предыдущий\s+)?вопрос\b|\bо\s+ч[её]м\s+я\s+(?:только\s+что\s+)?спрашивал\w*")
BEFORE_Q = re.compile(r"^\W*(?:а\s+)?(?:до\s+(?:него|этого)|перед\s+(?:ним|этим)|а\s+ещ[её]\s+раньше|ещ[её]\s+раньше)\W*$")
ORDINALS = {"перв": 0, "втор": 1, "трет": 2, "четверт": 3, "пят": 4, "шест": 5, "седьм": 6, "восьм": 7, "девят": 8, "десят": 9}
_ORDINAL = r"(" + "|".join(ORDINALS) + r"|прошл|предыдущ|последн)\w*"
FIRST_Q = re.compile(r"\bкакой\s+(?:был\s+)?(?:мой\s+|наш\s+)?(?:самый\s+)?(" + "|".join(ORDINALS) + r")\w*\s+вопрос\b")
# "что ты ответил на первый вопрос" (to_ty() has already turned "ответил" into "ответила")
NTH_A = re.compile(r"\bчто\s+ты\s+(?:мне\s+|нам\s+)?(?:ответила|сказала|написала)\s+на\s+(?:самый\s+)?(?:мой\s+|наш\s+)?" + _ORDINAL +
                   r"\s+вопрос|\bкакой\s+(?:был\s+)?(?:у\s+тебя\s+)?(?:твой\s+)?(?:самый\s+)?" + _ORDINAL + r"\s+ответ\b"
                   r"|\bповтори\s+(?:свой\s+|твой\s+)?(?:самый\s+)?" + _ORDINAL + r"\s+ответ\b")
LAST_A = re.compile(r"\bчто\s+ты\s+(?:только\s+что\s+)?ответила\b|\bповтори\s+(?:свой\s+|твой\s+)?(?:последний\s+|прошлый\s+|предыдущий\s+)?ответ\b|\bчто\s+ты\s+(?:сказала|написала)\s+(?:до\s+этого|в\s+прошлый\s+раз|только\s+что)")
COUNT_Q = re.compile(r"\bсколько\s+(?:вопросов|раз)\s+(?:тебе\s+)?(?:уже\s+)?(?:задали|спросили)\b")


@dataclass
class Answer:
    text: str
    route: str
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _grams(text):
    words = re.findall(r"[а-яёa-z0-9]+", norm(text))
    grams = Counter()
    for w in words:
        w = f"^{w}$"
        grams.update(w[i:i + 3] for i in range(max(1, len(w) - 2)))
    return grams


def similarity(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    return dot / math.sqrt(sum(v * v for v in ga.values()) * sum(v * v for v in gb.values()))


def last_context_turn(context):
    if isinstance(context, list) and context:
        item = context[-1]
        return str(item.get("text", item.get("content", ""))) if isinstance(item, dict) else str(item)
    return str(context or "")


TELL_VERBS = r"расскажи|расскажите|объясни|объясните|опиши|опишите|поясни|что\s+(?:ты\s+)?знаешь\s+(?:про|о|об)"
MAKE_VERBS = (r"напиши|напишите|придумай|придумайте|сочини|сочините|нарисуй|спой|перечисли|перечислите|назови|назовите|"
              r"посчитай|посчитайте|вычисли|реши|решите|переведи|переведите|дай|дайте|покажи|покажите|приведи|сформулируй|"
              r"процитируй|составь|составьте|подскажи|подскажите|скинь|пришли")
REQUEST = re.compile(r"^\W*(?:(?:а|ну|слушай|можешь|можете|пожалуйста|быстро)\W+)*(?:(" + TELL_VERBS + r")|(" + MAKE_VERBS +
                     r"))\b\W*(?:мне\s+|нам\s+)?(?:(?:про|о|об|по)\s+)?(.*?)\W*$", re.IGNORECASE)
DEFINE = re.compile(r"^\W*(?:а\s+)?(?:кто\s+так(?:ой|ая|ие)|что\s+так(?:ое|ие)|что\s+(?:значит|означает)|кто\s+это\s+такой|что\s+это\s+такое)\s+(.+?)\W*$",
                    re.IGNORECASE)
VERB_LIKE = re.compile(r"\b\w+(?:ть|ться|ешь|ишь|ет|ит|ут|ют|ат|ят|ем|им|ете|ите|ла|ло|ли|л|сь)\b|\b(?:я|ты|мы|вы|он|она|они|мне|тебе|меня|тебя)\b",
                       re.IGNORECASE)
TELL_ANSWERS = ["Про {x}? Сходу не расскажу, надо вспоминать.", "Про {x} я мало что знаю, если честно.",
                "Ой, про {x} так сразу не соображу. Давайте что-нибудь попроще."]
MAKE_ANSWERS = ["Ой, {x} я так сразу не выдам, честно.", "{X}? Это надо сесть и подумать, сходу не получится.",
                "{X} с ходу не вспомню, извините. Давайте следующий."]
DEFINE_ANSWERS = ["{X}? В общих чертах представляю, но объяснить нормально сходу не возьмусь.",
                  "Слышала, конечно, но своими словами объяснить не смогу. Не моя тема.",
                  "{X}... Примерно понимаю, но объяснять точно не буду, а то навру."]
FRAGMENT_ANSWERS = ["{X}? Сходу не скажу, честно.", "{X}... Это не совсем ко мне, если честно.",
                    "Хм, {x}. Тут я ничего умного не скажу."]

INTERJECTIONS = {"вот", "так", "ну", "и", "да", "нет", "капец", "блин", "жесть", "фу", "ммм", "мм", "правда", "реально", "серьезно",
                 "серьёзно", "точно", "ясно", "понятно", "ладно", "ок", "окей", "хорошо", "ого", "вау", "ничего", "себе", "интересно",
                 "круто", "класс", "супер", "норм", "нормально", "бывает", "ага", "угу", "уж", "же", "все", "всё", "ой", "эх", "ах"}


def topic_phrase(text):
    """The user's own words for the topic, cleaned for quoting back ("рецепт борща", "3 закона ньютона")."""
    t = re.sub(r"\s+", " ", re.sub(r"[?!.…]+$", "", text.strip())).strip(" ,:-—")
    if t.count("«") > t.count("»"):
        t += "»"     # the request regex stops before trailing punctuation: «слово «окно»» lost its closing quote
    words = t.split()
    if not words or len(words) > 5 or len(t) > 45 or re.search(r"[<>{}\\]", t):
        return None
    return t


ASKS_FOREIGN = re.compile(r"англ|english|переве|перевод|по-?казахски|қазақ|на\s+казахском|скажи\s+на\s+\w+ском", re.IGNORECASE)


def foreign_answer(answer, question):
    """An answer that is mostly not Cyrillic, for a question that did not ask for another language."""
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", answer)
    latin = sum(1 for c in letters if c.isascii())
    return bool(letters) and latin / len(letters) > 0.5 and not ASKS_FOREIGN.search(question)


# «Ты бот?» and «Ты человек?» are one topic, but they want opposite first words.
ASKS_HUMAN = re.compile(r"\b(?:человек\w*|живой|живая|настоящ\w*)\b")
ASKS_MACHINE = re.compile(r"\b(?:бот\w*|робот\w*|нейросет\w*|ии|chatgpt|программ\w*|машин\w*|искусствен\w*)\b")
NOT_HUMAN = re.compile(r"\bне\s+(?:\w+\s+)?(?:человек\w*|живой|живая|настоящ\w*)\b")
SAYS_NO = re.compile(r"^\W*нет\b|\bно\s+нет\b", re.IGNORECASE)


def human_question(question):
    """'ты человек?' / 'ты живой?' (yes means human); 'человек или бот?' is a choice: None otherwise."""
    q = to_ty(question)
    if not re.search(r"\bты\b", q) or not ASKS_HUMAN.search(q) or NOT_HUMAN.search(q):
        return None
    if ASKS_MACHINE.search(q):
        return "choice" if re.search(r"\bили\b", q) else None
    return "yes"


def bot_item(item):
    return item.get("topic") == "bot" or any(re.search(r"\b(?:бот|робот|человек)\b", e.lower()) for e in item.get("examples", []))


def fit_polarity(answers, question, item):
    """Answers written for «Ты бот?» begin with «Нет»; for «Ты человек?» that reads as a confession."""
    kind = human_question(question)
    if kind is None or not bot_item(item):
        return answers
    out = []
    for a in answers:
        m = re.match(r"^Нет,\s*(я\s+(?:человек|настоящ|жив).*)$", a)
        if m:
            out.append(("Да, " + m.group(1)) if kind == "yes" else m.group(1)[:1].upper() + m.group(1)[1:])
        elif not SAYS_NO.search(a):
            out.append(a)
    return out


# Content words: a confident topic must share some of them with what was asked.
CONTENT_STOP = frozenset("""что как какой какая какое какие какую где когда куда откуда почему зачем сколько кто чем кого кому
это этот эта эти тот там тут ты тебя тебе тобой твой твоя твое твои мне меня мой моя мое мои мы вы вас вам ваш
он она они его ее их бы же ли ну да нет не ни и а но или то так уже еще тоже очень было был была были быть есть
для про при над под без после перед через между если чтобы потому вообще сейчас самый самая самое просто можно
надо нужно может любишь думаешь считаешь скажи расскажи""".split())
QUOTED = re.compile(r"[«\"“„]([^«»\"“”„]{1,40})[»\"”“]")


def _stems(text, min_len=4):
    words = re.findall(r"[а-яё]+", str(text).lower().replace("ё", "е"))
    return {w[:5] for w in words if len(w) >= min_len and w not in CONTENT_STOP}


def lexical_fit(question, item):
    """False when a topic chosen by similarity or by the model shares too few words with the question:
    «Переведи слово «окно»» is not «Переведи: я устала», «стресс перед экзаменом» is not «смена перед экзаменом»."""
    pool = _stems(" ".join(item.get("examples", [])), 2)
    pool |= _stems(" ".join(re.sub(r"\\[wsbdWSBD]|[()?:|*+\\\[\]{}^$.]", " ", p) for p in item.get("patterns", [])), 2)
    if not pool:
        return True
    for span in QUOTED.findall(question):
        words = _stems(span, 2)
        if words and not words & pool:
            return False
    q = _stems(question)
    hit = len(q & pool)
    return not ((len(q) >= 3 and hit / len(q) < 0.34) or (len(q) == 2 and not hit))


COMPARE = re.compile(r"\bчем\s+(?:\w+\s+){0,3}отлича\w*|\bв\s+ч[её]м\s+(?:\w+\s+)?разниц\w*|\bразниц\w*\s+между\b|\bотличи[ея]\s+между\b")


class MiraRuntime:
    def __init__(self, persona_path, bank_path, fallbacks_path, index_path=None, *, retriever=None,
                 use_retrieval=True, retrieval_k=40, min_context_similarity=0.55, min_retrieval_score=3.5,
                 max_characters=280, deadline_seconds=30, seed=0, mode="hall", hall_fallbacks_path=None,
                 intent_path="runs/mira_intent/intent.pt", tokenizer_path="artifacts/tokenizer.json", typo_rate=0.0):
        self.profile = json.loads(Path(persona_path).read_text(encoding="utf-8-sig"))
        self.bank = Bank(bank_path)
        fallbacks = json.loads(Path(fallbacks_path).read_text(encoding="utf-8-sig"))
        self.mode = mode
        if mode == "hall":
            hall_path = Path(hall_fallbacks_path) if hall_fallbacks_path else Path(fallbacks_path).with_name("fallbacks_hall.json")
            if hall_path.is_file():
                fallbacks = json.loads(hall_path.read_text(encoding="utf-8-sig"))
        self.fallbacks = {k: v for k, v in fallbacks.items() if not k.startswith("_")}
        self.max_characters = min(300, int(self.profile.get("max_characters", max_characters)))
        self.deadline_seconds = float(self.profile.get("deadline_seconds", deadline_seconds))
        self.retrieval_k, self.min_context_similarity = retrieval_k, min_context_similarity
        self.min_retrieval_score = min_retrieval_score
        self.seed = seed
        self.typo_rate = typo_rate
        self.generic = Generic(self.profile)
        self.intent, self.intent_status = None, "off"
        if intent_path and Path(intent_path).is_file():
            try:
                from mira.intent_model import IntentModel
                self.intent = IntentModel(intent_path, tokenizer_path)
                self.intent_status = f"on ({len(self.intent.labels)} тем, порог {self.intent.threshold:.2f})"
            except Exception as error:   # rules still work without the model
                self.intent_status = f"error: {type(error).__name__}: {error}"
        self.retriever = retriever
        self.use_reranker_too = False
        if self.retriever is None and use_retrieval and index_path and HAVE_ARENA:
            from arena.retrieval import HybridRetriever
            self.retriever = HybridRetriever.load(index_path)
        if self.retriever is not None and getattr(self.retriever, "reranker", None):
            self.use_reranker_too = True
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.history = []          # [{"role","text"}]
        self.topics = []           # topic per assistant turn
        self.aliases = []          # team Q&A topics each question resembled
        self.used = set()          # normalized replies already said this round
        self.qa = []               # (normalized question, answer) for this round
        self.dialog = DialogMemory()
        self.pinned = {}           # topic key -> what Mira already answered on it this round
        self._pending_pin = None
        self._repeat_hit = False
        self.facts = UserFacts()
        self.counting = CountingState()
        if hasattr(self, "generic"):
            self.generic.reset()
        self.memory = ConversationMemory(persona={"facts": self.profile.get("facts", []),
                                                  "name": self.profile.get("name"),
                                                  "age": self.profile.get("age")}) if ConversationMemory else None

    def close(self):
        close = getattr(self.retriever, "close", None)
        if close:
            close()

    # ---------------------------------------------------------------- helpers
    def _pick(self, options, salt, allow_question=False):
        """Deterministic but varied choice of an option not yet said this round."""
        fresh = [o for o in options if norm(fill_gender(o, self.facts.gender)) not in self.used
                 and not (self.mode == "hall" and KAZAKH.search(o))
                 and not (self.mode == "hall" and not allow_question and hall_style(o).rstrip().endswith("?"))]
        if not fresh:
            return None
        digest = hashlib.sha256(f"{self.seed}|{salt}|{len(self.history)}".encode("utf-8")).digest()
        return fresh[digest[0] % len(fresh)]

    def _recent_topics(self):
        # topics of the last two answers, plus team Q&A topics those questions resembled
        # (so "А специальность?" follows "На каком ты курсе?" even when a rule answered it)
        return self.topics[-2:] + [a for group in self.aliases[-2:] for a in group]

    def _answers_for(self, item, text):
        """The item's answers that fit this wording: in Russian unless asked otherwise, «Да» for «ты человек?»."""
        return fit_polarity([a for a in item["answers"] if not foreign_answer(a, text)], text, item)

    def _key_for(self, item, text):
        key = self._pin_key("bank", item.get("pin") or item["id"], item.get("topic"))
        if key and human_question(text) and bot_item(item):
            key += ":" + human_question(text)   # «Ты бот? — Нет» must not come back as the answer to «Ты человек?»
        return key

    def _followup_reply(self, text, need=0.7):
        """A team-written follow-up ("Дальше:") right after its question: checked before the rules."""
        recent = set(self._recent_topics())
        if not recent:
            return None
        q = grams(text)
        best, score = None, 0.0
        for item in self.bank.intents:
            if item.get("source") != "team_qa" or not item.get("after") or not set(item["after"]) & recent:
                continue
            value = max((cosine(q, grams(e)) for e in item.get("examples", [])), default=0.0)
            if value > score:
                best, score = item, value
        if best is None or score < need or not lexical_fit(text, best):
            return None
        key = self._key_for(best, text)
        if key and key in self.pinned:
            return None
        answers = self._answers_for(best, text)
        if not answers:
            return None
        choice = self._pick(answers, best["id"] + text, allow_question=True)   # the team's own wording
        if not choice:
            return None
        self._pending_pin = key
        self._keep_question = choice.rstrip().endswith("?")
        return choice, f"followup:{best['id']}", best.get("topic")

    def _aliases_for(self, question, topic):
        q = grams(question)
        recent = set(self._recent_topics())
        found = []
        for item in self.bank.intents:
            if item.get("source") != "team_qa" or (item.get("after") and not set(item["after"]) & recent):
                continue
            value = max((cosine(q, grams(e)) for e in item.get("examples", [])), default=0.0)
            if value >= 0.6 and item["id"] != topic:
                found.append((value, item["id"]))
        return [ident for _, ident in sorted(found, reverse=True)[:4]]   # several teams may have written the same question

    VARYING = {"short", "thanks", "greet", "rude", "tease", "absurd", "support", "riddle", "kz", "joke", "compare",
               "opinion", "bye", "laugh", "flirt", "doing", "memory", "fallback", "retrieval", "tool", "clock"}

    def _pin_key(self, kind, ident, topic=None):
        if topic in self.VARYING:
            return None
        return f"{kind}:{ident}"

    def _pinned_repeat(self, key):
        """Same topic asked again in other words -> 'Я же уже говорила: …'."""
        if key and key in self.pinned:
            self._repeat_hit = True
            lead = ["Я же уже говорила:", "Уже отвечала:", "Как я уже говорила,"][len(self.history) // 2 % 3]
            return f"{lead} {self._lower_first(self.pinned[key])}"
        return None

    def _bank_reply(self, text, turn_index):
        greeting = re.match(r"^\W*(?:привет\w*|здравствуй\w*|хай|салем|салам\w*|добр\w+\s+(?:день|вечер|утро)|доброе утро)\W+(?=\w)", text, re.IGNORECASE)
        if greeting:
            rest, item = self._bank_reply(text[greeting.end():], turn_index)
            if rest:
                return ("Привет! " + rest if turn_index == 0 else rest), item
        compare = COMPARE.search(to_ty(text))
        for item in self.bank.match([to_ty(text), text], self._recent_topics(), turn_index, self.mode):
            if compare and item.get("priority", 0) <= 0 and not any(re.search(r"отлич|разниц", p) for p in item.get("patterns", [])):
                continue   # «Чем бешбармак отличается от плова?» is not the broad «бешбармак» topic («Пробовала, конечно»)
            key = self._key_for(item, text)
            again = self._pinned_repeat(key)
            if again:
                return again, item
            choice = self._pick(self._answers_for(item, text), item["id"] + text, allow_question=bool(item.get("ask")))
            if choice:
                self._pending_pin = key
                self._keep_question = bool(item.get("ask"))
                return choice, item
        return None, None

    def _understand(self, question, turn_index):
        """Favourites, bank, then generic handlers. Returns (text, route, topic) or Nones."""
        original, question = question, to_ty(question)
        strong = next(iter(self.bank.match([question, original], self._recent_topics(), turn_index, self.mode)), None)
        strong = strong is not None and strong.get("priority", 0) >= 2
        # after «Ты бот?» the team's follow-up «Докажи, что человек?» looks like «Ты человек?» letter by letter;
        # a strong rule for the question itself wins unless the follow-up is nearly word for word
        follow = self._followup_reply(original, need=0.85 if strong else 0.7)
        if follow:
            return follow
        told = self._reaction_to_user_fact(original)
        if told:
            return told
        fav = self.generic.favorite(question, self.seed)
        short = re.match(r"^\W*(?:ну\s+)?а\s+([а-яё]+(?:\s+[а-яё]+)?)\W*$", norm(question))
        if not fav and short and self.topics and self.topics[-1] in ("favorite", "football", "anime"):
            fav = self.generic.favorite("какой твой любимый " + short.group(1) + "?", self.seed)   # "а игрок?" after "любимый клуб"
        if fav:
            key = self._pin_key("fav", getattr(self.generic, "last_fav_key", fav))
            again = self._pinned_repeat(key)
            if again:
                return again, "repeat_topic", "favorite"
            if norm(fav) not in self.used:
                self._pending_pin = key
                fav_key = getattr(self.generic, "last_fav_key", "")
                return fav, "favorite", {"футболист": "football", "команда": "football", "спорт": "football",
                                         "аниме": "anime", "персонаж": "anime"}.get(fav_key, "favorite")
        candidate, item = self._bank_reply(original, turn_index)
        if (candidate and item.get("topic") in {"movies", "books", "games", "phone", "walk", "music", "sport"}
                and item.get("priority", 0) < 0 and "?" not in question
                and re.search(r"\b(?:я|мне|меня|мой|моя|мое|мои|у\s+меня)\b", question)):
            return None, None, None    # the asker's own story: acknowledge, don't lecture on the topic
        regex_item = item if candidate else None
        model = None if regex_item else self._model_reply(original, None, need=0.85, min_margin=0.3)  # very sure
        if model:
            return model
        if not candidate:
            near = self._example_reply(original)
            if near:
                return near
        if candidate and item.get("priority", 0) < 0:
            yes_no = self.generic.yes_no(question, self.seed)   # "ты когда-нибудь плакала…?" beats a broad topic
            if yes_no and norm(fill_gender(yes_no, self.facts.gender)) not in self.used:
                return yes_no, "yes_no", self.topics[-1] if self.topics else "yes_no"
        if candidate:
            return candidate, ("repeat_topic:" if self._repeat_hit else "bank:") + item["id"], item.get("topic")
        fkey = frame_key(question)
        if fkey is not None:
            again = self._pinned_repeat(self._pin_key("frame", fkey))
            if again:
                return again, "repeat_topic", "frame"
        framed = frame_answer(question, self.used, self.seed)
        if framed:
            self._pending_pin = self._pin_key("frame", fkey) if fkey is not None else None
            return framed, "frame", "frame"
        for name in ("either", "yes_no"):
            found = getattr(self.generic, name)(question, self.seed)
            if found and norm(fill_gender(found, self.facts.gender)) not in self.used:
                return found, name, self.topics[-1] if self.topics else name
        model = self._model_reply(original, regex_item)    # fairly sure, and nothing more exact fitted
        if model:
            return model
        for name in ("opinion", "why"):
            found = getattr(self.generic, name)(question, self.seed)
            if found and norm(fill_gender(found, self.facts.gender)) not in self.used:
                return found, name, self.topics[-1] if self.topics else name
        return None, None, None

    def _formal(self, question):
        if self.mode == "hall":
            return True      # at the contest Mira answers the hall politely
        said = " ".join([question] + [t["text"] for t in self.history if t["role"] == "user"]).lower()
        return bool(re.search(r"\b(?:вы|вас|вам|ваш\w*)\b", said))

    def _lower_first(self, text):
        first = re.match(r"[А-Яа-яЁёA-Za-z]+", text)
        if first and first.group(0).lower() in PROPER or (first and first.group(0).isupper() and len(first.group(0)) > 1):
            return text
        return text[:1].lower() + text[1:]

    def _repeat(self, question):
        """Same question again in this round -> 'Я же уже говорила: ...'."""
        q = re.sub(r"[^\w\s]", "", to_ty(question)).strip()
        if len(q.split()) < 2:
            return None
        for previous_q, previous_a in reversed(self.qa):
            if similarity(q, previous_q) >= 0.85:
                lead = ["Я же уже говорила:", "Уже отвечала:", "Как я уже говорила,"][len(self.qa) % 3]
                answer = previous_a.rstrip()
                return f"{lead} {self._lower_first(answer)}"
        return None

    def _meta_memory(self, question):
        q = to_ty(question)
        user_turns = [t["text"] for t in self.history if t["role"] == "user"]
        bot_turns = [t["text"] for t in self.history if t["role"] == "assistant"]
        if LAST_Q.search(q):
            if not user_turns:
                self._quoted_back = None
                return "Это первый вопрос, до него ничего не было"
            self._quoted_back = len(user_turns) - 1
            return f"Вы спрашивали: «{user_turns[-1]}»" if self.mode == "hall" else f"Ты спрашивал{{|а}}: «{user_turns[-1]}»"
        back = getattr(self, "_quoted_back", None)
        if BEFORE_Q.search(q) and back is not None and self.topics and self.topics[-1] == "memory":
            if back == 0:
                return "До него ничего не было, это был первый"
            self._quoted_back = back - 1
            return f"До этого: «{user_turns[back - 1]}»"
        nth = FIRST_Q.search(q)
        if nth and user_turns:
            index = ORDINALS[nth.group(1)]
            if index == 0:
                return f"Первым был вопрос: «{user_turns[0]}»"
            if index < len(user_turns):
                return f"Вы спрашивали: «{user_turns[index]}»" if self.mode == "hall" else f"Ты спрашивал{{|а}}: «{user_turns[index]}»"
        nth = NTH_A.search(q)
        if nth:
            if not bot_turns:
                return "Это первый вопрос, до него ничего не было"
            index = ORDINALS.get(next(g for g in nth.groups() if g), len(bot_turns) - 1)
            if index < len(bot_turns):
                return f"Я ответила: «{bot_turns[index]}»"
        if LAST_A.search(q) and bot_turns:
            return f"Я ответила: «{bot_turns[-1]}»"
        if COUNT_Q.search(q):
            n = len(user_turns)
            return f"Это уже {n + 1}-й вопрос, если считать этот" if n else "Это первый"
        return None

    def _vocabulary(self):
        if not hasattr(self, "_vocab"):
            words = set(re.findall(r"[а-я]{3,}", " ".join(p for item in self.bank.intents for p in item.get("patterns", []))))
            words |= set(re.findall(r"[а-я]{3,}", norm(" ".join(a for item in self.bank.intents for a in item["answers"]))))
            words |= {"тебе", "тебя", "сколько", "зовут", "дела", "делаешь", "любишь", "учишься", "работаешь", "живешь",
                      "откуда", "почему", "когда", "какой", "какая", "привет", "спасибо", "хорошо", "нормально"}
            self._vocab = words
        return self._vocab

    def _fix_typos(self, question):
        """One-letter typo fix (замена/перестановка/пропуск) against words the bank knows."""
        vocab = self._vocabulary()

        def close(a, b):
            if abs(len(a) - len(b)) > 1 or a[0] != b[0]:
                return False
            if len(a) == len(b):
                diff = [i for i in range(len(a)) if a[i] != b[i]]
                return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
            short, long_ = (a, b) if len(a) < len(b) else (b, a)
            return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))

        def fix(match):
            word = match.group(0)
            low = word.lower().replace("ё", "е")
            if len(low) < 4 or low in vocab:
                return word
            options = sorted((v for v in vocab if close(low, v)),
                             key=lambda v: (0 if len(v) == len(low) + 1 else 1 if len(v) == len(low) else 2, v))
            if not options:
                return word
            longer = [v for v in options if len(v) == len(low) + 1]
            return longer[0] if len(longer) == 1 else options[0] if len(options) == 1 else word

        return re.sub(r"[А-Яа-яЁё]+", fix, question)

    def _example_reply(self, text):
        """Team Q&A topics: the question is close to one of the team's example questions."""
        item, score = self.bank.nearest(text, self._recent_topics(), self.mode)
        if item is None:
            return None
        predicted = self._predict(text)
        agrees = predicted is not None and predicted[0] == item["id"] and predicted[1] >= 0.35
        need = 0.55 if item.get("after") else 0.62
        if item.get("source") == "retrieval_v2":
            need = 0.75   # new_topics: only near-exact matches, never as a guess
        content_words = [w for w in re.findall(r"[а-яё]+", to_ty(text)) if len(w) > 2]
        if len(content_words) < 2:
            need = max(need, 0.9)    # one word ("шарага?") is too little to recognise a topic
        if re.search(r"\bили\b", norm(text)) and not any(re.search(r"\bили\b", norm(x)) for x in item.get("examples", [])):
            need, agrees = 0.85, False   # "X или Y?" is a choice, not a question about X alone
        if score < need and not (agrees and score >= 0.4):
            return None
        if not lexical_fit(text, item):
            return None
        key = self._key_for(item, text)
        again = self._pinned_repeat(key)
        if again:
            return again, "repeat_topic:" + item["id"], item.get("topic")
        answers = self._answers_for(item, text)
        if not answers:
            return None
        choice = self._pick(answers, item["id"] + text, allow_question=True)   # the team's own wording
        if not choice:
            return None
        self._pending_pin = key
        self._keep_question = choice.rstrip().endswith("?")
        return choice, f"examples:{item['id']}", item.get("topic")

    def _predict(self, text):
        if self.intent is None:
            return None
        if self._prediction is None or self._prediction[0] != text:
            try:
                label, prob, margin = self.intent.predict(text)
            except Exception as error:
                self._last_model = ("error", f"{type(error).__name__}: {error}")
                return None
            self._prediction = (text, label, prob, margin)
            self._last_model = (label, round(prob, 3), round(margin, 3))
        return self._prediction[1:]

    def _model_reply(self, text, regex_item, need=None, min_margin=0.15):
        """The team's own neural intent model: used when rules found nothing or only a broad topic.

        need=None: the normal bar (at least 0.6; the calibrated threshold is measured on the
        team's own topics only, so out-of-topic questions need a safety margin)."""
        if self.intent is None or (regex_item is not None and regex_item.get("priority", 0) >= 0):
            return None
        if self._fix_typos(text) != text:
            return None   # a misspelt word: let the typo fix run first ("любимый футболст" is not "сериал")
        predicted = self._predict(text)
        if predicted is None:
            return None
        label, prob, margin = predicted
        if need is None:
            need = max(self.intent.threshold, 0.6) + (0.1 if regex_item is not None else 0.0)
        item = self.bank.by_id.get(label)
        if item is None or prob < need or margin < min_margin or (regex_item is not None and regex_item["id"] == label):
            return None
        if item.get("mode") and item["mode"] != self.mode:
            return None
        if item.get("after") and not set(item["after"]) & set(self._recent_topics()):
            return None   # a follow-up topic ("По какому предмету?") only right after its own question
        if prob < 0.9 and not lexical_fit(text, item):
            return None   # out of the team's topics the model can be sure and wrong: share some words or stay out
        key = self._key_for(item, text)
        again = self._pinned_repeat(key)
        if again:
            return again, "repeat_topic:" + item["id"], item.get("topic")
        choices = self._answers_for(item, text)
        if not choices:
            return None   # e.g. "I'm tired" for a question that never asked for English
        choice = self._pick(choices, item["id"] + text, allow_question=bool(item.get("ask")))
        if not choice:
            return None
        self._pending_pin = key
        self._keep_question = bool(item.get("ask"))
        return choice, f"model:{item['id']}", item.get("topic")

    def _reaction_to_user_fact(self, text):
        """The user shares something about themselves: react to it instead of answering a question nobody asked."""
        if "?" in text:
            return None
        q = norm(text)
        mine = re.match(r"^\W*(?:а\s+)?(?:мой|моя|мое|мои)\s+(?:самы[йяе]\s+)?любим\w*\s+([а-я]+)\s*[-—:]?\s*(?:это\s+)?(.+?)\W*$", q)
        if mine:
            hers = self.generic.favorite(f"какой твой любимый {mine.group(1)}?", self.seed)
            if hers:
                theirs = mine.group(2)
                hers_clean = re.sub(r"(?i)^(?:сейчас|наверное|наверно)\s+", "", hers.strip().rstrip(".")).strip()
                if theirs[:4] and theirs[:4] in norm(hers_clean):
                    return "О, у меня тоже!", "user_fact", "favorite"
                return f"Хороший выбор. А у меня {hers_clean}", "user_fact", "favorite"
        pet = re.match(r"^\W*(?:а\s+)?у\s+меня\s+(?:есть\s+)?(кот|кошка|котик|собака|пес|щенок|хомяк|попугай)\w*\s+(?:по\s+имени\s+|зовут\s+)?([а-я]{2,})\W*$", q)
        if pet:
            name = pet.group(2).capitalize()
            if pet.group(1).startswith("ко"):
                return f"О, котик! {name} — классное имя. У моих родителей тоже кошка, Мурка", "user_fact", "pets"
            return f"{name} — классное имя. А у моих родителей кошка Мурка", "user_fact", "pets"
        return None

    def _topic_fallback(self, question):
        """Last resort that still answers *about* what was asked, in the asker's own words."""
        q = FILLERS.sub("", question).strip() or question.strip()
        define = DEFINE.match(q)
        request = REQUEST.match(q)
        if define:
            x, answers, kind = topic_phrase(define.group(1)), DEFINE_ANSWERS, "define"
        elif request:
            x = topic_phrase(request.group(3)) if request.group(3) else None
            answers, kind = (TELL_ANSWERS if request.group(1) else MAKE_ANSWERS), "request_topic"
        elif ("?" not in q and 2 <= len(q.split()) <= 4 and not VERB_LIKE.search(q)
              and not re.search(r"\d+\s*[-+*/xх×:]\s*\d", q)
              and not all(w in INTERJECTIONS for w in re.findall(r"[а-яё]+", q.lower()))):
            x, answers, kind = topic_phrase(q), FRAGMENT_ANSWERS, "fragment"
        else:
            return None
        if not x:
            return None
        options = [a for a in answers if "{" in a] if x else answers
        text = self._pick(options, kind + x)
        if not text:
            return None
        return text.replace("{X}", x[:1].upper() + x[1:]).replace("{x}", x), kind

    def _kind(self, question):
        q = to_ty(FILLERS.sub("", question) or question)
        if ADVICE.search(q):
            return "advice"
        if "?" in q or QUESTION_WORD_START.search(q):
            wh = QUESTION_WORD.search(q) or QUESTION_WORD_START.search(q)
            if not wh:
                return "question_yesno"
            word = re.search(r"\b(что|как|какой|какая|какое|какие|какую|какого|каком|где|когда|сколько|кто|ч[её]м)\b", q)
            mapping = {"как": "q_how", "что": "q_what", "чем": "q_what", "чём": "q_what", "где": "q_where", "когда": "q_when",
                       "сколько": "q_howmuch", "кто": "q_who"}
            if word:
                return mapping.get(word.group(1), "q_which" if word.group(1).startswith("как") else "question_open")
            return "question_open"
        if re.match(r"^\W*(?:дай|расскажи|скажи|напиши|покажи|объясни|пришли|скинь|спой|опиши|продолжи|допиши|закончи|"
                    r"переведи|назови|перечисли|придумай|сочини|загадай|приведи|составь|повтори)\b", q):
            return "request"
        if re.search(r"\bу\s+меня\s+тоже\b|\bя\s+тоже\b", q) and NEGATIVE.search(q + " жуть"):
            return "statement_same"
        if re.search(r"\b(?:мне\s+кажется|по-моему|я\s+думаю|я\s+считаю|наверное\s+я)\b", q):
            return "statement_opinion"
        if re.search(r"\b(?:мне\s+(?:только\s+)?помогает|я\s+обычно|я\s+всегда)\b", q):
            return "statement_tip"
        if re.search(r"\b(?:сегодня|вчера|однажды|недавно|на\s+днях|представляешь)\b", q) and not NEGATIVE.search(q):
            return "statement_event"
        if re.search(r"\b(?:я|мне|меня|мой|моя|мое|мои|у\s+меня)\b", q) and not NEGATIVE.search(q) and not POSITIVE.search(q) \
                and not re.search(r"\b(?:кажется|думаю|считаю)\b", q):
            return "statement_personal"
        if NEGATIVE.search(q):
            return "statement_negative"
        if POSITIVE.search(q) or re.search(r"\b(?:мило|красиво|хлопали|весело|смешно)\b", q):
            return "statement_positive"
        if re.search(r"\b(?:сегодня|вчера|однажды|недавно|на\s+днях|видел\w*|случил\w*|представляешь)\b", q):
            return "statement_event"
        return "statement_neutral"

    def _soften_questions(self, text):
        """Do not end every reply with a counter-question."""
        asked_recently = sum(1 for turn in self.history[-4:] if turn["role"] == "assistant" and turn["text"].rstrip().endswith("?"))
        if asked_recently and text.rstrip().endswith("?"):
            parts = [p.strip() for p in SENTENCE.findall(text) if p.strip()]
            if len(parts) >= 2:
                head = " ".join(parts[:-1])
                return head
            match = re.match(r"^(.*?\))\s+[^)]*\?$", text)
            if match:
                return match.group(1)
        return text

    def _retrieve(self, question, start):
        if self.retriever is None:
            return [], {}
        diagnostics = {}
        rows, seen = [], set()
        modes = [False, True] if self.use_reranker_too else [None]
        for mode in modes:
            remaining = self.deadline_seconds - (time.monotonic() - start) - 3.0
            if remaining <= 0.2:
                break
            try:
                kwargs = {"history": self.history[-6:], "persona": self.profile.get("facts", []),
                          "k": self.retrieval_k, "timeout_seconds": min(6.0, remaining)}
                if mode is not None:
                    kwargs["use_reranker"] = mode
                found = self.retriever.search(question, **kwargs)
            except Exception as error:
                diagnostics[f"retrieval_error_{mode}"] = f"{type(error).__name__}: {error}"
                continue
            for rank, row in enumerate(found):
                key = row.get("id") or row.get("response")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({**row, "_rank": rank})
        return rows, diagnostics

    def _select(self, question, rows):
        history_text = " ".join(t["text"] for t in self.history[-4:])
        names = mentioned(self.history, question)
        user_text = " ".join(t["text"] for t in self.history if t["role"] == "user") + " " + question
        turn_index = sum(1 for t in self.history if t["role"] == "user")
        best, rejected = None, Counter()
        for row in rows:
            text = clean(bounded_text(row.get("response", ""), self.max_characters))
            problem = (output_problem(text) or turn_problem(text, question, turn_index, names)
                       or persona_problem(text, self.profile, user_text))
            if not problem and norm(text) in self.used:
                problem = "already_said"
            if not problem and similarity(text, question) > 0.8:
                problem = "echo"
            if not problem and not acceptable_response(text):
                problem = "quality"
            if not problem:
                verdict = evaluate_candidate(question, text, self.memory)
                if not verdict.get("accepted", True):
                    problem = "constraint"
            if problem:
                rejected[problem] += 1
                continue
            context_sim = similarity(last_context_turn(row.get("context")), question)
            words = re.findall(r"[а-яё]+", norm(text))
            if text.rstrip().endswith("?") and not re.search(r"[.!)]\s", text):
                rejected["counter_question_only"] += 1
                continue
            if len(words) <= 2 and context_sim < 0.75:
                rejected["too_short_for_match"] += 1
                continue
            if context_sim < self.min_context_similarity:
                rejected["weak_context"] += 1
                continue
            value = fit_score(text, question, history_text, row["_rank"]) + 4.0 * context_sim
            if best is None or value > best[0]:
                best = (value, text, row, context_sim)
        return best, dict(rejected)

    # ----------------------------------------------------------------- answer
    def answer(self, question: str) -> Answer:
        start = time.monotonic()
        question = re.sub(r"\s+", " ", str(question)).strip()
        if not question:
            raise ValueError("Question must not be empty")
        turn_index = sum(1 for t in self.history if t["role"] == "user")
        diagnostics = {}
        text, route, topic = None, None, None
        self._pending_pin, self._repeat_hit, self._keep_question = None, False, False
        self._last_model = None
        self._prediction = None
        hall = self.mode == "hall"

        tool = execute_local_tool(question) if len(question) < 400 else None
        if tool and tool.get("ok") and tool.get("text") is not None:
            text, route, topic = str(tool["text"]), "tool", "tool"
        if text is None:
            meta = self._meta_memory(question)
            if meta:
                text, route, topic = meta, "dialog_memory", "memory"
        if text is None:
            recalled_early = self.facts.recall(question) or self.dialog.recall_user(question, self._formal(question))
            if recalled_early and not recalled_early.startswith(("Вы мне этого", "Ты мне этого")):
                text, route, topic = recalled_early, "user_memory", "memory"
        if text is None:
            repeat = self._repeat(question)
            if repeat:
                text, route, topic = repeat, "repeat", "memory"
        if text is None:
            remembered = (self.dialog.recall_note(question) or self.dialog.recall_own(question)
                          or self.dialog.topics(question, self._formal(question)))
            if remembered:
                text, route, topic = remembered, "dialog_memory", "memory"
        if text is None:
            clock = clock_answer(question)
            if clock:
                text, route, topic = clock, "clock", "clock"
        if text is None:
            counted = self.counting.answer(question) or spoken_arithmetic(question)
            if counted:
                text, route, topic = counted, "counting", "riddle"
        if text is None:
            noted = self.facts.remember_definition(to_ty(question) if re.search(r"(?i)\bзапомните\b", question) else question)
            if noted:
                text, route, topic = noted, "user_memory", "memory"
        if text is None:
            recalled = self.facts.recall(question) or self.dialog.recall_user(question, self._formal(question))
            if recalled:
                text, route, topic = recalled, "user_memory", "memory"
                rest = [s for s in SENTENCE.findall(question) if s.strip() and not self.facts.recall(s)]
                if rest and len(rest) < len(SENTENCE.findall(question)):
                    extra, item = self._bank_reply(" ".join(rest), turn_index)
                    if extra:
                        text = f"{text} {self._soften_questions(extra)}"
        facts_before = copy.deepcopy(self.facts.facts)
        self.facts.observe(question)
        if text is None:
            intro = self.generic.introduction(question, facts_before, {**self.facts.facts, "_gender": self.facts.gender}, self.seed)
            if intro:
                text, route, topic = intro, "introduction", "pets" if self.facts.facts.get("pets") != facts_before.get("pets") else "intro"
        if text is None:
            picked = self.generic.pick_from_pair(question, self.seed)
            if picked:
                text, route, topic = picked, "pair_choice", "compare"
        self.generic.remember_pair(question)
        if text is None:
            sentences = [x.strip() for x in SENTENCE.findall(question) if x.strip()]
            asked = [x for x in sentences if x.endswith("?")]
            if len(sentences) > 1 and asked:
                last = asked[-1]
                text, route, topic = self._understand(FILLERS.sub("", last) or last, turn_index)
                if text:
                    before = " ".join(sentences[:sentences.index(last)])
                    if before:
                        prefix, item = self._bank_reply(FILLERS.sub("", before) or before, turn_index)
                        if (prefix and item.get("topic") in {"joke"} and len(prefix) <= 45
                                and "?" not in prefix and norm(prefix) not in self.used):
                            prefix = prefix if re.search(r"[.!?)(]$", prefix) else prefix + "."
                            text = f"{prefix} {text}"
            if text is None:
                text, route, topic = self._understand(FILLERS.sub("", question) or question, turn_index)
        if text is None:
            fixed = self._fix_typos(question)
            if fixed != question:
                text, route, topic = self._understand(FILLERS.sub("", fixed) or fixed, turn_index)
                if text:
                    route = route + "+typo"
                    diagnostics["typo_fixed"] = fixed
        if text is None:
            rows, retrieval_diag = self._retrieve(question, start)
            diagnostics.update(retrieval_diag)
            best, rejected = self._select(question, rows)
            diagnostics.update({"candidates": len(rows), "rejected": rejected})
            if best and best[0] >= self.min_retrieval_score:
                text, route, topic = best[1], "retrieval", "retrieval"
                diagnostics.update({"score": round(best[0], 3), "context_similarity": round(best[3], 3),
                                    "matched_context": last_context_turn(best[2].get("context"))[:160],
                                    "id": best[2].get("id")})
            elif best:
                diagnostics["best_rejected_score"] = round(best[0], 3)
        if text is None and self.dialog.user:
            remembered = self.dialog.recall_user(question, self._formal(question), relaxed=True)
            if remembered and not remembered.startswith(("Вы мне этого", "Ты мне этого")):
                text, route, topic = remembered, "user_memory", "user_memory"
        if text is None:
            topical = self._topic_fallback(question)
            if topical:
                text, kind = topical
                route, topic = "fallback:" + kind, "fallback"
        if text is None:
            kind = self._kind(question)
            text = self._pick(self.fallbacks.get(kind, []), kind) or self._pick(
                sum((v for v in self.fallbacks.values()), []), "any") or "Хм)"
            route, topic = "fallback:" + kind, "fallback"

        new_name = self.facts.facts.get("name")
        if new_name and new_name != facts_before.get("name") and route != "introduction":
            text = f"Очень приятно, {new_name}) {text}"
        if self.mode == "hall":
            from mira.hall import plural_slots
            text = plural_slots(text)
        text = fill_gender(text, self.facts.gender)
        rule = constraint(question)
        if rule and route not in ("tool", "counting", "clock", "dialog_memory") and not route.startswith(("followup", "examples")):
            text = apply_constraint(rule, text, question)
            route += "+" + rule
        if self.mode == "hall":
            if route not in ("dialog_memory",):
                said = " ".join([question] + [t["text"] for t in self.history if t["role"] == "user"]).lower()
                text = hall_style(text, keep_questions=self._keep_question, protect=frozenset(re.findall(r"[а-яё]+", said)))
        elif route not in ("tool", "counting", "user_memory"):
            text = self._soften_questions(text)
        text = bounded_text(clean(text), self.max_characters)
        if foreign_answer(text, question) and route not in ("tool", "counting", "clock", "dialog_memory", "user_memory", "repeat"):
            # the contest wants Russian: a team answer in English to a question that did not ask for it is a tell
            kind = self._kind(question)
            text = self._pick(self.fallbacks.get(kind, []), "lang" + kind) or self._pick(
                self.fallbacks.get("statement_neutral", []), "lang") or "Хм"
            route += "+lang_guard"
        if not text.strip(" .,!?)("):
            text = self._pick(self.fallbacks.get("statement_neutral", []), "empty") or "Ага"   # never send an empty message
        if text.endswith(".") and not text.endswith(".."):
            text = text[:-1]  # people rarely end chat messages with a full stop
        plain = text   # memory keeps the version without the typo
        if self.typo_rate and len(text) >= 15 and not route.startswith(("tool", "counting", "clock", "user_memory", "dialog_memory", "repeat")):
            import random
            from mira.hall import add_typo
            rnd = random.Random(hashlib.sha256(f"{self.seed}|typo|{len(self.history)}|{question}".encode("utf-8")).digest())
            if rnd.random() < self.typo_rate:
                said = frozenset(re.findall(r"[а-яё]+", " ".join([question] + [t["text"] for t in self.history if t["role"] == "user"]).lower()))
                text = add_typo(text, rnd, said)

        self.history += [{"role": "user", "text": question}, {"role": "assistant", "text": text}]
        self.aliases.append(self._aliases_for(question, topic))
        self.topics.append(topic)
        self.used.add(norm(plain))
        if self._pending_pin and not self._repeat_hit and route and not route.startswith(("repeat", "fallback")):
            self.pinned[self._pending_pin] = plain
        self.dialog.observe_user(question, turn_index)
        self.dialog.observe_mira(plain, question, topic, turn_index)
        if route not in ("dialog_memory", "user_memory") and not route.startswith(("fallback", "repeat")):
            self.qa.append((re.sub(r"[^\w\s]", "", to_ty(question)).strip(), plain))
        if self.memory is not None:
            self.memory.add_user(question)
            self.memory.add_assistant(text)
        if self._last_model:
            diagnostics["intent_model"] = self._last_model
        return Answer(text, route, round(time.monotonic() - start, 3), diagnostics)
