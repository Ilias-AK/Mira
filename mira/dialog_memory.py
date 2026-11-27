"""Memory of the current conversation (one round), rules only.

- every sentence the asker says about themselves is kept; a later question about
  it ("Где я работаю?", "Как зовут мою сестру?", "Помнишь, что я говорил про…?")
  is answered from the stored sentence: the words that answer the question are
  cut out ("Я работаю в банке" + "Где я работаю?" -> "В банке"), otherwise the
  sentence is quoted;
- "запомни/запомните …" notes ("число 7", "слово «яблоко»") are kept and recalled;
- Mira's own answers are kept: "Что ты говорила про работу?" quotes the answer;
- "О чём мы говорили?" names the topics of the round.
"""
from __future__ import annotations

import re

from mira.hall import to_ty

WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9-]+")
FIRST_PERSON = re.compile(r"\b(?:я|мне|меня|мной|мой|моя|мое|моё|мои|моего|мою|моей|моих|моим|у\s+меня)\b", re.IGNORECASE)
STOP = {"я", "мне", "меня", "мной", "мой", "моя", "мое", "моё", "мои", "моего", "мою", "моей", "моих", "моим", "у", "есть",
        "ты", "тебе", "тебя", "твой", "твоя", "твое", "твои", "твою", "твоего", "это", "вот", "же", "ли", "а", "и", "но",
        "что", "как", "какой", "какая", "какое", "какие", "какую", "какого", "где", "куда", "откуда", "когда", "кто", "кем",
        "чем", "сколько", "зовут", "помнишь", "напомни", "говорил", "говорила", "сказал", "сказала", "рассказывал",
        "рассказывала", "тебе", "уже", "очень", "просто", "ну", "так", "там", "тут", "был", "была", "было", "были", "не",
        "да", "нет", "вообще", "кстати", "слушай", "мира", "в", "во", "на", "с", "со", "по", "о", "об", "про", "из", "к",
        "ко", "за", "для", "от", "до", "раньше", "ей", "ему", "им", "его", "ее", "её", "их", "он", "она", "они", "лет", "год", "года", "сейчас", "сегодня", "запомни", "запомнить", "просил", "просила"}
RECALL_CUE = re.compile(
    r"\bпомнишь\b|\bнапомни\b|\bя\s+(?:тебе\s+)?(?:говорил\w*|сказал\w*|рассказывал\w*|писал\w*)\b"
    r"|\b(?:что|кем|где|куда|откуда|когда|сколько|чем|с\s+кем|как)\s+(?:же\s+)?(?:я|мне|меня)\b"
    r"|\bкак\s+зовут\s+мо\w+\b|\bкакой\s+(?:у\s+меня|мой)\b|\bкакая\s+(?:у\s+меня|моя)\b|\bкакое\s+(?:у\s+меня|мое)\b"
    r"|\bкакие\s+(?:у\s+меня|мои)\b|\bчто\s+у\s+меня\b|\bкто\s+(?:у\s+меня|мой|моя)\b"
    r"|\b(?:что|кем|где|куда|откуда|когда|сколько|чем|как|какой|какая|какое|кто)\b[^?]*\b(?:мой|моя|мое|моё|мои|моего|мою|моей|моему|моих|моим)\b"
    r"|\b(?:что|куда|где|когда|какой|какая|какое|какую|каким|с\s+кем|кого|чем|сколько|зачем|почему)\b[^?]*\bя\s+\w+(?:л|ла|ли|ю|у|усь|юсь)\b")
STRONG_CUE = re.compile(r"\bкак\s+зовут\s+мо\w+|\bпомнишь\b|\bнапомни\b|\bя\s+(?:тебе\s+)?(?:говорил\w*|рассказывал\w*)\b|\bкакой\s+мой\b|\bкакая\s+моя\b|\bкак\s+меня\s+зовут\b|\bсколько\s+мне\s+лет\b|\bоткуда\s+я\b")
QUOTE_CUE = re.compile(r"\bчто\s+я\s+(?:тебе\s+)?(?:говорил\w*|рассказывал\w*|сказал\w*|писал\w*)\s+(?:про|о|об|насчет)\b")
SYNONYMS = {"нрав": "люб", "обож": "люб", "любл": "люб", "любим": "люб", "работ": "работ", "учус": "уч", "учит": "уч",
            "учеб": "уч", "живу": "жив", "живе": "жив", "зову": "зов", "имя": "зов"}
NOTE = re.compile(r"\bзапомни\b\W*(?:что\s+)?(.{1,160}?)\s*[.!]?$")
ASK_NOTE = re.compile(r"\b(?:какое|какой|какую|что|какие)\s+(?:\w+\s+){0,2}?(?:я\s+)?(?:просил\w*|говорил\w*|сказал\w*|назвал\w*|загадал\w*)\s+(?:тебя\s+|тебе\s+)?(?:запомнить|запомни)\b"
                      r"|\bчто\s+(?:нужно|надо|ты\s+должна)\s+было\s+запомнить\b|\bчто\s+ты\s+запомнила\b"
                      r"|\bкакое\s+(?:слово|число)\s+я\s+(?:тебе\s+)?(?:загадал\w*|назвал\w*|говорил\w*|сказал\w*|просил\w*)")
ASK_OWN = re.compile(r"\bчто\s+ты\s+(?:говорила|сказала|рассказывала|писала|отвечала)\s+(?:про|о|об|насчет)\s+(.+?)\s*\??$"
                     r"|\bты\s+(?:же\s+)?(?:говорила|сказала),?\s+что\s+(.+?)\s*\??$")
ASK_TOPICS = re.compile(r"\bо\s+ч[её]м\s+(?:мы\s+)?(?:говорили|разговаривали|болтали|беседовали)\b|\bкакие\s+вопросы\s+(?:тебе\s+)?(?:задавали|были)\b")
TOPIC_NAMES = {"study": "учёбу", "work": "работу", "bot": "то, бот я или нет", "food": "еду", "family": "семью", "music": "музыку",
               "movies": "фильмы", "city": "Астану", "origin": "то, откуда я", "sleep": "сон", "love": "отношения", "home": "жильё",
               "future": "планы", "philosophy": "серьёзные вещи", "if": "всякие «если бы»", "memory": "память", "here": "то, как я здесь",
               "riddle": "загадки", "pets": "животных", "about": "меня", "hobby": "хобби", "sport": "спорт", "travel": "путешествия",
               "weather": "погоду", "mood": "настроение", "creative": "стихи и шутки", "know": "всякие факты", "kz": "языки",
               "friends": "друзей", "kursovaya": "курсовую", "senses": "ощущения", "joke": "шутки", "age": "возраст", "name": "имя"}


def stem(word: str) -> str:
    w = word.lower().replace("ё", "е")
    for key, value in SYNONYMS.items():
        if w.startswith(key):
            return value
    return w[:4] if len(w) >= 6 else w[:3]


def content(text: str):
    return [w for w in WORD.findall(text) if w.lower().replace("ё", "е") not in STOP and len(w) > 1]


class DialogMemory:
    def __init__(self):
        self.user = []      # [{"text", "turn"}]
        self.notes = []     # strings
        self.mira = []      # [{"text", "question", "topic", "turn"}]

    def observe_user(self, text: str, turn: int):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            note = NOTE.search(to_ty(sentence))
            if note and not ASK_NOTE.search(to_ty(sentence)):
                start = re.search(r"(?i)запомни\w*\W*(?:что\s+)?", sentence)
                self.notes.append(sentence[start.end():].strip(" .!") if start else note.group(1))
            if FIRST_PERSON.search(sentence) and not sentence.endswith("?"):
                self.user.append({"text": sentence, "turn": turn})

    def observe_mira(self, text: str, question: str, topic, turn: int):
        self.mira.append({"text": text, "question": question, "topic": topic, "turn": turn})

    # ------------------------------------------------------------------ recall
    def recall_note(self, question: str):
        if ASK_NOTE.search(to_ty(question)) and self.notes:
            note = re.sub(r"(?i)^(?:слово|число)\s*[:—-]?\s*", "", self.notes[-1])
            quoted = re.search(r"[«\"']([^»\"']+)[»\"']", note)
            return quoted.group(1).capitalize() if quoted else note[:1].upper() + note[1:]
        return None

    def recall_user(self, question: str, formal: bool, relaxed: bool = False):
        """relaxed: the question has no "я/мой" but clearly continues something the user said
        ("по какому предмету экзамен?" after "завтра у меня экзамен по физике")."""
        q = to_ty(question)
        if re.search(r"\b(?:ты|тебя|тебе|твой|твоя|твое|твои|твою|твоего)\b", q):
            return None     # a question about Mira, not about the asker
        if not RECALL_CUE.search(q):
            if not relaxed or not self.user or not re.search(r"\?|^\W*(?:что|какой|какая|какое|какую|какие|по\s+какому|на\s+какой|в\s+какой|где|куда|когда|сколько|кто|кого|с\s+кем)\b", q):
                return None
            q_long = {stem(w) for w in content(q) if len(w) >= 5}
            # without "я/мой" only a fresh statement counts, otherwise old notes leak into unrelated questions
            about_me = re.search(r"\b(?:я|мне|меня|мой|моя|мое|моё|мои|моего|моей|моих)\b", q)
            if not any(q_long & {stem(w) for w in content(item["text"])} for item in self.user[-8:] if about_me or item is self.user[-1]):
                return None
        not_told = "Вы мне этого не говорили" if formal else "Ты мне этого не говорил{|а}"
        if not self.user:
            first_person = re.search(r"\b(?:я|мне|меня|мной|мой|моя|мое|мои|моего|мою|моей|моих|мы|нас|наш\w*)\b", q)
            return not_told if STRONG_CUE.search(q) and first_person else None
        q_stems = {stem(w) for w in content(q)}
        best = None
        for item in reversed(self.user):
            clauses = []
            for clause in re.split(r",\s*(?:а|но|и)?\s*|\s+а\s+|\s+но\s+", item["text"]):
                clauses.append(clause)
                parts = re.split(r"\s+и\s+", clause)
                if len(parts) > 1:
                    clauses.extend(parts)
            for clause in clauses:
                words = content(clause)
                overlap = len(q_stems & {stem(w) for w in words})
                rest = len([w for w in words if stem(w) not in q_stems])
                rank = (overlap, -rest)
                if overlap and rest and (best is None or rank > best[0]):
                    best = (rank, clause, item["text"])
        if best is None:
            first_person = re.search(r"\b(?:я|мне|меня|мной|мой|моя|мое|мои|моего|мою|моей|моих|мы|нас|наш\w*)\b", q)
            return not_told if STRONG_CUE.search(q) and first_person else None
        said = "вы же говорили" if formal else "ты же говорил{|а}"
        if re.search(r"\bсколько\b", q):
            number = re.search(r"\b\d+\b", best[2])
            if number:
                years = " лет" if re.search(r"\bлет\b", q) else ""
                return f"{number.group(0)}{years}, {said}"
        clause = best[1]
        if re.search(r"\bс\s+кем\b", q):
            company = re.search(r"\bс[о]?\s+((?:[а-яё]+\s+)?[А-ЯЁа-яё][а-яё]+)\s*$|\bс[о]?\s+((?:[а-яё]+\s+)?[А-ЯЁ][а-яё]+)", clause)
            if company:
                return f"С {next(g for g in company.groups() if g)}, {said}"
        tokens = WORD.findall(clause)
        time_words = {"вчера", "сегодня", "завтра", "утром", "вечером", "днем", "ночью", "недавно", "позавчера"}
        keep = [i for i, w in enumerate(tokens) if w.lower().replace("ё", "е") not in STOP and stem(w) not in q_stems and len(w) > 1
                and (w.lower() not in time_words or re.search(r"\bкогда\b", q))]
        if keep and len(keep) <= 4 and not QUOTE_CUE.search(q):
            start, end = keep[0], keep[-1]
            if start > 0 and tokens[start - 1].lower() in {"в", "во", "на", "из", "с", "со", "у", "к", "до", "по"}:
                start -= 1
            slot = " ".join(tokens[start:end + 1])
            return f"{slot[:1].upper()}{slot[1:]}, {said}"
        quote = best[2].strip(" .!")
        return f"{'Вы говорили' if formal else 'Ты говорил{|а}'}: «{quote}»"

    def recall_own(self, question: str):
        q = to_ty(question)
        m = ASK_OWN.search(q)
        if not m or not self.mira:
            return None
        about = {stem(w) for w in content(next(g for g in m.groups() if g))}
        best = None
        for item in self.mira:
            score = len(about & {stem(w) for w in content(item["text"] + " " + item["question"])})
            if score and (best is None or score > best[0]):
                best = (score, item)
        if best is None:
            return "Не помню, чтобы я про это говорила"
        return f"Я говорила: «{best[1]['text']}»"

    def topics(self, question: str, hall: bool):
        if not ASK_TOPICS.search(to_ty(question)) or not self.mira:
            return None
        def listed(items):
            items = items[-4:]
            return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]

        names = []
        for item in self.mira:
            name = TOPIC_NAMES.get(item.get("topic") or "")
            if name and name not in names and item.get("topic") != "memory":
                names.append(name)
        told = []
        for item in self.user:
            low = item["text"].lower()
            for cue, name in ((r"работ", "работу"), (r"сестр|брат|мам|пап|семь", "семью"), (r"\bкот|кошк|собак|питом", "питомцев"),
                              (r"люблю|нравится|любим", "то, что ты любишь"), (r"живу|родом|родил", "то, откуда ты"),
                              (r"кино|фильм", "кино"), (r"учусь|универ|школ", "учёбу"), (r"зовут", "имя")):
                if re.search(cue, low) and name not in told:
                    told.append(name)
        if not names and not told:
            return "Да обо всём понемногу"
        parts = []
        if told:
            parts.append(f"{'Вы рассказывали' if hall else 'Ты рассказывал{|а}'} про {listed(told)}")
        if names:
            parts.append(f"{'спрашивали' if hall else 'спрашивал{|а}'} про {listed(names)}" if told else
                         f"{'Вы спрашивали' if hall else 'Ты спрашивал{|а}'} про {listed(names)}")
        return ", а ещё ".join(parts)
