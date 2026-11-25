"""Facts the user states about themselves, and answers to "do you remember ...".

Complements arena.memory (which already handles the user's name). Rules only;
the stored value is always the user's own words, never invented.
"""
from __future__ import annotations

import re

PETS = r"(кот\w*|кошк\w*|котик\w*|котенк\w*|котёнк\w*|пс\w*|собак\w*|пес\w*|пёс\w*|щен\w*|хомяк\w*|попуга\w*|черепах\w*|крыс\w*|рыбк\w*)"
PET_NAME = re.compile(r"(?:мо(?:его|ю|й|я|ему|им)|у меня|наш(?:его|у|а|)?)\s+(?:\w+\s+){0,2}?" + PETS +
                      r"(?:[^.!?]{0,25}?)\b(?:зовут|кличка|по имени|назвали)\s+([А-ЯЁA-Z][а-яёa-z\-]+|[а-яё\-]{3,})", re.IGNORECASE)
PET_NAME_REVERSED = re.compile(r"(?:зовут|кличка)\s+([А-ЯЁ][а-яё\-]+)[^.!?,]{0,20}?\b(?:мо(?:его|ю|й|я)|у меня)\s+" + PETS, re.IGNORECASE)
ORIGIN = re.compile(r"(?:\bя\s+(?:сам[аи]?\s+)?(?:из|родом из|родилс[яь] в|родилась в)|\bродом\s+из|\bродил(?:ся|ась)\s+в)\s+([А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+)?)", re.IGNORECASE)
LIVES = re.compile(r"\bя\s+(?:сейчас\s+)?живу\s+в\s+([А-ЯЁ][а-яё\-]+)", re.IGNORECASE)
STUDIES = re.compile(r"\bучусь\s+((?:в|на)\s+(?:[А-ЯЁA-Z][\wё\-]*|[а-яё\-]+)(?:\s+(?:на|в)\s+[а-яё\-]+|\s+[а-яё\-]+)?)", re.IGNORECASE)
WORKS = re.compile(r"\bработаю\s+([а-яё\-]+(?:\s+[а-яё\-]+){0,3})", re.IGNORECASE)
AGE = re.compile(r"\bмне\s+(\d{1,2})\s*(?:год|года|лет)?\b", re.IGNORECASE)
LIKES = re.compile(r"\bя\s+(?:очень\s+)?(?:люблю|обожаю)\s+((?:(?!\s+(?:и|а|но)\s+(?:не\s+люблю|ненавижу|терпеть))[^.,!?]){2,40})", re.IGNORECASE)
DISLIKES = re.compile(r"\b(?:я|и|а|но)\s+(?:вот\s+)?(?:не\s+люблю|терпеть\s+не\s+могу|ненавижу)\s+([^.,!?]{2,40})", re.IGNORECASE)

ASK_PET = re.compile(r"\b(?:как|помнишь)\b[^?]*\bзовут\b[^?]*\bмо(?:его|ю|й|я)\b\s+" + PETS +
                     r"|\bкличк\w*\b[^?]*\bмо(?:его|ю|й|я)\b\s+" + PETS +
                     r"|\bимя\s+(?:моего\s+|моей\s+)?" + PETS + r"|\b" + PETS + r"\s+(?:моего\s+)?как\s+зовут", re.IGNORECASE)
ASK_ORIGIN = re.compile(r"\bоткуда\s+я\b|\bа\s+я\s+откуда\b|\bгород\s+мой\b|\bмой\s+город\b|\bгде\s+я\s+родил\w*|\bиз\s+какого\s+(?:я\s+)?города\s+я\b|\bмой\s+родной\s+город\b", re.IGNORECASE)
ASK_LIVES = re.compile(r"\bгде\s+я\s+живу\b", re.IGNORECASE)
ASK_STUDIES = re.compile(r"\bгде\s+я\s+учусь\b|\bна\s+кого\s+я\s+учусь\b", re.IGNORECASE)
ASK_WORK = re.compile(r"\bгде\s+я\s+работаю\b|\bкем\s+я\s+работаю\b", re.IGNORECASE)
ASK_AGE = re.compile(r"\bсколько\s+мне\s+лет\b", re.IGNORECASE)
ASK_LIKES = re.compile(r"\bчто\s+я\s+(?:люблю|обожаю)\b", re.IGNORECASE)
ASK_DISLIKES = re.compile(r"\bчто\s+я\s+(?:не\s+люблю|ненавижу|терпеть\s+не\s+могу)\b", re.IGNORECASE)
ASK_SUMMARY = re.compile(r"\bчто\s+(?:ты\s+)?(?:знаешь|помнишь|запомнила)\s+(?:обо\s+мне|про\s+меня)\b|\bчто\s+(?:ты\s+)?(?:обо\s+мне|про\s+меня)\s+(?:знаешь|помнишь|запомнила)\b|\bрасскажи\s+(?:обо\s+мне|про\s+меня)\b|\bчто\s+я\s+(?:о\s+себе\s+)?(?:рассказал\w*|говорил\w*)\s*\??$", re.IGNORECASE)

COLOR = re.compile(r"(?i:мой\s+любимый\s+цвет|любимый\s+цвет\s+у\s+меня)\s*[-—:]?\s*(?:это\s+)?([а-яёА-ЯЁ]+)")
ASK_COLOR = re.compile(r"\bкакой\s+(?:я\s+)?цвет\s+(?:я\s+)?люблю\b|\bмой\s+любимый\s+цвет\s*\?|\bкакой\s+у\s+меня\s+любимый\s+цвет\b", re.IGNORECASE)
LIKES_AND = re.compile(r"\bи\s+(?:очень\s+)?(?:люблю|обожаю)\s+([^.,!?]{2,40})", re.IGNORECASE)
ASK_FOOD = re.compile(r"\bлюбимая\s+еда\b[^?]*\bу\s+меня\b|\bчто\s+я\s+люблю\s+(?:есть|кушать)\b|\bкакая\s+у\s+меня\s+любимая\s+еда\b|\bмоя\s+любимая\s+еда\s*\?", re.IGNORECASE)
CITY_GEN = {"шымкент": "Шымкента", "караганда": "Караганды", "караганды": "Караганды", "семей": "Семея", "павлодар": "Павлодара",
            "астана": "Астаны", "костанай": "Костаная", "тараз": "Тараза", "уральск": "Уральска", "петропавловск": "Петропавловска",
            "усть-каменогорск": "Усть-Каменогорска", "туркестан": "Туркестана", "кызылорда": "Кызылорды", "талдыкорган": "Талдыкоргана",
            "экибастуз": "Экибастуза", "жезказган": "Жезказгана", "актобе": "Актобе", "атырау": "Атырау", "актау": "Актау",
            "кокшетау": "Кокшетау", "алматы": "Алматы", "темиртау": "Темиртау", "бишкек": "Бишкека", "ташкент": "Ташкента",
            "москва": "Москвы", "кокшетауа": "Кокшетау"}


def from_city(value):
    """'Шымкент' -> 'Шымкента' (genitive for "из ..."); unknown forms stay as the user wrote them."""
    return CITY_GEN.get(value.lower().replace("ё", "е"), value)


USER_NAME = re.compile(r"(?i:меня\s+зовут|зовут\s+меня|мо[её]\s+имя|зови\s+меня|называй\s+меня|можно\s+просто)\s*[-—:]?\s*(?:это\s+)?([А-ЯЁ][а-яё]+|[а-яё]{3,}(?=\W*$))")
NOT_NAMES = {"так", "просто", "никак", "все", "всё", "это", "тоже", "типа", "вообще", "как", "кто", "что"}
USER_NAME_SHORT = re.compile(r"(?i:знакомиться|знакомы|познакомимся)\W+я\s+([А-ЯЁ][а-яё]+)|^\W*(?i:я)\s+([А-ЯЁ][а-яё]+)\W*$")
STUDY_CITY = re.compile(r"(?i:учусь)\b[^.!?]{0,40}?\b(?i:в)\s+([А-ЯЁ][а-яё]+)(?![\wА-ЯЁ])")
ASK_STUDY_CITY = re.compile(r"\bв\s+каком\s+городе\s+я\s+учусь\b|\bгде\s+я\s+учусь\b", re.IGNORECASE)
DEFINE = re.compile(r"(?i:запомни|запиши|знай|кстати)?\W*(?i:слово|термин)?\W*[«\"']?([а-яёa-z-]{2,})[»\"']?\s*(?:[-—:]\s*(?:это\s+)?(?:значит\s+|означает\s+)?|\s+(?:это\s+(?:значит|означает|когда)|это|значит|означает)\s+)(.{2,160}?)\s*[.!]?$")
MEMO = re.compile(r"^\W*(?i:запомни|запиши|не\s+забудь)\W*(?:(?i:что)\s+)?(.{2,160}?)\s*[.!]?$")
ASK_DEF = re.compile(r"(?i:что\s+(?:такое|значит|означает)|помнишь,?\s+что\s+(?:такое|значит))\s+[«\"']?([а-яёa-z-]{2,})[»\"']?|^\W*[«\"']?([а-яёa-z-]{2,})[»\"']?\s+(?i:это\s+что|что\s+это|—\s+это\s+что)\W*$|^\W*[«\"']?([а-яёa-z-]{2,})[»\"']?\s*\??\s*$")
ASK_MEMO = re.compile(r"(?i:что\s+я\s+(?:просил\w*|говорил\w*)\s+(?:тебя\s+)?запомнить|помнишь,?\s+что\s+я\s+просил\w*\s+(?:тебя\s+)?запомнить)")
ASK_NAME = re.compile(r"\bкак\s+меня\s+зовут\b|\bпомнишь\W+как\s+меня\b|\bмо[её]\s+имя\b", re.IGNORECASE)
_PAST = r"(?:[иаяеыуоё]л(?:ся)?)"
MASC = re.compile(r"\b[а-яё]{2,}" + _PAST + r"\b|\b(?:рад|готов|уверен|согласен|должен|сам|один|занят|свободен)\b")
FEM = re.compile(r"\b[а-яё]{2,}[иаяеыуоё]л(?:а|ась)\b|\b(?:рада|готова|уверена|согласна|должна|сама|одна|занята|свободна)\b")
NOT_VERB = re.compile(r"\b(?:зал|бал|канал|финал|сигнал|материал|идеал|подвал|вокзал|журнал|сериал|аппетит|стол|пол|футбол|волейбол|баскетбол|"
                      r"угол|орел|осел|котел|козел|дятел|пепел|минерал|генерал|сила|пила|школа|пчела|игла|скала|вилла|дела|мила|юла|"
                      r"смола|метла|стрела|ангела|зола|глагола)\b")
THIRD = re.compile(r"\b(?:он|она|они|ты|вы|мама|папа|брат|сестра|друг|подруга|кот|кошка|собака|сосед|соседи|он[аи]?|кто|все|бабушка|дедушка)\b")


# Russian genitive for "из <город>" is not derivable by rules; keep the user's own form.


class UserFacts:
    def __init__(self):
        self.facts = {}
        self.gender_votes = 0

    @property
    def gender(self):
        """'m', 'f' or None, from the user's own first-person verb forms."""
        if self.gender_votes > 0:
            return "m"
        if self.gender_votes < 0:
            return "f"
        return None

    def _vote_gender(self, text):
        for sentence in re.split(r"[.!?\n]+", text.lower().replace("ё", "е")):
            sentence = NOT_VERB.sub(" ", sentence)
            if not sentence.strip() or THIRD.search(sentence):
                continue
            if FEM.search(sentence):
                self.gender_votes -= 1
            elif MASC.search(sentence):
                self.gender_votes += 1

    def observe(self, text):
        text = str(text)
        self._vote_gender(text)
        match = USER_NAME.search(text)
        short = USER_NAME_SHORT.search(text)
        if match or short:
            name = match.group(1) if match else (short.group(1) or short.group(2))
        if (match or short) and name.lower() not in NOT_NAMES:
            self.facts["name"] = name[:1].upper() + name[1:]
            if self.gender_votes == 0 and re.search(r"[ая]$", self.facts["name"]) and self.facts["name"].lower() not in {
                    "илья", "никита", "саша", "женя", "миша", "дима", "ваня", "коля", "петя", "лёша", "леша", "кузьма", "фома"}:
                self.gender_votes = -1
        for match in PET_NAME.finditer(text):
            self.facts.setdefault("pets", {})[match.group(1).lower()[:3]] = match.group(2)
        for match in ([] if PET_NAME.search(text) else PET_NAME_REVERSED.finditer(text)):
            self.facts.setdefault("pets", {})[match.group(2).lower()[:3]] = match.group(1)
        for key, pattern in (("origin", ORIGIN), ("lives", LIVES), ("studies", STUDIES), ("works", WORKS)):
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if key == "studies" and re.match(r"(?i)на\s+\w+(?:ом|ем)\b(?:\s+курсе)?$", value):
                    continue  # "на втором курсе" is a year, not a speciality
                self.facts[key] = value
        match = COLOR.search(text)
        if match:
            self.facts["color"] = match.group(1).lower()
        for match in LIKES_AND.finditer(text):
            value = match.group(1).strip()
            if value and value not in self.facts.setdefault("likes", []):
                self.facts["likes"].append(value)
        match = STUDY_CITY.search(text)
        if match:
            self.facts["place"] = match.group(1)
        match = AGE.search(text)
        if match:
            self.facts["age"] = match.group(1)
        for key, pattern in (("likes", LIKES), ("dislikes", DISLIKES)):
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and value not in self.facts.setdefault(key, []):
                    self.facts[key].append(value)

    def remember_definition(self, text):
        """'запомни слово: X это значит Y' -> stores X; returns an acknowledgement or None."""
        if not re.search(r"(?i)\b(?:запомни|запиши|знай)\b", text):
            return None
        match = None if re.search(r"(?i)\bзовут\b|\bя\s+из\b|\bлюблю\b|\bмой\s+любимый\b", text) else DEFINE.search(text)
        if match and match.group(1).lower() not in {"что", "это", "меня", "мне", "я", "ты", "слово"}:
            self.facts.setdefault("definitions", {})[match.group(1).lower().replace("ё", "е")] = match.group(2).strip()
            return "Запомнила) Интересное слово"
        memo = MEMO.search(text)
        if memo:
            self.facts.setdefault("memos", []).append(memo.group(1).strip())
            options = ["Так, записала в голове)", "Запомнила)"]
            if re.search(r"(?i)\bкот|\bкошк", text):
                options.append("Хорошо, кот — главное ахах")
            return options[len(self.facts["memos"]) % len(options)]
        return None

    def summary(self):
        """Everything the user told about themselves, in one short message (or None)."""
        f, parts = self.facts, []
        if f.get("name"):
            parts.append(f"тебя зовут {f['name']}")
        if f.get("age"):
            parts.append(f"тебе {f['age']}")
        if f.get("origin"):
            parts.append(f"ты из {f['origin']}")
        if f.get("studies"):
            parts.append(f"учишься {f['studies']}")
        if f.get("works"):
            parts.append(f"работаешь {f['works']}")
        for kind, name in (f.get("pets") or {}).items():
            word = {"кот": "кот", "кош": "кошка", "соб": "собака", "пес": "пёс", "пёс": "пёс", "хом": "хомяк", "поп": "попугай"}.get(kind, "питомец")
            parts.append(f"у тебя {word} {name}")
        if f.get("color"):
            parts.append(f"любимый цвет {f['color']}")
        if f.get("likes"):
            parts.append("любишь " + f["likes"][0])
        if not parts:
            return None
        text = ", ".join(parts[:7])
        return text[:1].upper() + text[1:] + ". Пока всё, что успел{|а} рассказать)"

    def recall(self, question):
        q = str(question)
        if ASK_SUMMARY.search(q):
            return self.summary() or "Пока почти ничего, ты о себе не рассказывал{|а})"
        definitions = self.facts.get("definitions", {})
        if definitions:
            match = ASK_DEF.search(q)
            if match:
                word = next(g for g in match.groups() if g).lower().replace("ё", "е")
                if word in definitions:
                    return f"Ты же говорил{{|а}}: {word} — это {definitions[word]})"
        if ASK_MEMO.search(q):
            if self.facts.get("memos"):
                return f"Ты просил{{|а}} запомнить: «{self.facts['memos'][-1]}»)"
            if definitions:
                word, meaning = list(definitions.items())[-1]
                return f"Слово «{word}» — это {meaning})"
        if ASK_NAME.search(q) and ASK_COLOR.search(q) and self.facts.get("name") and self.facts.get("color"):
            return f"{self.facts['name']}, а цвет {self.facts['color']}"
        if ASK_COLOR.search(q) and self.facts.get("color"):
            return f"{self.facts['color'].capitalize()}, ты же говорил{{|а}})"
        if ASK_FOOD.search(q) and self.facts.get("likes"):
            food = self.facts["likes"][-1]
            return f"{food[0].upper()}{food[1:]} любишь) Видишь, всё помню"
        if ASK_NAME.search(q) and self.facts.get("name"):
            return f"{self.facts['name']}, конечно)"
        match = ASK_PET.search(q)
        if match:
            kind = next(g for g in match.groups() if g)
            name = self.facts.get("pets", {}).get(kind.lower()[:3])
            if name:
                return f"{name[0].upper()}{name[1:]}, конечно)"
            return None
        if ASK_STUDY_CITY.search(q) and self.facts.get("place"):
            return f"В {self.facts['place']} же)"
        if ASK_ORIGIN.search(q) and self.facts.get("origin"):
            return f"Из {from_city(self.facts['origin'])})"
        if ASK_LIVES.search(q) and self.facts.get("lives"):
            return f"Ты же живёшь в {self.facts['lives']}"
        if ASK_STUDIES.search(q) and self.facts.get("studies"):
            return f"Ты же учишься {self.facts['studies']}"
        if ASK_WORK.search(q) and self.facts.get("works"):
            return f"Ты же работаешь {self.facts['works']}"
        if ASK_AGE.search(q) and self.facts.get("age"):
            return f"{self.facts['age']}, если я ничего не путаю"
        if ASK_DISLIKES.search(q) and self.facts.get("dislikes"):
            return f"{self.facts['dislikes'][-1][0].upper()}{self.facts['dislikes'][-1][1:]} точно не любишь)"
        if ASK_LIKES.search(q) and self.facts.get("likes"):
            return f"Ты же любишь {self.facts['likes'][-1]})"
        return None
