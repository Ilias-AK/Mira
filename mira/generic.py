"""Generic human-like handlers that work for unseen questions.

All rules, no model:
- favourites ("какой твой любимый цвет?", "... а у тебя?") from persona.json
- either/or choices ("кофе или чай?") — picks one, using persona preferences
- yes/no about Mira ("ты умеешь плавать?") — first-person echo of the verb,
  remembered so the same question gets the same polarity
- opinion and "why" questions — short committed human hedges
- self-introduction by the user — greets by name, reacts to city/pet
"""
from __future__ import annotations

import hashlib
import re

from mira.bank import norm

VERB_1SG = {
    "знаешь": "знаю", "умеешь": "умею", "любишь": "люблю", "смотришь": "смотрю", "слушаешь": "слушаю",
    "читаешь": "читаю", "играешь": "играю", "ходишь": "хожу", "ездишь": "езжу", "веришь": "верю",
    "пьешь": "пью", "ешь": "ем", "хочешь": "хочу", "можешь": "могу", "занимаешься": "занимаюсь",
    "катаешься": "катаюсь", "рисуешь": "рисую", "поешь": "пою", "танцуешь": "танцую", "готовишь": "готовлю",
    "помнишь": "помню", "понимаешь": "понимаю", "бываешь": "бываю", "куришь": "курю", "водишь": "вожу",
    "носишь": "ношу", "спишь": "сплю", "говоришь": "говорю", "увлекаешься": "увлекаюсь", "болеешь": "болею",
    "учишь": "учу", "собираешься": "собираюсь", "планируешь": "планирую", "скучаешь": "скучаю",
    "боишься": "боюсь", "любила": "любила", "мечтала": "мечтала",
}
PAST_FEM = re.compile(r"^[а-я]{2,}(?:ла|лась)$")
OPINION_VERBS = {"думаешь", "считаешь", "относишься", "полагаешь"}
YESNO = re.compile(r"^\W*(?:а\s+)?(?:ты\s+)(?:бы\s+)?(?:вообще\s+|когда-нибудь\s+|часто\s+|сама\s+|уже\s+|тоже\s+|хоть\s+раз\s+|еще\s+)?(?:бы\s+)?([а-я]+)\b(.*)$")
YESNO_PAST = re.compile(r"^\W*(?:а\s+)?(?:сама\s+)?([а-я]{2,}(?:ла|лась))\s*(бы)?\b(.*)$")
HAVE = re.compile(r"^\W*(?:а\s+)?(?:есть\s+ли\s+у\s+тебя\s+|у\s+тебя\s+(?:есть|был|была|были)\s+)(.+?)\s*\?")
YESNO_NO_TY = re.compile(r"^\W*(?:а\s+)?(?:вообще\s+|когда-нибудь\s+)?([а-я]+(?:ешь|ишь|ешься|ишься))\b(.*)$")
EITHER = re.compile(r"((?:(?:никогда|всегда|не)\s+){0,2}[а-яa-z-]+)\s+или\s+([а-яa-z-]+(?:\s+[а-яa-z-]+){0,2})\s*(?:[?,.!]|$)")
OPINION = re.compile(r"\b(?:как\s+(?:ты\s+)?(?:думаешь|считаешь|относишься)|что\s+(?:ты\s+)?(?:думаешь|скажешь)\s+(?:о|об|про)|твое\s+мнение|как\s+тебе)\b")
WHY = re.compile(r"^\W*(?:а\s+|ого,?\s+а\s+|и\s+)?(?:почему|зачем|с\s+чего)\b")
FAV = re.compile(r"\b(?:любим\w+|самый\s+любимый|больше\s+всего\s+нравится)\b")
BACK = re.compile(r"\bа\s+(?:у\s+)?(?:тебя|ты|тебе)\s*\??\s*$")


def _hash(*parts) -> int:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[0]


def infinitive_to_past_fem(word: str) -> str | None:
    irregular = {"пойти": "пошла", "идти": "шла", "прийти": "пришла", "уйти": "ушла", "есть": "поела",
                 "ехать": "поехала", "остаться": "осталась", "забить": "забила", "написать": "написала"}
    if word in irregular:
        return irregular[word]
    if word.endswith("ться"):
        return word[:-4] + "лась"
    if word.endswith("ть"):
        return word[:-2] + "ла"
    return None


class Generic:
    def __init__(self, profile: dict):
        self.profile = profile
        self.favorites = profile.get("favorites", {})
        self.preferences = {norm(k): v for k, v in profile.get("preferences", {}).items()}
        yn = profile.get("yes_no", {})
        self.known_true = [norm(x) for x in yn.get("true", [])]
        self.known_false = [norm(x) for x in yn.get("false", [])]
        self.said_polarity = {}
        self.last_pair = None

    def reset(self):
        self.said_polarity = {}
        self.last_pair = None

    def preference(self, word: str) -> int:
        word = norm(word)
        if word in self.preferences:
            return self.preferences[word]
        for key, value in self.preferences.items():
            if len(word) >= 3 and len(key) >= 3 and key[:4] == word[:4]:
                return value
        return 1

    def remember_pair(self, question: str):
        q = norm(question)
        m = re.search(r"([а-я-]+)\s+(?:намного\s+|гораздо\s+)?лучше\s+([а-я-]+)", q) or EITHER.search(q)
        if m:
            self.last_pair = (m.group(1), m.group(2).split()[0])

    def pick_from_pair(self, question: str, salt=0) -> str | None:
        q = norm(question)
        if not self.last_pair or not re.search(r"\b(?:что|кого|какое|какой)\s+(?:ты\s+)?(?:больше|сама\s+больше)\s+(?:любишь|нравится|выбираешь)|\bа\s+ты\s+(?:что\s+)?выбираешь\b|\bа\s+тебе\s+что\b", q):
            return None
        left, right = self.last_pair
        lv, rv = self.preference(left), self.preference(right)
        choice = left if lv >= rv else right
        base = {"зимы": "зиму", "лета": "лето", "весны": "весну", "осени": "осень", "кошек": "кошек", "собак": "собак"}.get(choice, choice)
        return ["Я больше {}) ".format(base) + "Хотя у каждого свои плюсы", "Наверное, {})".format(base)][_hash(q, salt) % 2]

    # -------------------------------------------------------------- favourites
    def favorite(self, question: str, salt=0) -> str | None:
        q = norm(question)
        if not (FAV.search(q) or (BACK.search(q) and re.search(r"\bмо[йяеи]\s+любим", q))):
            return None
        short_ask = re.match(r"^\W*(?:а\s+)?(?:тво[йяеи]\s+)?(?:самы[йея]\s+)?любим\w*\s+\w+\W*$", q)
        if not short_ask and "?" not in q and not BACK.search(q) and not re.search(r"\b(?:какой|какая|какое|какие|назови|скажи|расскажи)\b", q):
            return None   # "мой любимый фильм — X" is the asker's fact, not a question to Mira
        best = None
        for key, item in self.favorites.items():
            for stem in item.get("stems", []):
                if re.search(r"\b" + re.escape(norm(stem)), q) and (best is None or len(stem) > best[0]):
                    best = (len(stem), item)
        if best is None:
            return None
        self.last_fav_key = next(k for k, v in self.favorites.items() if v is best[1])
        answers = best[1]["answers"]
        return answers[_hash(q, salt) % len(answers)]

    # ------------------------------------------------------------------ either
    def either(self, question: str, salt=0) -> str | None:
        q = norm(question)
        if re.search(r"\bили\s+(?:нет|не)\b", q):
            return None
        m = EITHER.search(q)
        if not m:
            return None
        left, right = m.group(1), m.group(2).strip()
        if left in {"и", "а", "но", "что", "то", "это", "ты", "я"}:
            return None
        past = infinitive_to_past_fem(right.split()[0])
        if past and right.split()[0].endswith(("ть", "ться", "ти")):
            return ["Я бы {}) Но решать тебе".format(past + (" " + " ".join(right.split()[1:]) if len(right.split()) > 1 else "")),
                    "Я бы, наверное, {})".format(past)][_hash(q, salt) % 2]
        lv, rv = self.preferences.get(left, 1), self.preferences.get(right.split()[0], 1)
        choice = left if lv > rv else right if rv > lv else (left, right)[_hash(q, salt) % 2]
        word = choice[0].upper() + choice[1:]
        templates = ["{}, конечно)", "{}, наверное)", "Однозначно {}", "{}) Даже думать не надо"]
        text = templates[_hash(q, salt, 1) % len(templates)]
        return text.format(word if not text.startswith("Однозначно") else choice)

    # ------------------------------------------------------------------ yes/no
    def yes_no(self, question: str, salt=0) -> str | None:
        q = norm(question)
        first = re.split(r"(?<=\?)", q)[0]
        if not first.endswith("?") or re.search(r"\b(?:что|как|какой|какая|какое|какие|какого|каком|где|когда|куда|откуда|почему|зачем|сколько|кто|чем|кого|кому)\b(?!-)", first):
            return None
        q = first
        about = re.match(r"^\W*(?:а\s+)?(?:тебя|тебе)\s+(?:это\s+)?(не\s+)?([а-я]+(?:ет|ит|ят|ут|ют))\b", q)
        if about:
            verb = about.group(2)
            positive = _hash(q, "polarity") % 100 < 55
            cap = verb[0].upper() + verb[1:]
            return [f"{cap} немного, если честно)", f"Ну, {verb} чуть-чуть)"][_hash(q, salt) % 2] if positive else \
                   [f"Не особо {verb}) А тебя?", "Да нет, не особо"][_hash(q, salt) % 2]
        have = HAVE.match(q)
        if have:
            thing = have.group(1)
            table = {norm(k): v for k, v in self.profile.get("has", {}).items()}
            known = next((v for k, v in table.items() if thing.startswith(k) or k.startswith(thing.split()[0][:5])), None)
            positive = known if known is not None else _hash(thing, "has") % 100 < 50
            return (["Есть) ", "Да, есть) "][_hash(q, salt) % 2] + (table.get("_detail_" + thing.split()[0], "") or "")).strip() if positive \
                else ["Нет, пока нет)", "Нет(", "Не-а)"][_hash(q, salt) % 3]
        conditional = bool(re.search(r"\bбы\b", q))
        m = YESNO.match(q) or YESNO_NO_TY.match(q) or YESNO_PAST.match(q)
        if not m:
            return None
        verb, rest = m.group(1), m.groups()[-1].strip(" ?!.,")
        if conditional and PAST_FEM.match(verb):
            positive = _hash(q, "polarity") % 100 < 70
            cap = verb[0].upper() + verb[1:]
            return [f"{cap} бы, наверное)", f"{cap} бы, конечно)"][_hash(q, salt) % 2] if positive else \
                   ["Не знаю... наверное, нет)", "Хм, вряд ли)"][_hash(q, salt) % 2]
        if verb in OPINION_VERBS or verb in {"бот", "робот", "человек", "живая", "серьезно"}:
            return None
        if verb in VERB_1SG:
            mine = VERB_1SG[verb]
        elif PAST_FEM.match(verb):
            mine = verb
        elif re.search(r"(?:аешь|яешь|еешь|уешь)$", verb):
            mine = verb[:-3] + "ю"
        else:
            return None
        phrase = norm(f"{mine} {rest}".strip())
        key = phrase[:40]
        if key in self.said_polarity:
            positive = self.said_polarity[key]
        elif any(phrase.startswith(x) or x.startswith(phrase) for x in self.known_true if x.split()[0] == mine):
            positive = True
        elif any(phrase.startswith(x) or x.startswith(phrase) for x in self.known_false if x.split()[0] == mine):
            positive = False
        else:
            positive = _hash(phrase, "polarity") % 100 < 65
        self.said_polarity[key] = positive
        cap = mine[0].upper() + mine[1:]
        if positive:
            options = [f"{cap}, но не очень хорошо)", f"Да, {mine}) А ты?", f"{cap}, конечно)", f"Ну да, {mine}. Иногда)"]
            if PAST_FEM.match(mine):
                options = [f"{cap}, один раз)", f"Да, {mine}) Давно, правда", f"{cap}, конечно) А ты?"]
        else:
            options = [f"Не {mine}, если честно)", f"Нет, не особо", f"Не-а) А ты?"]
            if PAST_FEM.match(mine):
                options = [f"Не {mine} пока(", f"Нет, ни разу) А ты?"]
        return options[_hash(q, salt) % len(options)]

    # ----------------------------------------------------------------- opinion
    def opinion(self, question: str, salt=0) -> str | None:
        q = norm(question)
        if not OPINION.search(q):
            return None
        options = ["Неоднозначно) В чём-то хорошо, в чём-то не очень", "Нормально отношусь, без фанатизма)",
                   "Мне кажется, тут всё от человека зависит", "Честно, не задумывалась. Но скорее положительно)"]
        return options[_hash(q, salt) % len(options)]

    def why(self, question: str, salt=0) -> str | None:
        q = norm(question)
        if not WHY.search(q) or len(q.split()) > 7:
            return None
        if not re.search(r"\b(?:ты|тебя|тебе|так|именно|передумала|бросила|решила|выбрала|ушла|не\s+стала)\b|^\W*(?:а\s+)?почему\W*$", q):
            return None   # a question about the world, not about Mira: "Ну так вышло" would be absurd
        options = ["Даже не знаю, просто так сложилось)", "Не знаю, как-то само получилось)",
                   "Ну так вышло) Долгая история", "Хм, просто захотелось)"]
        return options[_hash(q, salt) % len(options)]

    # ------------------------------------------------------------ introduction
    def introduction(self, question: str, facts_before: dict, facts_after: dict, salt=0) -> str | None:
        """React to new facts the user just told about themselves."""
        new = {k: v for k, v in facts_after.items() if facts_before.get(k) != v}
        if not new or question.strip().endswith("?") and not BACK.search(norm(question)):
            return None
        parts = []
        if "name" in new and facts_before.get("name"):
            parts.append("Ладно, тогда {}".format(new["name"]))
        elif "name" in new and re.search(r"\bзапомни\b", norm(question)):
            parts.append("Хорошо, {}".format(new["name"]))
        elif "name" in new:
            parts.append(["Приятно познакомиться, {}) Я Мира".format(new["name"]),
                          "Очень приятно, {}) Меня Мира зовут".format(new["name"])][_hash(question, salt) % 2])
        if "color" in new and not parts:
            parts.append("Это тоже запомнила" if facts_before.get("name") else ["Запомнила)", "О, а мой синий)"][_hash(question, salt) % 2])
        pets = new.get("pets") or {}
        if pets:
            kind, name = next(iter(pets.items()))
            parts.append(["{} — классное имя)".format(name), "{}, ахах, мило)".format(name)][_hash(name) % 2])
        if "origin" in new and "name" not in new:
            parts.append(["О, а я из Кокшетау)", "Ого, ни разу там не была)"][_hash(question, salt) % 2])
        if "studies" in new:
            value = new["studies"]
            if norm(value).startswith("на "):
                went = "пошла" if facts_after.get("_gender") == "f" else "пошёл"
                parts.append(f"А почему тогда {went} {value}?" if re.search(r"\bхочу\b", norm(question)) else f"О, {value}) Нравится?")
            elif re.search(r"\sна\s", value):
                parts.append(f"О, {value}. А я в ЕНУ, на учёте и аудите)")
            else:
                parts.append("А на кого учишься?")
        elif "origin" in new and len(parts) < 2:
            parts.append("Давно оттуда уехал{}?".format("а" if facts_after.get("_gender") == "f" else ""))
        if "age" in new and not parts:
            mine = int(self.profile.get("age", 20))
            theirs = int(new["age"])
            parts.append(f"О, так ты младше меня) Мне {mine}" if theirs < mine - 1 else
                         f"О, почти ровесники) Мне {mine}" if theirs <= mine + 1 else f"О, ты старше) Мне {mine}")
        if "likes" in new and not parts and not BACK.search(norm(question)):
            parts.append("Ого, классно)")
        parts = [p if re.search(r"[.!?)(]$", p) else p + "." for p in parts]
        return " ".join(parts) if parts else None
