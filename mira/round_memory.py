"""Память раунда: списки, числа, кодовые слова и порядок реплик.

Мира не придумывает здесь новых фраз: она повторяет слова собеседника или свои прошлые ответы
в уже существующих формах («…, вы же говорили», «Вы спрашивали: «…»», «Я ответила: «…»»,
«Запомнила»), а числа и длины считает. Всё это — факты раунда, а не реплики персонажа.
"""
from __future__ import annotations

import re

ORDINALS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10,
}
NUMBER_WORDS = {
    "ноль": 0, "один": 1, "одну": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "пятьдесят": 50, "сто": 100,
}
ORD = r"(перв\w*|втор\w*|трет\w*|четв[её]рт\w*|пят\w*|шест\w*|седьм\w*|восьм\w*|девят\w*|десят\w*|последн\w*|прошл\w*|предыдущ\w*|\d+)"

LIST_DECL = re.compile(r"(?:запомни\w*|задам|назову|вот|скажу)\b[^:]{0,40}:\s*(.+)$")
NUMBER_DECL = re.compile(r"\bзапомни\w*\s+(?:число|цифру|номер)\s+(-?\d+)")
CODE_DECL = re.compile(r"\b(?:пароль|кодовое\s+слово|код|секретное\s+слово)\b[^«\"]{0,30}[«\"]([^»\"]{1,40})[»\"]")
CODE_DECL_DASH = re.compile(r"\b(?:пароль|кодовое\s+слово|секретное\s+слово)\b[^—:-]{0,20}[—:-]\s*([а-яёa-z0-9-]{2,30})")
NUMBER_OP = re.compile(r"\b(прибав\w*|добав\w*|отним\w*|вычт\w*|умнож\w*|раздел\w*|подели\w*)\b[^0-9а-я]*(?:к\s+нему\s+|из\s+него\s+|его\s+на\s+|на\s+)?(-?\d+|[а-я]+)")

ITEM_AT = re.compile(r"\b(?:что|какое|какая|какой|кто)\b[^?]{0,25}?\b(?:было|был|была|шло|шла|шел|стояло|стоял|стояла|идет|идёт)?\s*" + ORD + r"\b"
                     r"|^\W*(?:а\s+)?(?:какое|какая|какой|что)?\s*" + ORD + r"\W*$|^\W*а\s+" + ORD + r"\W*$")
WHOLE_LIST = re.compile(r"\bповтори\w*\s+(?:весь\s+|полный\s+)?список\b|\bперечисли\w*\s+(?:исходные\s+|все\s+)?(?:темы|слова|имена|пункты|все)\b"
                        r"|\bназови\w*\s+(?:оба|все|обе)\s+\w+|\bназови\w*\s+новый\s+порядок\b|\bкакой\s+(?:сейчас\s+|теперь\s+)?порядок\b"
                        r"|\bпорядок\s+был\s+исходным\b|\bисходн\w+\s+(?:порядок|список|темы|слова)\b")
APPEND = re.compile(r"\bдобав\w*\s+(?:в\s+конец|в\s+список|ещ[её])\s+(?:слово\s+|имя\s+)?[«\"]?([а-яёa-z0-9-]{2,30})")
SWAP = re.compile(r"\bпомен\w*\s+(?:их\s+)?местами\b")
CODE_Q = re.compile(r"\bкако\w+\s+(?:было\s+)?(?:кодовое\s+слово|пароль|код|секретное\s+слово)\b|\bназови\w*\s+(?:его|слово|пароль)\s*(?:снова|ещ[её]\s+раз)?\b|\bкакое\s+слово\s+я\s+(?:просил\w*\s+)?запомнить")
LETTERS_Q = re.compile(r"\bсколько\s+в\s+н[её]м\s+букв\b|\bсколько\s+букв\b")
NUMBER_Q = re.compile(r"\bкакое\s+число\s+было\s+(исходным|первым|сначала|в\s+начале)\b|\bисходное\s+число\b|\bрезультат\w*\b|\bоба\s+числа\b|\bвсе\s+числа\b")
USER_Q_AT = re.compile(r"\b(?:повтори\w*|сформулируй\w*|напомни\w*|назови\w*)\b[^?]{0,30}\b(?:мой|мо[её])\s+(?:самый\s+)?" + ORD + r"\s+вопрос"
                       r"|\b(?:как\w*|что)\s+(?:был\w*\s+)?(?:мо[йяеё])\s+" + ORD + r"\s+(?:вопрос|команд\w*|просьб\w*|сообщени\w*|реплик\w*|фраз\w*)"
                       r"|\b(?:повтори\w*|сформулируй\w*)\s+(?:самый\s+)?" + ORD + r"\s+(?:мой\s+)?вопрос"
                       r"|\bкакой\s+(?:был\s+)?(?:самый\s+)?" + ORD + r"\s+вопрос"
                       r"|\bчто\s+я\s+спросил\w*\s+(?:во|в|на)\s+" + ORD + r"\s+(?:вопросе|раз)"
                       r"|\bкакой\s+вопрос\s+я\s+задал\w*\s+(?:во|в|на)\s+" + ORD)
USER_Q_FIRST = re.compile(r"\bчто\s+я\s+спросил\w*\s+(?:сначала|в\s+начале|первым)\b"
                          r"|\b(?:что|какой\s+факт)\s+я\s+(?:сказал|сообщил|назвал)\w*\s+(?:первым|сначала|в\s+начале)\b"
                          r"|\bповтори\w*\s+исходн\w+\s+(?:ситуаци\w*|услови\w*|фраз\w*|утверждени\w*)|\bчто\s+я\s+(?:попросил\w*|спросил\w*)\s+(?:сделать\s+)?в\s+самом\s+начале\b")
USER_Q_BEFORE = re.compile(r"\bкакой\s+вопрос\s+был\s+(?:перед\s+этим|до\s+этого|передо\s+мной)\b")
USER_Q_AFTER_FIRST = re.compile(r"\bкакой\s+вопрос\s+я\s+задал\w*\s+после\s+первого\b")
OWN_AT = re.compile(r"\bчто\s+ты\s+(?:ответила|сказала|написала)\s+(?:на|во|в)\s+" + ORD + r"\s+(?:вопрос|ответ)"
                    r"|\bкакое\s+слово\s+было\s+(?:во|в)\s+" + ORD + r"\s+ответе\b|\bчто\s+(?:ты\s+)?(?:сказала|ответила)\s+во?\s+" + ORD + r"\s+ответе\b")
OWN_AT_CHOICE = re.compile(r"\bнапомни\w*\s+(?:твой|свой)\s+" + ORD + r"\s+(?:выбор|ответ)")
OWN_ABOUT = re.compile(r"\bчто\s+ты\s+(?:ответила|сказала|говорила)\s+(?:про|о|об|насч[её]т)\s+(.+?)\W*$")
OWN_LAST_ADVICE = re.compile(r"\bкакой\s+совет\s+ты\s+(?:только\s+что\s+)?дала\b")
ASKED_ADVICE = re.compile(r"посовет|подскаж|что\s+(?:мне\s+)?делать|на\s+мо[её]м\s+месте|как\s+(?:мне\s+)?(?:лучше\s+)?поступить|"
                          r"что\s+бы\s+(?:ты|вы)\s+сделал")
NOTE_DECL = re.compile(r"^\W*запомни\w*\s*:\s*(.{4,120}?)\W*$")
ASKS_SLOT = re.compile(r"\b(?:где|когда|во\s+сколько|сколько|какое|какой|какая|какие|что)\b")
TO_YOU = re.compile(r"\b(?:ты|тебе|тебя|тобой|твой|твоя|твое|твоё|твои|твоего|твоей)\b")
EITHER_TAIL = re.compile(r"^\W*(?:а\s+)?(?:это\s+)?([а-яё]{3,})\s+или\s+([а-яё]{3,})\s*\?\W*$")
NOTE_STOP = {"запомни", "запомните", "мой", "моя", "мое", "мои", "любимый", "любимая", "любимое", "это", "для", "что", "как"}


def _stems(text):
    return {w[:5] for w in re.findall(r"[а-яё]+", text.lower().replace("ё", "е")) if len(w) >= 4 and w not in NOTE_STOP}
COUNT_ASKED = re.compile(r"\bсколько\s+(?:вопросов|раз)\s+(?:я\s+)?(?:уже\s+)?(?:задал\w*|спросил\w*)\b|\bсколько\s+вопросов\s+было\s+до\s+этого\b")


def ordinal(word: str, total: int):
    if word.startswith(("последн", "прошл", "предыдущ")):
        return total - 1 if total else None
    if word.isdigit():
        n = int(word)
        return n - 1 if 1 <= n <= total else None
    for stem, n in ORDINALS.items():
        if word.startswith(stem):
            return n - 1 if n <= total else None
    return None


def to_number(token: str):
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return NUMBER_WORDS.get(token)


def plural_letters(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} буква"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} буквы"
    return f"{n} букв"


def split_items(text: str):
    text = re.sub(r"[.!?…]+\s*$", "", text.strip())
    parts = [p.strip(" «»\"'.") for p in re.split(r",\s*|\s+и\s+", text)]
    return [p for p in parts if p and len(p) <= 40]


class RoundMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.items, self.original_items, self.numbers, self.code, self.notes = [], [], [], None, []

    def observe(self, question: str, q_ty: str):
        """Сохраняет списки, числа и кодовые слова из реплики собеседника. Возвращает подтверждение или None."""
        low = question.lower().replace("ё", "е")
        code = CODE_DECL.search(low) or CODE_DECL_DASH.search(low)
        if code:
            self.code = code.group(1).strip()
        number = NUMBER_DECL.search(low)
        if number:
            self.numbers.append(int(number.group(1)))
        decl = LIST_DECL.search(low)
        listed = False
        if decl and not number and not code:
            items = split_items(decl.group(1))
            if len(items) >= 2:
                self.items, self.original_items = list(items), list(items)
                listed = True
        note = NOTE_DECL.search(low)
        if note and not number and not code and not listed:
            start = low.find(note.group(1))
            self.notes.append(question[start:start + len(note.group(1))].strip())   # as the person wrote it

    def answer(self, question: str, q_ty: str, user_turns, bot_turns):
        q = q_ty
        if self.items:
            add = APPEND.search(q)
            if add:
                self.items.append(add.group(1))
                return "Запомнила"
            if SWAP.search(q) and len(self.items) >= 2:
                self.items = self.items[::-1]
                return "Запомнила"
            if WHOLE_LIST.search(q) and not re.search(r"\bчисл", q):
                chosen = self.original_items if re.search(r"исходн", q) else self.items
                return ", ".join(chosen).capitalize() + ", вы же говорили"
            if not re.search(r"\bвопрос|\bответ|\bчисл", q):
                at = ITEM_AT.search(q)
                if at:
                    word = next(g for g in at.groups() if g)
                    index = ordinal(word, len(self.items))
                    if index is not None:
                        return self.items[index].capitalize() + ", вы же говорили"
        if self.code:
            if LETTERS_Q.search(q):
                return plural_letters(len(re.sub(r"[^а-яёa-z]", "", self.code)))
            if CODE_Q.search(q):
                return self.code.capitalize() + ", вы же говорили"
        if self.numbers:
            op = NUMBER_OP.search(q)
            if op:
                value = to_number(op.group(2))
                if value is not None:
                    base, verb = self.numbers[-1], op.group(1)
                    if verb.startswith(("прибав", "добав")):
                        result = base + value
                    elif verb.startswith(("отним", "вычт")):
                        result = base - value
                    elif verb.startswith("умнож"):
                        result = base * value
                    elif value:
                        result = base / value
                        result = int(result) if result == int(result) else round(result, 2)
                    else:
                        return None
                    self.numbers.append(result)
                    return str(result)
            num = NUMBER_Q.search(q)
            if num:
                if re.search(r"оба|все", num.group(0)):
                    return " и ".join(str(n) for n in self.numbers)
                if re.search(r"результат", num.group(0)):
                    return str(self.numbers[-1]) if len(self.numbers) > 1 else None
                return str(self.numbers[0]) + ", вы же говорили"
        # «Запомните: встреча в шесть вечера у входа.» -> «Во сколько встреча?» (не про саму Миру)
        if self.notes and ASKS_SLOT.search(q) and not TO_YOU.search(q):
            asked = _stems(q)
            for note in reversed(self.notes):
                if asked & _stems(note):
                    return note[:1].upper() + note[1:] + ", вы же говорили"
        # «Это кот или собака?» после «У меня кот по имени Марс»: вариант, который назвал сам собеседник
        either = EITHER_TAIL.search(q)
        if either and user_turns and not TO_YOU.search(q):
            statements = " ".join(t for t in user_turns if "?" not in t).lower().replace("ё", "е")
            said = {w[:5] for w in re.findall(r"[а-яё]{3,}", statements)}
            first, second = either.group(1), either.group(2)
            hits = [w for w in (first, second) if w[:5] in said]
            if len(hits) == 1:
                return hits[0].capitalize() + ", вы же говорили"
        # порядок собственных реплик собеседника и Миры
        if user_turns:
            m = USER_Q_AT.search(q)
            if m:
                word = next(g for g in m.groups() if g)
                index = ordinal(word, len(user_turns))
                if index is not None:
                    return f"Вы спрашивали: «{user_turns[index]}»"
            if USER_Q_FIRST.search(q):
                return f"Вы спрашивали: «{user_turns[0]}»"
            if USER_Q_BEFORE.search(q):
                return f"Вы спрашивали: «{user_turns[-1]}»"
            if USER_Q_AFTER_FIRST.search(q) and len(user_turns) > 1:
                return f"Вы спрашивали: «{user_turns[1]}»"
            if COUNT_ASKED.search(q):
                n = len(user_turns)
                return f"Это уже {n + 1}-й вопрос, если считать этот"
        if bot_turns:
            m = OWN_AT.search(q)
            if m:
                word = next(g for g in m.groups() if g)
                index = ordinal(word, len(bot_turns))
                if index is not None:
                    return f"Я ответила: «{bot_turns[index]}»"
            m = OWN_AT_CHOICE.search(q)
            if m:
                index = ordinal(m.group(1), len(bot_turns))
                if index is not None:
                    return f"Я ответила: «{bot_turns[index]}»"
            if OWN_LAST_ADVICE.search(q):
                for i in range(min(len(user_turns), len(bot_turns)) - 1, -1, -1):
                    if ASKED_ADVICE.search(user_turns[i].lower().replace("ё", "е")):
                        return f"Я ответила: «{bot_turns[i]}»"
                return None
            m = OWN_ABOUT.search(q)
            if m:
                about = _stems(m.group(1))
                for i in range(min(len(user_turns), len(bot_turns)) - 1, -1, -1):
                    if about & _stems(user_turns[i]) and not OWN_ABOUT.search(user_turns[i].lower()):
                        return f"Я ответила: «{bot_turns[i]}»"
        return None
