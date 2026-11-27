"""Contest ("hall") mode helpers.

At the Turing Arena the host reads audience questions to three screens; answers
are written, in Russian, up to 300 characters. People behind a screen answer the
hall: no counter-questions, no chat smileys or "ахах", no unprompted complaints.

- to_ty(): understand the host's polite "вы" by mapping it to the "ты" forms the
  bank patterns are written in (matching only; the answer text is untouched).
- hall_style(): turn a chat-style reply into a calm written answer.
- clock_answer(): date / weekday / time questions from the local clock.
- constraint(): "ответьте одним словом", "да или нет".
"""
from __future__ import annotations

import datetime as _dt
import re

IMPERATIVE = {
    "скажите": "скажи", "расскажите": "расскажи", "опишите": "опиши", "напишите": "напиши", "назовите": "назови",
    "представьте": "представь", "придумайте": "придумай", "ответьте": "ответь", "посчитайте": "посчитай",
    "попробуйте": "попробуй", "докажите": "докажи", "повторите": "повтори", "выберите": "выбери", "угадайте": "угадай",
    "продолжите": "продолжи", "объясните": "объясни", "поделитесь": "поделись", "вспомните": "вспомни",
    "загадайте": "загадай", "приведите": "приведи", "оцените": "оцени", "сравните": "сравни", "посоветуйте": "посоветуй",
    "подскажите": "подскажи", "признайтесь": "признайся", "признавайтесь": "признавайся", "запомните": "запомни",
    "назовите-ка": "назови", "напомните": "напомни", "простите": "прости", "извините": "извини", "здравствуйте": "здравствуй",
    "покажите": "покажи", "закончите": "закончи", "переведите": "переведи", "составьте": "составь", "давайте": "давай", "представьтесь": "представься", "назовитесь": "назовись", "опишите-ка": "опиши",
}
PRONOUNS = {"вы": "ты", "вас": "тебя", "вам": "тебе", "вами": "тобой", "ваш": "твой", "ваша": "твоя", "ваше": "твое",
            "ваши": "твои", "вашего": "твоего", "вашей": "твоей", "вашем": "твоем", "вашу": "твою", "ваших": "твоих",
            "вашим": "твоим", "вашими": "твоими", "себя": "себя"}
NOT_VERBS = {"дети", "сети", "сайте", "конфете", "газете", "планете", "ракете", "карете", "диете", "котлете", "анкете",
             "комнате", "кассете", "пакете", "билете", "предмете", "ответе", "совете", "свете", "цвете", "лете", "мете",
             "бите", "сите", "сюите", "граните", "зените", "ките", "элите", "магните", "кредите", "визите", "колите",
             "отите", "учите"}


def _norm(text):
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def to_ty(text: str) -> str:
    """'Что вы любите есть?' -> 'что ты любишь есть?' (lowercase, ё→е)."""
    q = re.sub(r"([а-я])(\d)", r"\1 \2", _norm(text))   # "рождения7" -> "рождения 7"
    q = re.sub(r"\bч[её]\b|\bчо\b|\bшо\b", "что", q)                 # "че ты не ешь" -> "что ты не ешь"
    words = re.split(r"(\W+)", q)
    plural = any(w in ("вы", "вас", "вам", "вами") or w.startswith("ваш") for w in words)
    out = []
    for i, w in enumerate(words):
        if w in IMPERATIVE:
            out.append(IMPERATIVE[w])
            continue
        if w in PRONOUNS:
            before = "".join(words[max(0, i - 4):i])
            if w in ("вас", "вами") and re.search(r"(?:\bиз|\bсреди|\bмежду)\s*$", before):
                out.append(w)          # "кто из вас" — the group of three screens, keep plural
                continue
            out.append(PRONOUNS[w])
            continue
        if plural and len(w) >= 3 and w not in ("или", "ли", "али") and re.search(r"[аеиыяу]ли$", w) and "вы" in words[max(0, i - 6):i]:
            out.append(w[:-1] + "а")      # вы были -> ты была (Mira is female)
            continue
        if plural and len(w) > 4 and w not in NOT_VERBS:
            if w.endswith("етесь"):
                w = w[:-5] + "ешься"
            elif w.endswith("итесь"):
                w = w[:-5] + "ишься"
            elif w.endswith("ете"):
                w = w[:-3] + "ешь"
            elif w.endswith("ите"):
                w = w[:-3] + "ишь"
            elif w not in ("или", "ли", "али") and re.search(r"[аеиыяу]ли$", w) and any(p in ("вы",) for p in words[max(0, i - 6):i]):
                w = w[:-1] + "а"          # вы были -> ты была (Mira is female)
            elif re.search(r"[аеиыяу]лись$", w) and "вы" in words[max(0, i - 6):i]:
                w = w[:-3] + "ась"
        out.append(w)
    return _feminine("".join(out))


# Зал часто спрашивает в мужском роде («ты ел?», «что ты ответил?»), а правила банка написаны в женском.
_NOT_VERBS_L = {"стол", "пол", "зал", "бал", "мол", "кол", "футбол", "волейбол", "баскетбол", "гол", "осел", "котел", "дятел",
                "ангел", "посол", "орел", "козел", "пепел", "уголь", "сокол", "тыл", "пыл", "мел", "вокзал", "журнал", "сигнал",
                "канал", "финал", "фестиваль", "материал", "идеал", "генерал", "персонал", "капитал", "интервал", "скандал",
                "сериал", "минерал", "шакал", "пенал", "автомобиль", "алкоголь", "контроль", "роль", "боль", "соль", "раздел",
                "предел", "отдел", "удел", "ел?"}
_FEM_IRREGULAR = {"был": "была", "мог": "могла", "смог": "смогла", "помог": "помогла", "лег": "легла", "слег": "слегла", "пек": "пекла",
                  "испек": "испекла", "сам": "сама", "один": "одна", "должен": "должна", "рад": "рада", "готов": "готова",
                  "уверен": "уверена", "согласен": "согласна", "занят": "занята", "голоден": "голодна", "прав": "права",
                  "женат": "замужем", "влюблен": "влюблена", "счастлив": "счастлива", "доволен": "довольна", "устал": "устала"}


def _feminine(q: str) -> str:
    """«ты … ел / ответил / сам / смог» -> женский род: первый подходящий глагол (и «сам») в пределах 4 слов
    после «ты», не переходя через знак препинания."""
    tokens = re.split(r"([а-яё]+)", q)        # чётные — разделители, нечётные — слова
    changed_verb = False
    i = 1
    while i < len(tokens):
        if tokens[i] == "ты":
            j, seen = i + 2, 0
            while j < len(tokens) and seen < 5:
                if re.search(r"[.,!?;:—()]", tokens[j - 1]) and j - 1 != i + 1:
                    break
                word = tokens[j]
                if word == "сам":
                    tokens[j] = "сама"
                elif not changed_verb:
                    new = _fem_word(word)
                    if new != word:
                        tokens[j] = new
                        changed_verb = True
                j += 2
                seen += 1
        i += 2
    return "".join(tokens)


def _fem_word(word: str) -> str:
    if word in _FEM_IRREGULAR:
        return _FEM_IRREGULAR[word]
    if word in _NOT_VERBS_L or len(word) < 2:
        return word
    if re.search(r"ш[её]л$", word):
        return re.sub(r"ш[её]л$", "шла", word)
    if re.search(r"[аеиоуыяё]л$", word):
        return word + "а"
    if len(word) > 3 and re.search(r"(?:[аеиоуыяё]|г|к|з|с)ли$", word):   # «ты бы смогли» после замены «вы» -> «ты»
        return word[:-1] + "а"
    return word


VY_PRONOUNS = {"ты": "вы", "тебя": "вас", "тебе": "вам", "тобой": "вами", "твой": "ваш", "твоя": "ваша", "твое": "ваше",
               "твоё": "ваше", "твои": "ваши", "твоего": "вашего", "твоей": "вашей", "твою": "вашу", "твоих": "ваших",
               "твоим": "вашим", "твоими": "вашими", "твоему": "вашему"}
VY_IMPERATIVE = {"снизь": "снизьте", "ищи": "ищите", "угадай": "угадайте", "вспоминай": "вспоминайте", "спроси": "спросите", "посмотри": "посмотрите", "попробуй": "попробуйте", "держись": "держитесь", "переживай": "переживайте",
                 "береги": "берегите", "напиши": "напишите", "давай": "давайте", "отдохни": "отдохните", "подожди": "подождите",
                 "представь": "представьте", "спеши": "спешите", "пиши": "пишите", "езжай": "езжайте", "иди": "идите",
                 "лечись": "лечитесь", "выздоравливай": "выздоравливайте", "подготовь": "подготовьте", "выдохни": "выдохните",
                 "полежи": "полежите", "добивай": "добивайте", "обращай": "обращайте", "обращайся": "обращайтесь",
                 "оставайся": "оставайтесь", "грузись": "грузитесь", "выдавай": "выдавайте", "позорь": "позорьте",
                 "начинай": "начинайте", "уговаривай": "уговаривайте", "скажи": "скажите", "расскажи": "расскажите",
                 "заходи": "заходите", "приходи": "приходите", "звони": "звоните", "спроси": "спросите", "подумай": "подумайте"}
VY_VERBS = {"ешь": "едите", "хочешь": "хотите", "пересдашь": "пересдадите", "сдашь": "сдадите", "дашь": "дадите",
            "продашь": "продадите", "передашь": "передадите", "бежишь": "бежите", "даёшь": "даёте", "даешь": "даете"}
NOT_2SG = {"лишь", "тишь", "мышь", "камыш", "малыш", "шалаш", "кишмиш", "глушь", "чушь", "ешь"}


def plural_slots(text: str) -> str:
    """'понял{|а}' -> 'поняли', '{расстроен|расстроена}' -> 'расстроены' (addressing people politely)."""
    def full(m):
        fem = m.group(2)
        if fem in ("самой", "самому") or m.group(1) in ("самому",):
            return "самим"
        if fem.endswith("лась"):
            return fem[:-4] + "лись"
        if fem.endswith("ла"):
            return fem[:-2] + "ли"
        if fem.endswith("а"):
            return fem[:-1] + "ы"
        return m.group(1)
    text = re.sub(r"(\w+)\{\|а\}", lambda m: m.group(1) + ("и" if m.group(1).endswith(("л", "м")) else "ы"), text)
    return re.sub(r"\{([^{}|]*)\|([^{}|]*)\}", full, text)


def to_vy(text: str, protect=frozenset()) -> str:
    """Mira's reply addressed to the hall: 'ты/твой/знаешь' -> 'вы/ваш/знаете'."""
    def fix(m):
        word = m.group(0)
        low = word.lower()
        new = None
        if low in VY_PRONOUNS:
            new = VY_PRONOUNS[low]
        elif low in VY_IMPERATIVE:
            new = VY_IMPERATIVE[low]
        elif low in VY_VERBS:
            new = VY_VERBS[low]
        elif low not in NOT_2SG and low not in protect and len(low) >= 5:
            for old, repl in (("ешься", "етесь"), ("ишься", "итесь"), ("ёшь", "ёте"), ("ешь", "ете"), ("ишь", "ите")):
                if low.endswith(old):
                    new = low[: -len(old)] + repl
                    break
        if new is None:
            return word
        return new[0].upper() + new[1:] if word[0].isupper() else new
    text = re.sub(r"[А-Яа-яЁё]+", fix, text)
    text = re.sub(r"\b(вы|Вы)(\s+)(?:сам|сама)\b", r"\1\2сами", text)                 # "ты сам" -> "вы сами"
    nouns = {"стол", "пол", "мол", "зал", "мел", "бал", "вал", "тыл", "пыл"}

    def past(m):
        word, end = m.group(3), m.group(4)
        if (word + "л").lower() in nouns and word.lower() != "бы":
            return m.group(0)
        return m.group(1) + m.group(2) + word + "ли"
    # "вы (это уже) решил/решила" -> "вы решили": a past verb up to three words after "вы"
    text = re.sub(r"\b(вы|Вы)((?:\s+[а-яё]+){0,3}?\s+)([а-яё]{1,}?[аеиоуыя])(ла|л)\b", past, text)
    text = re.sub(r"\b(вы|Вы)((?:\s+[а-яё]+){0,3}?\s+)([а-яё]+?[аеиоуыя])(лась|лся)\b",
                  lambda m: m.group(1) + m.group(2) + m.group(3) + "лись", text)
    short = {"прав": "правы", "права": "правы", "доволен": "довольны", "довольна": "довольны", "уверен": "уверены",
             "уверена": "уверены", "готов": "готовы", "готова": "готовы", "рад": "рады", "рада": "рады",
             "согласен": "согласны", "согласна": "согласны", "должен": "должны", "должна": "должны"}
    text = re.sub(r"\b(вы|Вы)((?:\s+(?:же|точно|уже|тоже|не))*\s+)([а-яё]+)\b",
                  lambda m: m.group(1) + m.group(2) + short.get(m.group(3).lower(), m.group(3)), text)
    # imperative right after "вы": "вы ищи" -> "вы ищите" (not past plural "спросили")
    return re.sub(r"\b(вы|Вы)(\s+)([а-яё]+(?:[^л]и|ь))\b",
                  lambda m: m.group(1) + m.group(2) + (m.group(3) + "те" if m.group(3).lower() not in NOT_2SG
                                                       and not m.group(3).lower().endswith(("лись", "лась", "лся", "лось"))
                                                       and m.group(3).lower() not in ("сами", "мои", "свои", "ваши", "они", "им")
                                                       else m.group(3)), text)


SMILE_END = re.compile(r"\s*[)(]+\s*$")
SMILE_MID = re.compile(r"\s*[)(]{1,3}(?=\s+[А-ЯЁA-Z])")
LAUGH = re.compile(r"(?i)[,.]?\s*\b(?:а?х[ах]{2,}\w*|хех\w*|лол)\b[,.]?")
BACK_Q = re.compile(r"(?i)\b(?:а\s+ты|а\s+у\s+тебя|а\s+тебе|а\s+вы|а\s+у\s+вас|а\s+вам|ты\s+как|тебя|тебе|твой|твоя|твои|сам[аи]?|у\s+тебя)\b")


def hall_style(text: str, keep_questions: bool = False, protect=frozenset()) -> str:
    """Chat reply -> written answer for the hall. protect: the asker's own words, never changed."""
    t = to_vy(plural_slots(str(text).strip()), protect)
    t = LAUGH.sub("", t)
    t = SMILE_MID.sub(".", t)
    t = SMILE_END.sub("", t)
    t = re.sub(r"[)(]{2,}", "", t)
    t = re.sub(r"\s*\)\s*([,.])", r"\1", t)
    t = re.sub(r"\s*\(\s*", " ", t)
    t = re.sub(r"\s*\)\s*", ". ", t)
    t = re.sub(r"(?<!\.)\.\s+\.(?!\.)", ".", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;—-")
    # drop a trailing counter-question addressed to the asker when something else remains
    parts = re.findall(r"[^.!?]+[.!?]*", t)
    popped = False
    while not keep_questions and len(parts) > 1 and parts[-1].strip().endswith("?"):
        parts.pop()
        popped = True
    if popped:
        t = "".join(parts).strip()
    t = re.sub(r"\.{2,}$", "...", t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
SEASONS = {12: "зима", 1: "зима", 2: "зима", 3: "весна", 4: "весна", 5: "весна", 6: "лето", 7: "лето", 8: "лето",
           9: "осень", 10: "осень", 11: "осень"}


def clock_answer(question: str, now: _dt.datetime | None = None) -> str | None:
    q = to_ty(question)
    if re.search(r"\b(?:самый|самое|самая|любим\w*|нелюбим\w*|тяжел\w*|лучш\w*|был\s+бы|если|запомн\w*|просил\w*|загадал\w*|выбрал\w*|назвал\w*|сказал\w*|говорил\w*|в\s+день|в\s+неделю|в\s+месяц|сидиш\w*|сидите|трат\w*|занима\w*|уход\w*|провод\w*)\b", q):
        return None
    now = now or _dt.datetime.now()
    if re.search(r"\b(?:какое|какой)\s+(?:сегодня\s+)?(?:число|дата)\b|\bкакое\s+сегодня\b|\bсегодняшн\w+\s+дат", q):
        return f"Сегодня {now.day} {MONTHS[now.month - 1]}"
    if re.search(r"\bкакой\s+(?:сегодня\s+)?день\s+недели\b|\bкакой\s+сегодня\s+день\b", q):
        return f"Сегодня {WEEKDAYS[now.weekday()]}"
    if re.search(r"\bкоторый\s+час\b|\bсколько\s+(?:сейчас\s+)?времени\b|\bсколько\s+время\b", q):
        return f"Примерно {now.hour}:{now.minute:02d}, если мои часы не врут"
    if re.search(r"\bкакой\s+(?:сейчас\s+)?(?:год|месяц)\b", q):
        return f"{now.year} год, {['январь','февраль','март','апрель','май','июнь','июль','август','сентябрь','октябрь','ноябрь','декабрь'][now.month-1]}" \
            if "год" in q else f"{['январь','февраль','март','апрель','май','июнь','июль','август','сентябрь','октябрь','ноябрь','декабрь'][now.month-1].capitalize()}"
    if re.search(r"\bкакое\s+(?:сейчас\s+)?время\s+года\b", q):
        return f"{SEASONS[now.month].capitalize()}, хотя в Астане это понятие растяжимое"
    return None


def constraint(question: str) -> str | None:
    """'одним словом' / 'да или нет' -> 'one_word' / 'yes_no' / None."""
    q = to_ty(question)
    if re.search(r"\bодним\s+словом\b|\bв\s+одно\s+слово\b", q):
        return "one_word"
    if re.search(r"\b(?:только\s+)?да\s+или\s+нет\b", q):
        return "yes_no"
    return None


def apply_constraint(kind: str, answer: str, question: str) -> str:
    if kind == "yes_no":
        low = _norm(answer)
        if re.match(r"^(?:нет|не\b|неа|вряд ли|никогда|скорее нет)", low):
            return "Нет"
        return "Да"
    if kind == "one_word":
        words = re.findall(r"[А-Яа-яЁё-]+", answer)
        skip = {"ну", "да", "нет", "хм", "наверное", "честно", "если", "конечно", "скорее", "я", "мне", "это", "вообще",
                "думаю", "кажется", "сейчас", "чаще", "наверно", "пожалуй", "ой", "вот", "так", "сложно", "сказать"}
        for w in words:
            if w.lower() not in skip and len(w) > 2:
                return w[0].upper() + w[1:].lower()
        return "Сложно"
    return answer


# --- редкие опечатки: соседняя клавиша на русской раскладке (ЙЦУКЕН) ---
KEY_ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
NEIGHBORS = {}
for _row in KEY_ROWS:
    for _i, _ch in enumerate(_row):
        NEIGHBORS[_ch] = _row[max(0, _i - 1):_i] + _row[_i + 1:_i + 2]


def add_typo(text, rnd, protect=frozenset()):
    """Replace one inner letter of one ordinary word with a neighbouring key.

    Names (capitalised), quoted titles, numbers and the asker's own words are left alone.
    """
    if "«" in text:
        return text
    words = [m for m in re.finditer(r"(?<![\w-])[а-яё]{5,}(?![\w-])", text) if m.group(0) not in protect]
    if not words:
        return text
    m = rnd.choice(words)
    word = m.group(0)
    i = rnd.randrange(1, len(word) - 1)
    near = NEIGHBORS.get(word[i])
    if not near:
        return text
    word = word[:i] + rnd.choice(near) + word[i + 1:]
    return text[:m.start()] + word + text[m.end():]
