"""Candidate checks for the Mira persona: pure Python, no model weights.

A retrieved line is a real human reply from another conversation. It is usable
only if it could plausibly come from Mira (21, female, student, unmarried, no
children) and fits the conversation turn. These rules reject lines that would
expose the bot (foreign biography, wrong gender, farewells, other people's
names) and score the rest; they are heuristics, not a semantic guarantee.
"""
from __future__ import annotations

import re

WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
STOP = set("""а и но да нет не ни же ли бы то это как что кто где когда куда откуда почему зачем
сколько какой какая какие какое каким ты тебя тебе тобой я меня мне мной мы нас нам вы вас вам
он она они его ее её их у в во на с со к ко о об от до по за из для при про над под без
так там тут здесь вот ну еще ещё уже очень просто всё все весь вся мой моя мои твой твоя
свой своя быть есть был была было были будет буду бывает можно нужно надо тоже также или
если чтобы чем раз вообще сейчас сегодня""".split())

FAREWELL = re.compile(r"\b(?:пока|спокойной ночи|до встречи|до свидания|мне пора|пора бежать|"
                      r"спасибо за (?:беседу|общение|разговор)|всего (?:хорошего|доброго)|удачи тебе|"
                      r"был[аи]? рад[аы]? (?:пообщаться|поговорить)|приятно было пообщаться)\b")
GREETING = re.compile(r"^\W*(?:привет\w*|здравствуй\w*|добрый (?:день|вечер)|доброе утро|хай|приветствую)\b")
FORMAL_YOU = re.compile(r"\b(?:вы|вас|вам|вами|ваш|ваша|ваше|ваши|вашего|вашей|вашем|вашу|ваших)\b")
FOREIGN_LIFE = re.compile(
    r"\b(?:мо(?:й|его|ему|им) (?:муж|сын|внук|ребен\w*|ребён\w*|парн\w*)|моя (?:жена|дочь|дочка|внучка|девушка)|"
    r"мои (?:дети|внуки|сыновья|дочки|дочери)|у меня (?:есть )?(?:муж|жена|сын|дочь|дети|внук\w*|двойня|ребен\w*|ребён\w*)|"
    r"(?:я )?(?:замужем|женат|разв[её]л\w*|холост|на пенсии|пенсионер\w*)|"
    r"мужу|жене|детям|детей|внукам|внуков|"
    r"я (?:школьни\w+|в школе|учусь в школе|в \d+ классе)|"
    r"работаю (?:слесар|автослесар|инженер|адвокат|учител|врач|водител|программист|каменщик|оператор|"
    r"дворник|бухгалтер|менеджер|продав|юрист|фельдшер|медсестр|тамад|проводник|стюардесс|банкир|"
    r"на заводе|на фабрике|в офисе)\w*|"
    r"я (?:слесарь|инженер|адвокат|учитель|врач|водитель|программист|каменщик|дворник|бухгалтер|"
    r"юрист|фельдшер|медсестра|тамада|проводник|стюардесса|банкир|домохозяйка|фрилансер|музыкант|"
    r"спортсмен|ветеринар|военн\w*|мама|папа|бабушка|дедушка|мужчина|парень))\b")
KIN = re.compile(r"\b(?:сын\w*|доч(?:ь|ери|ерью|ка|ки|ку|кой|енька)\w*|внук\w*|внучк\w*|реб[её]н\w*|дет(?:и|ей|ям|ьми|ках|ки|ишк\w*)|"
                 r"муж(?:а|у|ем|е|ья|ем)?|жен(?:а|ы|е|у|ой)|супруг\w*|свекр\w*|т[её]щ\w*|дальнобой\w*|пенси\w*|"
                 r"в армии|служил\w*|на заводе|на смене|моя девушка|мой парень)\b")
MALE_IMPLICIT = re.compile(
    r"(?:^|[.!?)]\s*)(?:[а-яё]+(?:[^аеиоуыэюяё\s]ел|[аяиуыо]л|ёл)|согласен|понял|рад|готов|уверен|должен|занят|"
    r"привык|смог|мог|помог|устал|отош[её]л|приш[её]л|уш[её]л|наш[её]л|пош[её]л)\b(?!\s*(?:бы|ли)\b)")
MALE_ADJ = re.compile(r"\bя\s+(?:\S+\s+){0,3}?(?:высокий|низкий|молодой|старый|взрослый|занят|свободен|счастлив|болен|"
                      r"одинок|влюбл[её]н|голоден|доволен|привык|смог|мог|помог|сам)\b")
FIRST_PERSON_PLACE = re.compile(r"\b(?i:я|мы|у нас|живу|жила|родом|родилась|родился|переехал\w*)\b[^.!?]{0,30}?\b(?:из|в|во)\s+([А-ЯЁ][а-яё]{2,})")
MALE_SELF = re.compile(
    r"\bя\s+(?:(?:уже|тоже|сам|вчера|сегодня|когда-то|раньше|недавно|давно|не|просто|только|ещё|еще|"
    r"так|бы|вот|там|тут|тогда|сразу|как-то|даже|всегда|никогда|часто|редко)\s+){0,2}"
    r"(?!был[аио]?\b)[а-яё]{2,}(?:[^аеиоуыэюяё\s]ел|[аяиуыо]л)\b|\bя\s+(?:сам|был|рад|готов|должен|уверен|согласен|женат)\b")
MALE_PAST = re.compile(r"\b(?:я|никогда|ни\s+разу|сам)\s+(?:\S+\s+){0,2}?(?:не\s+)?(?:был|ходил|ездил|видел|пробовал|делал|знал|думал|хотел|любил|жил|работал|учился|играл|смотрел|читал|слышал|пошел|пришел)\b")
NAME_CALL = re.compile(r"(?:^|[,.!?]\s*)([А-ЯЁ][а-яё]{2,})(?=[,!])")
AGE = re.compile(r"\bмне\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:год|года|лет)\b")
CITY = re.compile(r"\bя (?:из|живу в|родом из) ([А-ЯЁ][а-яё]+)")
SELF_NAME = re.compile(r"\b(?:меня зовут|я\s*[-—]\s*|зовут меня)\s*([А-ЯЁ][а-яё]+)")
JUNK = re.compile(r"\b(?:end|and/)\b|\*\*|```|https?://|www\.|@\w+|#\w+", re.IGNORECASE)


def norm(text):
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def content_words(text):
    return {w[:6] for w in WORD.findall(norm(text)) if w not in STOP and len(w) > 2}


def clean(text):
    """Light surface cleanup that keeps the human wording."""
    text = EMOJI.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?)])", r"\1", text)
    return text.strip(" -—")


def persona_problem(text, profile, user_text=""):
    t = norm(text)
    if FOREIGN_LIFE.search(t):
        return "foreign_biography"
    if KIN.search(t):
        return "foreign_family_or_job"
    if MALE_SELF.search(t) or MALE_ADJ.search(t) or MALE_IMPLICIT.search(t) or MALE_PAST.search(t):
        return "male_self_reference"
    for match in AGE.finditer(t):
        number = int(match.group(1) or match.group(2))
        if 5 <= number <= 99 and number != profile.get("age") and ("мне" in match.group(0) or "я" in t.split()):
            return "foreign_age"
    for match in SELF_NAME.finditer(text):
        if match.group(1).lower() not in {profile.get("name", "").lower()}:
            return "foreign_name"
    allowed = {c.lower() for c in profile.get("cities", [])} | {w.lower() for w in re.findall(r"[А-ЯЁ][а-яё]+", user_text)}
    for pattern in (CITY, FIRST_PERSON_PLACE):
        for match in pattern.finditer(text):
            if match.group(1).lower() not in allowed:
                return "foreign_city"
    if FORMAL_YOU.search(t) and not FORMAL_YOU.search(norm(user_text)):
        return "formal_you"
    return None


def turn_problem(text, question, turn_index, mentioned_names):
    t = norm(text)
    if JUNK.search(text):
        return "junk"
    if FAREWELL.search(t):
        return "farewell"
    if GREETING.search(t) and not GREETING.search(norm(question)):
        return "unprompted_greeting"
    for match in NAME_CALL.finditer(text):
        name = match.group(1)
        if name.lower() not in mentioned_names and name not in {"Да", "Нет", "Ну", "Ого", "Ой", "Хм", "Ага",
                                                              "Круто", "Классно", "Здорово", "Понятно", "Конечно",
                                                              "Спасибо", "Ахах", "Хаха", "Блин", "Слушай"}:
            return "addresses_other_person"
    if len(EMOJI.findall(str(text))) > 2:
        return "emoji_heavy"
    words = WORD.findall(t)
    if not words:
        return "no_words"
    if len(words) > 45:
        return "too_long"
    return None


def score(candidate, question, history_text, rank):
    """Higher is better. Combines retrieval rank with turn-fit signals."""
    text = candidate
    q, c = content_words(question), content_words(text)
    overlap = len(q & c)
    context_overlap = len(content_words(history_text) & c)
    words = len(WORD.findall(text))
    value = -0.35 * rank + 1.6 * overlap + 0.4 * min(context_overlap, 3)
    if len(q) >= 2 and overlap == 0:
        value -= 1.2  # nothing in the reply touches the question's content words
    asks = question.strip().endswith("?")
    yes_no = asks and bool(re.match(r"^\W*(?:ты|а ты|тебе|у тебя|ты же|любишь|умеешь|будешь|хочешь|"
                                    r"нравится|можешь|была|был|смотришь|играешь|знаешь)\b", norm(question)))
    if yes_no and re.match(r"^\W*(?:да|нет|конечно|не|ага|неа|угу|еще бы|ещё бы|бывает|иногда|редко|часто|люблю|обожаю)\b", norm(text)):
        value += 2.0
    if asks and text.strip().endswith("?") and not re.search(r"[.!)]\s", text):
        value -= 1.5  # answers a question only with a counter-question
    if 2 <= words <= 25:
        value += 0.8
    elif words == 1:
        value -= 0.6
    elif words > 35:
        value -= 1.0
    return value


def mentioned(history, question):
    names = set()
    for turn in list(history) + [{"text": question}]:
        for match in re.finditer(r"[А-ЯЁ][а-яё]{2,}", str(turn.get("text", ""))):
            names.add(match.group(0).lower())
    return names
