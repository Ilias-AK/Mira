"""Tiny symbolic solver for everyday counting questions ("было три яблока, одно отдал").

Rules only, no model. Keeps the last result so a follow-up ("а если вернул
половину?") can continue the same count.
"""
from __future__ import annotations

import re
from fractions import Fraction

NUMS = {
    "ноль": 0, "один": 1, "одно": 1, "одна": 1, "одну": 1, "одного": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3, "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "пятнадцать": 15,
    "двадцать": 20, "половину": Fraction(1, 2), "половина": Fraction(1, 2),
}
WORDS = {0: "Ноль", 1: "Одно", 2: "Два", 3: "Три", 4: "Четыре", 5: "Пять", 6: "Шесть", 7: "Семь",
         8: "Восемь", 9: "Девять", 10: "Десять", 11: "Одиннадцать", 12: "Двенадцать"}
TAKE = r"(?:отдал\w*|съел\w*|потерял\w*|подарил\w*|продал\w*|выбросил\w*|забрал\w*|уронил\w*|сломал\w*|отдала|съела)"
GIVE = r"(?:купил\w*|нашел\w*|нашла|дали|подарили|добавил\w*|принес\w*|получил\w*)"
NUM = r"(\d+|" + "|".join(sorted(NUMS, key=len, reverse=True)) + r")"


def _value(token):
    token = token.lower()
    return Fraction(int(token)) if token.isdigit() else Fraction(NUMS[token])


def say(value: Fraction) -> str:
    if value.denominator == 1 and int(value) in WORDS:
        return WORDS[int(value)]
    if value.denominator == 2 and int(value) in WORDS:
        whole = int(value)
        return (WORDS[whole] + " с половиной") if whole else "Половина"
    return str(float(value)).rstrip("0").rstrip(".").replace(".", ",")


class CountingState:
    def __init__(self):
        self.result = None
        self.moved = None

    def answer(self, question: str) -> str | None:
        q = question.lower().replace("ё", "е")
        start = re.search(r"\b(?:было|есть|имеется|лежало|лежит)\s+" + NUM + r"\b", q) or re.search(r"\bу\s+(?:тебя|меня|него|нее)\s+" + NUM + r"\b", q)
        asks = re.search(r"\bсколько\b", q)
        if start and asks:
            total = _value(start.group(1))
            for sign, verbs in ((-1, TAKE), (1, GIVE)):
                m = re.search(NUM + r"\s+(?:\w+\s+){0,2}?" + verbs, q) or re.search(verbs + r"\s+(?:\w+\s+){0,2}?" + NUM, q)
                if m:
                    moved = _value(next(g for g in m.groups() if g))
                    self.result, self.moved = total + sign * moved, moved
                    return f"{say(self.result)}) Было {say(total).lower()}, {say(moved).lower()} {'ушло' if sign < 0 else 'добавилось'}"
            return None
        if self.result is not None and re.search(r"\bверн\w*\b", q):
            part = Fraction(1, 2) if re.search(r"\bполовин\w*\b", q) else (self.moved or 0)
            if re.search(r"\bполовин\w*\b", q) and self.moved:
                part = self.moved / 2
            self.result += part
            return f"Тогда {say(self.result).lower()}) Половинка же вернулась" if part.denominator != 1 else f"Тогда {say(self.result).lower()})"
        return None


UNITS = {"ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
         "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
         "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
         "девятнадцать": 19}
TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
        "восемьдесят": 80, "девяносто": 90, "сто": 100}
NUMBER = r"(\d+|(?:(?:" + "|".join(TENS) + r")(?:\s+(?:" + "|".join(UNITS) + r"))?)|(?:" + "|".join(UNITS) + r"))"
OPS = {"плюс": "+", "прибавить": "+", "минус": "-", "отнять": "-", "умножить на": "*", "умножить": "*",
       "разделить на": "/", "поделить на": "/", "делить на": "/"}
FACTS = [
    (r"сколько\s+минут\s+в\s+сутк", "1440) 24 на 60"), (r"сколько\s+часов\s+в\s+сутк", "24, конечно)"),
    (r"сколько\s+секунд\s+в\s+час", "3600"), (r"сколько\s+минут\s+в\s+час", "60)"),
    (r"сколько\s+дней\s+в\s+(?:обычном\s+)?году", "365, а в високосном 366"), (r"сколько\s+дней\s+в\s+недел", "Семь)"),
    (r"сколько\s+месяцев\s+в\s+году", "Двенадцать)"), (r"сколько\s+часов\s+в\s+недел", "168, если не ошибаюсь"),
    (r"сколько\s+дней\s+в\s+високосн", "366"), (r"сколько\s+секунд\s+в\s+минут", "60"),
    (r"сколько\s+секунд\s+в\s+сутк", "86400, если я правильно посчитала"),
    (r"сколько\s+метров\s+в\s+километр", "Тысяча"), (r"сколько\s+сантиметров\s+в\s+метр", "Сто"),
    (r"сколько\s+миллиметров\s+в\s+сантиметр", "Десять"), (r"сколько\s+граммов?\s+в\s+килограмм", "Тысяча"),
    (r"сколько\s+килограммов?\s+в\s+тонн", "Тысяча"), (r"сколько\s+градусов\s+в\s+прям\w+\s+угл", "90"),
    (r"сколько\s+градусов\s+в\s+(?:развернут|развёрнут)\w*\s+угл", "180"), (r"сколько\s+градусов\s+в\s+круг", "360"),
    (r"сколько\s+сторон\s+у\s+шестиугольник", "Шесть"), (r"сколько\s+сторон\s+у\s+пятиугольник", "Пять"),
    (r"сколько\s+сторон\s+у\s+треугольник", "Три, это же треугольник"), (r"сколько\s+углов\s+у\s+квадрат", "Четыре"),
    (r"сколько\s+минут\s+в\s+полутора\s+час", "90"), (r"сколько\s+недель\s+в\s+году", "52 с хвостиком"),
]


def to_number(token):
    token = token.strip()
    if token.isdigit():
        return int(token)
    total = 0
    for word in token.split():
        total += TENS.get(word, UNITS.get(word, 0))
    return total


def _eval_expr(expr):
    """['2','+','2','*','2'] with usual precedence, exact fractions."""
    values = [Fraction(int(expr[0]))]
    ops = []
    for op, num in zip(expr[1::2], expr[2::2]):
        n = Fraction(int(num))
        if op == "*":
            values[-1] *= n
        elif op == "/":
            values[-1] /= n
        else:
            ops.append(op)
            values.append(n)
    total = values[0]
    for op, v in zip(ops, values[1:]):
        total = total + v if op == "+" else total - v
    return total


def spoken_arithmetic(question: str) -> str | None:
    q = question.lower().replace("ё", "е")
    rev = re.search(r"\b(?:напиши|скажи|произнеси|запиши)\s+(?:слово\s+)?[«\"']?([а-яa-z-]+)[»\"']?\s+(?:наоборот|задом\s+наперед)", q)
    if rev:
        return rev.group(1)[::-1]
    letters = re.search(r"\bсколько\s+букв\s+в\s+слове\s+[«\"']?([а-яa-z-]+)", q)
    if letters:
        return str(len(letters.group(1).replace("-", "")))
    for pattern, answer in FACTS:
        if re.search(pattern, q):
            return answer
    sisters = re.search(NUMBER + r"\s+сест[её]?р\w*\b.*\bу\s+каждой\b.*\b(?:один|одна|1|по\s+одному)\s+брат", q)
    if sisters:
        n = to_number(sisters.group(1)) + 2
        return f"{n}: она сама, её сёстры и один брат на всех"
    siblings = re.search(NUMBER + r"\s+брат\w*\b.*\bу\s+каждого\b.*\b(?:одна|1)\s+сестр", q)
    if siblings:
        n = to_number(siblings.group(1)) + 1
        return f"{n}) Сестра же у всех одна и та же"
    root = re.search(r"корень\s+(?:квадратный\s+)?из\s+" + NUMBER, q)
    if root:
        value = to_number(root.group(1))
        r = round(value ** 0.5)
        return f"{r})" if r * r == value else f"Примерно {value ** 0.5:.1f}".replace(".", ",")
    ops = "|".join(sorted(OPS, key=len, reverse=True))
    chain = re.search(NUMBER + r"(?:\s+(?:" + ops + r")\s+" + NUMBER + r"){2,}", q)
    if chain and re.search(r"\b(?:сколько|посчитай|будет|равно|получится)\b|\?", q):
        tokens = re.findall(NUMBER + r"|" + ops, chain.group(0))
        flat = [x if isinstance(x, str) else next(g for g in x if g) for x in re.findall(r"(" + ops + r")|" + NUMBER, chain.group(0))]
        expr = []
        for part in re.split(r"\s+(" + ops + r")\s+", chain.group(0)):
            expr.append(OPS[part] if part in OPS else str(to_number(part)))
        try:
            value = _eval_expr(expr)
        except (ZeroDivisionError, ValueError):
            return None
        text = str(int(value)) if value.denominator == 1 else f"{float(value):.2f}".rstrip("0").replace(".", ",")
        note = ", сначала умножение" if ("*" in expr or "/" in expr) and ("+" in expr or "-" in expr) else ""
        return text + note
    m = re.search(NUMBER + r"\s+(" + ops + r")\s+" + NUMBER, q)
    if not m or not re.search(r"\b(?:сколько|посчитай|будет|равно|получится)\b|\?", q):
        return None
    a, op, b = to_number(m.group(1)), OPS[m.group(2)], to_number(m.group(3))
    if op == "/" and b == 0:
        return "На ноль же делить нельзя)"
    value = {"+": a + b, "-": a - b, "*": a * b, "/": Fraction(a, b) if b else 0}[op]
    text = str(value) if not isinstance(value, Fraction) or value.denominator == 1 else f"{float(value):.2f}".rstrip("0").replace(".", ",")
    if isinstance(value, Fraction) and value.denominator == 1:
        text = str(int(value))
    return text
