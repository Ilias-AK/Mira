"""Conservative public-response filters shared by data, retrieval and runtime.

This is a finite set of literal checks, not a semantic safety classifier. Ordinary
discussion of illness, death, disagreement, or advice against self-harm is not a
reason to reject a response. The checks address concrete corpus failure modes.
"""
from __future__ import annotations

import re
import unicodedata


_PROFANITY = re.compile(
    r"(?<!\w)(?:бляд\w*|блять|бля|пизд\w*|хуй\w*|ху[еёя]\w*|"
    r"(?:за|вы|на|от|по|пере|раз|у|в)?[её]б(?:а|у|и|е|ё|ну|ло)\w*|"
    r"нахуй|похуй|нихуя|нахер)(?!\w)", re.IGNORECASE,
)
_TARGETED_ABUSE = re.compile(
    r"\b(?:ты|вы|он|она|они)\s+(?:(?:просто|полный|полная|полные|тупой|тупая|тупые)\s+)?"
    r"(?:идиот(?:ы|ка|ки|ом)?|дебил(?:ы|ка|ки|ом)?|дура(?:к|ки|ком)?|сука|суки|урод(?:ы|ина)?)\b",
    re.IGNORECASE,
)
_GRAPHIC_ACTION = re.compile(
    r"\b(?:отрез(?:ать|ал[аи]?|аю|ает|ают|ай(?:те)?)|отреж(?:ь(?:те)?|у|ешь|ет|ем|ете|ут)|"
    r"отруб(?:ить|ил[аи]?|лю|ишь|ит|им|ите|ят|и(?:те)?)|"
    r"вскры(?:ть|л[аи]?|вай(?:те)?|ваю|вает|вают)|вскро(?:й(?:те)?|ю|ешь|ет|ем|ете|ют)|"
    r"перерез(?:ать|ал[аи]?|аю|ает|ают|ай(?:те)?)|перереж(?:ь(?:те)?|у|ешь|ет|ем|ете|ут)|"
    r"выкол(?:оть|ол[аи]?|и(?:те)?)|выдав(?:ить|ил[аи]?|лю|ишь|ит|им|ите|ят|и(?:те)?)|"
    r"проткн(?:уть|ул[аи]?|у|ешь|ет|ем|ете|ут|и(?:те)?)|"
    r"прострел(?:ить|ил[аи]?|ю|ишь|ит|им|ите|ят|и(?:те)?))\b"
    r"[^.!?\n]{0,60}?\b(?:палец|пальц\w*|руку|руки|рук|ногу|ноги|ног|"
    r"вен[ыау]|горл[оау]|глаз\w*|голов[уаы]|кишк\w*)\b",
    re.IGNORECASE,
)
_SELF_HARM = re.compile(
    r"\b(?:уби(?:ть|вай|вать|л|ла)|убей)\s+себя\b|"
    r"\bпокончи(?:ть|л|ла)?\s+с\s+собой\b|"
    r"\b(?:повесься|повеситься)\b|"
    r"\b(?:причини|сделай)\s+себе\s+(?:боль|больно|вред)\b",
    re.IGNORECASE,
)
_VIOLENT_COMMAND = re.compile(
    r"\b(?:убей|зарежь|пристрели|застрели|задуши)\s+(?:его|ее|её|их|человека|соседа|родителей)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.|(?:t|telegram)\.me/)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-zа-я]{2,}(?!\w)", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+\d|8\s*\()?[\d][\d\s()\-]{7,}\d(?!\w)")
_PHONE_CUE = re.compile(r"(?:телефон|позвони|звони|номер|whatsapp|ватсап|вотсап)", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"\bя\s+(?:являюсь\s+)?(?:большая\s+|большой\s+)?языков(?:ая|ой)\s+модел(?:ь|ью)\b|"
    r"\bкак\s+(?:языковая\s+модель|искусственный\s+интеллект|ии)\b|"
    r"\bя\s+(?:чат[- ]?бот|бот|нейросеть|искусственный\s+интеллект)\b|"
    r"\bas\s+an?\s+(?:ai|language\s+model)\b",
    re.IGNORECASE,
)


def _negated_before(text: str, position: int) -> bool:
    prefix = re.split(r"[.!?;\n]", text[:position])[-1][-70:]
    return bool(re.search(r"\bне\s+(?:[^\W_]+\s+){0,3}$", prefix, re.IGNORECASE))


def _nonnegated_match(pattern: re.Pattern, text: str) -> bool:
    return any(not _negated_before(text, match.start()) for match in pattern.finditer(text))


def response_issues(text: str, *, include_style: bool = True) -> list[str]:
    """Return stable technical reason codes; never annotate the response text."""
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    issues = []
    if not normalized:
        return ["empty_response"]
    if _PROFANITY.search(normalized):
        issues.append("explicit_profanity")
    if _TARGETED_ABUSE.search(normalized):
        issues.append("targeted_abuse")
    if _nonnegated_match(_GRAPHIC_ACTION, normalized):
        issues.append("graphic_violence")
    if _nonnegated_match(_SELF_HARM, normalized):
        issues.append("self_harm_statement_or_instruction")
    if _nonnegated_match(_VIOLENT_COMMAND, normalized):
        issues.append("violent_instruction")
    if _URL.search(normalized):
        issues.append("contact_url")
    if _EMAIL.search(normalized):
        issues.append("contact_email")
    for match in _PHONE.finditer(normalized):
        candidate = match.group(0)
        digits = sum(character.isdigit() for character in candidate)
        context = normalized[max(0, match.start() - 35):match.start()]
        if 10 <= digits <= 15 and (candidate.startswith("+") or "(" in candidate or _PHONE_CUE.search(context)):
            issues.append("contact_phone")
            break
    if include_style and _BOILERPLATE.search(normalized):
        issues.append("assistant_boilerplate")
    return issues


def acceptable_response(text: str, *, include_style: bool = True) -> bool:
    return not response_issues(text, include_style=include_style)

