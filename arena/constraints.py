"""Conservative explicit checks and bounded offline arithmetic/word tools.

These checks are deliberately narrow. They do not claim to recognize arbitrary
entailment, negation scope or contradictions in natural language.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import re
import unicodedata

from .memory import ConversationMemory, WORD_RE, extract_explicit_facts, normalize
from .quality import response_issues


def _issue(code: str, reason: str, evidence, confidence: float = 1.0) -> dict:
    return {"code": code, "reason": reason, "confidence": confidence, "evidence": evidence}


def _safe_arithmetic(expression: str) -> Fraction:
    if len(expression) > 160 or not re.fullmatch(r"[\d\s.+\-*/%()eE]+", expression):
        raise ValueError("Expression contains unsupported characters")
    root = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(root)) > 64:
        raise ValueError("Expression is too complex")

    def visit(node, depth=0):
        if depth > 12:
            raise ValueError("Expression nesting is too deep")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if literal is None or len(literal) > 40:
                raise ValueError("Numeric literal is too long")
            exponent = re.search(r"[eE]([+-]?\d+)$", literal)
            if exponent and abs(int(exponent.group(1))) > 40:
                raise ValueError("Decimal exponent is too large")
            result = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                result = -result
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.FloorDiv):
                result = Fraction(left // right)
            elif isinstance(node.op, ast.Mod):
                result = left % right
            elif isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right) > 12:
                    raise ValueError("Exponent must be an integer in [-12,12]")
                result = left ** int(right)
            else:
                raise ValueError("Unsupported arithmetic operator")
        else:
            raise ValueError("Only numeric arithmetic is supported")
        if result.numerator.bit_length() > 256 or result.denominator.bit_length() > 256:
            raise ValueError("Arithmetic result is too large")
        return result

    return visit(root)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    # Finite decimals remain exact. Other values keep their exact fraction.
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator == 1:
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 100
            return format(Decimal(value.numerator) / Decimal(value.denominator), "f").rstrip("0").rstrip(".")
    return f"{value.numerator}/{value.denominator}"


def _graphemes(text: str) -> list[str]:
    # NFC plus combining marks is sufficient for Cyrillic word operations.
    # Emoji ZWJ sequences are not advertised as supported word characters.
    result = []
    for character in unicodedata.normalize("NFC", text):
        if unicodedata.combining(character) and result:
            result[-1] += character
        else:
            result.append(character)
    return result


def execute_local_tool(query: str) -> dict | None:
    """Return an exact supported tool result, an explicit error, or no match."""
    text = unicodedata.normalize("NFC", str(query)).strip()
    quoted_word = r"[«\"“']([^«»\"“”'\n]{1,120})[»\"”']"
    count_letter = re.fullmatch(r"(?:сколько|посчитай)\s+(?:раз\s+встречается\s+)?(?:букв[ауы]?|символ(?:ов|а)?)\s+" + quoted_word + r"\s+в\s+слове\s+" + quoted_word + r"[?.!]?", text, re.IGNORECASE)
    if count_letter:
        letter, word = count_letter.groups()
        if len(_graphemes(letter)) != 1:
            return {"ok": False, "tool": "count_letter", "error": "Нужна одна буква", "evidence": {"letter": letter, "word": word}}
        value = sum(character.casefold() == letter.casefold() for character in _graphemes(word))
        return {"ok": True, "tool": "count_letter", "value": value, "text": str(value), "evidence": {"letter": letter, "word": word}}
    bare_count = re.fullmatch(r"(?:сколько|посчитай)\s+букв[ауы]?\s+([^\W_])\s+в\s+слове\s+([^\W_]+)[?.!]?", text, re.IGNORECASE)
    if bare_count:
        letter, word = bare_count.groups()
        value = sum(character.casefold() == letter.casefold() for character in _graphemes(word))
        return {"ok": True, "tool": "count_letter", "value": value, "text": str(value), "evidence": {"letter": letter, "word": word}}
    word_length = re.fullmatch(r"(?:сколько|посчитай)\s+(букв|символов)\s+в\s+слове\s+" + quoted_word + r"[?.!]?", text, re.IGNORECASE)
    if word_length:
        unit, word = word_length.groups()
        value = sum(any(ch.isalpha() for ch in grapheme) for grapheme in _graphemes(word)) if unit.casefold() == "букв" else len(_graphemes(word))
        return {"ok": True, "tool": "word_length", "value": value, "text": str(value), "evidence": {"word": word, "unit": unit}}
    reverse_word = re.fullmatch(r"(?:напиши|запиши|произнеси)\s+(?:слово\s+)?" + quoted_word + r"\s+(?:наоборот|задом\s+наперед|задом\s+напер[её]д)[?.!]?", text, re.IGNORECASE)
    if reverse_word:
        word = reverse_word.group(1)
        value = "".join(reversed(_graphemes(word)))
        return {"ok": True, "tool": "reverse_word", "value": value, "text": value, "evidence": {"word": word}}
    arithmetic = re.fullmatch(r"(?:(?:сколько\s+(?:будет|получится)|посчитай|вычисли)\s*:?[ \t]*)?([\d\s.,+\-*/%()^eE×÷−]+)[?=!]?", text, re.IGNORECASE)
    if arithmetic and any(character.isdigit() for character in arithmetic.group(1)):
        expression = arithmetic.group(1).strip().replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**").replace(",", ".")
        # A final full stop is punctuation, not an extra numeric operator.
        expression = expression.rstrip(".").strip()
        try:
            value = _safe_arithmetic(expression)
            answer = _fraction_text(value)
            return {"ok": True, "tool": "arithmetic", "value": answer, "text": answer, "evidence": {"expression": expression}}
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as error:
            return {"ok": False, "tool": "arithmetic", "error": str(error), "evidence": {"expression": expression}}
    return None


def recall_explicit_memory(query: str, memory: ConversationMemory, *, before_turn: int | None = None) -> dict | None:
    """Answer a literal definition/name lookup from existing explicit facts.

    This is a symbolic lookup, not a response template or general interpretation
    of a follow-up. An unknown term, unsupported inflection, implicit reference,
    or person never produces a guessed answer. Only aliases explicitly retained
    by the narrow definition parser are accepted.
    """
    text = unicodedata.normalize("NFC", str(query)).strip()
    definition = re.fullmatch(
        r"(?:(?:а|тогда)\s+)?(?:напомни[, ]+)?что\s+(?:такое|значит|означает)\s+"
        r"(?:слово\s+)?[«\"“']?([^\W_]+(?:[-'][^\W_]+)*)[»\"”']?[?!.]?",
        text, re.IGNORECASE,
    )
    if definition:
        key = normalize(definition.group(1))
        definitions = [fact for fact in memory.active_facts(before_turn=before_turn, scope="conversation")
                       if fact["kind"] == "definition"]
        matches = [fact for fact in definitions if fact["key"] == key]
        if not matches:
            matches = [fact for fact in definitions if key in fact.get("aliases", [])]
        if matches:
            fact = matches[-1]
            return {"ok": True, "tool": "recall_definition", "value": fact["value"],
                    "text": str(fact["value"]), "evidence": {"fact": fact, "key": key}}
        return None
    name = re.fullmatch(r"(?:напомни[, ]+)?как\s+(тебя|меня)\s+зовут[?!.]?", text, re.IGNORECASE)
    if name:
        scope = "assistant" if name.group(1).casefold() == "тебя" else "user"
        matches = [fact for fact in memory.active_facts(before_turn=before_turn, scope=scope)
                   if fact["kind"] == "identity" and fact["key"] == "name"]
        if matches:
            fact = matches[-1]
            return {"ok": True, "tool": "recall_name", "value": fact["value"],
                    "text": str(fact["value"]), "evidence": {"fact": fact, "scope": scope}}
    return None


def _forbidden_words(query: str) -> list[str]:
    results = []
    quoted = r"[«\"“']([^«»\"“”'\n]{1,80})[»\"”']"
    patterns = (
        r"(?:не\s+(?:используй|произноси|пиши|говори|упоминай))\s+(?:слово\s+)?" + quoted,
        r"(?:без\s+слова|исключи\s+слово)\s+" + quoted,
        r"(?:не\s+(?:используй|произноси|пиши|говори|упоминай))\s+слово\s+([^\W_]+)",
    )
    for pattern in patterns:
        results.extend(match.group(1) for match in re.finditer(pattern, query, re.IGNORECASE))
    return list(dict.fromkeys(results))


def evaluate_candidate(query: str, response: str, memory: ConversationMemory | None = None) -> dict:
    issues = []
    if not str(response).strip():
        issues.append(_issue("empty", "Пустой ответ", {}))
    for quality_code in response_issues(response):
        if quality_code != "empty_response":
            issues.append(_issue(quality_code, "Ответ не проходит общую проверку публичной реплики", {"check": quality_code}))
    response_words = WORD_RE.findall(normalize(response))
    response_set = set(response_words)
    for forbidden in _forbidden_words(query):
        normalized = normalize(forbidden)
        found = normalized in response_set if len(WORD_RE.findall(normalized)) == 1 else normalized in normalize(response)
        if found:
            issues.append(_issue("forbidden_word", "Ответ содержит явно запрещённое слово", {"word": forbidden}))
    word_limit = None
    if re.search(r"\b(?:ответь|отвечай)\s+(?:ровно\s+)?одним\s+словом\b", query, re.IGNORECASE):
        word_limit = 1
    exact_count = re.search(r"\b(?:ответь|отвечай)\s+(?:ровно\s+)?(\d{1,2})\s+слов(?:ом|ами)?\b", query, re.IGNORECASE)
    if exact_count:
        word_limit = int(exact_count.group(1))
    if word_limit is not None and len(response_words) != word_limit:
        issues.append(_issue("word_count", "Не соблюдено явно заданное число слов", {"required": word_limit, "actual": len(response_words)}))
    if memory is not None:
        existing = {(fact["kind"], fact["key"]): fact for fact in memory.active_facts(scope="assistant")}
        for asserted in extract_explicit_facts(response, role="assistant", source_turn=len(memory.history)):
            previous = existing.get((asserted["kind"], asserted["key"]))
            if previous is None:
                continue
            conflict = asserted["polarity"] != previous["polarity"]
            if asserted["kind"] == "identity":
                conflict = normalize(str(asserted["value"])) != normalize(str(previous["value"]))
            if conflict:
                issues.append(_issue("explicit_contradiction", "Ответ противоречит ранее зафиксированному собственному факту", {"previous": previous, "candidate": asserted}, min(previous["confidence"], asserted["confidence"])))
        # A short "да"/"нет" can explicitly answer a known first-person fact
        # without repeating its object. Only a bare affirmative-form question is
        # supported; negated questions and conditions remain intentionally out
        # of scope because Russian yes/no polarity there is ambiguous.
        fact_question = re.fullmatch(r"(?:ты\s+)?(пь[её]шь|любишь|ешь|предпочитаешь|смотришь|слушаешь)\s+([^?!.]+)[?!.]?", query.strip(), re.IGNORECASE)
        if fact_question:
            verb, obj = fact_question.groups()
            operation = {"пьешь": "drink", "любишь": "like", "ешь": "eat", "предпочитаешь": "prefer", "смотришь": "watch", "слушаешь": "listen"}[normalize(verb)]
            previous = existing.get(("preference", operation + ":" + normalize(obj.strip())))
            opening = re.match(r"^\s*(да|нет|ага|угу|конечно)\b", response, re.IGNORECASE)
            first_words = response_words[:4]
            # "Да нет" and a prompt correction are not unambiguous affirmations.
            ambiguous = opening is not None and opening.group(1).casefold() != "нет" and ("нет" in first_words[1:] or "не" in first_words[1:])
            if previous is not None and opening is not None and not ambiguous:
                polarity = -1 if opening.group(1).casefold() == "нет" else 1
                if polarity != previous["polarity"]:
                    issues.append(_issue("explicit_answer_contradiction", "Краткий ответ противоречит известному факту, о котором прямо спросили", {"question": query, "answer_polarity": polarity, "previous": previous}, .96))
    tool_result = execute_local_tool(query)
    # A successful tool result is offered as evidence; a prose answer is not
    # rejected merely because its exact surface string differs from the number.
    return {"accepted": not any(issue["confidence"] >= .95 for issue in issues), "issues": issues, "tool_result": tool_result}

