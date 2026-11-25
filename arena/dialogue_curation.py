"""Deterministic dialogue pools for a controlled continuation experiment.

Selection is deliberately conservative for noisy single-turn Telegram pairs.
Topic and lexical checks are weak heuristics, not semantic quality labels.
Source utterances, identities, personas and conversation boundaries are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import shutil
import time

from .data import normalized_fingerprint_text, record_fingerprint, sha256_file
from .data_expansion import candidate_problem
from .formatting import question_and_history
from .runtime import output_problem
from .tokenization import ArenaTokenizer


TOPICS = {
    "books": r"книг\w*|чита\w*|прочит\w*|писател\w*|роман(?:ы|ов|а)?|рассказ\w*|библиотек\w*|литератур\w*|фантастик\w*|детектив\w*|стих\w*|поэзи\w*",
    "music": r"музык\w*|песн\w*|песен\w*|концерт\w*|гитар\w*|пианин\w*|фортепиан\w*|джаз\w*|рок|попс\w*|мелоди\w*|классик\w*",
    "film": r"фильм\w*|кино\w*|сериал\w*|мультфильм\w*|мультик\w*|комеди\w*|драм\w*|режиссер\w*|режиссёр\w*|театр\w*|спектакл\w*",
    "cooking": r"готов(?:ить|ит|ят|ишь|им|лю|ил\w*|к\w*|л\w*)|блюд\w*|рецепт\w*|кухн\w*|вкусн\w*|пирог\w*|торт\w*|суп(?:а|у|ом|е|ы|ов|чик\w*)?|салат\w*|картош\w*|картофел\w*|макарон\w*|рис|каш(?:а|и|у|е|ей|ами|ах|ка|ку|ки)|яйц\w*|яичниц\w*|омлет\w*|печен\w*|печён\w*|блин\w*|пицц\w*|овощ\w*|фрукт\w*|мяс\w*|рыб\w*|чай\w*|кофе\w*|завтрак\w*|ужин\w*|обед\w*",
    "travel": r"путешеств\w*|поездк\w*|поезд\w*|туризм\w*|турист\w*|отпуск\w*|отдых\w*|отдох\w*|мор[еяю]|пляж\w*|гор[ыау]|горы|палатк\w*|поход\w*|чемодан\w*|рюкзак\w*|экскурси\w*|музе\w*",
    "pets": r"кошк\w*|кот(?:а|у|ом|ы|ов|ик\w*|ят\w*|ён\w*)?|собак\w*|щен\w*|питом\w*|попуга\w*|хомяк\w*|аквариум\w*|рыбк\w*|животн\w*",
    "home": r"квартир\w*|комнат\w*|дом(?:а|у|ом|е|ик\w*)?|уют\w*|мебел\w*|диван\w*|кресл\w*|ремонт\w*|обо[иья]\w*|балкон\w*|цвет\w*|растен\w*|дач\w*|сад\w*|огород\w*|яблон\w*|уборк\w*|поряд\w*",
    "hobbies": r"хобби|увлеч\w*|увлек\w*|рисов\w*|рису\w*|картин\w*|фото\w*|вяза\w*|вяж\w*|вышив\w*|рукодел\w*|коллекц\w*|мастер\w*|пазл\w*|шахмат\w*|шашк\w*|игр\w*|игра\w*",
    "outdoors": r"прогул\w*|гуля\w*|парк\w*|лес\w*|природ\w*|погод\w*|дожд\w*|солнц\w*|солнеч\w*|снег\w*|зим\w*|лето|летом|осен\w*|осень|весн\w*|велосипед\w*|лыж\w*|коньк\w*|плав\w*|бассейн\w*|бег\w*|спорт\w*|трениров\w*",
    "learning": r"учеб\w*|учёб\w*|учит\w*|учусь|учим\w*|изуч\w*|науч\w*|язык\w*|английск\w*|запомин\w*|памят\w*|экзамен\w*|урок\w*|школ\w*|курс\w*|студент\w*|университет\w*|математик\w*",
    "daily_life": r"выходн\w*|вечер\w*|утро|утром|свободн\w*|скуч\w*|сон|спать|сплю|высп\w*|праздник\w*|подар\w*|поздрав\w*|друз\w*",
}
TOPIC_PATTERNS = {key: re.compile(r"\b(?:" + value + r")\b", re.I) for key, value in TOPICS.items()}
_OUT_OF_SCOPE = re.compile(r"\b(?:секс\w*|половы\w*|эрот\w*|интим\w*|измен\w*|развод\w*|любовниц\w*|алкогол\w*|водк\w*|пив\w*|вино|коньяк\w*|курени\w*|курит\w*|сигарет\w*|наркот\w*|травк\w*|таблет\w*|лекарств\w*|лечени\w*|лечит\w*|болезн\w*|болит\w*|врач\w*|диагноз\w*|витамин\w*|беремен\w*|гормон\w*|похуде\w*|диет\w*|самоуб\w*|суицид\w*|убить\w*|убий\w*|насили\w*|войн\w*|военн\w*|оруж\w*|политик\w*|путин\w*|украин\w*|наци\w*|фаши\w*|депутат\w*|президент\w*|церков\w*|религи\w*|православ\w*|мусульм\w*|бог|бога|грех\w*|кредит\w*|инвест\w*|договор\w*|налог\w*|закон\w*|штраф\w*|адвокат\w*|алименты|зарплат\w*)\b", re.I)
_RUDE = re.compile(r"\b(?:дурак\w*|дур[аы]\w*|идиот\w*|дебил\w*|туп\w*|урод\w*|долб\w*|кретин\w*|ненавиж\w*|ненавист\w*|дохн\w*|сдох\w*|заткни\w*|отвали\w*|жри\w*|сожри\w*|рожа\w*|рожи\w*|морда\w*|рж[ау]\w*|хрен\w*|фига\w*|фигн\w*|сра\w*|гадост\w*)\b", re.I)
_GENERALIZATION = re.compile(r"\b(?:мужчин\w*|женщин\w*|баб[аы]\w*|мужик\w*|тёлк\w*|телк\w*)\b", re.I)
_META = re.compile(r"\b(?:тдв|нф|ответах|ответы\s+мейл|мейл|мэйл|аватар\w*|никнейм\w*|вопросы\s+лидеры|скрин\w*|форум\w*|чатик\w*|админ\w*|модератор\w*)\b", re.I)
_TAIL = re.compile(r"\b(?:и|или|но|потому|чтобы|если|который|которая|которые|так\s+как|например|сам|сама)\s*[,.!?…]*$", re.I)
_RETORT = re.compile(r"^(?:а\s+(?:тебе|вам)\s+(?:какое|зачем)|сам(?:а)?\s+(?:подумай|знаешь)|не\s+твое\s+дело|не\s+твоё\s+дело)", re.I)
_MORE_NOISE = re.compile(r"\b(?:дерьм\w*|говн\w*|фуу+\w*|фу|гадк\w*|мерзк\w*|уродств\w*|пошл\w*|кастр\w*|извращ\w*|чмо\w*|сволоч\w*|шлюх\w*|проститут\w*)\b", re.I)
_SPECIALIST = re.compile(r"\b(?:вылеч\w*|леч(?:у|ат|ит|ить|ишь|им|и|ись|ился|илась)|кашл\w*|кашель|простуд\w*|боле(?:ть|л|ла|ли|ет|ют)|боляч\w*|давлен\w*|кров\w*|онкол\w*|онкоз\w*|заболев\w*|тонометр\w*|привив\w*|симптом\w*|инфекц\w*|аллерг\w*|здоров\w*|антибиот\w*|полици\w*|милици\w*|войск\w*|приватизац\w*|прописк\w*|регистрац\w*|кондиционер\w*|электрик\w*|проводк\w*|парол\w*|логин\w*|вконтакт\w*|контакт\w*|неисправ\w*)\b", re.I)
_ROMANCE = re.compile(r"\b(?:парн\w*|парен\w*|девуш\w*|любов\w*|любви|влюб\w*|ревност\w*|ревнив\w*|флирт\w*)\b", re.I)
_FOREIGN_IDENTITY = re.compile(
    r"\b(?:я|меня|мне|мой|моя|мо[её]|мои)\b[^.!?\n]{0,100}"
    r"\b(?:языков\w*\s+(?:ai[- ]?)?модел\w*|(?:ai|ии)[- ]?модел\w*|laion|openassistant|chatgpt|создател\w*|обуча[лн]\w*|запрограммир\w*)\b", re.I)
_EXTRA_EXCLUDE = re.compile(r"\b(?:бух\w*|кабак\w*|вин[аоуы]|шампан\w*|узи|яйцеклет\w*|фолликул\w*|пенис\w*|сперм\w*|эрект\w*|рак|рака|раку|печен\w*|сустав\w*|гинек\w*|верующ\w*|христиан\w*|язычник\w*|свадьб\w*|женить\w*|замуж\w*|свидан\w*|зодиак\w*|гороскоп\w*|самц\w*|самок|нижн\w*|пист\w*|педик\w*|пиндос\w*|хохл\w*|москал\w*|чурк\w*|жид\w*|нигг\w*|сцук\w*|сцуко|сук[аи]\w*|понос\w*|госпереворот\w*|власт\w*|электорат\w*|фпс|дот[аыуе]|дотер\w*|дотёр\w*)\b", re.I)
_PERSONAL_QUESTION = re.compile(r"\b(?:ты|вы|тебе|вам|тебя|вас|твой|твоя|твои|ваш\w*|любите|любишь|предпочитаете|предпочитаешь|умеете|умеешь|бывали|бывал|бывала|читали|читал|читала|смотрели|смотрел|смотрела)\b", re.I)
_PERSONAL_ANSWER = re.compile(r"\b(?:я|мне|меня|мой|моя|мо[её]|мои\w*|мы|нам|нас|люблю|нравится|нравятся|предпочитаю|выбираю|обычно|часто|иногда|смотрю|читаю|готовлю|гуляю|хожу|пою|играю|собираю|катаюсь|рисую|делаю|езжу|путешествую|мечтаю|планирую|работаю|учу|учусь|стараюсь)\b", re.I)
_CONCRETE_REQUEST = re.compile(r"(?:\b(?:что|как)\s+(?:можно\s+|лучше\s+|бы\s+)?(?:приготов\w*|готов\w*|свар\w*|испеч\w*|почит\w*|посмотр\w*)|\b(?:посовет\w*|порекоменд\w*)[^?]{0,70}\b(?:книг\w*|фильм\w*|блюд\w*|рецепт\w*)|\b(?:как|где|какую|какой|какие)\b[^?]{0,120}\b(?:готовить|приготовить|сварить|испечь|почитать|прочитать|посмотреть|выбрать|сделать|научиться|потренировать|запомнить|вырастить|посадить|ухаживать|погулять|сходить)\b)", re.I)
_ACTIONABLE_ANSWER = re.compile(r"\b(?:возьм\w*|добав\w*|нареж\w*|смеш\w*|вари\w*|жар\w*|готов\w*|чита\w*|выбер\w*|полож\w*|взбей\w*|купи\w*|ищи\w*|найд\w*|сход\w*|попроб\w*|начн\w*|начат\w*|повтор\w*|обрат\w*|проверь\w*|использ\w*|постав\w*|перейд\w*|сдел\w*|совет\w*|можно|нужно|лучше|подойд\w*|стоит)\b", re.I)
_MISSING_REFERENCE = re.compile(r"(?:\b(?:как\s+вам|что\s+за|что\s+на|оцени\w*)\b[^?]{0,45}\b(?:фото|картин\w*|рисун\w*|изображен\w*)|\b(?:на\s+фото|на\s+картинке|в\s+профиле|в\s+моем\s+мире|в\s+моём\s+мире|в\s+агенте)\b)", re.I)
_ABSTRACT_NOISE = re.compile(r"(?:\b(?:счасть\w*|судьб\w*|личност\w*|ясновид\w*|телепат\w*)\b|читать\s+мысли|играешь\s+во\s+взрослого)", re.I)
_WORDS = re.compile(r"[а-яё]+", re.I)
_STOP = set("почему потому когда тогда какой какая какие каких сколько только можно нужно будет будут было были есть если чтобы такой такая такие этого эту этот меня тебе тебя вами сами очень просто вообще всего всегда никогда каждый человек люди здесь туда этом этим свои своим себя сказал скажи могу может хотел хочу хочется люблю нравится самый самое лучше именно вопрос ответ чем кто что как где куда зачем какую которой который которое которые это для про при под над без или уже ещё еще раз два три вот так там мне вам она они оно мой моя мои твой твоя твои наш нас вас его её своих свой своей тоже даже хотя хорошо плохо много мало сейчас потом самом сама сам".split())


def text_hash(text):
    return hashlib.sha256(normalized_fingerprint_text(text).encode("utf-8")).hexdigest()


def topics(text):
    return {key for key, pattern in TOPIC_PATTERNS.items() if pattern.search(text)}


def content_prefixes(text):
    return {word[:5] for word in _WORDS.findall(text.casefold()) if len(word) >= 5 and word not in _STOP}


def telegram_problem(question, answer):
    """Restrict noisy pairs to everyday, textually connected conversations."""
    initial = candidate_problem(question, answer, strict_chat=True)
    if initial:
        return initial
    if len(question) > 240 or len(answer) > 260:
        return "long_single_turn"
    combined = question + "\n" + answer
    if "?" not in question and not re.match(r"^(?:посовет\w*|подскаж\w*|расскаж\w*)\b", question, re.I):
        return "not_explicit_question_or_request"
    for name, pattern in (("unreviewed_domain", _OUT_OF_SCOPE), ("rude", _RUDE),
                          ("gender_generalization_scope", _GENERALIZATION), ("platform_context", _META),
                          ("rude_or_adult_scope", _MORE_NOISE), ("specialist_unreviewed", _SPECIALIST),
                          ("romantic_generalization_scope", _ROMANCE)):
        if pattern.search(combined):
            return name
    if _TAIL.search(answer):
        return "unfinished_answer"
    if _RETORT.search(answer):
        return "retort"
    question_topics = topics(question)
    if not question_topics:
        return "outside_everyday_topics"
    answer_topics = topics(answer)
    shared = content_prefixes(question) & content_prefixes(answer)
    if not (question_topics & answer_topics) and not shared:
        return "no_topic_or_content_link"
    if "?" in answer and len(_WORDS.findall(answer)) < 8:
        return "short_counterquestion"
    if output_problem(answer):
        return "runtime_" + output_problem(answer)
    return None


def precision_class(question, answer):
    """Return a narrow dialogue type or None; still not semantic certification."""
    combined = question + "\n" + answer
    if _EXTRA_EXCLUDE.search(combined) or _MISSING_REFERENCE.search(question) or _ABSTRACT_NOISE.search(combined):
        return None
    if "?" in answer or len(_WORDS.findall(answer)) < 5:
        return None
    if _PERSONAL_QUESTION.search(question) and _PERSONAL_ANSWER.search(answer):
        return "personal_experience"
    if _CONCRETE_REQUEST.search(question) and _ACTIONABLE_ANSWER.search(answer) and len(_WORDS.findall(answer)) >= 7:
        return "concrete_help"
    return None


class HeldoutProtection:
    def __init__(self, paths):
        self.ids, self.groups, self.pairs, self.questions, self.utterances = set(), set(), set(), set(), set()
        self.paths = {}
        for path in paths:
            path = Path(path)
            self.paths[str(path)] = sha256_file(path)
            for record in read_jsonl(path):
                if record.get("kind") != "dialogue":
                    continue
                self.ids.add(record["id"])
                self.groups.add(record["group_id"])
                self.pairs.add(record_fingerprint(record))
                q, _ = question_and_history(record)
                self.questions.add(text_hash(q))
                for text in [record["response"], *[turn["text"] for turn in record["context"]]]:
                    if len(normalized_fingerprint_text(text)) >= 24:
                        self.utterances.add(text_hash(text))

    def problem(self, record, *, strict_text=True):
        if record["id"] in self.ids or record["group_id"] in self.groups:
            return "heldout_identity"
        if record_fingerprint(record) in self.pairs:
            return "heldout_pair"
        # Existing corpus can legitimately share generic greetings between groups.
        # New sources additionally reject any exact held-out question or longer turn.
        if strict_text:
            q, _ = question_and_history(record)
            if text_hash(q) in self.questions:
                return "heldout_question"
            if any(text_hash(text) in self.utterances for text in
                   [record["response"], *[turn["text"] for turn in record["context"]]]):
                return "heldout_utterance"
        return None


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finalize_identity_filter(output_dir):
    """Remove incompatible assistant self-identities without rewriting a target.

    This explicit final pass is also usable for a completed curation made by an
    earlier selector. Previous manifests and removed records remain auditable.
    """
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity_filter_complete"):
        return manifest
    before_bytes = manifest_path.read_bytes()
    (root / "manifest_before_identity.json").write_bytes(before_bytes)
    removed, excluded_ids = [], set()
    for name in ("oasst", "oasst2"):
        path = root / "pools" / f"{name}.jsonl"
        records = list(read_jsonl(path))
        kept = []
        for record in records:
            if _FOREIGN_IDENTITY.search(record["response"]):
                removed.append(record)
                excluded_ids.add(record["id"])
            else:
                kept.append(record)
        if len(kept) != len(records):
            path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in kept), encoding="utf-8", newline="\n")
        manifest["counts"][name]["rejected_foreign_assistant_identity"] = len(records) - len(kept)
        manifest["counts"][name]["retained"] = len(kept)
        manifest["counts"][name]["multi_turn"] = sum(len(r["context"]) > 1 for r in kept)
        entry = f"pools/{name}.jsonl"
        manifest["outputs"].pop(entry.replace("/", "\\"), None)
        manifest["outputs"][entry] = {"sha256": sha256_file(path), "records": len(kept)}
    if excluded_ids:
        tokenizer = ArenaTokenizer.load(manifest.get("tokenizer_path", "artifacts/tokenizer.json"))
        for name in ("oasst", "oasst2"):
            token_count = sum(len(tokenizer.encode(r["response"])) for r in read_jsonl(root / "pools" / f"{name}.jsonl"))
            manifest["counts"][name]["target_tokens"] = token_count
            manifest["counts"][name]["mean_target_tokens"] = token_count / max(1, manifest["counts"][name]["retained"])
        train = root / "train.jsonl"
        rows = [r for r in read_jsonl(train) if r["id"] not in excluded_ids]
        train.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8", newline="\n")
        manifest["outputs"]["train.jsonl"] = {"sha256": sha256_file(train), "records": len(rows),
                                                "sources": dict(Counter(r["source"] for r in rows))}
    samples = {}
    for name in ("toloka", "oasst", "oasst2", "telegram_pairs"):
        reservoir, rng = [], random.Random(f"{manifest['seed']}:{name}")
        for i, record in enumerate(read_jsonl(root / "pools" / f"{name}.jsonl")):
            if len(reservoir) < 100:
                reservoir.append(record)
            else:
                index = rng.randrange(i + 1)
                if index < 100:
                    reservoir[index] = record
        samples[name] = reservoir
    (root / "audit_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Deterministic random reservoir sample after selection; not a quality certificate."]
    for name, rows in samples.items():
        lines.append("\nSOURCE: " + name)
        for index, record in enumerate(rows):
            question, _ = question_and_history(record)
            lines.extend([f"\n{index}. {record['id']}", "Q: " + question, "A: " + record["response"]])
    (root / "audit_sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "identity_exclusions.json").write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("audit_sample.json", "audit_sample.txt", "identity_exclusions.json", "manifest_before_identity.json"):
        manifest["outputs"][name] = {"sha256": sha256_file(root / name)}
    manifest["identity_filter_complete"] = True
    manifest["identity_filter_code_sha256"] = sha256_file(__file__)
    manifest["identity_excluded_records"] = len(removed)
    manifest["identity_filter_note"] = "Only incompatible first-person assistant identity targets removed; no generated replacements. Earlier selection source hash is retained separately."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_precision_audit(output_dir):
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    grouped = {"personal_experience": [], "concrete_help": []}
    for record in read_jsonl(root / "pools/telegram_pairs.jsonl"):
        question, _ = question_and_history(record)
        category = precision_class(question, record["response"])
        if category is None:
            raise ValueError(f"Unexpected non-precision row: {record['id']}")
        grouped[category].append(record)
    samples, lines = {}, []
    for category, records in grouped.items():
        # Stable independent samples per class, separate from the combined reservoir.
        samples[category] = random.Random(f"precision-audit:{manifest['seed']}:{category}").sample(records, min(100, len(records)))
        lines.append("CLASS: " + category)
        for index, record in enumerate(samples[category]):
            q, _ = question_and_history(record)
            lines.extend([f"\n{index}. {record['id']}", "Q: " + q, "A: " + record["response"]])
        class_path = root / "pools" / f"telegram_{category}.jsonl"
        class_path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records), encoding="utf-8", newline="\n")
        manifest["outputs"][class_path.relative_to(root).as_posix()] = {"sha256": sha256_file(class_path), "records": len(records), "subset_of": "pools/telegram_pairs.jsonl"}
    (root / "precision_audit.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "precision_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["precision_class_counts"] = {name: len(records) for name, records in grouped.items()}
    for name in ("precision_audit.json", "precision_audit.txt"):
        manifest["outputs"][name] = {"sha256": sha256_file(root / name)}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def apply_manual_exclusions(output_dir, exclusions_path):
    """Apply documented human/agent review decisions, retaining the pre-review audit."""
    root, decisions_path = Path(output_dir), Path(exclusions_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision_hash = sha256_file(decisions_path)
    if manifest.get("manual_exclusions_sha256") == decision_hash:
        return manifest
    if manifest.get("manual_exclusions_sha256"):
        raise ValueError("Manual review was already applied; make a separate new curation for a new decision set")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))["exclusions"]
    excluded_ids = {row["id"] for row in decisions}
    if len(excluded_ids) != len(decisions) or any(not row.get("reason") for row in decisions):
        raise ValueError("Manual decisions need unique IDs and nonempty reasons")
    train_rows = list(read_jsonl(root / "train.jsonl"))
    unknown = excluded_ids - {row["id"] for row in train_rows}
    if unknown:
        raise ValueError(f"Manual decisions reference missing training IDs: {sorted(unknown)}")
    (root / "manifest_before_manual.json").write_bytes(manifest_path.read_bytes())
    for name in ("audit_sample.json", "audit_sample.txt", "precision_audit.json", "precision_audit.txt"):
        if (root / name).is_file():
            backup = root / (Path(name).stem + "_before_manual" + Path(name).suffix)
            backup.write_bytes((root / name).read_bytes())
            manifest["outputs"][backup.name] = {"sha256": sha256_file(backup)}
    tokenizer = ArenaTokenizer.load(manifest.get("tokenizer_path", "artifacts/tokenizer.json"))
    samples = {}
    for name in ("toloka", "oasst", "oasst2", "telegram_pairs"):
        pool = root / "pools" / f"{name}.jsonl"
        original = list(read_jsonl(pool))
        removed = [row for row in original if row["id"] in excluded_ids]
        kept = [row for row in original if row["id"] not in excluded_ids]
        if removed:
            pool.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in kept), encoding="utf-8", newline="\n")
            counter = manifest["counts"][name]
            counter["rejected_manual_review"] = len(removed)
            counter["retained"] = len(kept)
            counter["multi_turn"] -= sum(len(r["context"]) > 1 for r in removed)
            counter["target_tokens"] -= sum(len(tokenizer.encode(r["response"])) for r in removed)
            counter["mean_target_tokens"] = counter["target_tokens"] / max(1, len(kept))
            if name == "telegram_pairs":
                for record in removed:
                    q, _ = question_and_history(record)
                    for topic in topics(q):
                        counter["topic_" + topic] -= 1
                    counter["selection_class_" + str(precision_class(q, record["response"]))] -= 1
            manifest["outputs"][pool.relative_to(root).as_posix()] = {"sha256": sha256_file(pool), "records": len(kept)}
        reservoir, rng = [], random.Random(f"{manifest['seed']}:{name}")
        for i, record in enumerate(kept):
            if len(reservoir) < 100:
                reservoir.append(record)
            else:
                index = rng.randrange(i + 1)
                if index < 100:
                    reservoir[index] = record
        samples[name] = reservoir
    train_rows = [r for r in train_rows if r["id"] not in excluded_ids]
    train = root / "train.jsonl"
    train.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in train_rows), encoding="utf-8", newline="\n")
    manifest["outputs"]["train.jsonl"] = {"sha256": sha256_file(train), "records": len(train_rows), "sources": dict(Counter(r["source"] for r in train_rows))}
    (root / "audit_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Post-review reservoir: regenerated after exclusions; no semantic grade is implied."]
    for name, rows in samples.items():
        lines.append("\nSOURCE: " + name)
        for index, record in enumerate(rows):
            q, _ = question_and_history(record)
            lines.extend([f"\n{index}. {record['id']}", "Q: " + q, "A: " + record["response"]])
    (root / "audit_sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "manual_exclusions.json").write_bytes(decisions_path.read_bytes())
    for name in ("audit_sample.json", "audit_sample.txt", "manual_exclusions.json", "manifest_before_manual.json"):
        manifest["outputs"][name] = {"sha256": sha256_file(root / name)}
    manifest["manual_exclusions_sha256"] = decision_hash
    manifest["manual_exclusions_count"] = len(excluded_ids)
    manifest["manual_filter_code_sha256"] = sha256_file(__file__)
    manifest["manual_review_note"] = "Audit grades refer to pre-exclusion samples. Removal of reviewed bad rows must not inflate an independent quality estimate."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return write_precision_audit(root)


def reviewed_exclusions(expansion):
    """Known problematic training rows are excluded, not rewritten as good answers."""
    excluded = {}
    for source in ("telegram", "oasst2"):
        path = Path(expansion) / source / "sample_audit.json"
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")).get("examples", []):
            if row["assessment"].startswith("exclude_"):
                excluded[row["id"]] = row["assessment"]
    return excluded


def prepare(frozen_dir, expansion_dir, output_dir, *, tokenizer_path="artifacts/tokenizer.json", seed=20261121, telegram_limit=0, toloka_limit=0, from_pools=None, telegram_profile="precision"):
    started = time.perf_counter()
    frozen, expansion, output = Path(frozen_dir), Path(expansion_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite curation output: {output}")
    tokenizer = ArenaTokenizer.load(tokenizer_path)
    protection = HeldoutProtection([frozen / "val.jsonl", frozen / "test.jsonl",
                                   expansion / "oasst2/val.jsonl", expansion / "oasst2/test.jsonl"])
    excludes = reviewed_exclusions(expansion)
    output.mkdir(parents=True)
    pools = output / "pools"
    pools.mkdir()
    counts, samples, seen_pairs, seen_ids, response_counts = {}, {}, set(), set(), Counter()
    inputs = {}
    outputs = {}
    sources = [("toloka", frozen / "train.jsonl"), ("oasst", frozen / "train.jsonl"),
               ("oasst2", expansion / "oasst2/train.jsonl"), ("telegram_pairs", expansion / "telegram/train.jsonl")]
    parent_manifest = None
    if from_pools is not None:
        previous = Path(from_pools)
        parent_manifest = {"path": str(previous / "manifest.json"), "sha256": sha256_file(previous / "manifest.json")}
        parent_data = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        for name, _ in sources:
            relative = f"pools/{name}.jsonl"
            entry = parent_data["outputs"].get(relative, parent_data["outputs"].get(relative.replace("/", "\\")))
            if entry is None or sha256_file(previous / relative) != entry["sha256"]:
                raise ValueError(f"Parent curation pool changed: {relative}")
        sources = [(name, previous / "pools" / f"{name}.jsonl") for name, _ in sources]
    for source_name, source_path in sources:
        inputs[str(source_path)] = sha256_file(source_path)
        counter, sample, rng = Counter(), [], random.Random(f"{seed}:{source_name}")
        destination = pools / f"{source_name}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in read_jsonl(source_path):
                if record.get("kind") != "dialogue" or record.get("source") != source_name:
                    continue
                if record.get("split") != "train":
                    raise ValueError(f"Non-training record in training source: {record.get('id')}")
                counter["input"] += 1
                problem = protection.problem(record, strict_text=source_name in ("oasst2", "telegram_pairs"))
                question, _ = question_and_history(record)
                answer = record["response"]
                if not problem and record["id"] in excludes:
                    problem = "reviewed_" + excludes[record["id"]]
                if not problem and source_name == "telegram_pairs":
                    problem = telegram_problem(question, answer)
                    if not problem and telegram_profile == "precision" and precision_class(question, answer) is None:
                        problem = "outside_precision_dialogue_classes"
                elif not problem:
                    problem = candidate_problem(question, answer)
                    if not problem and output_problem(answer):
                        problem = "runtime_" + output_problem(answer)
                pair = record_fingerprint(record)
                if not problem and (record["id"] in seen_ids or pair in seen_pairs):
                    problem = "duplicate_pair_or_id"
                answer_key = text_hash(answer)
                if not problem and source_name == "telegram_pairs" and response_counts[answer_key] >= 3:
                    problem = "generic_response_cap"
                target_tokens = 0
                if not problem:
                    target_tokens = len(tokenizer.encode(answer))
                    counter["otherwise_eligible"] += 1
                    counter["otherwise_eligible_tokens"] += target_tokens
                    counter["otherwise_eligible_le127"] += target_tokens <= 127
                    if target_tokens > 95:
                        problem = "target_tokens_over95"
                if problem:
                    counter["rejected_" + problem] += 1
                    continue
                seen_ids.add(record["id"])
                seen_pairs.add(pair)
                response_counts[answer_key] += 1
                counter["retained"] += 1
                counter["multi_turn"] += len(record["context"]) > 1
                counter["target_tokens"] += target_tokens
                if source_name == "telegram_pairs":
                    counter.update("topic_" + topic for topic in topics(question))
                    counter["selection_class_" + str(precision_class(question, answer))] += 1
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if len(sample) < 100:
                    sample.append(record)
                else:
                    index = rng.randrange(counter["retained"])
                    if index < 100:
                        sample[index] = record
        counts[source_name] = dict(counter) | {"mean_target_tokens": counter["target_tokens"] / max(1, counter["retained"])}
        samples[source_name] = sample
        outputs[destination.relative_to(output).as_posix()] = {"sha256": sha256_file(destination), "records": counter["retained"]}
    # Optional deterministic caps choose rows by identity hash, not source file position.
    # The pools remain complete; the mixture never repeats examples to inflate its size.
    chosen = []
    for source_name, _ in sources:
        path = pools / f"{source_name}.jsonl"
        limit = telegram_limit if source_name == "telegram_pairs" else toloka_limit if source_name == "toloka" else 0
        if limit:
            chosen.extend(heapq.nsmallest(limit, read_jsonl(path), key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).digest()))
        else:
            chosen.extend(read_jsonl(path))
    random.Random(seed).shuffle(chosen)
    train_path = output / "train.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in chosen:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    mixture_counts = Counter(row["source"] for row in chosen)
    outputs["train.jsonl"] = {"sha256": sha256_file(train_path), "records": len(chosen), "sources": dict(mixture_counts)}
    # Keep the old validation bit-identical for comparable NLL and review IDs.
    for split in ("val", "test"):
        shutil.copyfile(frozen / f"{split}.jsonl", output / f"{split}.jsonl")
        shutil.copyfile(expansion / f"oasst2/{split}.jsonl", output / f"oasst2_{split}.jsonl")
        outputs[f"{split}.jsonl"] = {"sha256": sha256_file(output / f"{split}.jsonl"), "unchanged_frozen": True}
        outputs[f"oasst2_{split}.jsonl"] = {"sha256": sha256_file(output / f"oasst2_{split}.jsonl"), "unchanged_expansion": True}
    (output / "audit_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Deterministic random reservoir sample after selection; not a quality certificate."]
    for name, rows in samples.items():
        lines.append("\nSOURCE: " + name)
        for index, record in enumerate(rows):
            question, _ = question_and_history(record)
            lines.extend([f"\n{index}. {record['id']}", "Q: " + question, "A: " + record["response"]])
    (output / "audit_sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["audit_sample.json"] = {"sha256": sha256_file(output / "audit_sample.json")}
    outputs["audit_sample.txt"] = {"sha256": sha256_file(output / "audit_sample.txt")}
    manifest = {"version": 2, "status": "experimental_curated_not_quality_certified", "seed": seed,
                "filters": {"telegram_topics": list(TOPICS), "telegram_profile": telegram_profile, "telegram_limit": telegram_limit, "toloka_limit": toloka_limit,
                            "lexical_relevance_is_heuristic": True, "known_bad_review_rows_removed": len(excludes)},
                "inputs": inputs, "parent_curation": parent_manifest, "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": sha256_file(tokenizer_path), "max_answer_tokens": 95,
                "protected_heldout": protection.paths, "counts": counts, "outputs": outputs,
                "selection_code_sha256": sha256_file(__file__), "seconds": time.perf_counter() - started,
                "limits": ["No synthetic dialogue generation or source-text rewriting.",
                           "Telegram source positives and topic overlap do not certify semantic relevance.",
                           "Telegram lacks conversation IDs: original-thread disjointness cannot be proved.",
                           "Old train rows preserve the frozen group-based split; generic cross-split text can remain.",
                           "New sources block exact held-out questions and longer utterances; paraphrases remain possible.",
                           "Sampling caps change source exposure; choose training weights only after independent quality review."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = finalize_identity_filter(output)
    return write_precision_audit(output) if telegram_profile == "precision" else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", default="data/processed")
    parser.add_argument("--expansion-dir", default="data/expansion/20261121")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--from-pools", default=None, help="Further restrict an earlier curation; never restores previously excluded rows")
    parser.add_argument("--telegram-profile", choices=("precision", "everyday"), default="precision")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--telegram-limit", type=int, default=0)
    parser.add_argument("--toloka-limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.telegram_limit, args.toloka_limit) < 0:
        parser.error("Source limits must be nonnegative (0 means uncapped)")
    result = prepare(args.frozen_dir, args.expansion_dir, args.output_dir, tokenizer_path=args.tokenizer, seed=args.seed,
                     telegram_limit=args.telegram_limit, toloka_limit=args.toloka_limit, from_pools=args.from_pools,
                     telegram_profile=args.telegram_profile)
    print(json.dumps({"counts": result["counts"], "mixture": result["outputs"]["train.jsonl"], "seconds": result["seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
