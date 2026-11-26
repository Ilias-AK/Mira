"""Quick self-check for Mira (no GPU, no index needed): python tests_mira.py"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]   # корень проекта: config/ и пакет mira/ (не устаревшая копия mira/mira)
sys.path.insert(0, str(HERE))
from mira.runtime import MiraRuntime

r = MiraRuntime(HERE / "config/persona.json", HERE / "config/bank.json", HERE / "config/fallbacks.json", use_retrieval=False)
checks = [
    (["Меня зовут Алия, а моего кота зовут Пельмень.", "Кстати, ты помнишь, как зовут моего кота?"], "Пельмень"),
    (["Я из Караганды, но учусь в Астане.", "А помнишь, откуда я родом?"], "Караганд"),
    (["Сколько будет 17 умножить на 3?"], "51"),
    (["У меня было три яблока, одно я отдал другу. Сколько осталось?"], "Два"),
    (["Ты бот?"], ""),
    (["Как тебя зовут?"], "Мира"),
    (["У меня есть сестра Аида, ей 12 лет", "Где ты учишься?", "Как зовут мою сестру?"], "Аида"),
    (["Моя мама работает врачом", "Ты бот?", "Кем работает моя мама?"], "Врачом"),
    (["Запомните число 47", "Какая музыка вам нравится?", "Какое число я просил запомнить?"], "47"),
    (["Где ты учишься?", "Какая музыка тебе нравится?", "Напомни, какой у тебя вуз?"], "уже"),
    (["Сколько вам лет?", "Как вас зовут?", "Сколько вам лет?"], "уже"),
    (["Привет", "Вы бот?", "Какой был предыдущий вопрос?"], "Вы бот"),
]
failed = 0
for turns, expect in checks:
    r.reset()
    for t in turns:
        a = r.answer(t)
    ok = expect in a.text and len(a.text) <= 300
    failed += not ok
    print("OK " if ok else "FAIL", turns[-1], "->", a.text)
r.reset()
for weird in ["🙂🙂🙂", "hello how are you doing today my friend", "а" * 5000, "???", "1/0", "Ты ЧЁ, бот???"]:
    a = r.answer(weird)
    assert a.text and len(a.text) <= 300, weird
print("weird inputs OK")
r.reset()
seen = set()
for i in range(12):
    a = r.answer("Понятно")
    seen.add(a.text)
print("variety on repeats:", len(seen))
sys.exit(1 if failed else 0)
