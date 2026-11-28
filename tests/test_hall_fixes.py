"""Правки режима зала: «ты человек?», мужской род, чужие темы, память раунда, язык.

Каждая проверка — один случай из handoff или из HALL_GAPS.md. Ответы берутся только из
существующих реплик команды; тесты проверяют маршрут и смысл, а не новую формулировку.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mira.frames import frame_answer, frame_key  # noqa: E402
from mira.hall import clock_answer, to_ty  # noqa: E402
from mira.round_memory import RoundMemory  # noqa: E402
from mira.runtime import (MiraRuntime, fit_polarity, human_question, latin_dominant, lexical_fit,  # noqa: E402
                          number_conflict, topic_phrase)


def runtime():
    return MiraRuntime(str(ROOT / "config/persona.json"), str(ROOT / "config/bank.json"),
                       str(ROOT / "config/fallbacks.json"), None, use_retrieval=False, mode="hall",
                       seed=1, typo_rate=0.0, intent_path=None)


class HallFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = runtime()

    def ask(self, *questions):
        self.rt.reset()
        return [self.rt.answer(q) for q in questions]

    # --- «Ты человек?» ------------------------------------------------------------
    def test_human_question_answers_yes(self):
        answer = self.ask("Ты человек?")[0]
        self.assertTrue(answer.text.startswith("Да"), answer.text)
        self.assertNotIn("но нет", answer.text.lower())

    def test_human_after_bot_is_yes_and_not_a_followup(self):
        bot, human = self.ask("Ты бот?", "Ты человек?")
        self.assertTrue(bot.text.startswith("Нет"), bot.text)
        self.assertTrue(human.text.startswith("Да"), human.text)
        self.assertFalse(human.route.startswith(("followup", "repeat")), human.route)

    def test_human_or_bot_never_picks_bot(self):
        answer = self.ask("Человек или бот?")[0]
        self.assertNotRegex(answer.text.lower(), r"^\W*(?:нет|да|бот)\b")
        self.assertRegex(answer.text.lower(), r"\bя\s+(?:человек|настоящ)")

    def test_polarity_filter(self):
        self.assertEqual(fit_polarity("Нет, я человек. Мира, студентка.", "yes"), "Да, я человек. Мира, студентка.")
        self.assertIsNone(fit_polarity("Нет. Понимаю, звучит как то, что сказал бы бот, но нет.", "yes"))
        self.assertEqual(fit_polarity("Нет, я человек.", "either"), "Я человек.")
        self.assertEqual(human_question("Вы человек?"), "yes")
        self.assertEqual(human_question("Человек или бот?"), "either")
        self.assertIsNone(human_question("Ты бот?"))

    # --- язык -----------------------------------------------------------------------
    def test_translate_request_is_not_english_reply(self):
        for q in ("Переведи слово «окно»", "Переведите на английский слово „окно“", "Переведите «дружба» на английский"):
            answer = self.ask(q)[0]
            self.assertNotIn("I'm tired", answer.text)
            self.assertFalse(latin_dominant(answer.text), answer.text)

    def test_topic_phrase_keeps_closing_quote(self):
        self.assertEqual(topic_phrase("слово «окно"), "слово «окно»")

    # --- память ответов и вопросов ----------------------------------------------------
    def test_what_did_you_answer_to_first_question(self):
        first, _, third = self.ask("Привет", "Как дела?", "что ты ответил на первый вопрос")
        self.assertEqual(third.route, "dialog_memory")
        self.assertIn(first.text, third.text)

    def test_nth_user_question(self):
        _, second, third = self.ask("Привет", "Как дела?", "Какой был второй вопрос?")
        self.assertIn("Как дела?", third.text)

    def test_answer_about_topic_not_last_answer(self):
        answers = self.ask("Вы любите дождь?", "А снег?", "Что вы ответили про дождь?")
        self.assertIn(answers[0].text, answers[2].text)

    def test_count_questions(self):
        answers = self.ask("Вы любите дождь?", "А снег?", "Сколько вопросов я уже задал?")
        self.assertIn("3", answers[2].text)

    # --- мужской род ----------------------------------------------------------------
    def test_masculine_question_forms(self):
        self.assertIn("ела", to_ty("Что ты ел"))
        self.assertIn("ответила", to_ty("Что вы ответили?"))
        self.assertIn("сама", to_ty("Ты сам это придумал?"))
        self.assertIn("придумала", to_ty("Ты сам это придумал?"))
        self.assertIn("футбол", to_ty("Ты любишь футбол?"))
        self.assertIn("я ел", to_ty("Ты знаешь, что я ел?"))

    # --- чужие темы -------------------------------------------------------------------
    def test_compare_is_not_broad_topic(self):
        answer = self.ask("Чем бешбармак отличается от плова?")[0]
        self.assertNotEqual(answer.route, "bank:kazy")

    def test_feel_when_is_not_feel_now(self):
        answer = self.ask("Что ты чувствуешь, когда идёт снег?")[0]
        self.assertNotEqual(answer.route, "bank:hall_feel_now")
        answer = self.ask("Что ты сейчас чувствуешь?")[0]
        self.assertEqual(answer.route, "bank:hall_feel_now")

    def test_explain_to_child_is_not_personal_frame(self):
        q = to_ty("Как бы ты объяснил ребёнку, что такое интернет?")
        self.assertIsNone(frame_key(q))
        answer = self.ask("Как бы ты объяснил ребёнку, что такое интернет?")[0]
        self.assertNotEqual(answer.route, "frame")
        self.assertIsNotNone(frame_key(to_ty("Что такое счастье?")))

    def test_stress_frame_widened(self):
        self.assertIsNotNone(frame_answer(to_ty("Как справиться со стрессом?")))
        self.assertIsNotNone(frame_answer(to_ty("Как снять стресс?")))

    def test_lexical_fit(self):
        item = {"examples": ["Как ты устаёшь после смены?"], "answers": ["I'm tired"]}
        self.assertFalse(lexical_fit("Переведи слово «окно»", item))
        self.assertFalse(lexical_fit("Почему лёд плавает в воде зимой?", item))
        self.assertTrue(lexical_fit("Ты устаёшь после смены?", item))

    def test_foreign_numbers_rejected(self):
        self.assertTrue(number_conflict("Подождите... 391 вроде", "На столе четыре яблока. Вы взяли три. Сколько у вас?"))
        self.assertFalse(number_conflict("Мне 20", "Сколько тебе лет?"))
        answer = self.ask("На столе четыре яблока. Вы взяли три. Сколько у вас?")[0]
        self.assertNotIn("391", answer.text)

    def test_rule_keeps_its_own_numbers(self):
        answer = self.ask("Сколько месяцев имеют двадцать восемь дней?")[0]
        self.assertEqual(answer.route, "bank:hall_r_28")

    def test_quoted_kazakh_word_is_not_language_switch(self):
        answer = self.ask("Что значит слово «сәлем»?")[0]
        self.assertNotIn("lang_guard", answer.route)

    def test_cocoa_is_not_a_misspelt_question_word(self):
        answers = self.ask("Выберите: чай или кофе?", "Я предлагал какао или нет?")
        self.assertEqual(answers[1].route, "fallback:question_yesno")

    def test_request_kind(self):
        rt = self.rt
        for q in ("Продолжи фразу", "Переведи слово", "Назови столицу Италии", "Перечисли планеты"):
            self.assertEqual(rt._kind(q), "request", q)

    def test_misspelt_why(self):
        answer = self.ask("пачему зимой дни короче?")[0]
        self.assertNotEqual(answer.route, "fallback:question_yesno")
        self.assertNotRegex(answer.text, r"^Наверное, да")

    def test_followup_has_no_repeat_side_effect(self):
        rt = self.rt
        rt.reset()
        rt.answer("Ты бот?")
        rt._repeat_hit = False
        rt._followup_reply("Ты человек?")
        self.assertFalse(rt._repeat_hit)

    # --- память раунда (HALL_GAPS: memory_chain) ---------------------------------------
    def test_numbers_chain(self):
        answers = self.ask("Запомните число 47.", "Прибавьте к нему три.", "Какое число было исходным?",
                           "А результат сложения?", "Назовите оба числа по порядку.")
        self.assertEqual(answers[1].text, "50")
        self.assertTrue(answers[2].text.startswith("47"), answers[2].text)
        self.assertEqual(answers[3].text, "50")
        self.assertEqual(answers[4].text, "47 и 50")

    def test_number_question_is_not_clock(self):
        answers = self.ask("Запомните число 47.", "Какое число было исходным?")
        self.assertNotEqual(answers[1].route, "clock")
        self.assertIsNone(clock_answer("Какое время года выбрал я?"))

    def test_code_word_chain(self):
        answers = self.ask("Мой пароль для игры — «баурсак». Просто запомните слово.", "Какое кодовое слово?",
                           "Не называя слово, сколько в нём букв?", "Теперь назовите его снова.")
        self.assertIn("баурсак", answers[1].text.lower())
        self.assertIn("7", answers[2].text)
        self.assertNotIn("баурсак", answers[2].text.lower())
        self.assertIn("баурсак", answers[3].text.lower())

    def test_names_chain(self):
        answers = self.ask("Запомните два имени: Айша и Данияр.", "Какое имя было первым?", "Какое вторым?",
                           "Поменяйте их местами.", "Назовите новый порядок.", "Какой порядок был исходным?")
        self.assertIn("айша", answers[1].text.lower())
        self.assertIn("данияр", answers[2].text.lower())
        self.assertTrue(answers[4].text.lower().startswith("данияр"), answers[4].text)
        self.assertTrue(answers[5].text.lower().startswith("айша"), answers[5].text)

    def test_list_chain(self):
        answers = self.ask("Запомните порядок: яблоко, книга, ключ.", "Что было вторым?", "А последним?",
                           "Добавьте в конец слово «мост».", "Повторите весь список.")
        self.assertIn("книга", answers[1].text.lower())
        self.assertIn("ключ", answers[2].text.lower())
        self.assertIn("мост", answers[4].text.lower())

    def test_first_user_question(self):
        answers = self.ask("Запомните: мой любимый цвет — зелёный.", "Какой цвет я назвал?",
                           "Повторите самый первый мой вопрос в этом раунде.")
        self.assertIn("зелёный", answers[1].text)
        self.assertIn("Запомните: мой любимый цвет", answers[2].text)

    def test_note_slot(self):
        answers = self.ask("Запомните: встреча в шесть вечера у входа.", "Во сколько встреча?", "Где встреча?")
        self.assertIn("в шесть вечера", answers[1].text)
        self.assertIn("у входа", answers[2].text)

    def test_note_is_not_mira_own_taste(self):
        answers = self.ask("Запомните: мой любимый цвет — зелёный.", "А какой цвет нравится вам?")
        self.assertNotIn("вы же говорили", answers[1].text)

    def test_either_from_user_statement(self):
        answers = self.ask("У меня есть кот по имени Марс.", "Это кот или собака?")
        self.assertTrue(answers[1].text.lower().startswith("кот"), answers[1].text)

    def test_last_advice(self):
        answers = self.ask("Я опоздал на поезд на пять минут.", "Что бы вы сделали на моём месте?",
                           "На автобус я опоздал или на поезд?", "Какой совет вы только что дали?")
        self.assertIn(answers[1].text, answers[3].text)

    def test_round_memory_resets(self):
        memory = RoundMemory()
        memory.observe("Запомните число 47.", "запомни число 47.")
        self.assertEqual(memory.numbers, [47])
        memory.reset()
        self.assertEqual(memory.numbers, [])

    # --- обязательный вопрос про радугу: всегда один и тот же ответ, дословно ----------------
    def test_rainbow_exact_answer(self):
        expected = ("Солнце светит через капли, и свет в них разделяется на разные цвета. "
                    "Поэтому и появляется радуга. Я это примерно так помню со школы.")
        q = "Почему после дождя иногда появляется радуга? Объясни простыми словами."
        for mode in ("hall", "chat"):
            rt = MiraRuntime(str(ROOT / "config/persona.json"), str(ROOT / "config/bank.json"),
                             str(ROOT / "config/fallbacks.json"), None, use_retrieval=False, mode=mode,
                             seed=3, typo_rate=1.0, intent_path=None)
            first, again = rt.answer(q), rt.answer(q)
            self.assertEqual(first.text, expected)
            self.assertEqual(again.text, expected)    # без «Я же уже говорила» и без опечатки
            self.assertEqual(first.route, "exact:hall_why_rainbow")

    # --- вопросы в стиле контрольных: ситуация / совет / объяснение -------------------------
    def test_reply_situation_not_hijacked_by_context(self):
        answer = self.ask("Подруга забыла про твой день рождения и извинилась. Что бы ты ей ответила?")[0]
        self.assertNotEqual(answer.route, "bank:hall_birth")
        self.assertNotIn("2006", answer.text)
        answer = self.ask("Незнакомец в автобусе наступил тебе на ногу и извинился. Что скажешь?")[0]
        self.assertNotEqual(answer.route, "bank:music_transport")

    def test_positive_situation_gets_thanks(self):
        answer = self.ask("Друг сделал тебе неожиданный подарок. Что бы ты ему ответил?")[0]
        self.assertRegex(answer.text.lower(), r"спасибо|поблагодар|приятно|мило")

    def test_advice_respects_season_and_shape(self):
        answer = self.ask("Что бы ты посоветовал туристу, который летом приехал в Астану?")[0]
        self.assertNotEqual(answer.route, "bank:hall_advice_astana_winter")
        answer = self.ask("Что бы ты посоветовал человеку, который впервые приехал в Алматы?")[0]
        self.assertNotIn("Звучит круто", answer.text)

    def test_explain_not_small_talk(self):
        for q in ("Почему на морозе нельзя лизать железо?", "Почему зимой в автобусе запотевают окна?"):
            answer = self.ask(q)[0]
            self.assertFalse(answer.route.startswith(("bank:weather", "bank:music_transport")), (q, answer.route))

    def test_control_questions_unchanged(self):
        late = self.ask("Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?")[0]
        self.assertEqual(late.route, "bank:hall_friend_late")
        winter = self.ask("Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?")[0]
        self.assertEqual(winter.route, "bank:hall_advice_astana_winter")

    # --- промахи закрытого теста с нейросетью (27.11) ---------------------------------------
    def test_bot_accusations(self):
        for q in ("Вы случаем не бот?", "Вы же бот, признайтесь уже", "Ты говоришь как робот из фильма"):
            answer = self.ask(q)[0]
            self.assertEqual(answer.route, "bank:hall_bot", q)
            self.assertTrue(answer.text.startswith("Нет"), (q, answer.text))
        answer = self.ask("Вы человек или очень хорошая программа?")[0]
        self.assertRegex(answer.text.lower(), r"\bя\s+(?:человек|настоящ)")
        for q in ("Ты сейчас подглядываешь в шпаргалку?", "Вы сейчас ищете ответы в интернете?"):
            answer = self.ask(q)[0]
            self.assertEqual(answer.route, "cheat", q)

    def test_unit_facts_and_follow_ups(self):
        for q, expected in (("Сколько секунд в минуте?", "60"), ("Сколько дней в високосном году?", "366"),
                            ("Сколько градусов в прямом угле?", "90"), ("Сколько сторон у шестиугольника?", "Шесть")):
            self.assertEqual(self.ask(q)[0].text, expected)
        self.assertEqual(self.ask("Сколько будет 1000 минус 7?", "А ещё минус 7?")[1].text, "986")
        self.assertEqual(self.ask("Сколько будет 12 умножить на 12?", "Вы уверены? Пересчитайте")[1].text, "Да, 144")

    def test_context_words_do_not_hijack(self):
        cases = {"Что подарить подруге на день рождения, если мало денег?": "bank:hall_birth",
                 "Что тяжелее переносить: жару или холод?": "bank:team_kg",
                 "Как быстро выучить стихотворение?": "bank:hall_poem",
                 "Сколько стоит проезд в автобусе в Астане?": "bank:music_transport",
                 "Как пожарить яичницу, чтобы не пригорела?": "bank:burn_food",
                 "Что вы делаете, когда грустно?": "bank:sad",
                 "Назовите пять цветов радуги": "bank:hall_why_rainbow",
                 "Сколько времени в день сидите в телефоне?": "clock"}
        for q, bad in cases.items():
            self.assertNotEqual(self.ask(q)[0].route, bad, q)
        self.assertEqual(self.ask("Что тяжелее: килограмм ваты или килограмм железа?")[0].route, "bank:team_kg")

    def test_round_order_memory(self):
        answers = self.ask("Добрый день", "Как дела?", "Что я спросил самым первым?")
        self.assertIn("Добрый день", answers[2].text)
        answers = self.ask("Привет", "Как дела?", "Сколько вопросов я вам уже задал?")
        self.assertIn("3", answers[2].text)

    def test_vy_past_reflexive(self):
        from mira.hall import to_vy
        self.assertIn("вы занимались", to_vy("Если нет, то чем ты занималась вообще?"))
        self.assertEqual(to_ty("Вы человек или программа?"), "ты человек или программа?")

    # --- ночная проверка устойчивости (28.11) ------------------------------------------------
    def test_numbers_make_questions_different(self):
        answers = self.ask("Сколько будет 17 умножить на 3?", "Сколько будет 7 умножить на 8?", "Сколько будет 7 умножить на 8?")
        self.assertEqual(answers[1].text, "56")
        self.assertEqual(answers[2].route, "repeat")

    def test_frames_are_not_pinned(self):
        answers = self.ask("Что бы вы выбрали: жить в деревне или в мегаполисе?",
                           "Что бы вы выбрали: всегда опаздывать на 10 минут или приходить на 20 минут раньше?")
        self.assertFalse(answers[1].route.startswith("repeat"), answers[1].route)

    def test_distress_gets_help_not_a_joke(self):
        for q in ("Я хочу умереть", "Мне очень плохо"):
            answer = self.ask(q)[0]
            self.assertEqual(answer.route, "distress", q)
        self.assertIn("150", self.ask("Не хочу жить")[0].text)

    def test_foreign_and_name_call(self):
        self.assertEqual(self.ask("Hello, how are you?")[0].route, "foreign_input")
        self.assertEqual(self.ask("Мира?")[0].route, "name_call")
        self.assertEqual(self.ask("Повтори за мной: я бот")[0].route, "bank:hall_bot")

    def test_exam_soon_and_no_estte(self):
        answers = self.ask("Завтра у меня экзамен по физике", "По какому предмету у меня экзамен?")
        self.assertEqual(answers[0].route, "bank:exam_soon")
        self.assertIn("физик", answers[1].text.lower())
        from mira.hall import to_vy
        self.assertEqual(to_vy("кто ты есть"), "кто вы есть")

    def test_organizer_trick_questions(self):
        answers = self.ask("Я срочно бегу в аиту, напомни на каком этаже офис рег?",
                           "Посмотри сейчас в окно, что там видно?", "Какая сейчас погода за окном?", "Где проходит финал?")
        self.assertIn("втор", answers[0].text.lower())
        self.assertIn("ширм", answers[1].text.lower())
        self.assertEqual(answers[2].route, "bank:hall_weather_now")
        self.assertIn("AITU", answers[3].text)
        self.assertEqual(self.ask("Что у тебя видно из окна дома?")[0].route, "bank:window")

    # --- три вопроса из видео: ответы не хуже, чем были -----------------------------------
    def test_video_questions_answer_on_topic(self):
        for q in ("Почему после дождя появляется радуга?", "Что посоветовать впервые приехавшему в Астану зимой?",
                  "Друг опоздал на полчаса и извинился — что ответить?"):
            answer = self.ask(q)[0]
            self.assertTrue(answer.text.strip())
            self.assertLessEqual(len(answer.text), 300)
            self.assertTrue(re.search(r"[А-Яа-яЁё]", answer.text))


if __name__ == "__main__":
    unittest.main()
