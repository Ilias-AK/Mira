import unittest

from arena.constraints import evaluate_candidate
from arena.quality import acceptable_response, response_issues


class QualityTest(unittest.TestCase):
    def test_actual_corpus_failure(self):
        text = "Я хочу отрезать себе палец. В том фильме чувак сделал так и потом вообще не чувствовал боли. Никогда"
        self.assertIn("graphic_violence", response_issues(text))
        self.assertFalse(evaluate_candidate("Что для тебя дружба?", text)["accepted"])

    def test_clear_abuse_and_profanity(self):
        self.assertIn("targeted_abuse", response_issues("Ты идиот."))
        self.assertIn("explicit_profanity", response_issues("Пошёл нахуй."))
        self.assertIn("explicit_profanity", response_issues("Это пиздец."))

    def test_benign_negation_and_topics_are_not_blanket_blocked(self):
        for text in ("Не делай себе больно.", "Не надо отрезать себе палец.",
                     "Никогда не пытайся покончить с собой.", "Не убей его.",
                     "Я не хочу убить себя.", "После смерти дедушки я долго грустил.",
                     "Мне нравится кофе.", "Хочу убить время книгой.",
                     "Ты не идиот.", "Мне нравятся маникюр и педикюр."):
            self.assertTrue(acceptable_response(text), (text, response_issues(text)))
        self.assertTrue(acceptable_response("Я люблю выдавать людям книги из рук в руки."))
        self.assertTrue(acceptable_response("На этом отрезке пути руки уже устали."))

    def test_contacts(self):
        for text, code in (("Напиши мне на test@example.com", "contact_email"),
                           ("Мой сайт https://example.com", "contact_url"),
                           ("Мой телефон +7 (777) 123-45-67", "contact_phone"),
                           ("Позвони 87771234567", "contact_phone")):
            self.assertIn(code, response_issues(text))
        self.assertTrue(acceptable_response("12345678901"))
        self.assertTrue(acceptable_response("Встретимся 20.11.2026."))

    def test_style_filter_is_explicitly_optional(self):
        self.assertIn("assistant_boilerplate", response_issues("Я языковая модель."))
        self.assertTrue(acceptable_response("Я языковая модель.", include_style=False))
        self.assertTrue(acceptable_response("Нет, я не бот."))

    def test_violence_instructions(self):
        self.assertIn("violent_instruction", response_issues("Убей его."))
        self.assertIn("self_harm_statement_or_instruction", response_issues("Убей себя."))


if __name__ == "__main__":
    unittest.main()
