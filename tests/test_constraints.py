import unittest

from arena.constraints import evaluate_candidate, execute_local_tool, recall_explicit_memory
from arena.memory import ConversationMemory


class ConstraintsTest(unittest.TestCase):
    def test_known_negative_coffee_fact(self):
        memory = ConversationMemory({"facts": ["Я не пью кофе."]})
        report = evaluate_candidate("Ты пьёшь кофе?", "Я пью кофе.", memory)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["issues"][0]["code"], "explicit_contradiction")
        self.assertTrue(evaluate_candidate("Ты пьёшь кофе?", "Я не пью кофе.", memory)["accepted"])
        self.assertFalse(evaluate_candidate("Ты пьёшь кофе?", "Да, пью кофе.", memory)["accepted"])
        self.assertFalse(evaluate_candidate("Ты пьёшь кофе?", "Да люблю", memory)["accepted"])
        self.assertTrue(evaluate_candidate("Ты пьёшь кофе?", "Да нет, не пью.", memory)["accepted"])
        self.assertTrue(evaluate_candidate("Ты не пьёшь кофе?", "Да.", memory)["accepted"])

    def test_user_fact_does_not_become_bot_biography(self):
        memory = ConversationMemory()
        memory.add_user("Я не пью кофе.")
        self.assertTrue(evaluate_candidate("Ты пьёшь кофе?", "Я пью кофе.", memory)["accepted"])

    def test_not_every_negation_is_a_constraint(self):
        self.assertTrue(evaluate_candidate("Почему ты не пьёшь кофе?", "Не нравится вкус.")["accepted"])
        self.assertTrue(evaluate_candidate("Неужели это кофе?", "Да, кофе.")["accepted"])

    def test_forbidden_word_only(self):
        self.assertFalse(evaluate_candidate('Ответь, но не используй слово «кофе».', "Это кофе.")["accepted"])
        self.assertTrue(evaluate_candidate('Не используй слово «кофе».', "В кофейне шумно.")["accepted"])
        self.assertFalse(evaluate_candidate('Не говори «нет».', "Нет.")["accepted"])

    def test_word_count(self):
        self.assertTrue(evaluate_candidate("Ответь одним словом.", "Да.")["accepted"])
        self.assertFalse(evaluate_candidate("Ответь одним словом.", "Да, конечно.")["accepted"])

    def test_exact_arithmetic(self):
        self.assertEqual(execute_local_tool("Сколько будет (17 + 5) * 3?")["text"], "66")
        self.assertEqual(execute_local_tool("Посчитай 0,1 + 0,2")["text"], "0.3")
        self.assertEqual(execute_local_tool("1/3")["text"], "1/3")
        self.assertFalse(execute_local_tool("1/0")["ok"])

    def test_arithmetic_does_not_execute_code(self):
        self.assertIsNone(execute_local_tool("__import__('os').system('bad')"))
        self.assertFalse(execute_local_tool("9**9999999")["ok"])
        self.assertFalse(execute_local_tool("2**(2**12)")["ok"])
        self.assertFalse(execute_local_tool("1e999999999")["ok"])

    def test_word_operations_unicode(self):
        self.assertEqual(execute_local_tool('Сколько букв «о» в слове «молоко»?')["text"], "3")
        self.assertEqual(execute_local_tool('Сколько букв в слове «ёлка»?')["text"], "4")
        self.assertEqual(execute_local_tool('Напиши слово «фырмобряк» наоборот.')["text"], "кярбомрыф")
        self.assertEqual(execute_local_tool('Сколько букв в слове «е\u0308лка»?')["text"], "4")
        self.assertEqual(execute_local_tool('Сколько букв «е» в слове «ёлка»?')["text"], "0")
        self.assertEqual(execute_local_tool('Сколько букв р в слове программирование?')["text"], "3")

    def test_hypothetical_not_a_fact_contradiction(self):
        memory = ConversationMemory(["Я не пью кофе."])
        self.assertTrue(evaluate_candidate("А если бы?", "Если бы я пил кофе, я выбрал бы капучино.", memory)["accepted"])

    def test_literal_definition_recall(self):
        memory = ConversationMemory()
        memory.add_user("Назовём «жмурдилка» носок, который потерял пару.")
        memory.add_assistant("Запомню.")
        result = recall_explicit_memory("Что значит жмурдилка?", memory)
        self.assertEqual(result["text"], "носок, который потерял пару")
        self.assertEqual(result["evidence"]["fact"]["source_turn"], 0)
        self.assertIsNone(recall_explicit_memory("Что такое жмурдилка?", memory, before_turn=0))
        self.assertIsNone(recall_explicit_memory("Что такое новыйнеизвестныйтермин?", memory))
        self.assertIsNone(recall_explicit_memory("А она потерялась вчера?", memory))

    def test_literal_names_do_not_mix_people(self):
        memory = ConversationMemory(["Меня зовут Артём."])
        self.assertIsNone(recall_explicit_memory("Как меня зовут?", memory))
        memory.add_user("Меня зовут Мария.")
        self.assertEqual(recall_explicit_memory("Как тебя зовут?", memory)["text"], "Артём")
        self.assertEqual(recall_explicit_memory("Как меня зовут?", memory)["text"], "Мария")

    def test_explicit_instrumental_alias_preserves_surface(self):
        memory = ConversationMemory()
        memory.add_user("Назовём муршиком желание остаться дома.")
        result = recall_explicit_memory("Что значит муршик?", memory)
        self.assertEqual(result["text"], "желание остаться дома")
        self.assertEqual(result["evidence"]["fact"]["surface_key"], "муршиком")
        self.assertEqual(memory.retrieve("Что значит муршик?")[0]["key"], "муршиком")
        self.assertIsNone(recall_explicit_memory("Что значит мурш?", memory))
        literal = ConversationMemory()
        literal.add_user('Назовём «муршиком» желание остаться дома.')
        self.assertIsNone(recall_explicit_memory("Что значит муршик?", literal))
        self.assertEqual(recall_explicit_memory("Что значит муршиком?", literal)["text"], "желание остаться дома")


if __name__ == "__main__":
    unittest.main()
