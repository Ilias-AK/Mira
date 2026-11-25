import unittest

from arena.memory import ConversationMemory


class MemoryTest(unittest.TestCase):
    def test_exact_newword_recalled_later(self):
        memory = ConversationMemory()
        memory.add_user("Назовём «жмурдилка» носок, который потерял пару.")
        memory.add_assistant("Буду знать.")
        memory.add_user("А ты любишь чай?")
        memory.add_assistant("Да, чёрный.")
        facts = memory.retrieve("Что такое жмурдилка?")
        self.assertEqual(facts[0]["surface_key"], "жмурдилка")
        self.assertEqual(facts[0]["value"], "носок, который потерял пару")
        self.assertEqual(facts[0]["source_turn"], 0)
        self.assertEqual(facts[0]["scope"], "conversation")

    def test_persona_and_speakers_are_separate(self):
        memory = ConversationMemory(["Я не пью кофе."])
        memory.add_user("Я пью кофе.")
        assistant_facts = memory.active_facts(scope="assistant")
        user_facts = memory.active_facts(scope="user")
        self.assertEqual(assistant_facts[0]["polarity"], -1)
        self.assertEqual(user_facts[0]["polarity"], 1)

    def test_no_future_turns_leakage(self):
        memory = ConversationMemory()
        memory.add_user("Привет.")
        memory.add_user("Назовём «фырма» кружку без ручки.")
        self.assertEqual(memory.retrieve("Что такое фырма?", before_turn=1), [])
        self.assertTrue(memory.retrieve("Что такое фырма?", before_turn=2))

    def test_no_inference_from_question_or_hypothetical(self):
        memory = ConversationMemory()
        memory.add_assistant("Если бы я любил кофе, я пил бы его утром.")
        memory.add_assistant("Я люблю кофе?")
        self.assertEqual(memory.active_facts(scope="assistant"), [])

    def test_snapshot_roundtrip_and_copy_safety(self):
        memory = ConversationMemory({"facts": ["Меня зовут Артём.", "Мне 20 лет."]})
        memory.add_user("Назовём зюбря потерянный ключ.")
        restored = ConversationMemory.from_snapshot(memory.snapshot())
        self.assertEqual(restored.facts, memory.facts)
        self.assertEqual(restored.history, memory.history)
        history = memory.history
        history.append({"role": "assistant", "text": "будущее"})
        self.assertEqual(len(memory.history), 1)

    def test_explicit_correction_supersedes_same_scope_only(self):
        memory = ConversationMemory()
        memory.add_user("Я люблю чай.")
        memory.add_assistant("Я не люблю чай.")
        memory.add_user("Я не люблю чай.")
        user_facts = memory.active_facts(scope="user")
        self.assertEqual(len(user_facts), 1)
        self.assertEqual(user_facts[0]["polarity"], -1)
        self.assertEqual(user_facts[0]["source_turn"], 2)

    def test_names_stop_at_conjunction(self):
        memory = ConversationMemory(["Меня зовут Артём и мне 20 лет."])
        names = [fact for fact in memory.active_facts() if fact["key"] == "name"]
        self.assertEqual(names[0]["value"], "Артём")

    def test_pronoun_is_not_invented_word_definition(self):
        memory = ConversationMemory()
        memory.add_user("Это значит конец разговора.")
        self.assertFalse([fact for fact in memory.facts if fact["kind"] == "definition"])


if __name__ == "__main__":
    unittest.main()
