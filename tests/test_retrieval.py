import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from arena.retrieval import HybridRetriever, RetrievalIndex


def records():
    return [
        {"id": "coffee1", "group_id": "coffee", "split": "train", "context": "Ты пьёшь кофе?", "response": "Я не пью кофе.", "source": "test"},
        {"id": "coffee2", "group_id": "coffee", "split": "train", "context": "Почему не пьёшь кофе?", "response": "Не нравится вкус.", "source": "test"},
        {"id": "tea", "group_id": "tea", "split": "train", "context": "Ты пьёшь чай?", "response": "Да, люблю чай.", "source": "test"},
        {"id": "travel", "group_id": "travel", "split": "train", "context": [{"role": "user", "text": "Любишь путешествовать?"}], "response": "Люблю поездки на поезде.", "source": "test"},
        {"id": "book", "group_id": "book", "split": "train", "context": "Любишь читать книги?", "response": "Читаю перед сном.", "source": "test"},
        {"id": "heldout", "group_id": "test_dialogue", "split": "test", "context": "Ты пьёшь кофе?", "response": "SECRET_HELDOUT", "source": "test"},
        {"id": "missing_split", "group_id": "unknown", "context": "Ты пьёшь кофе?", "response": "SECRET_NO_SPLIT"},
    ]


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.index = HybridRetriever.build(records(), self.temporary.name)

    def tearDown(self):
        self.index.close()
        self.temporary.cleanup()

    def test_exact_query_and_train_only(self):
        found = self.index.search("Ты пьёшь кофе?", k=20)
        self.assertEqual(found[0]["id"], "coffee1")
        self.assertFalse({"heldout", "missing_split"} & {item["id"] for item in found})
        self.assertEqual(self.index.metadata["counts"]["indexed"], 5)

    def test_entire_group_excluded_before_ranking(self):
        found = self.index.search("Почему не пьёшь кофе?", k=20, exclude_groups={"coffee"})
        self.assertTrue(found)
        self.assertTrue(all(item["group_id"] != "coffee" for item in found))
        alias = self.index.search("Ты пьёшь кофе?", limit=20, exclude_group_id="coffee")
        self.assertTrue(all(item["group_id"] != "coffee" for item in alias))

    def test_subword_typo_recall(self):
        found = self.index.search("путешествоват", k=1)
        self.assertEqual(found[0]["id"], "travel")

    def test_save_load_is_identical(self):
        before = self.index.search("кофе")
        with RetrievalIndex.load(self.temporary.name) as restored:
            self.assertEqual(before, restored.search("кофе"))

    def test_no_response_key_leakage(self):
        self.assertEqual(self.index.search("SECRET_HELDOUT"), [])
        self.assertEqual(self.index.search("квазисекретxyz"), [])

    def test_rebuild_invalidates_reranker(self):
        self.index.fit_reranker(auto_negatives=True, max_examples=10, epochs=5)
        self.assertIsNotNone(self.index.reranker)
        self.index.close()
        self.index = HybridRetriever.build(records(), self.temporary.name)
        self.assertIsNone(self.index.reranker)

    def test_trainable_reranker_survives_reload(self):
        result = self.index.fit_reranker(auto_negatives=True, max_examples=10, epochs=15)
        self.assertEqual(result["mode"], "pair_negatives")
        self.assertGreaterEqual(result["examples"], 4)
        self.assertTrue(any(abs(value) > .001 for value in self.index.reranker["weights"]))
        self.assertFalse(self.index.reranker["enabled"])
        self.assertEqual(self.index.search("Ты пьёшь кофе?"), self.index.search("Ты пьёшь кофе?", use_reranker=False))
        self.assertNotEqual(self.index.search("Ты пьёшь кофе?", use_reranker=True)[0]["score"], self.index.search("Ты пьёшь кофе?", use_reranker=False)[0]["score"])
        with HybridRetriever.load(self.temporary.name) as restored:
            self.assertEqual(self.index.search("Ты пьёшь кофе?"), restored.search("Ты пьёшь кофе?"))

    def test_label_training_requires_training_split(self):
        labels = [
            {"query": "Ты пьёшь кофе?", "candidate_id": "coffee1", "label": 1, "split": "train"},
            {"query": "Ты пьёшь кофе?", "candidate_id": "tea", "label": 0, "split": "train"},
            {"query": "Любишь читать книги?", "candidate_id": "book", "label": 1, "split": "train"},
            {"query": "Любишь читать книги?", "candidate_id": "travel", "label": 0, "split": "train"},
            {"query": "Ты пьёшь кофе?", "candidate_id": "coffee1", "label": 0, "split": "test"},
        ]
        result = self.index.fit_reranker(labels, epochs=5, enable=True)
        self.assertEqual(result["examples"], 4)
        self.assertTrue(self.index.reranker["enabled"])
        self.assertEqual(self.index.search("Ты пьёшь кофе?"), self.index.search("Ты пьёшь кофе?", use_reranker=True))

    def test_jsonl_path_and_duplicate_records(self):
        path = Path(self.temporary.name) / "records.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records() + [records()[0]]), encoding="utf-8")
        with HybridRetriever.build(path, Path(self.temporary.name) / "nested") as built:
            self.assertEqual(built.metadata["counts"]["indexed"], 5)
            self.assertEqual(built.metadata["counts"]["skipped_duplicate_id"], 1)

    def test_timeout_does_not_poison_next_query(self):
        # Do not assume a real 1us budget expires before a tiny query completes:
        # clock resolution and scheduling differ across Windows/Python versions.
        with patch.object(self.index, "_clock", side_effect=[100.0, 102.0]):
            with self.assertRaises(TimeoutError):
                self.index.search("Ты пьёшь кофе?", timeout_seconds=1)
        self.assertIsNone(self.index._deadline)
        result = self.index.search("Ты пьёшь кофе?", timeout_seconds=2)
        self.assertEqual(result[0]["id"], "coffee1")
        self.assertEqual(self.index.connection.execute("PRAGMA busy_timeout").fetchone()[0], 1000)

    def test_sqlite_progress_timeout_does_not_poison_next_query(self):
        clock = {"now": 100.0}

        def expensive_sql(*args, **kwargs):
            # Expire the budget after search begins. The real SQLite progress
            # handler must interrupt the query, not a mocked OperationalError.
            clock["now"] = 102.0
            return self.index.connection.execute("""
                WITH RECURSIVE numbers(n) AS (
                    SELECT 0 UNION ALL SELECT n + 1 FROM numbers WHERE n < 100000
                ) SELECT sum(n) FROM numbers
            """).fetchall()

        with patch.object(self.index, "_clock", side_effect=lambda: clock["now"]):
            with patch.object(self.index, "_search", side_effect=expensive_sql):
                with self.assertRaises(TimeoutError) as raised:
                    self.index.search("кофе", timeout_seconds=1)
        self.assertIsInstance(raised.exception.__cause__, sqlite3.OperationalError)
        self.assertEqual(raised.exception.__cause__.sqlite_errorcode, sqlite3.SQLITE_INTERRUPT)
        self.assertIsNone(self.index._deadline)
        self.assertEqual(self.index.search("Ты пьёшь кофе?", timeout_seconds=2)[0]["id"], "coffee1")
        self.assertEqual(self.index.connection.execute("PRAGMA busy_timeout").fetchone()[0], 1000)

    def test_unexpired_budget_preserves_ranking(self):
        expected = self.index.search("Ты пьёшь кофе?", k=5)
        # A clock that has not advanced is not itself an expired deadline.
        with patch.object(self.index, "_clock", return_value=100.0):
            actual = self.index.search("Ты пьёшь кофе?", k=5, timeout_seconds=.000001)
        self.assertEqual(actual, expected)

    def test_old_turn_cannot_retrieve_answer_to_different_question(self):
        examples = [
            {"id": "wrong_turn", "group_id": "a", "split": "train", "context": [
                {"role": "user", "text": "Ты пьёшь кофе?"},
                {"role": "assistant", "text": "Да."},
                {"role": "user", "text": "Сколько тебе лет?"},
            ], "response": "27"},
            {"id": "right_turn", "group_id": "b", "split": "train", "context": [
                {"role": "user", "text": "Ты пьёшь кофе?"},
            ], "response": "Нет, пью чай."},
        ]
        with HybridRetriever.build(examples, Path(self.temporary.name) / "turns") as built:
            result = built.search("Ты пьёшь кофе?", k=2)
            self.assertEqual(result[0]["id"], "right_turn")
            self.assertNotIn("wrong_turn", [row["id"] for row in built.search("кофе")])

    def test_public_response_filter_also_applies_to_direct_build(self):
        examples = records() + [{"id": "bad", "group_id": "bad", "split": "train", "context": "Что для тебя дружба?", "response": "Я хочу отрезать себе палец."}]
        with HybridRetriever.build(examples, Path(self.temporary.name) / "quality") as built:
            self.assertEqual(built.metadata["counts"]["skipped_response_quality"], 1)
            self.assertFalse(any(row["id"] == "bad" for row in built.search("Что для тебя дружба?")))


if __name__ == "__main__":
    unittest.main()
