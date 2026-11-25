"""Offline answer orchestration with a killable generation worker.

The console receives only the answer. Diagnostics are returned separately.
No downloads, remote inference, or network clients are used by this module.
"""

from __future__ import annotations

import json
import contextlib
from difflib import SequenceMatcher
import multiprocessing as mp
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def bounded_text(text: str, limit: int = 280) -> str:
    """Finish at a boundary; fit both Unicode characters and UTF-16 UI units."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    units = sum(2 if ord(character) > 0xFFFF else 1 for character in text)
    if units <= limit:
        return text
    used, end_index = 0, 0
    for character in text:
        width = 2 if ord(character) > 0xFFFF else 1
        if used + width > limit:
            break
        used += width
        end_index += 1
    prefix = text[:end_index]
    end = max(prefix.rfind(". "), prefix.rfind("! "), prefix.rfind("? "))
    if end >= limit // 3:
        return prefix[:end + 1]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip(" ,;:-")[:limit]


def output_problem(text: str) -> str | None:
    if not text.strip():
        return "empty"
    if "\ufffd" in text:
        return "invalid_unicode"
    if "```" in text or re.search(r"(?:^|\n)\s*~~~(?:[A-Za-z0-9_+-]*)\s*(?:\n|$)", text):
        return "code_fence"
    if not re.search(r"[А-Яа-яЁё]", text):
        latin_words = re.findall(r"[A-Za-z]+", text)
        if len(latin_words) >= 3 and sum(map(len, latin_words)) >= 15:
            return "non_russian_text"
    if len(text) > 8 and not re.search(r"\w", text, flags=re.UNICODE):
        return "no_content"
    if len(text) > 20 and re.search(r"(.{1,16})\1{4,}", text, flags=re.IGNORECASE):
        return "repetition"
    if re.search(r"<\|?/?(?:pad|bos|eos|user|assistant|question|history|retrieved|memory|persona)\b", text):
        return "special_tokens"
    words = re.findall(r"[\wёЁ]+", text.lower())
    if len(words) >= 9:
        triples = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if triples and max(triples.count(t) for t in set(triples)) >= 3:
            return "repetition"
    return None


def repeated_response(text: str, question: str, history: list[dict]) -> int | None:
    """Find a near-identical recent answer, not a semantic relevance verdict.

    Short factual answers, repeated questions and explicit repeat requests are
    allowed. Compare whole replies so a shared greeting alone is insufficient.
    The returned index refers to the original history, useful in diagnostics.
    """
    from arena.memory import normalize

    def words(value):
        return re.findall(r"\w+", normalize(value))

    current = words(text)
    query = words(question)
    if len(current) < 5 or re.search(
        r"\b(?:повтори|повторите|повторить|процитируй|процитируйте)\b", normalize(question)
    ):
        return None
    previous_question = None
    start = max(0, len(history) - 12)
    for index, turn in enumerate(history):
        if turn.get("role") == "user":
            previous_question = words(turn.get("text", ""))
        elif index >= start and turn.get("role") == "assistant":
            previous = words(turn.get("text", ""))
            if len(previous) < 5 or previous_question == query:
                continue
            match = SequenceMatcher(None, current, previous, autojunk=False).find_longest_match()
            if match.size >= 5 and match.size / max(len(current), len(previous)) >= .8:
                return index
    return None


def _generation_worker(connection, checkpoint, tokenizer_path, device, threads):
    # Imports happen in the child so retrieval-only operation does not need torch.
    try:
        import torch
        from arena.training import load_model_checkpoint
        from arena.formatting import format_example

        torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, metadata = load_model_checkpoint(
            checkpoint, device=device, tokenizer_path=tokenizer_path)
        del metadata
        model.eval()
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else None
        connection.send({"status": "ready", "device": device})
    except BaseException as error:
        connection.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        connection.close()
        return
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            break
        if request is None:
            break
        try:
            formatted = format_example(
                request["record"], tokenizer,
                max_source_length=min(request.get("max_source_tokens", 512), model.config.max_source_length),
                max_target_length=min(request.get("max_new_tokens", 96), model.config.max_target_length),
                retrieved=request.get("retrieved"), memory=request.get("memory"),
                tools=request.get("tools"))
            kwargs = {k: torch.tensor([formatted[k]], device=device) for k in
                ["input_ids", "attention_mask", "copy_mask", "retrieval_mask"]}
            amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
            with torch.inference_mode(), amp:
                ids = model.generate(**kwargs,
                    max_new_tokens=request.get("max_new_tokens", 96),
                    bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                    temperature=request.get("temperature", 0.0),
                    no_repeat_ngram_size=request.get("no_repeat_ngram_size", 0),
                    deadline_seconds=request["seconds"])
            text = tokenizer.decode(ids[0].detach().cpu().tolist())
            connection.send({"status": "ok", "text": text})
        except Exception as error:
            connection.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
    connection.close()


class IsolatedGenerator:
    """One model copy, persistent child, parent-side wall-clock timeout."""

    def __init__(self, checkpoint, tokenizer_path, device="auto", threads=2, startup_timeout=120):
        self.checkpoint, self.tokenizer_path = str(checkpoint), str(tokenizer_path)
        self.device, self.threads = device, threads
        self.process = self.connection = None
        self.startup_timeout = startup_timeout
        self.error = None
        self.start()

    def start(self):
        self.close()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_generation_worker,
            args=(child, self.checkpoint, self.tokenizer_path, self.device, self.threads), daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(self.startup_timeout):
            self.error = "Model startup timed out"
            self.close()
            return
        try:
            message = parent.recv()
        except (EOFError, OSError) as error:
            self.error = str(error)
            self.close()
            return
        self.error = message.get("error") if message.get("status") != "ready" else None
        if self.error:
            self.close()

    def generate(self, seconds, **request):
        if not self.process or not self.process.is_alive() or seconds <= .05:
            return {"status": "unavailable", "error": self.error}
        try:
            self.connection.send({**request, "seconds": max(.01, seconds - .1)})
            if self.connection.poll(seconds):
                return self.connection.recv()
            self.error = "Generation deadline exceeded; worker stopped"
        except (EOFError, OSError, BrokenPipeError) as error:
            self.error = str(error)
        self.close()
        return {"status": "timeout", "error": self.error}

    def close(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=.2)
            if self.process.is_alive() and hasattr(self.process, "kill"):
                self.process.kill()
                self.process.join(timeout=.2)
            if not self.process.is_alive():
                self.process.close()
        if self.connection:
            self.connection.close()
        self.process = self.connection = None


@dataclass
class Answer:
    text: str
    route: str
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ArenaRuntime:
    def __init__(self, index_path, persona=None, generator=None, max_characters=280,
                 deadline_seconds=30, max_source_tokens=512, max_new_tokens=96,
                 use_retrieval=True, use_memory=True, temperature=0.0,
                 reject_repeated_responses=False, no_repeat_ngram_size=0):
        if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
        from arena.retrieval import HybridRetriever
        from arena.memory import ConversationMemory
        self.retriever = HybridRetriever.load(index_path)
        self.persona = list(persona or [])
        self.memory = ConversationMemory(persona=self.persona)
        self.generator = generator
        self.max_characters = min(300, max(20, max_characters))
        self.deadline_seconds = min(40.0, max(1.0, deadline_seconds))
        self.max_source_tokens, self.max_new_tokens = max_source_tokens, max_new_tokens
        self.use_retrieval, self.use_memory = use_retrieval, use_memory
        self.temperature = temperature
        self.reject_repeated_responses = reject_repeated_responses
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def reset(self):
        from arena.memory import ConversationMemory
        self.memory = ConversationMemory(persona=self.persona)

    def answer(self, question: str) -> Answer:
        from arena.constraints import evaluate_candidate, execute_local_tool, recall_explicit_memory
        from arena.quality import acceptable_response

        start = time.monotonic()
        question = str(question).strip()
        if not question:
            raise ValueError("Question must not be empty")
        if len(question) > 10000:
            return Answer(bounded_text("Вопрос очень длинный. Сократи его, пожалуйста.", self.max_characters),
                          "emergency", time.monotonic() - start)
        history = list(self.memory.history)
        self.memory.add_user(question)
        memories = self.memory.retrieve(question, limit=6) if self.use_memory else []
        diagnostics = {"candidate_count": 0, "rejected": []}
        tool = execute_local_tool(question)
        if not tool and self.use_memory:
            tool = recall_explicit_memory(question, self.memory)
        if tool:
            for key in ["answer", "text", "result"]:
                if key in tool and tool[key] is not None:
                    answer = bounded_text(str(tool[key]), self.max_characters)
                    if answer and acceptable_response(answer):
                        self.memory.add_assistant(answer)
                        return Answer(answer, "tool", time.monotonic() - start)
                    diagnostics["tool_rejected"] = True
        candidates = []
        if self.use_retrieval:
            try:
                candidates = self.retriever.search(question, history=history[-6:],
                    persona=self.persona, k=12,
                    timeout_seconds=max(.01, min(5.0, self.deadline_seconds - (time.monotonic() - start) - 2.0)))
            except Exception as error:
                diagnostics["retrieval_error"] = f"{type(error).__name__}: {error}"
        diagnostics["candidate_count"] = len(candidates)
        eligible = []
        for row in candidates:
            text = bounded_text(row.get("response", ""), self.max_characters)
            problem = output_problem(text)
            if not problem and self.reject_repeated_responses and repeated_response(text, question, history) is not None:
                problem = "repeated_response"
            verdict = evaluate_candidate(question, text, self.memory if self.use_memory else None)
            if not problem and verdict.get("accepted", False):
                eligible.append({**row, "response": text})
            else:
                diagnostics["rejected"].append({"id": row.get("id"),
                    "problem": problem, "issues": verdict.get("issues", [])})
        # Keep one known response ready before requesting neural generation.
        fallback = eligible[0]["response"] if eligible else None
        generated = None
        remaining = self.deadline_seconds - (time.monotonic() - start) - .5
        if self.generator and remaining > .2:
            record = {"id": "live", "group_id": "live", "kind": "dialogue",
                "context": history[-12:] + [{"role": "user", "text": question}],
                "response": "", "persona": self.persona}
            response = self.generator.generate(seconds=remaining, record=record,
                retrieved=eligible[:1], memory=memories, tools=[],
                max_source_tokens=self.max_source_tokens, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, no_repeat_ngram_size=self.no_repeat_ngram_size)
            diagnostics["generation_status"] = response.get("status")
            if response.get("error"):
                diagnostics["generation_error"] = response["error"]
            if response.get("status") == "ok":
                candidate = bounded_text(response.get("text", ""), self.max_characters)
                verdict = evaluate_candidate(question, candidate, self.memory if self.use_memory else None)
                problem = output_problem(candidate)
                if not problem and self.reject_repeated_responses and repeated_response(candidate, question, history) is not None:
                    problem = "repeated_response"
                if not problem and verdict.get("accepted", False):
                    generated = candidate
                else:
                    diagnostics["generation_rejected"] = {"problem": problem,
                        "issues": verdict.get("issues", [])}
        if generated:
            answer, route = generated, "generator"
        elif fallback:
            answer, route = fallback, "retrieval"
        else:
            # One technical reserve, not a topic decision tree.
            answer, route = "Я запутался. Можешь сформулировать покороче?", "emergency"
        answer = bounded_text(answer, self.max_characters)
        self.memory.add_assistant(answer)
        return Answer(answer, route, time.monotonic() - start, diagnostics)

    def close(self):
        if self.generator:
            self.generator.close()
        close = getattr(self.retriever, "close", None)
        if close:
            close()


def load_persona(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {"facts": data}

