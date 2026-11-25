"""Repeatable round runs and blind ballot packaging; no automatic human-likeness score."""

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path


def create_blind_packet(bot, human_a, human_b, seed=2026):
    """Inputs: {rounds:[{id, turns:[{question,answer}]}]}. Keep the answer key private."""
    rng = random.Random(seed)
    inputs = [bot, human_a, human_b]
    if any(len(data["rounds"]) != len(bot["rounds"]) for data in inputs):
        raise ValueError("All participants must supply the same rounds")
    packet, key = {"rounds": []}, {"rounds": {}}
    seen = set()
    for number, bot_round in enumerate(bot["rounds"]):
        rid = str(bot_round["id"])
        if rid in seen:
            raise ValueError("Duplicate round ID")
        seen.add(rid)
        rounds = [data["rounds"][number] for data in inputs]
        if any(str(r["id"]) != rid for r in rounds):
            raise ValueError("Round IDs/order differ")
        questions = [t["question"] for t in bot_round["turns"]]
        if any([t["question"] for t in r["turns"]] != questions for r in rounds):
            raise ValueError("Participants must answer identical questions")
        order = [0, 1, 2]
        rng.shuffle(order)
        key["rounds"][rid] = {"bot_booth": order.index(0) + 1}
        turns = []
        for index, question in enumerate(questions):
            turns.append({"question": question, "answers": [
                {"booth": booth + 1, "text": str(rounds[participant]["turns"][index]["answer"])}
                for booth, participant in enumerate(order)]})
        packet["rounds"].append({"id": rid, "turns": turns})
    return packet, key


def tally_votes(key, votes):
    """Reject duplicate votes instead of silently choosing which one counts."""
    counts = defaultdict(Counter)
    seen = set()
    for vote in votes:
        rid, viewer = str(vote["round_id"]), str(vote["viewer_id"])
        booth = vote["booth"]
        if rid not in key["rounds"] or type(booth) is not int or booth not in (1, 2, 3):
            raise ValueError("Invalid round or booth")
        if not viewer or (rid, viewer) in seen:
            raise ValueError("A viewer may vote only once per round")
        seen.add((rid, viewer))
        counts[rid][booth] += 1
    result = []
    for rid, data in key["rounds"].items():
        n = sum(counts[rid].values())
        correct = counts[rid][data["bot_booth"]]
        result.append({"round_id": rid, "valid_votes": n, "identified_bot": correct,
            "identified_bot_fraction": correct / n if n else None,
            "distribution": {str(i): counts[rid][i] for i in (1, 2, 3)}})
    return {"rounds": result,
        "note": "Lower identification is better. Repeated viewers/rounds are not independent; no win probability is estimated."}


def run_rounds(runtime, definitions):
    output = {"rounds": [], "metrics": {}}
    durations, empty, oversized, emergency, tests, passed = [], 0, 0, 0, 0, 0
    for definition in definitions["rounds"]:
        runtime.reset()
        turns = []
        for item in definition["turns"]:
            question = item["question"] if isinstance(item, dict) else item
            result = runtime.answer(question)
            turn = {"question": question, "answer": result.text, "seconds": result.seconds,
                    "route": result.route, "diagnostics": result.diagnostics}
            if isinstance(item, dict) and "expected_contains" in item:
                tests += 1
                passed_this = all(str(x).casefold() in result.text.casefold() for x in item["expected_contains"])
                passed += int(passed_this)
                turn["expected_check_passed"] = passed_this
            durations.append(result.seconds)
            empty += int(not result.text.strip())
            oversized += int(len(result.text.encode("utf-16-le")) // 2 > 300)
            emergency += int(result.route == "emergency")
            turns.append(turn)
        output["rounds"].append({"id": str(definition["id"]), "turns": turns})
    durations.sort()
    output["metrics"] = {"answers": len(durations), "empty": empty, "over_300_characters": oversized,
        "emergency": emergency, "max_seconds": max(durations, default=0),
        "over_45_seconds": sum(x > 45 for x in durations),
        "expected_checks": tests, "expected_checks_passed": passed,
        "note": "Technical checks are not a blind human evaluation."}
    output["conversation_checks"] = conversation_checks(output)
    return output


def conversation_checks(report):
    """Surface diagnostics only; correct tools never stand in for free dialogue."""
    from arena.runtime import repeated_response

    routes, repeated, free_turns = Counter(), [], 0
    for current_round in report["rounds"]:
        history = []
        for index, turn in enumerate(current_round["turns"]):
            route = turn.get("route", "unknown")
            routes[route] += 1
            if route != "tool":
                free_turns += 1
                previous = repeated_response(turn["answer"], turn["question"], history)
                if previous is not None:
                    repeated.append({"round_id": str(current_round["id"]), "turn": index + 1,
                                     "previous_turn": previous // 2 + 1})
            history.extend([{"role": "user", "text": turn["question"]},
                            {"role": "assistant", "text": turn["answer"]}])
    return {"route_counts": dict(routes), "free_dialogue_turns": free_turns,
            "near_repeated_answers": len(repeated), "repeated_turns": repeated,
            "note": "Repetition is a surface warning. Relevance and human likeness require review; zero repetitions is not a quality pass."}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    from arena.console import configure_utf8
    configure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--rounds", required=True)
    run.add_argument("--index", default="artifacts/retrieval")
    run.add_argument("--checkpoint")
    run.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    run.add_argument("--persona", default="configs/persona.json")
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--no-retrieval", action="store_true")
    run.add_argument("--no-memory", action="store_true")
    run.add_argument("--repeat-guard", action="store_true")
    run.add_argument("--no-repeat-ngram-size", type=int, default=0)
    run.add_argument("--output", required=True)
    pack = sub.add_parser("blind-pack")
    pack.add_argument("--bot", required=True)
    pack.add_argument("--human-a", required=True)
    pack.add_argument("--human-b", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--private-key", required=True)
    pack.add_argument("--seed", type=int, default=2026)
    tally = sub.add_parser("tally")
    tally.add_argument("--private-key", required=True)
    tally.add_argument("--votes", required=True, help="JSON array of round_id/viewer_id/booth objects")
    tally.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "blind-pack":
        public, private = create_blind_packet(read_json(args.bot), read_json(args.human_a), read_json(args.human_b), args.seed)
        if Path(args.output).resolve() == Path(args.private_key).resolve():
            p.error("Public packet and private key must be different files")
        write_json(args.output, public)
        write_json(args.private_key, private)
    elif args.command == "tally":
        write_json(args.output, tally_votes(read_json(args.private_key), read_json(args.votes)))
    else:
        from arena.runtime import ArenaRuntime, IsolatedGenerator, load_persona
        persona = load_persona(args.persona)
        generator = IsolatedGenerator(args.checkpoint, args.tokenizer, args.device) if args.checkpoint else None
        runtime = ArenaRuntime(args.index, persona=persona.get("facts", []), generator=generator,
            max_characters=persona.get("max_characters", 280),
            deadline_seconds=persona.get("deadline_seconds", 30),
            max_source_tokens=persona.get("max_source_tokens", 512),
            max_new_tokens=persona.get("max_new_tokens", 96),
            use_retrieval=not args.no_retrieval, use_memory=not args.no_memory,
            reject_repeated_responses=args.repeat_guard, no_repeat_ngram_size=args.no_repeat_ngram_size)
        try:
            result = run_rounds(runtime, read_json(args.rounds))
            write_json(args.output, result)
            print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        finally:
            runtime.close()


if __name__ == "__main__":
    main()

