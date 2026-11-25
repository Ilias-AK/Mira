"""Train and use a local byte-level BPE tokenizer; no pretrained weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers import __version__ as tokenizers_version


SPECIAL_TOKENS = (
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<question>",
    "<history>", "<persona>", "<retrieved>", "<memory>", "<tool>",
    "<user>", "<assistant>", "<mask>",
)


def _tokenizer_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "tokenizer.json" if path.is_dir() else path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArenaTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        for index, token in enumerate(SPECIAL_TOKENS):
            actual = tokenizer.token_to_id(token)
            if actual != index:
                raise ValueError(f"Invalid special token {token}: expected {index}, got {actual}")

    @classmethod
    def load(cls, path: str | Path) -> "ArenaTokenizer":
        return cls(Tokenizer.from_file(str(_tokenizer_file(path))))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def token_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise KeyError(f"Unknown tokenizer token: {token}")
        return value

    @property
    def pad_id(self) -> int:
        return self.token_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.token_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.token_id("<eos>")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, path: str | Path) -> None:
        path = _tokenizer_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


def iter_training_texts(input_paths: Iterable[str | Path], max_chars: int = 0) -> Iterator[str]:
    """Only training records enter vocabulary fitting; no implicit markup is added."""
    consumed = 0
    for path in input_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if record.get("split") != "train":
                    continue
                if record.get("kind") == "text":
                    texts = [record.get("text", "")]
                else:
                    texts = [turn.get("text", "") for turn in record.get("context", [])]
                    texts += record.get("persona", [])
                    texts += [record.get("response", "")]
                for text in texts:
                    if not isinstance(text, str) or not text:
                        continue
                    if max_chars and consumed >= max_chars:
                        return
                    if max_chars:
                        text = text[:max_chars - consumed]
                    consumed += len(text)
                    yield text


def train_tokenizer(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    max_chars: int = 0,
    show_progress: bool = True,
) -> ArenaTokenizer:
    if vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError("vocab_size must include all special tokens and 256 byte symbols")
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one training JSONL is required")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    stats = {"text_segments": 0, "characters": 0}

    def measured() -> Iterator[str]:
        next_report = 50_000_000
        for text in iter_training_texts(paths, max_chars=max_chars):
            stats["text_segments"] += 1
            stats["characters"] += len(text)
            if show_progress and stats["characters"] >= next_report:
                print(f"Tokenizer read {stats['characters']:,} train characters", flush=True)
                next_report += 50_000_000
            yield text

    tokenizer.train_from_iterator(measured(), trainer=trainer)
    if not stats["characters"]:
        raise ValueError("No nonempty train-split text found; tokenizer was not saved")
    result = ArenaTokenizer(tokenizer)
    result.save(output_path)
    output = _tokenizer_file(output_path)
    metadata = {
        "format_version": 1,
        "inputs": [{"path": str(p), "sha256": _file_hash(p)} for p in paths],
        "tokenizers_version": tokenizers_version,
        "split": "train",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": result.vocab_size,
        "min_frequency": min_frequency,
        "max_chars": max_chars,
        "special_tokens": {token: i for i, token in enumerate(SPECIAL_TOKENS)},
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        **stats,
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    from arena.console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--vocab-size", type=int, default=8192)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--max-chars", type=int, default=0)
    args = parser.parse_args(argv)
    print(f"Training byte-level BPE from train records in {len(args.input)} file(s)", flush=True)
    tokenizer = train_tokenizer(
        args.input, args.output, args.vocab_size, args.min_frequency, args.max_chars,
    )
    print(json.dumps({"tokenizer": args.output, "vocab_size": tokenizer.vocab_size}))


if __name__ == "__main__":
    main()
