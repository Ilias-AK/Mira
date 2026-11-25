"""UTF-8 output for Windows terminals, pipes, and JSONL clients."""

import sys


def configure_utf8():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

