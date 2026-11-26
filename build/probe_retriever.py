import json
class ProbeRetriever:
    """Replays saved retrieval candidates (container testing only)."""
    reranker = None
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.by_q = {}
        for t in d["turns"]:
            self.by_q[t["question"]] = [{"id": c["id"], "response": c["response"], "context": [{"text": c.get("matched_context", "")}]} for c in t.get("default", [])]
    def search(self, q, **kw):
        return self.by_q.get(q, [])
