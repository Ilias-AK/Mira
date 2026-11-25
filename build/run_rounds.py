import sys, json
sys.path.insert(0, "build"); sys.path.insert(0, ".")
from probe_retriever import ProbeRetriever
from mira.runtime import MiraRuntime
from collections import Counter
r = MiraRuntime("config/persona.json", "config/bank.json", "config/fallbacks.json",
                retriever=ProbeRetriever("/home/claude/n4r/runs/night_v4/reviews/retrieval_probe_dev.json"))
c = Counter()
for item in json.load(open(sys.argv[1]))["rounds"]:
    r.reset(); print("==", item["id"])
    for q in item["turns"]:
        a = r.answer(q); c[a.route.split(":")[0]] += 1
        print("  U:", q); print("  M:", a.text, "   [", a.route, "]")
print(c)
