# save as check_json.py and run:  python check_json.py
import json, io, itertools

PATH = "data/main_json.json"

try:
    with open(PATH, "r", encoding="utf-8") as f:
        json.load(f)
    print("Looks valid JSON ✅")
except json.JSONDecodeError as e:
    print(f"JSON error at line {e.lineno}, col {e.colno} (char {e.pos}) — {e.msg}")
    with open(PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(0, e.lineno - 6)
    end   = min(len(lines), e.lineno + 5)
    excerpt = "".join(f"{i+1:>6}: {line}" for i, line in enumerate(lines[start:end], start))
    print("\n--- Surrounding lines ---")
    print(excerpt)
