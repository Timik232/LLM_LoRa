"""
Find all examples where VIKA asks a question but executes an action
These should be using "Разговор" instead
"""

import json
from pathlib import Path

data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru_extended.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

# Find problematic examples
problems = []
confirmation_keywords = ["уверены", "точно", "правильно", "подтверд", "действительно"]

for key, ex in examples.items():
    msg = ex.get("answer", {}).get("MessageText", "")
    action = ex.get("answer", {}).get("Content", {}).get("Action", "")

    # Check if message ends with '?' and action is not "Разговор"
    if action != "Разговор":
        # Case 1: Message ends with question mark
        if msg.strip().endswith("?"):
            problems.append((key, msg, action, "ends_with_question"))
        # Case 2: Contains confirmation keywords
        elif any(kw in msg.lower() for kw in confirmation_keywords):
            if "?" in msg:  # Only if it's a question
                problems.append((key, msg, action, "confirmation_question"))

print("=" * 70)
print("PROBLEMATIC EXAMPLES FOUND")
print("=" * 70)
print(f"\nTotal: {len(problems)} examples with questions executing actions\n")

# Group by action type
from collections import defaultdict

by_action = defaultdict(list)
for key, msg, action, reason in problems:
    by_action[action].append((key, msg, reason))

for action, items in sorted(by_action.items()):
    print(f"\n{action}: {len(items)} examples")
    for i, (key, msg, reason) in enumerate(items[:5], 1):
        msg_short = msg[:70] + "..." if len(msg) > 70 else msg
        print(f"  {i}. {key}")
        print(f'     "{msg_short}"')
    if len(items) > 5:
        print(f"     ... and {len(items) - 5} more")

print("\n" + "=" * 70)
print("EXAMPLES TO FIX:")
print("=" * 70)
for key, msg, action, reason in problems:
    print(f"\n{key}:")
    print(f'  Message: "{msg}"')
    print(f"  Current action: {action}")
    print("  Should be: Разговор")
    print(f"  Reason: {reason}")
