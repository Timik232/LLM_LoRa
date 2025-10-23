"""
Find examples where VIKA announces future action but executes it immediately
VIKA saying "я выключу", "я закрою", "отключу" etc. should use "Разговор", not execute
"""

import json
from pathlib import Path

data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru_extended.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

print("=" * 70)
print("FINDING FUTURE TENSE ANNOUNCEMENTS WITH IMMEDIATE EXECUTION")
print("=" * 70)

# Future tense verbs that indicate announcement, not execution
future_patterns = [
    "выключу",
    "включу",
    "закрою",
    "открою",
    "отключу",
    "перекрою",
    "активирую",
    "запущу",
    "восстановлю",
    "буду",
    "сделаю",
    "выполню",
]

problems = []

for key, ex in examples.items():
    msg = ex.get("answer", {}).get("MessageText", "")
    action = ex.get("answer", {}).get("Content", {}).get("Action", "")

    if action != "Разговор":
        msg_lower = msg.lower()
        for pattern in future_patterns:
            if pattern in msg_lower:
                # This is announcing a future action but executing it now
                problems.append((key, msg, action, pattern))
                break

print(f"\nFound {len(problems)} examples with future tense + immediate execution\n")

# Group by action
from collections import defaultdict

by_action = defaultdict(list)
for key, msg, action, pattern in problems:
    by_action[action].append((key, msg, pattern))

for action, items in sorted(by_action.items()):
    print(f"\n{action}: {len(items)} examples")
    for key, msg, pattern in items[:3]:
        msg_short = msg[:80] + "..." if len(msg) > 80 else msg
        print(f"  {key}")
        print(f'    "{msg_short}"')
        print(f"    Pattern: '{pattern}'")
    if len(items) > 3:
        print(f"    ... and {len(items) - 3} more")

print("\n" + "=" * 70)
print("DETAILED LIST:")
print("=" * 70)

for key, msg, action, pattern in problems:
    print(f"\n{key}:")
    print(f'  Message: "{msg}"')
    print(f"  Pattern matched: '{pattern}'")
    print(f"  Current action: {action}")
    print("  Should be: Разговор (announcing, not executing)")

print(f"\n{'='*70}")
print(f"TOTAL: {len(problems)} examples need fixing")
print("=" * 70)
