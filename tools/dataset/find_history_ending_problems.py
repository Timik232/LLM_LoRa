"""
Find examples where History ends with 'user:' instead of 'system:' or 'VIKA:'
The last message in History should ALWAYS be system or VIKA, never user
(because the user's last message is in UserInput)
"""

import json
from pathlib import Path

data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru_extended.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

print("=" * 70)
print("FINDING HISTORY ENDING PROBLEMS")
print("=" * 70)

problems = []

for key, ex in examples.items():
    history = ex.get("prompt", {}).get("History", [])

    if len(history) == 0:
        continue

    last_msg = history[-1]

    # Check if last message starts with 'user:'
    if last_msg.strip().startswith("user:"):
        user_input = ex.get("prompt", {}).get("UserInput", "")
        problems.append((key, history, user_input))

print(f"\nFound {len(problems)} examples with History ending in 'user:'\n")

if problems:
    print("=" * 70)
    print("DETAILED LIST:")
    print("=" * 70)

    for key, history, user_input in problems:
        print(f"\n{key}:")
        print(f"  History ({len(history)} messages):")
        for i, msg in enumerate(history):
            msg_short = msg[:80] + "..." if len(msg) > 80 else msg
            print(f"    [{i}] {msg_short}")
        print(f"  Last message: {history[-1][:100]}...")
        print(
            f"  UserInput: {user_input[:80]}..."
            if len(user_input) > 80
            else f"  UserInput: {user_input}"
        )
        print("  PROBLEM: History ends with 'user:' but should end with 'system:' or 'VIKA:'")

print(f"\n{'='*70}")
print(f"TOTAL: {len(problems)} examples need fixing")
print("=" * 70)
