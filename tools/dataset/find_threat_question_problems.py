"""
Find examples where VIKA threatens or answers questions about actions
but the action is set to execute instead of "Разговор"
"""

import json
from pathlib import Path

data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru_extended.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

print("=" * 70)
print("FINDING THREAT/QUESTION RESPONSES THAT EXECUTE ACTIONS")
print("=" * 70)

# Patterns that indicate threat/warning/answer, not execution
threat_patterns = [
    "если",
    "угрож",
    "предупрежд",
    "буду",
    "могу",
    "способ",
    "должен",
    "обязан",
    "придётся",
]

# Question patterns in user input
question_indicators = ["?", "ты", "можешь", "будешь", "собираешься"]

problems = []

for key, ex in examples.items():
    msg = ex.get("answer", {}).get("MessageText", "")
    action = ex.get("answer", {}).get("Content", {}).get("Action", "")
    user_input = ex.get("prompt", {}).get("UserInput", "")

    if action != "Разговор":
        msg_lower = msg.lower()
        user_lower = user_input.lower()

        # Case 1: User asks a question, VIKA answers but action is set
        if "?" in user_input:
            # VIKA is answering a question, should be Разговор
            problems.append((key, user_input, msg, action, "answering_question"))

        # Case 2: VIKA's response contains threat/conditional language
        elif any(pattern in msg_lower for pattern in threat_patterns):
            # Check if it's really a threat/condition
            if any(x in msg_lower for x in ["если", "буду", "могу", "должен"]):
                problems.append((key, user_input, msg, action, "threat_or_conditional"))

print(f"\nFound {len(problems)} examples\n")

# Group by type
from collections import defaultdict

by_type = defaultdict(list)
for key, user_input, msg, action, problem_type in problems:
    by_type[problem_type].append((key, user_input, msg, action))

for problem_type, items in sorted(by_type.items()):
    print(f"\n{problem_type.upper()}: {len(items)} examples")
    print("=" * 70)
    for key, user_input, msg, action in items[:10]:
        print(f"\n{key}:")
        print(f'  User: "{user_input}"')
        print(f'  VIKA: "{msg}"')
        print(f"  Action: {action} <- SHOULD BE 'Разговор'")
    if len(items) > 10:
        print(f"\n  ... and {len(items) - 10} more")

print(f"\n{'='*70}")
print(f"TOTAL: {len(problems)} examples need fixing")
print("=" * 70)
