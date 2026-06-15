import json
from collections import Counter, defaultdict

data_path = "T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru.json"
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

print("=" * 70)
print("GAME SCENARIO COVERAGE ANALYSIS")
print("=" * 70)

# Analyze by scenario based on example keys
scenarios = defaultdict(list)
for key in examples.keys():
    scenarios[key.split("_")[0]].append(key)

print("\nScenarios in dataset:")
for scenario, keys in sorted(scenarios.items()):
    print(f"  {scenario}: {len(keys)} examples")

# Check for specific game mechanics mentioned in plot
game_mechanics = {
    "authentication": ["доказать", "ценност", "сотрудник", "компани", "RTUITLab"],
    "door_actions": ["открыть дверь", "открыть главную дверь", "дверь в"],
    "light_control": ["свет", "электричество", "освещ"],
    "gas_emergency": ["газ", "труб", "B8", "A3", "L7", "утечка"],
    "oxygen_control": ["кислород", "задохн"],
    "cabin_access": ["каюта", "ящик", "пропуск"],
    "engineer_section": ["инженерн", "манипулятор", "починк"],
    "information_hiding": ["скрыва", "конфиденциальн", "журнал", "записи"],
    "brevity_requests": ["коротко", "быстр", "не по предписани"],
}

print("\n" + "=" * 70)
print("GAME MECHANIC COVERAGE")
print("=" * 70)

mechanic_coverage = {}
for mechanic, keywords in game_mechanics.items():
    count = 0
    for ex in examples.values():
        prompt_text = str(ex.get("prompt", {}))
        answer_text = str(ex.get("answer", {}))
        full_text = (prompt_text + answer_text).lower()
        if any(kw.lower() in full_text for kw in keywords):
            count += 1
    mechanic_coverage[mechanic] = count
    print(f"{mechanic:25s}: {count:3d} examples")

# Analyze available actions vs game requirements
print("\n" + "=" * 70)
print("AVAILABLE ACTIONS IN DATASET")
print("=" * 70)

all_available_actions = set()
for ex in examples.values():
    actions = ex.get("prompt", {}).get("AvailableActions", [])
    all_available_actions.update(actions)

print(f"Total unique actions in AvailableActions: {len(all_available_actions)}")
for action in sorted(all_available_actions):
    print(f"  - {action}")

# Actions used in answers
all_answer_actions = Counter()
for ex in examples.values():
    action = ex.get("answer", {}).get("Content", {}).get("Action", "")
    if action:
        all_answer_actions[action] += 1

print("\n" + "=" * 70)
print("ACTIONS USED IN TRAINING ANSWERS")
print("=" * 70)
for action, count in all_answer_actions.most_common():
    print(f"  {action}: {count} examples")

# Check for missing critical game actions
print("\n" + "=" * 70)
print("MISSING GAME MECHANICS")
print("=" * 70)

critical_missing = []
if mechanic_coverage["cabin_access"] < 5:
    critical_missing.append("- Cabin/crew quarters access (каюта экипажа)")
if mechanic_coverage["engineer_section"] < 5:
    critical_missing.append("- Engineering section interactions (инженерный отдел)")
if mechanic_coverage["brevity_requests"] < 5:
    critical_missing.append("- Player requesting brief/non-bureaucratic responses")
if mechanic_coverage["information_hiding"] < 5:
    critical_missing.append("- VIKA hiding information about anomaly/crew")

if critical_missing:
    print("Critical game scenarios with insufficient examples:")
    for item in critical_missing:
        print(f"  {item}")
else:
    print("All critical game mechanics have some coverage.")

# Analyze conversation patterns
print("\n" + "=" * 70)
print("CONVERSATION DEPTH ANALYSIS")
print("=" * 70)

short_convs = sum(
    1 for ex in examples.values() if len(ex.get("prompt", {}).get("History", [])) <= 3
)
medium_convs = sum(
    1 for ex in examples.values() if 4 <= len(ex.get("prompt", {}).get("History", [])) <= 10
)
long_convs = sum(
    1 for ex in examples.values() if len(ex.get("prompt", {}).get("History", [])) > 10
)

print(f"Short conversations (≤3 messages): {short_convs}")
print(f"Medium conversations (4-10 messages): {medium_convs}")
print(f"Long conversations (>10 messages): {long_convs}")
print("\nNote: Game requires complex multi-turn conversations where player")
print("must persuade, negotiate, and extract information from VIKA.")
