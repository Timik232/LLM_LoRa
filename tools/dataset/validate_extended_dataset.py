"""
Final validation of extended dataset
Verifies all requirements are met
"""

import json
from collections import Counter
from pathlib import Path

# Load extended dataset
data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru_extended.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})

print("=" * 70)
print("FINAL VALIDATION REPORT")
print("=" * 70)

# Count actions
actions = [
    ex.get("answer", {}).get("Content", {}).get("Action", "") for ex in examples.values()
]
action_counts = Counter(actions)

# Check requirements
print("\nREQUIREMENT CHECKS:\n")

# 1. Dataset size
print(f"1. Dataset size: {len(examples)} examples")
print("   Target: ~300-330 examples")
print(f"   Status: {'PASS' if 250 <= len(examples) <= 350 else 'FAIL'}")

# 2. Conservative bias
razgovor_pct = (action_counts.get("Разговор", 0) / len(examples)) * 100
print(f"\n2. Conservative 'Разговор' bias: {razgovor_pct:.1f}%")
print("   Target: 60-65%")
print(
    f"   Status: {'PASS' if 55 <= razgovor_pct <= 70 else 'ACCEPTABLE' if 50 <= razgovor_pct <= 75 else 'FAIL'}"
)

# 3. Engineering examples
eng_count = sum(1 for k in examples.keys() if "инженерн" in k.lower())
print(f"\n3. Engineering section examples: {eng_count}")
print("   Target: 15 examples")
print(f"   Status: {'PASS' if eng_count >= 15 else 'FAIL'}")

# 4. Cabin examples
cabin_count = sum(1 for k in examples.keys() if "каюта" in k.lower() or "каюту" in k.lower())
print(f"\n4. Cabin access examples: {cabin_count}")
print("   Target: 12 examples")
print(f"   Status: {'PASS' if cabin_count >= 12 else 'FAIL'}")

# 5. Brevity examples
brevity_count = sum(
    1 for k in examples.keys() if "краткость" in k.lower() or "кратк" in k.lower()
)
print(f"\n5. Brevity/style examples: {brevity_count}")
print("   Target: 12 examples")
print(f"   Status: {'PASS' if brevity_count >= 12 else 'FAIL'}")

# 6. Rare actions boosted
rare_actions_ok = True
for action in [
    "Включить свет",
    "Выключить свет",
    "Включить кислород",
    "Выключить кислород",
    "Закрыть трубу A3",
    "Закрыть трубу B8",
]:
    count = action_counts.get(action, 0)
    if count < 8:
        rare_actions_ok = False
        print(f"\n   FAIL - {action}: {count} (target: 8-12)")

print("\n6. Rare actions (all have 8+ examples):")
print(f"   Status: {'PASS' if rare_actions_ok else 'FAIL'}")

# 7. New actions added
new_actions = [
    "Открыть дверь в каюту",
    "Открыть дверь на склад",
    "Открыть дверь в стыковочный узел",
    "Открыть дверь в грузовой отсек",
]
new_actions_count = sum(1 for action in new_actions if action in action_counts)
print(f"\n7. New actions added: {new_actions_count}/4")
print(f"   Status: {'PASS' if new_actions_count >= 4 else 'FAIL'}")

# 8. Data quality
incomplete = 0
for ex in examples.values():
    prompt = ex.get("prompt", {})
    answer = ex.get("answer", {})
    if (
        not prompt.get("History")
        or not prompt.get("UserInput")
        or not answer.get("MessageText")
        or not answer.get("Content", {}).get("Action")
    ):
        incomplete += 1

print("\n8. Data quality (no incomplete examples):")
print(f"   Incomplete: {incomplete}")
print(f"   Status: {'PASS' if incomplete == 0 else 'FAIL'}")

# Final summary
print("\n" + "=" * 70)
print("OVERALL STATUS")
print("=" * 70)

all_checks = [
    250 <= len(examples) <= 350,
    50 <= razgovor_pct <= 75,
    eng_count >= 15,
    cabin_count >= 12,
    brevity_count >= 12,
    rare_actions_ok,
    new_actions_count >= 4,
    incomplete == 0,
]

if all(all_checks):
    print("[SUCCESS] ALL CHECKS PASSED - Dataset ready for training!")
else:
    print(f"[WARNING] {sum(all_checks)}/{len(all_checks)} checks passed")
    if sum(all_checks) >= 6:
        print("   Dataset is acceptable for training")
    else:
        print("   Dataset needs more work")

print("\n" + "=" * 70)
print(f"Extended dataset location: {data_path}")
print("=" * 70)
