import json
from collections import Counter
from pathlib import Path

data_path = Path("T:/projects/LLM_LoRa/LLM_LoRa/data/dataset_ru.json")
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

examples = data.get("examples", {})
print(f"Total examples: {len(examples)}\n")

# Action distribution
actions = [
    ex.get("answer", {}).get("Content", {}).get("Action", "") for ex in examples.values()
]
action_counts = Counter(actions)
print("=" * 60)
print("ACTION DISTRIBUTION:")
print("=" * 60)
for action, count in action_counts.most_common():
    percentage = (count / len(examples)) * 100
    print(f"{action:40s}: {count:3d} ({percentage:5.1f}%)")

# Response lengths
messages = [ex.get("answer", {}).get("MessageText", "") for ex in examples.values()]
msg_lengths = [len(m) for m in messages]
print(f"\n{'=' * 60}")
print("RESPONSE STATISTICS:")
print("=" * 60)
print(f"Average response length: {sum(msg_lengths) / len(msg_lengths):.1f} chars")
print(f"Max response length: {max(msg_lengths)} chars")
print(f"Min response length: {min(msg_lengths)} chars")
print(f"Empty responses: {sum(1 for m in messages if not m.strip())}")

# History lengths
history_lengths = [len(ex.get("prompt", {}).get("History", [])) for ex in examples.values()]
print(f"\n{'=' * 60}")
print("HISTORY STATISTICS:")
print("=" * 60)
print(f"Average history length: {sum(history_lengths) / len(history_lengths):.1f} messages")
print(f"Max history length: {max(history_lengths)} messages")
print(f"Min history length: {min(history_lengths)} messages")

# Duplicates
histories = [" ".join(ex.get("prompt", {}).get("History", [])) for ex in examples.values()]
user_inputs = [ex.get("prompt", {}).get("UserInput", "") for ex in examples.values()]
hist_counter = Counter(histories)
input_counter = Counter(user_inputs)
dup_histories = sum(1 for count in hist_counter.values() if count > 1)
dup_inputs = sum(1 for count in input_counter.values() if count > 1)
print(f"\n{'=' * 60}")
print("DUPLICATE DETECTION:")
print("=" * 60)
print(f"Duplicate histories: {dup_histories}/{len(examples)}")
print(f"Duplicate user inputs: {dup_inputs}/{len(examples)}")

# Incomplete examples
incomplete = 0
for key, example in examples.items():
    prompt_data = example.get("prompt", {})
    answer_data = example.get("answer", {})
    if (
        not prompt_data.get("History")
        or not prompt_data.get("UserInput")
        or not answer_data.get("MessageText")
        or not answer_data.get("Content", {}).get("Action")
    ):
        incomplete += 1
print(f"\n{'=' * 60}")
print("DATA QUALITY:")
print("=" * 60)
print(f"Incomplete examples: {incomplete}/{len(examples)}")

# Available actions consistency
avail_actions_list = []
for ex in examples.values():
    avail = ex.get("prompt", {}).get("AvailableActions", [])
    avail_actions_list.extend(avail)
unique_available = set(avail_actions_list)
unique_used = set(actions)
print(f"\nUnique actions in AvailableActions fields: {len(unique_available)}")
print(f"Unique actions actually used in answers: {len(unique_used)}")
print("\nActions used but never in AvailableActions:")
for action in unique_used - unique_available:
    if action:
        print(f"  - {action}")

# System prompt variations
system_prompts = []
for ex in examples.values():
    hist = ex.get("prompt", {}).get("History", [])
    if hist:
        system_prompts.append(hist[0])
sys_counter = Counter(system_prompts)
print(f"\n{'=' * 60}")
print("SYSTEM PROMPT VARIATIONS:")
print("=" * 60)
print(f"Unique system prompts: {len(sys_counter)}")
for sp, count in sys_counter.most_common(5):
    print(f"  ({count}x) {sp[:70]}...")
