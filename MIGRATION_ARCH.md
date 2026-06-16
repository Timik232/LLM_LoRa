# Миграция LLM_LoRa + Hermes Agent на Arch Linux

**Дата:** 2026-06-16
**Железо:** i5-12600KF, RTX 3090 24GB, 64GB RAM (то же устройство, другая ОС)

---

## ЧАСТЬ 1. Миграция проекта LLM_LoRa

### 1.1. Что переносим

| Компонент | Источник (Windows) | Назначение (Arch) |
|---|---|---|
| Git-репозиторий | `T:\pythonProject\LLM_LoRa` | `~/projects/LLM_LoRa` |
| Ветка | `main` (854ecbd) + незакоммиченные правки | `main` |
| Данные | `data/dataset_ru_extended.json` (194 примера) | тот же путь |
| DVC remote | `s3://dvc-storage` @ `minio.komolov.synology.me` | без изменений |
| MLflow | `https://mlflow.komolov.synology.me` | без изменений |
| Git remotes | `origin` (GitHub), `rtuitlab` (GitLab) | без изменений |

### 1.2. Незакоммиченные правки (ВАЖНО!)

На ветке `main` есть **незакоммиченные изменения** в двух файлах:

**`conf/config.yaml`** — изменено:
- LoRA: `r: 8 → 32`, `alpha: 16 → 32` (= r, по рекомендации Unsloth)
- epochs: `1.5 → 10`
- eval_steps: `50 → 20`
- warmup_steps: `5 → 10`
- learning_rate: `3.26e-05 → 5e-5`
- weight_decay: `0.03 → 0.01`
- max_grad_norm: добавлен `1.0`
- optim: `paged_adamw_8bit → adamw_torch`
- gradient_checkpointing: `true → false`

**`training_model/one_file_train.py`** — изменено:
- `generate_prompt()`: добавлен `add_generation_prompt=False` + try/except fallback
- `generate_and_tokenize_prompt()`: добавлен `"text": full_prompt` для SFTTrainer
- `attach_lora_adapters()` + `run_sft_training()`: target_modules расширены с 3 до 7 (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
- `SFTConfig`: убран `skip_prepare_dataset=True` (совместимость с TRL 1.6)
- `SFTConfig`: добавлен `max_grad_norm=cfg.training.max_grad_norm`
- `merge_adapter_from_checkpoint()`: автоудаление MTP-ключей (`mtp_num_hidden_layers`) из config.json

**⇒ Перед миграцией нужно закоммитить эти правки!** (инструкция ниже)

### 1.3. Предварительные шаги (на Windows, перед переездом)

```bash
cd /t/pythonProject/LLM_LoRa

# 1. Закоммитить незакоммиченные правки
git add conf/config.yaml training_model/one_file_train.py
git commit -m "Qwen3.5: LoRA r=32/alpha=32, 7 target_modules, adamw_torch, remove skip_prepare_dataset, MTP cleanup"

# 2. Удалить временные debug-скрипты (не нужны на Arch)
git clean -fdx -- test_*.py debug_*.py *.log train_log.txt *.stackdump

# 3. Запушить
git push origin main

# 4. Проверить что DVC данные запушены
dvc status -c
dvc push   # если есть незапушенные данные
```

### 1.4. Установка на Arch Linux

#### Системные пакеты

```bash
sudo pacman -Syu

# Базовые инструменты
sudo pacman -S base-devel git python python-pip curl wget jq

# CUDA (NVIDIA драйвер + toolkit)
sudo pacman -S nvidia nvidia-utils cuda cuda-tools

# Дополнительно для сборки
sudo pacman -S cmake ninja openmp

# Опционально: poetry
sudo pacman -S poetry
# ИЛИ через pip:
# pip install poetry
```

#### Проверка GPU

```bash
nvidia-smi
# Должно показать RTX 3090, 24GB, driver 570+ (или новее)
nvcc --version
# CUDA toolkit 12.x
```

#### Клонирование проекта

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Timik232/LLM_LoRa.git
cd LLM_LoRa

# Добавить GitLab remote
git remote add rtuitlab https://gitlab.rtuitlab.dev/ser13volk/LLM_LoRa.git
```

#### Poetry окружение

```bash
cd ~/projects/LLM_LoRa

# ВАЖНО: pyproject.toml указывает torch 2.6.0 для cu124
# На Arch можно использовать системный CUDA или conda-managed
# Вариант A: через poetry (как на Windows)
poetry env use python3.11
poetry install

# Вариант B: если poetry тянет неправильный torch:
# 1. Установить torch вручную: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 2. Затем poetry install --no-root
```

### 1.5. ⭐ КРИТИЧЕСКИЙ ШАГ: flash-linear-attention

**Это то, что невозможно на Windows и решает `grad_norm: nan`.**

```bash
cd ~/projects/LLM_LoRa
poetry shell   # активировать venv

# Установка triton (доступен ТОЛЬКО на Linux!)
pip install triton

# Установка causal-conv1d (требует Linux + CUDA)
pip install causal-conv1d

# Установка flash-linear-attention
pip install flash-linear-attention

# Проверка
python -c "import triton; print('triton:', triton.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "import flash_linear_attention; print('FLA:', flash_linear_attention.__version__)"
```

**Ожидаемый результат:** все три импорта работают без ошибок.

**Проверка что fast path активирован:**
```bash
PYTHONUNBUFFERED=1 poetry run python -m training_model train 2>&1 | head -50
# НЕ должно быть предупреждения:
#   "The fast path is not available because one of the required library is not installed"
```

### 1.6. Настройка DVC

```bash
cd ~/projects/LLM_LoRa

# AWS credentials (НЕ в .dvc/config!)
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = ser13volk
aws_secret_access_key = 123654789
EOF
chmod 600 ~/.aws/credentials

# DVC remote уже настроен в .dvc/config (пушится в git)
# Проверка:
dvc status -c    # должно показать "Data and pipelines are up to date"
dvc pull         # скачать данные если ещё не скачаны
```

### 1.7. Сборка llama.cpp (для GGUF конверсии)

```bash
cd ~/projects/LLM_LoRa

# Клонировать llama.cpp (если не в репо)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Собрать с CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Проверить
./build/bin/llama-quantize --help

# Обновить путь в conf/config.yaml:
# quantized_path: "llama-quantize.exe" → "llama.cpp/build/bin/llama-quantize"
```

### 1.8. Запуск обучения

```bash
cd ~/projects/LLM_LoRa
PYTHONUNBUFFERED=1 poetry run python -m training_model train
```

**Что проверить в первых шагах:**
1. `grad_norm` — **должен быть числом**, не `nan`
2. `loss` — должен падать от ~2.8 вниз
3. Скорость — **должна быть быстрее** чем на Windows (ожидаемо 3–7 с/шаг, не 150+ с/шаг)
4. Предупреждение "fast path not available" — **НЕ должно появляться**

### 1.9. Конфигурация, требующая изменения на Arch

В `conf/config.yaml` нужно обновить путь к llama-quantize:

```yaml
paths:
  # Windows: "llama-quantize.exe"
  # Linux:   "llama.cpp/build/bin/llama-quantize"
  quantized_path: "llama.cpp/build/bin/llama-quantize"
```

Остальное (модель, данные, DVC, MLflow) — без изменений.

### 1.10. Рекомендации Unsloth для Qwen3.5 (применить после стабилизации)

| Параметр | Текущее значение | Рекомендация Unsloth |
|---|---|---|
| optim | `adamw_torch` | `adamw_8bit` (экономия памяти) |
| lora_alpha | `32` (= r) | `32` (= r) ✓ уже правильно |
| attn_implementation | `flash_attention_2` | `flash_attention_2` ✓ |
| gradient_checkpointing | `false` | `true` (если не хватает памяти) |
| QLoRA 4-bit | не используется | не рекомендуется для Qwen3.5 |

---

## ЧАСТЬ 2. Миграция Hermes Agent

### 2.1. Что переносим

| Компонент | Источник (Windows) | Назначение (Arch) |
|---|---|---|
| config.yaml | `~\AppData\Local\hermes\config.yaml` | `~/.config/hermes/config.yaml` |
| .env | `~\AppData\Local\hermes\.env` | `~/.config/hermes/.env` |
| gateway.json | `~\AppData\Local\hermes\gateway.json` | `~/.config/hermes/gateway.json` |
| memories | `~\AppData\Local\hermes\memories\` | `~/.config/hermes/memories/` |
| SOUL.md | `~\AppData\Local\hermes\SOUL.md` | `~/.config/hermes/SOUL.md` |
| sessions DB | `~\AppData\Local\hermes\state.db` | `~/.config/hermes/state.db` |
| skills | (встроенные, переустановятся) | автоматически |

### 2.2. Установка Hermes на Arch

```bash
# Установить Hermes Agent
pip install hermes-agent

# Или через UV (быстрее):
pip install uv
uv tool install hermes-agent

# Проверка
hermes --version
```

### 2.3. Настройка конфигурации

```bash
mkdir -p ~/.config/hermes

# config.yaml можно скопировать с Windows как есть,
# НО нужно изменить terminal.backend:
```

**Ключевые изменения в `config.yaml`:**

```yaml
terminal:
  backend: local          # осталось как есть (local)
  shell_init_files: []    # на Arch можно добавить ["~/.bashrc"]
  auto_source_bashrc: true

# Остальное — без изменений:
# - model: glm-5.2-chat через custom provider (litellm proxy)
# - web backend: exa
# - telegram: включён через gateway.json
```

### 2.4. Секреты (.env)

**Активные ключи на текущей системе:**

```bash
# Скопировать эти переменные в ~/.config/hermes/.env:

# Telegram
TELEGRAM_BOT_TOKEN=<токен из gateway.json>
TELEGRAM_ALLOWED_USERS=<твой Telegram user ID>

# LLM
GLM_API_KEY=<ключ>

# Web search
EXA_API_KEY=<ключ>

# Browser (если используется)
BROWSERBASE_PROXIES=true

# Terminal
TERMINAL_ENV=local
TERMINAL_TIMEOUT=180
TERMINAL_LIFETIME_SECONDS=300

# Modal (если используется)
TERMINAL_MODAL_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20

# Debug
WEB_TOOLS_DEBUG=false
VISION_TOOLS_DEBUG=false
MOA_TOOLS_DEBUG=false
IMAGE_TOOLS_DEBUG=false
```

**Значения можно получить командой (на Windows):**
```bash
cat ~/AppData/Local/hermes/.env | grep -v "^#" | grep -v "^$"
```

### 2.5. Telegram Gateway

```bash
# gateway.json — скопировать как есть
# Содержит Telegram bot token (дублирует TELEGRAM_BOT_TOKEN в .env)

cat > ~/.config/hermes/gateway.json << 'EOF'
{
  "platforms": {
    "telegram": {
      "enabled": true,
      "token": "<ТЕЛЕГРАМ_ТОКЕН>"
    }
  }
}
EOF
chmod 600 ~/.config/hermes/gateway.json
```

**⚠️ Важно:** при запуске Hermes на Arch, Telegram-бот **отключится на Windows** (бот может работать только в одном экземпляре — Telegram API разрешает только одно long-polling соединение). Нужно выключить Hermes на Windows перед запуском на Arch.

### 2.6. Memory (память агента)

```bash
# Скопировать файлы памяти (там сохранены факты о проектах)
cp /mnt/t/.../memories/MEMORY.md ~/.config/hermes/memories/MEMORY.md
cp /mnt/t/.../memories/USER.md   ~/.config/hermes/memories/USER.md
```

**Содержимое MEMORY.md включает:**
- Структуру проекта VK Styles
- Web Audio API pitfall
- Структуру проекта LLM_LoRa

**Содержимое USER.md включает:**
- Стиль общения пользователя (терск, русский, автономное выполнение)
- Предпочтения по форматированию

### 2.6.1. Обновить пути в MEMORY после переезда

После миграции нужно обновить пути в памяти:

```
# Старое:
# T:\pythonProject\LLM_LoRa → ~/projects/LLM_LoRa
# T:\pythonProject\vk-styles → ~/projects/vk-styles
# T:\llm_models → ~/llm_models
```

### 2.7. Первый запуск

```bash
# Запустить Hermes
hermes

# Или в gateway-режиме (для Telegram):
hermes gateway start
```

**Проверочный чек-лист:**
- [ ] Модель `glm-5.2-chat` отвечает через litellm proxy
- [ ] Telegram-бот подключается (получает сообщения)
- [ ] Terminal-команды работают (`ls`, `git`, `python`)
- [ ] Memory загружена (показывает контекст проектов)
- [ ] Web search работает (через Exa)

### 2.8. Опциональные улучшения для Arch

```yaml
# В config.yaml можно добавить:
terminal:
  backend: local
  shell_init_files:
    - "~/.bashrc"

# systemd service для автозапуска Hermes Gateway:
```

```bash
# /etc/systemd/system/hermes-gateway.service
[Unit]
Description=Hermes Agent Gateway
After=network.target

[Service]
Type=simple
User=%i
ExecStart=/usr/bin/hermes gateway start
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## ЧАСТЬ 3. Чек-лист миграции

### Подготовка (на Windows)
- [ ] Закоммитить правки в LLM_LoRa (`git add` + `git commit` + `git push`)
- [ ] Очистить debug-скрипты (`git clean -fdx -- test_*.py debug_*.py`)
- [ ] Запушить DVC данные (`dvc push`)
- [ ] Сохранить `.env` Hermes (выписать активные ключи)
- [ ] Сохранить `gateway.json` (Telegram токен)
- [ ] Сохранить `config.yaml` Hermes
- [ ] Сохранить `memories/` (MEMORY.md, USER.md)

### Установка (на Arch)
- [ ] Системные пакеты (`base-devel`, `cuda`, `python`, `git`)
- [ ] `nvidia-smi` работает, GPU виден
- [ ] Hermes установлен (`pip install hermes-agent`)
- [ ] LLM_LoRa клонирован (`git clone`)
- [ ] Poetry venv создан (`poetry install`)
- [ ] `flash-linear-attention` + `causal-conv1d` + `triton` установлены
- [ ] DVC работает (`dvc pull`)
- [ ] AWS credentials настроены (`~/.aws/credentials`)
- [ ] llama.cpp собран (`cmake -B build -DGGML_CUDA=ON`)
- [ ] Путь к llama-quantize обновлён в `conf/config.yaml`

### Запуск
- [ ] Обучение: `grad_norm` — число, не `nan`
- [ ] Обучение: `loss` падает
- [ ] Обучение: скорость < 10 с/шаг
- [ ] Нет предупреждения "fast path not available"
- [ ] Hermes отвечает в Telegram
- [ ] Terminal, web search, memory работают

---

## Приложение A. Текущая конфигурация обучения

```yaml
# conf/config.yaml (актуальные значения)
model:
  model_name: "Qwen/Qwen3.5-2B"
  new_model: "qwen3.5-2b-chinese-style"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  train_steps: 60
  lora:
    r: 32
    alpha: 32
    dropout: 0.05

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 6
  num_train_epochs: 10
  eval_steps: 20
  logging_steps: 5
  warmup_steps: 10
  learning_rate: 5e-5
  bf16: true
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 2048
  optim: "adamw_torch"
  gradient_checkpointing: false

logging:
  logging_backend: "mlflow"
  mlflow:
    experiment_name: "qwen3.5-2b-chinese-style"
    tracking_uri: "https://mlflow.komolov.synology.me"
```

## Приложение B. Данные

- **Задача:** style transfer — переписывание русского текста в «китайском» стиле
- **Train:** `data/dataset_ru_extended.json` — 194 примера
- **Test:** `data/test_ru_extended.json` — 20 примеров
- **Сырые:** `data/translate_expanded.csv` — ~194 пары
- **System prompt:** содержит `/no_think` для отключения think-блоков

## Приложение C. Решение проблемы grad_norm: nan

**Корневая причина:** Qwen3.5 имеет гибридную архитектуру (GatedDeltaNet — линейное внимание/SSM + стандартное Multi-Head Attention). При `sdpa` или даже `flash_attention_2` без библиотеки `flash-linear-attention`, DeltaNet слои падают в torch fallback → bf16 переполнение в Softmax → `grad_norm: nan`.

**Решение (только Linux):**
1. Установить `triton` (недоступен на Windows)
2. Установить `causal-conv1d` (недоступен на Windows)
3. Установить `flash-linear-attention` (зависит от triton + causal-conv1d)

**Ссылки:**
- GitHub issue: [QwenLM/Qwen3.5#107](https://github.com/QwenLM/Qwen3.5/issues/107)
- Transformers issue: [huggingface/transformers#44928](https://github.com/huggingface/transformers/issues/44928)
