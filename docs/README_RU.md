# 🚀 LLM-LoRa (Русская версия)

**Эффективное по памяти дообучение больших языковых моделей с использованием LoRa и продвинутого квантования**

[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](../docker-compose.yaml)

[Быстрый старт](#быстрый-старт) • [Возможности](#возможности) • [Установка](#установка) • [Документация](#документация) • [Примеры](#примеры)

---

## 🎯 Обзор

**LLM-LoRa** - это комплексная платформа для эффективного по памяти дообучения больших языковых моделей с использованием LoRa (Low-Rank Adaptation) и передовых техник квантования. Обучайте современные LLM со значительно меньшими требованиями к памяти, сохраняя при этом производительность.

### ✨ Ключевые преимущества

- **🧠 Эффективность памяти**: Обучение больших моделей с использованием на 70% меньше GPU памяти благодаря LoRa и квантованию
- **⚡ Несколько методов**: Поддержка SFT (Supervised Fine-Tuning) и GRPO (метод из DeepSeek)
- **🔧 Продвинутое квантование**: Встроенная поддержка квантования bitsandbytes (4-bit, 8-bit)
- **🐳 Готово к продакшену**: Полная поддержка Docker с ускорением GPU и интеграцией Ollama
- **📊 Отслеживание экспериментов**: Интегрированное логирование MLflow и Weights & Biases
- **⚙️ Гибкая конфигурация**: Управление конфигурацией на основе Hydra

## 🚀 Быстрый старт

Начните работу менее чем за 2 минуты:

### Использование Docker (Рекомендуется)

```bash
# Клонируйте репозиторий
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# Настройте параметры обучения
cp conf/config.yaml my_config.yaml
# Отредактируйте my_config.yaml со своими настройками

# Запустите обучение
docker-compose up
```

### Использование Poetry

```bash
# Установите зависимости
poetry install --no-root

# Запустите обучение
python main.py
```

### Пример быстрого обучения

```python
# Базовое обучение с настройками по умолчанию
python main.py model.model_name=microsoft/DialoGPT-small

# Кастомная конфигурация
python main.py \
  model.model_name=microsoft/DialoGPT-medium \
  model.train_steps=1000 \
  training.learning_rate=2e-5 \
  training.use_grpo=true
```

## ✨ Возможности

### 🧠 **Оптимизация памяти**
- Дообучение LoRa (Low-Rank Adaptation)
- 4-битное и 8-битное квантование
- Gradient checkpointing
- Эффективные по памяти паттерны внимания

### ⚡ **Методы обучения**
- **SFT**: Supervised Fine-Tuning (обычно дообучение с учителем)
- **GRPO**: Generalized Reward-based Policy Optimization (метод, который применялся в DeepSeek)
- Кастомные функции потерь и оптимизаторы
- Обучение со смешанной точностью

### 🛠️ **Продакшен возможности**
- Docker контейнеризация с поддержкой GPU
- Интеграция с Ollama для сервинга моделей
- Управление конфигурацией через Hydra
- DVC для версионирования данных и моделей

### 📊 **Мониторинг и оценка**
- Отслеживание экспериментов MLflow
- Интеграция с Weights & Biases
- Автоматизированная оценка DeepEval
- Кастомные метрики и логирование

## 📦 Установка

### Требования

- **GPU**: NVIDIA GPU с поддержкой CUDA
- **Память**: Минимум 8GB видеопамяти (зависит от размера модели)
- **Хранилище**: 70GB+ для полной установки Docker
- **Python**: 3.11 до 3.13

**Важно!** Образ может занимать до 70ГБ памяти, это связано с тем, что библиотеки для работы с CUDA достаточно тяжёлые, как и скачиваемая модель. Убедитесь, что на вашем диске достаточно места.

### Вариант 1: Docker (Рекомендуется)

**Предварительные требования:**
- GPU NVIDIA с поддержкой CUDA
- Docker и Docker Compose установлены
- NVIDIA Container Toolkit

```bash
# 1. Установка NVIDIA Container Toolkit (если не установлен)
# Ubuntu/Debian:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Проверьте доступ к GPU в Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. Клонируйте репозиторий
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# 4. Настройте обучение (опционально)
cp conf/config.yaml my_config.yaml
# Отредактируйте my_config.yaml со своими настройками

# 5. Соберите и запустите обучение
docker-compose build
docker-compose up llm_training

# 6. Запуск сервинга модели (после обучения)
docker-compose up ollama
```

**Docker Сервисы:**
- `llm_training`: Контейнер обучения с поддержкой GPU
- `ollama`: Контейнер сервинга модели (порт 11434)
- `mlflow`: UI отслеживания экспериментов (порт 5000)

### Вариант 2: Локальная установка

Рекомендую создать отдельно виртуальное окружение для Python для обучения модели.

```bash
# Клонируйте репозиторий
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# Установите Poetry (если не установлен)
pip install poetry

# Установите зависимости
poetry install --no-root

# Проверьте установку
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔧 Конфигурация

Проект использует Hydra для управления конфигурацией. В директории conf находится файл `config.yaml`, который позволяет настроить параметры обучения модели.

Ключевые конфигурационные файлы:
- `conf/config.yaml`: Основная конфигурация
- `conf/model/`: Настройки моделей
- `conf/training/`: Гиперпараметры обучения

### Пример конфигурации

```yaml
# conf/config.yaml
model:
  model_name: "microsoft/DialoGPT-medium"
  train_steps: 1000
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"

training:
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  use_grpo: false

paths:
  train_data: "data/train.json"
  output_dir: "models/output"
```

## 📖 Примеры использования

### Базовое обучение

```bash
# Обучение с настройками по умолчанию
python main.py

# Обучение конкретной модели
python main.py model.model_name=microsoft/DialoGPT-large

# Включение GRPO обучения
python main.py training.use_grpo=true
```

### Продвинутое обучение

```bash
# Полный pipeline с кастомными настройками
python main.py \
  model.model_name=microsoft/DialoGPT-medium \
  model.train_steps=2000 \
  training.learning_rate=1e-4 \
  training.batch_size=8 \
  model.lora.r=32 \
  logging.use_mlflow=true
```

### Конвертация RKLLM (Развертывание на устройствах)

Конвертируйте обученные модели в формат RKLLM для развертывания на аппаратуре Rockchip NPU:

```bash
# Включите конвертацию RKLLM во время обучения
python main.py \
  model.rkllm.enabled=true \
  model.rkllm.target_platform=rk3588 \
  model.rkllm.quantization=w8a8

# Доступные целевые платформы
model.rkllm.target_platform=rk3588    # RK3588 (рекомендуется)
model.rkllm.target_platform=rk3576    # RK3576

# Варианты квантования
model.rkllm.quantization=w8a8         # 8-битные веса, 8-битная активация (рекомендуется)
model.rkllm.quantization=w4a16        # 4-битные веса, 16-битная активация
model.rkllm.quantization=w4a16_g128   # 4-битные веса с группировкой

# Продвинутые настройки RKLLM
python main.py \
  model.rkllm.enabled=true \
  model.rkllm.target_platform=rk3588 \
  model.rkllm.quantization=w8a8 \
  model.rkllm.do_parallelize=true \
  model.rkllm.hybrid_quantization=true \
  model.rkllm.num_npu_core=3
```

**Местоположение вывода RKLLM**: `models/rkllm_models/`

**Поддерживаемые возможности**:
- Несколько платформ Rockchip (RK3588, RK3576)
- Различные стратегии квантования для оптимальных соотношений производительность/размер
- Распараллеливание модели для больших моделей
- Гибридное квантование для улучшенной точности
- Использование нескольких ядер NPU (1-3 ядра)

### Сервинг модели

```bash
# После обучения, запуск с Ollama
docker-compose up ollama

# Тестирование модели
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "custom-model", "prompt": "Привет, как дела?"}'
```

## 🏗️ Структура проекта

```
├── main.py                     # Основная точка входа
├── conf/                       # Файлы конфигурации
│   └── config.yaml            # Главная конфигурация Hydra
├── training_model/            # Логика обучения
│   ├── __main__.py            # Точка входа обучения
│   ├── one_file_train.py      # Основная реализация обучения
│   ├── grpo_train.py          # Метод обучения GRPO
│   ├── dpo_train.py           # Метод обучения DPO
│   ├── data_preparation.py    # Предобработка данных
│   ├── auth_utils.py          # Утилиты аутентификации
│   ├── logging_utils.py       # Конфигурация логирования
│   ├── optuna.py              # Оптимизация гиперпараметров
│   ├── exceptions.py          # Пользовательские исключения
│   └── types.py               # Определения типов
├── testing_model/             # Оценка модели
│   ├── __main__.py            # Точка входа тестирования
│   └── models.py              # Пользовательские реализации моделей
├── evaluation/                # Фреймворки оценки
│   ├── deepeval_integration.py # Интеграция фреймворка DeepEval
│   ├── model_evaluation.py     # Основные функции оценки
│   └── game_evaluation.py      # Оценка для игр
├── tests/                     # Набор тестов
│   ├── unit/                  # Модульные тесты
│   ├── integration/           # Интеграционные тесты
│   └── test_requirements.txt  # Зависимости тестов
├── data/                      # Наборы данных для обучения (отслеживается DVC)
├── models/                    # Выходные модели и чекпоинты
├── docs/                      # Документация
├── llama.cpp/                 # Интеграция llama.cpp (локальная)
├── pyproject.toml             # Зависимости Poetry
├── docker-compose.yaml        # Docker сервисы
├── Dockerfile                 # Основной контейнер обучения
├── run_pipeline.sh            # Скрипт pipeline обучения
└── CLAUDE.md                  # Инструкции проекта
```

## 📊 Производительность

### Сравнение использования памяти

| Метод | Размер модели | GPU память | Время обучения |
|-------|---------------|------------|----------------|
| Полное дообучение | 7B | 28GB | 6ч |
| LoRa + 4-bit | 7B | 8GB | 4ч |
| LoRa + 8-bit | 7B | 12GB | 5ч |

Проверить, хватит ли видеопамяти для обучения, можно по этой ссылке: https://huggingface.co/spaces/Vokturz/can-it-run-llm

### Поддерживаемые модели

- **Microsoft DialoGPT** (Small, Medium, Large)
- **Meta LLaMA/LLaMA 2** (7B, 13B, 70B)
- **Mistral** (7B, Mixtral)
- **Любая модель HuggingFace transformer**

## 🔧 Дополнительная информация

Папка с конфигом Hydra синхронизируется с Docker-контейнером, поэтому если вы хотите изменить гипер-параметры, то можете сделать это в папке `conf` и не пересобирать контейнер. Также синхронизируются данные в папке `data` и модели после обучения с папкой `models`.

По умолчанию веса полученной модели будут располагаться в директории `models`. Её можно заменить, для этого нужно в конфиге поменять `final_weights_path` и `output_dir`. Две директории необходимы для возможности указать отдельно директорию для квантизованной модели и для весов модели.

## 🐛 Решение проблем

### Распространенные проблемы

**CUDA Out of Memory**
```bash
# Уменьшите размер батча или включите накопление градиентов
python main.py training.per_device_train_batch_size=1 training.gradient_accumulation_steps=8

# Включите градиентные чекпоинты для дополнительной экономии памяти
python main.py training.gradient_checkpointing=true
```

**Проблемы с Docker GPU**
```bash
# 1. Проверьте установку драйвера NVIDIA
nvidia-smi

# 2. Проверьте NVIDIA Container Toolkit
nvidia-container-cli info

# 3. Проверьте доступ к GPU в Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. Если GPU недоступна, перезапустите демон Docker
sudo systemctl restart docker

# 5. Проверьте конфигурацию GPU в Docker Compose
docker-compose config
```

**Проблемы со сборкой Docker**
```bash
# Очистите кэш Docker и пересоберите
docker system prune -a
docker-compose build --no-cache

# Проверьте свободное место на диске (требуется 70GB+)
df -h

# Отслеживайте прогресс сборки с подробным выводом
docker-compose build --progress=plain
```

**Проблемы с интеграцией llama.cpp**
```bash
# Проверьте символические ссылки llama.cpp в контейнере
docker-compose exec llm_training ls -la /app/llama.cpp
docker-compose exec llm_training ls -la /app/llama.cpp/llama-quantize.exe

# Проверьте исполняемый файл квантования
docker-compose exec llm_training file /llama.cpp/build/bin/llama-quantize
```

**Ошибки загрузки/конвертации модели**
```bash
# Проверьте доступное дисковое пространство для загрузки и конвертации моделей
du -sh models/
df -h .

# Проверьте аутентификацию HuggingFace (для приватных моделей)
docker-compose exec llm_training python -c "from huggingface_hub import whoami; print(whoami())"

# Проверьте логи конвертации GGUF
docker-compose logs llm_training | grep -i gguf

# Проверьте конвертацию RKLLM (если включена)
docker-compose logs llm_training | grep -i rkllm
```

**Проблемы с конфигурацией**
```bash
# Проверьте конфигурацию Hydra
python main.py --config-path=conf --config-name=config --help

# Проверьте синтаксис переопределения конфигурации
python main.py model.train_steps=10 --dry-run

# Отладка загрузки данных
python main.py training.logging_steps=1 training.eval_steps=5
```

**Проблемы с сервисами контейнера**
```bash
# Проверьте статус всех сервисов
docker-compose ps

# Посмотрите логи конкретного сервиса
docker-compose logs llm_training
docker-compose logs ollama
docker-compose logs mlflow

# Перезапустите конкретный сервис
docker-compose restart llm_training

# Доступ к shell контейнера для отладки
docker-compose exec llm_training bash
```

**Проблемы с производительностью**
```bash
# Отслеживайте использование GPU во время обучения
nvidia-smi -l 1

# Проверьте использование ресурсов контейнера
docker stats

# Оптимизируйте под доступную память GPU
python main.py training.per_device_train_batch_size=1 training.gradient_accumulation_steps=16 training.fp16=true
```

### Получение помощи

1. **Сначала проверьте логи**: `docker-compose logs llm_training`
2. **Проверьте системные требования**: GPU NVIDIA, 70GB+ дискового пространства, CUDA toolkit
3. **Обновите драйверы**: Убедитесь, что установлены последние драйверы NVIDIA
4. **GitHub Issues**: Сообщайте о багах в [GitHub Issues](https://github.com/timik232/LLM_LoRa/issues)
5. **Включите диагностику**: информация о GPU (`nvidia-smi`), версия Docker, логи ошибок

Больше проблем можно найти на странице [Issues](https://github.com/timik232/LLM_LoRa/issues).

## 📄 Лицензия

Этот проект лицензирован под лицензией MIT - см. файл [LICENSE](../LICENSE) для деталей.

## 🙏 Благодарности

- **HuggingFace Transformers** за реализации трансформеров
- **Microsoft PEFT** за реализацию LoRa
- **bitsandbytes** за поддержку квантования
- **TRL** за методы обучения с подкреплением
- Сообщество разработчиков ML с открытым исходным кодом

---

**Поставьте звезду ⭐ этому репозиторию, если он вам помог!**

[Сообщить об ошибке](https://github.com/timik232/LLM_LoRa/issues) • [Запросить функцию](https://github.com/timik232/LLM_LoRa/issues) • [Обсуждения](https://github.com/timik232/LLM_LoRa/discussions)
