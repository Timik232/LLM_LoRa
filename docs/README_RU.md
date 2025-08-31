# 🚀 LLM-LoRa (Русская версия)

**Эффективное по памяти дообучение больших языковых моделей с использованием LoRa и продвинутого квантования**

[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
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
git clone https://github.com/your-username/LLM_LoRa.git
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

Если контейнер не запускается, то вам потребуется скачать [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). Он необходим для работы Docker с видеокартой.

```bash
# Соберите и запустите
docker-compose build
docker-compose up
```

### Вариант 2: Локальная установка

Рекомендую создать отдельно виртуальное окружение для Python для обучения модели.

```bash
# Клонируйте репозиторий
git clone https://github.com/your-username/LLM_LoRa.git
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
│   ├── config.yaml            # Главная конфигурация
│   ├── model/                 # Конфигурации моделей
│   └── training/              # Конфигурации обучения
├── training_model/            # Логика обучения
│   ├── one_file_train.py     # Основное обучение
│   ├── grpo_train.py         # Реализация GRPO
│   └── data_preparation.py   # Предобработка данных
├── testing_model/            # Логика оценки
│   ├── test.py              # Реализации тестов
│   └── deepeval_func.py     # Интеграция DeepEval
├── data/                     # Наборы данных для обучения
├── models/                   # Выходные модели
├── docs/                     # Документация
└── docker-compose.yaml      # Конфигурация Docker
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
python main.py training.batch_size=2 training.gradient_accumulation_steps=8
```

**Проблемы с Docker GPU**
```bash
# Проверьте NVIDIA container toolkit
nvidia-container-cli info
```

**Ошибки загрузки модели**
- Убедитесь в достаточном дисковом пространстве (70GB+ для полной установки)
- Проверьте токен HuggingFace для приватных моделей
- Проверьте совместимость с CUDA

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

[Сообщить об ошибке](https://github.com/your-username/LLM_LoRa/issues) • [Запросить функцию](https://github.com/your-username/LLM_LoRa/issues) • [Обсуждения](https://github.com/your-username/LLM_LoRa/discussions)
