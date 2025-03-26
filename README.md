# LLM Training
# EN
## Introduction
This repository can be used to train a language model (LLM) in json or csv format.
## Build

# RU
## Вступление
Этот репозиторий можно использовать для обучения модели языковой модели (LLM) в **json** или **csv** формате.
В проекте для обучения используется Peft для обучения LoRa моделей для экономии памяти. Также используется
bitsandbytes, что позволяет квантовать веса и опять же расходовать меньше памяти для обучения модели.
Обучать можно двумя методами: SFT (обычно дообучение с учителем) и GRPO (метод, который
применялся в DeepSeek).
Однако стоит учесть, что даже с использованием этих методов, обучение модели всё ещё
может потребовать значительного объёма видеопамяти. Проверить, хватит ли видеопамяти для обучения,
можно проверить по этой ссылке: https://huggingface.co/spaces/Vokturz/can-it-run-llm
## Запуск
Настройки обучения. В директории conf находится файл `config.yaml`, который позволяет настроить
параметры обучения модели. Параметры, которые нежелательно менять, помечены комментариями.
### Запуск через Docker
Обучение можно запустить через Docker. Для этого необходимо настроить параметры обучения в файле, после чего
выполнить следующие команды:
```bash
docker-compose build
```
И затем:
```bash
docker-compose up
```
После чего дождаться обучения модели. /
По умолчанию веса полученной модели будут располагаться в директории `models`. Её можно заменить,
для этого нужно в конфиге поменять `final_weights_path` и `output_dir`. Две директории необходимы для
возможности указать отдельно директорию для квантизованной модели и для весов модели.
### Запуск локально
Рекомендую создать отдельно виртуальное окружение для Python для обучения модели. После чего
нужно установить poetry командой: `pip install poetry`. После чего выполнить команду:
```bash
poetry install --no-root
```
Затем можно запустить `main.py` и дождаться результата.
