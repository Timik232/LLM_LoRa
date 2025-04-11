# LLM Training
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Timik232/LLM_LoRa?utm_source=oss&utm_medium=github&utm_campaign=Timik232%2FLLM_LoRa&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
# EN
## Introduction
This repository can be used to train a Language model (LLM) in **json** or **csv** format.
The training project uses Peft to train LoRa models to save memory. It is also used
bitsandbytes, which allows you to quantize weights and, again, use less memory to train the model.
There are two methods of teaching: SFT (usually additional training with a teacher) and GRPO (the method that
was used in DeepSeek).
However, it is worth considering that even using these methods, model training is still
it may require a significant amount of video memory. To check if there is enough video memory for training,
you can check this link: https://huggingface.co/spaces/Vokturz/can-it-run-llm
## Launch
Training settings. The conf directory contains the file `config.yaml`, which allows you to configure
the training parameters of the model. \
<u>**Important!**</u> The image can take up to 70 GB of memory, this is due to the fact that the libraries for working with
CUDA are quite heavy, as is the downloaded model. Make sure that there is enough space on your disk.
In addition, if the container does not start, then you will need to download
[Nvidia Toolkit](https://developer.nvidia.com/cuda-toolkit). It is necessary for
the Docker to work with the graphics card.
### Launch via Docker
Training can be started via Docker. To do this, configure the training parameters in the file, and then
run the following commands:
```bash
docker-compose build
```
And then:
```bash
docker-compose up
```
Then wait for the model to be trained. /
By default, the weights of the resulting model will be located in the `models` directory. It can be replaced,
to do this, change the `final_weights_path` and `output_dir` in the config. Two directories are required to
be able to specify a separate directory for the quantized model and for the weights of the model.
### Launch locally
I recommend creating a separate virtual environment for Python to train the model. After that
, you need to install poetry with the command: `pip install poetry`. Then run the command:
``bash
poetry install --no-root
``
Then you can run `main.py `and wait for the result.

### Additional information
The folder with the hydra config is synchronized with the docker container, so if you want to change
the hyper-parameters, you can do it in the `conf` folder and not rebuild the container.
The data in the `data` folder and the models after training are also synchronized with the `models` folder.

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
параметры обучения модели. \
<u>**Важно!**</u> Образ может занимать до 70ГБ памяти, это связано с тем, что библиотеки для работы с
CUDA достаточно тяжёлые, как и скачиваемая модель. Убедитесь, что на вашем диске достаточно места.
Кроме того, если контейнер не запускается, то вам потребуется скачать
[Nvidia Toolkit](https://developer.nvidia.com/cuda-toolkit). Он необходим для
работы Докера с видеокартой.
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

### Дополнительная информация
Папка с конфигом гидры синхронизируется с докер-контейнером, поэтому если вы хотите изменить
гипер-параметры, то вы можете сделать это в папке `conf` и не пересобирать контейнер.
Также синхронизируются данные в папке `data` и модели после обучения с папкой `models`.
