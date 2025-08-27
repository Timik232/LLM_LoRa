import fire
from training_model import LLMLoRaCLI

if __name__ == "__main__":
    cli = LLMLoRaCLI()
    fire.Fire(cli)
