"""Model implementations for testing and evaluation framework"""

import asyncio
import time

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from mistralai import Mistral


class CustomLocalModel(DeepEvalBaseLLM):
    """
    A custom local model implementation for DeepEval testing.

    Attributes:
        model (ChatOpenAI): The underlying language model.
        model_name (str): Name of the model.
    """

    def __init__(
        self,
        model: str = "vikhr-yandexgpt-5-lite-8b-it_gguf",
        url: str = "http://localhost:1234/v1/",
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Initialize the custom local model.

        Args:
            model (str, optional): Name of the model.
                Defaults to "vikhr-yandexgpt-5-lite-8b-it_gguf".
            url (str, optional): Base URL for the model.
                Defaults to "http://localhost:1234/v1/".
        """
        self.model = ChatOpenAI(
            base_url=url,
            api_key="dummy",
            model=model,
        )
        self.model_name = model

    def load_model(self) -> ChatOpenAI:
        """
        Load and return the model.

        Returns:
            ChatOpenAI: The loaded language model.
        """
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated model response.
        """
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response for the given prompt.

        Args:
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated model response.
        """
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: Model name.
        """
        return self.model_name


class CustomMistralModel(DeepEvalBaseLLM):
    """
    A custom Mistral model implementation for DeepEval testing with rate limiting.

    Attributes:
        client (Mistral): Mistral API client.
        model_name (str): Name of the model.
        temperature (float): Sampling temperature.
        last_request_time (Optional[float]): Timestamp of last API request.
        rate_limit_delay (float): Minimum delay between requests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Initialize the custom Mistral model.

        Args:
            api_key (str): API key for Mistral service.
            model (str, optional): Name of the model. Defaults to "mistral-small-latest".
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
        """
        self.client = Mistral(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.last_request_time: float | None = None
        self.rate_limit_delay = 1.2  # 1.2 seconds to stay safely under limit

    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting by introducing a delay between API requests.
        Ensures at least 1 second between requests.
        """
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    async def _aenforce_rate_limit(self) -> None:
        """
        Asynchronous version of rate limiting.
        Ensures at least 1 second between API requests.
        """
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated model response.
        """
        self._enforce_rate_limit()
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response for the given prompt.

        Args:
            prompt (str): Input prompt for the model.

        Returns:
            str: Generated model response.
        """
        await self._aenforce_rate_limit()
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: Model name.
        """
        return self.model_name

    def load_model(self) -> Mistral:
        """
        Load and return the Mistral client.

        Returns:
            Mistral: The Mistral API client.
        """
        return self.client
