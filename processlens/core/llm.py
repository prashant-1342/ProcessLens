import json
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel
from processlens.core.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        settings: Optional[Settings] = None,
        mock_response: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.default_llm_provider).lower()
        self.mock_response = mock_response
        self._groq_client = None
        self._openai_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                api_key = self.settings.groq_api_key
                if not api_key:
                    raise ValueError("GROQ_API_KEY is not set")
                self._groq_client = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq package is required for Groq provider")
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
                api_key = self.settings.openai_api_key
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is not set")
                self._openai_client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")
        return self._openai_client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        if self.mock_response is not None:
            return self.mock_response

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.provider == "groq":
            client = self._get_groq_client()
            chosen_model = model or self.settings.groq_model
            response_format = {"type": "json_object"} if json_mode else None
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "openai":
            client = self._get_openai_client()
            chosen_model = model or self.settings.openai_model
            response_format = {"type": "json_object"} if json_mode else None
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            return response.choices[0].message.content or ""

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        if self.mock_response is not None:
            data = json.loads(self.mock_response)
            return response_model.model_validate(data)

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            (system_prompt + "\n\n" if system_prompt else "")
            + f"Output valid JSON conforming to this JSON schema:\n{schema_json}\nReturn ONLY the JSON object, without markdown formatting or commentary."
        )

        raw = self.generate(
            prompt=prompt,
            system_prompt=augmented_system,
            json_mode=True,
            model=model,
            temperature=temperature,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed_json = json.loads(cleaned)
        return response_model.model_validate(parsed_json)


def get_llm_client(provider: Optional[str] = None, mock_response: Optional[str] = None) -> LLMClient:
    return LLMClient(provider=provider, mock_response=mock_response)
