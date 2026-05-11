from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMProvider(Protocol):
    model: str

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
    ) -> dict[str, Any]:
        ...

    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(slots=True)
class OpenAICompatibleLLMProvider:
    """Small OpenAI-compatible chat completions client.

    The provider only assumes the common ``/chat/completions`` endpoint and a
    JSON-object response format. It deliberately avoids depending on the
    OpenAI SDK so the package remains lightweight.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 45.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("an API key is required for OpenAI-compatible LLM calls")
        if not self.model:
            raise ValueError("model is required for OpenAI-compatible LLM calls")

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
    ) -> dict[str, Any]:
        content = self._chat(
            messages,
            response_format={"type": "json_object"},
            metadata={"schema_name": schema_name},
        )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{schema_name} response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{schema_name} response must be a JSON object")
        return payload

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages, response_format=None, metadata=None).strip()

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, str] | None,
        metadata: dict[str, str] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if metadata:
            payload["metadata"] = metadata

        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        choices = response_payload.get("choices")
        if not choices:
            raise RuntimeError("LLM response did not include choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM response content was not text")
        return content


def make_llm_provider(
    provider: str | LLMProvider | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
) -> LLMProvider | None:
    if provider is None:
        return None
    if isinstance(provider, str):
        name = provider.lower()
        if name not in {"openai", "openai-compatible", "compatible"}:
            raise ValueError(f"unsupported LLM provider: {provider}")
        if not model:
            raise ValueError("llm_model is required when using an OpenAI-compatible provider")
        return OpenAICompatibleLLMProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    return provider
