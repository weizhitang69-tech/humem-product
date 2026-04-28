from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(slots=True)
class OpenAIEmbeddingProvider:
    """Small standard-library OpenAI embeddings client.

    This deliberately avoids adding the OpenAI SDK as a dependency. Set
    ``OPENAI_API_KEY`` in the environment or pass ``api_key`` directly.
    """

    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
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
            raise RuntimeError(f"OpenAI embeddings request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc.reason}") from exc

        data = sorted(response_payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def make_embedding_provider(
    provider: str | EmbeddingProvider | None,
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider | None:
    if provider is None:
        return None
    if isinstance(provider, str):
        if provider.lower() != "openai":
            raise ValueError(f"unsupported embedding provider: {provider}")
        return OpenAIEmbeddingProvider(model=model, api_key=api_key, dimensions=dimensions)
    return provider


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
