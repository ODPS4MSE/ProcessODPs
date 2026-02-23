from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

import requests


class LLMClient(ABC):
    @abstractmethod
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        text = response.json().get("response", "{}")
        return _safe_json_loads(text)


class HuggingFaceTextGenClient(LLMClient):
    """Inference API client for open-source models hosted on Hugging Face."""

    def __init__(self, model_id: str, hf_token: str) -> None:
        self.model_id = model_id
        self.hf_token = hf_token

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 700, "temperature": 0.2, "return_full_text": False},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "{}")
        else:
            text = str(data)
        return _safe_json_loads(text)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
