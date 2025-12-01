import os
import requests
from typing import List, Dict, Optional


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_SITE_URL")
        if referer:
            self.headers["HTTP-Referer"] = referer
        title = os.environ.get("OPENROUTER_SITE_NAME")
        if title:
            self.headers["X-Title"] = title

    def chat_completion(self, model: str, messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.0) -> Dict:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]
