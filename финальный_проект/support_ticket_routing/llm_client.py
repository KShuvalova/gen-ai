from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def get_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")


def get_auth_token() -> str | None:
    token = os.getenv("LLM_AUTH_TOKEN")
    if not token or token == "your_token_here":
        return None
    return token


def get_model_name() -> str:
    return os.getenv("LLM_MODEL", "deepseek-v4-flash")


def llm_config_available() -> bool:
    return get_auth_token() is not None


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:500]}")

    return json.loads(match.group(0))


def chat_json(prompt: str) -> dict[str, Any]:
    token = get_auth_token()
    if token is None:
        raise RuntimeError("LLM_AUTH_TOKEN is not set in .env")

    url = f"{get_base_url()}/chat/completions"

    payload = {
        "model": get_model_name(),
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Return valid JSON only. Do not use markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {error.code}: {error_body}") from error

    response_json = json.loads(raw)
    content = response_json["choices"][0]["message"]["content"]

    return extract_json(content)
