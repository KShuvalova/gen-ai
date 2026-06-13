from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


load_dotenv()

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


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


def _raw_chat_json(prompt: str) -> dict[str, Any]:
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


def chat_json(
    prompt: str,
    response_model: type[ResponseModelT] | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | ResponseModelT:
    """
    Calls an OpenAI-compatible chat completion endpoint and returns validated JSON.

    If response_model is provided, the JSON is validated through a Pydantic model.
    If validation fails, the request is retried with a correction instruction.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    current_prompt = prompt
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            data = _raw_chat_json(current_prompt)

            if response_model is None:
                return data

            return response_model.model_validate(data)

        except (ValidationError, ValueError, KeyError, RuntimeError) as error:
            last_error = error

            if attempt == max_retries:
                break

            current_prompt = (
                prompt
                + "\n\nYour previous response did not match the required JSON schema."
                + "\nReturn JSON only and fix the following validation/parsing error:"
                + f"\n{repr(error)}"
            )

            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"LLM response failed validation after {max_retries} attempts: {last_error}"
    )
