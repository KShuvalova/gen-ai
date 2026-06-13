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


def get_cost_rates() -> tuple[float, float]:
    input_rate = float(os.getenv("LLM_INPUT_COST_PER_1M_USD", "0.0"))
    output_rate = float(os.getenv("LLM_OUTPUT_COST_PER_1M_USD", "0.0"))
    return input_rate, output_rate


def cost_pricing_configured() -> bool:
    input_rate, output_rate = get_cost_rates()
    return input_rate > 0 or output_rate > 0


def empty_usage() -> dict[str, float | int | bool]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_pricing_configured": cost_pricing_configured(),
        "llm_attempts": 0,
    }


def normalize_usage(raw_usage: dict[str, Any] | None) -> dict[str, int]:
    raw_usage = raw_usage or {}

    prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(raw_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(left: dict[str, float | int | bool], right: dict[str, int]) -> dict[str, float | int | bool]:
    left["prompt_tokens"] = int(left["prompt_tokens"]) + int(right["prompt_tokens"])
    left["completion_tokens"] = int(left["completion_tokens"]) + int(right["completion_tokens"])
    left["total_tokens"] = int(left["total_tokens"]) + int(right["total_tokens"])
    left["llm_attempts"] = int(left["llm_attempts"]) + 1
    left["estimated_cost_usd"] = estimate_cost_usd(
        int(left["prompt_tokens"]),
        int(left["completion_tokens"]),
    )
    left["cost_pricing_configured"] = cost_pricing_configured()
    return left


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    input_rate, output_rate = get_cost_rates()

    cost = (prompt_tokens / 1_000_000) * input_rate
    cost += (completion_tokens / 1_000_000) * output_rate

    return round(cost, 8)


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


def _raw_chat_json(prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
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
    usage = normalize_usage(response_json.get("usage"))

    return extract_json(content), usage


def chat_json_with_usage(
    prompt: str,
    response_model: type[ResponseModelT] | None = None,
    max_retries: int = 3,
) -> tuple[dict[str, Any] | ResponseModelT, dict[str, float | int | bool]]:
    """
    Calls an OpenAI-compatible chat completion endpoint and returns validated JSON with token/cost usage.

    If response_model is provided, the JSON is validated through a Pydantic model.
    If validation fails, the request is retried with a correction instruction.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    current_prompt = prompt
    last_error: Exception | None = None
    cumulative_usage = empty_usage()

    for attempt in range(1, max_retries + 1):
        try:
            data, usage = _raw_chat_json(current_prompt)
            cumulative_usage = add_usage(cumulative_usage, usage)

            if response_model is None:
                return data, cumulative_usage

            return response_model.model_validate(data), cumulative_usage

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


def chat_json(
    prompt: str,
    response_model: type[ResponseModelT] | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | ResponseModelT:
    result, _usage = chat_json_with_usage(
        prompt=prompt,
        response_model=response_model,
        max_retries=max_retries,
    )
    return result
