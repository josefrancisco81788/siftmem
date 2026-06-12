#!/usr/bin/env python3
"""Pluggable LLM JSON generation for optional Siftmem features."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def resolve_provider(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().lower()
    value = os.environ.get("SIFTMEM_LLM_PROVIDER", "none").strip().lower()
    return value or "none"


def resolve_model(provider: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env_model = os.environ.get("SIFTMEM_LLM_MODEL", "").strip()
    if env_model:
        return env_model
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return ""


def _parse_json_response(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _gemini_generate_json(prompt: str, user_content: str, *, model: str) -> Any | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{prompt}\n\n---\n\n{user_content}"}],
            }
        ],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    candidates = payload.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    return _parse_json_response(text)


def _openai_generate_json(prompt: str, user_content: str, *, model: str) -> Any | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception:  # noqa: BLE001
        return None

    text = (response.choices[0].message.content or "").strip()
    return _parse_json_response(text)


def generate_json(
    prompt: str,
    user_content: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Any | None:
    """Generate parsed JSON from an LLM. Returns None when provider is none or call fails."""
    resolved_provider = resolve_provider(provider)
    if resolved_provider == "none":
        return None
    resolved_model = resolve_model(resolved_provider, model)

    if resolved_provider == "gemini":
        return _gemini_generate_json(prompt, user_content, model=resolved_model)
    if resolved_provider == "openai":
        return _openai_generate_json(prompt, user_content, model=resolved_model)
    return None


def gemini_generate_json(prompt: str, user_content: str, *, model: str = DEFAULT_GEMINI_MODEL) -> Any | None:
    """Backward-compatible shim. Prefer generate_json() with explicit provider."""
    return _gemini_generate_json(prompt, user_content, model=model)


def llm_available(provider: str | None = None) -> bool:
    resolved = resolve_provider(provider)
    if resolved == "none":
        return False
    if resolved == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if resolved == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
    return False
