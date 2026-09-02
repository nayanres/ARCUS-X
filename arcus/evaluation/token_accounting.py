"""Deterministic token accounting."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def measure_tokens(
    prompt: str,
    completion: str,
    provider_response: Any = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure prompt/completion/total tokens using the canonical pipeline."""
    prompt_text = prompt or ""
    completion_text = completion or ""

    accounting_trace = []
    usage = _extract_provider_usage(provider_response)
    if usage is not None:
        prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
        completion_tokens = _coerce_int(usage.get("completion_tokens"))
        total_tokens = _coerce_int(usage.get("total_tokens"))
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "token_source": "provider_usage",
                "token_confidence": "authoritative",
                "token_metadata": {
                    "provider_usage": usage,
                    "provider_response_type": type(provider_response).__name__ if provider_response is not None else None,
                    "accounting_trace": [
                        {"method": "provider_usage", "status": "success", "reason": "provider usage statistics were present"}
                    ],
                },
            }

    accounting_trace.append({
        "method": "provider_usage",
        "status": "unavailable",
        "reason": "provider usage statistics were not available",
    })

    official_result = _try_official_tokenizer(prompt_text, completion_text, model_name)
    if official_result is not None:
        return {
            "prompt_tokens": official_result["prompt_tokens"],
            "completion_tokens": official_result["completion_tokens"],
            "total_tokens": official_result["total_tokens"],
            "token_source": "official_tokenizer",
            "token_confidence": "high",
            "token_metadata": {
                "tokenizer_name": official_result["tokenizer_name"],
                "model_name": model_name,
                "prompt_length": len(prompt_text),
                "completion_length": len(completion_text),
                "accounting_trace": [
                    *accounting_trace,
                    {"method": "official_tokenizer", "status": "success", "reason": "official tokenizer was available"},
                ],
            },
        }

    compatible_result = _try_compatible_tokenizer(prompt_text, completion_text, model_name)
    if compatible_result is not None:
        return {
            "prompt_tokens": compatible_result["prompt_tokens"],
            "completion_tokens": compatible_result["completion_tokens"],
            "total_tokens": compatible_result["total_tokens"],
            "token_source": "compatible_tokenizer",
            "token_confidence": "medium",
            "token_metadata": {
                "tokenizer_name": compatible_result["tokenizer_name"],
                "model_name": model_name,
                "prompt_length": len(prompt_text),
                "completion_length": len(completion_text),
                "accounting_trace": [
                    *accounting_trace,
                    {"method": "official_tokenizer", "status": "unavailable", "reason": "official tokenizer was not available"},
                    {"method": "compatible_tokenizer", "status": "success", "reason": "compatible tokenizer was available"},
                ],
            },
        }

    heuristic_result = _heuristic_token_count(prompt_text, completion_text)
    return {
        "prompt_tokens": heuristic_result["prompt_tokens"],
        "completion_tokens": heuristic_result["completion_tokens"],
        "total_tokens": heuristic_result["total_tokens"],
        "token_source": "heuristic",
        "token_confidence": "low",
        "token_metadata": {
            "characters": heuristic_result["characters"],
            "words": heuristic_result["words"],
            "punctuation": heuristic_result["punctuation"],
            "character_estimate": heuristic_result["character_estimate"],
            "word_estimate": heuristic_result["word_estimate"],
            "heuristic_formula": heuristic_result["heuristic_formula"],
            "model_name": model_name,
            "accounting_trace": [
                *accounting_trace,
                {"method": "official_tokenizer", "status": "unavailable", "reason": "official tokenizer was not available"},
                {"method": "compatible_tokenizer", "status": "unavailable", "reason": "compatible tokenizer was not available"},
                {"method": "heuristic", "status": "success", "reason": "falling back to deterministic heuristic"},
            ],
        },
    }


def _extract_provider_usage(provider_response: Any) -> Optional[Dict[str, Any]]:
    """Extract token usage from common provider response shapes."""
    if provider_response is None:
        return None

    if isinstance(provider_response, dict):
        usage = provider_response.get("usage")
        if isinstance(usage, dict):
            return usage
        if {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(provider_response.keys()):
            return {
                "prompt_tokens": provider_response.get("prompt_tokens"),
                "completion_tokens": provider_response.get("completion_tokens"),
                "total_tokens": provider_response.get("total_tokens"),
            }
        return None

    usage = getattr(provider_response, "usage", None)
    if isinstance(usage, dict):
        return usage
    if hasattr(provider_response, "prompt_tokens") and hasattr(provider_response, "completion_tokens"):
        return {
            "prompt_tokens": getattr(provider_response, "prompt_tokens"),
            "completion_tokens": getattr(provider_response, "completion_tokens"),
            "total_tokens": getattr(provider_response, "total_tokens", None),
        }
    return None


def _try_official_tokenizer(prompt: str, completion: str, model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Attempt official tokenizer implementations when available."""
    if not model_name:
        return None

    # OpenAI / compatible models via tiktoken.
    try:
        import tiktoken  # type: ignore
    except ImportError:
        tiktoken = None

    if tiktoken is not None:
        encoding_name = _select_tiktoken_encoding(model_name)
        if encoding_name:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                return {
                    "prompt_tokens": len(encoding.encode(prompt, disallowed_special=())),
                    "completion_tokens": len(encoding.encode(completion, disallowed_special=())),
                    "total_tokens": len(encoding.encode(prompt + completion, disallowed_special=())),
                    "tokenizer_name": f"tiktoken:{encoding_name}",
                }
            except Exception:
                pass

    return None


def _try_compatible_tokenizer(prompt: str, completion: str, model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Attempt a compatible tokenizer from transformers when available."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name or "gpt2", trust_remote_code=True)
    except Exception:
        return None

    try:
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokenizer_name": getattr(tokenizer, "name_or_path", "transformers.AutoTokenizer"),
        }
    except Exception:
        return None


def _heuristic_token_count(prompt: str, completion: str) -> Dict[str, Any]:
    """Deterministic fallback heuristic for token counting."""
    combined_text = f"{prompt}{completion}"
    characters = len(combined_text)
    words = len(re.findall(r"\S+", combined_text))
    punctuation = len(re.findall(r"[^\w\s]", combined_text))

    estimated_char_tokens = characters / 4.0
    estimated_word_tokens = words * 1.3
    estimated_tokens = 0.5 * estimated_char_tokens + 0.5 * estimated_word_tokens
    rounded = int(round(estimated_tokens))
    clamped = max(1, rounded)

    prompt_chars = len(prompt)
    prompt_words = len(re.findall(r"\S+", prompt))
    prompt_punctuation = len(re.findall(r"[^\w\s]", prompt))
    completion_chars = len(completion)
    completion_words = len(re.findall(r"\S+", completion))
    completion_punctuation = len(re.findall(r"[^\w\s]", completion))

    prompt_estimate = _estimate_text_tokens(prompt)
    completion_estimate = _estimate_text_tokens(completion)

    return {
        "prompt_tokens": prompt_estimate,
        "completion_tokens": completion_estimate,
        "total_tokens": prompt_estimate + completion_estimate,
        "characters": characters,
        "words": words,
        "punctuation": punctuation,
        "character_estimate": estimated_char_tokens,
        "word_estimate": estimated_word_tokens,
        "heuristic_formula": "0.5 * (characters / 4) + 0.5 * (words * 1.3)",
        "prompt_details": {
            "characters": prompt_chars,
            "words": prompt_words,
            "punctuation": prompt_punctuation,
        },
        "completion_details": {
            "characters": completion_chars,
            "words": completion_words,
            "punctuation": completion_punctuation,
        },
    }


def _estimate_text_tokens(text: str) -> int:
    characters = len(text)
    words = len(re.findall(r"\S+", text))
    punctuation = len(re.findall(r"[^\w\s]", text))
    estimated_char_tokens = characters / 4.0
    estimated_word_tokens = words * 1.3
    estimated_tokens = 0.5 * estimated_char_tokens + 0.5 * estimated_word_tokens
    rounded = int(round(estimated_tokens))
    return max(1, rounded)


def _select_tiktoken_encoding(model_name: str) -> Optional[str]:
    """Select a reasonable tiktoken encoding for known model families."""
    name = (model_name or "").lower()
    if "gpt-4.1" in name or "gpt-4o" in name or "o1" in name or "o3" in name:
        return "o200k_base"
    if "gpt-4" in name or "gpt-3.5" in name or "text-embedding" in name:
        return "cl100k_base"
    if "codex" in name or "davinci" in name:
        return "p50k_base"
    if "gpt-3" in name:
        return "r50k_base"
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
