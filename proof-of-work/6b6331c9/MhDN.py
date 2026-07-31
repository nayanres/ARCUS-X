"""Provider/model-agnostic API compatibility layer for ARCUS-X.

This module makes the API execution layer resilient to differences in how
providers name their request parameters (e.g. ``max_tokens`` vs
``max_completion_tokens`` vs ``max_output_tokens``) and to the *categories* of
errors providers return.

Design goals (see task spec):

1. **No hardcoded model-name mapping.** Whether a model needs
   ``max_completion_tokens`` is discovered at runtime, not looked up in a table
   keyed by model name. The same model may accept ``max_tokens`` through one
   provider (OpenRouter) but reject it through another (Azure), so the decision
   is made per ``(provider, model)`` combination, dynamically.

2. **Dynamic fallback chain.** We attempt the default token parameter first and,
   on an ``UNSUPPORTED_PARAMETER`` error, automatically retry with the next
   compatible alternative. The chain is data-driven so new parameters
   (``max_output_tokens`` or provider-specific ones) can be added without
   rewriting the execution layer.

3. **Structured error classification.** Failures are categorised into
   ``UNSUPPORTED_PARAMETER``, ``AUTHENTICATION_ERROR``, ``RATE_LIMIT``,
   ``DEPLOYMENT_ERROR``, ``CONTEXT_LIMIT``, ``TIMEOUT``, ``INVALID_REQUEST`` and
   ``UNKNOWN_PROVIDER_ERROR``. Each carries provider/model/HTTP status/provider
   error code/raw message/retryable status so nothing is collapsed into a single
   opaque ``[API_ERROR]``.

4. **Capability cache.** Once a ``(provider, model)`` combination is observed to
   require a specific token parameter, that fact is cached for the remainder of
   the process so subsequent requests skip the failed attempt.

5. **Adaptation logging.** Every compatibility change is recorded with
   structured metadata (provider, model, adjustment) so it can be surfaced in the
   run manifest, the raw stream, and the post-run analyzer output.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-parameter fallback chain
# ---------------------------------------------------------------------------
# Ordered list of parameter names to try for the *maximum completion length*.
# The first entry is the universal default; subsequent entries are fallbacks
# used when a provider rejects the previous one as unsupported.
#
# To add future compatibility (e.g. ``max_output_tokens`` or a provider-specific
# parameter) simply append to this list -- no execution-layer rewrite required.
DEFAULT_TOKEN_PARAM = "max_tokens"
TOKEN_PARAM_FALLBACK_CHAIN: List[str] = [
    "max_tokens",
    "max_completion_tokens",
    # Reserved for future providers/models:
    # "max_output_tokens",
]


# ---------------------------------------------------------------------------
# Structured API error classification
# ---------------------------------------------------------------------------
class APIErrorCategory(str, Enum):
    """Structured categories for provider API failures.

    Each category preserves enough metadata (provider, model, HTTP status,
    provider error code, raw message, retryable flag) to support debugging and
    downstream reporting without collapsing everything into a single
    ``[API_ERROR]`` bucket.
    """

    UNSUPPORTED_PARAMETER = "UNSUPPORTED_PARAMETER"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    DEPLOYMENT_ERROR = "DEPLOYMENT_ERROR"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"


@dataclass
class APIError:
    """A structured, classified provider API error.

    Attributes:
        category: One of :class:`APIErrorCategory`.
        provider: Provider slug (e.g. ``"azure"``, ``"openrouter"``).
        model: Model identifier used in the request.
        http_status: HTTP status code (or ``None`` for non-HTTP failures).
        provider_code: Provider-specific error code/type if available.
        message: Raw error message from the provider.
        retryable: Whether the caller may safely retry the request.
    """

    category: APIErrorCategory
    provider: Optional[str]
    model: Optional[str]
    http_status: Optional[int]
    provider_code: Optional[str]
    message: str
    retryable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "provider": self.provider,
            "model": self.model,
            "http_status": self.http_status,
            "provider_code": self.provider_code,
            "message": self.message,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"[{self.category.value}] provider={self.provider} model={self.model} "
            f"status={self.http_status} code={self.provider_code} "
            f"retryable={self.retryable}: {self.message}"
        )


# ---------------------------------------------------------------------------
# Capability cache
# ---------------------------------------------------------------------------
@dataclass
class CapabilityRecord:
    """A discovered capability for a ``(provider, model)`` combination."""

    provider: str
    model: str
    token_parameter: str


class CapabilityCache:
    """Process-wide cache of discovered provider/model capabilities.

    The cache is keyed by ``(provider, model)``. Once a token parameter is
    discovered (either because the default worked, or because a fallback was
    required), it is stored so future requests skip the failed attempt.

    Thread-safe so it can be shared across concurrent probe evaluations.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token_params: Dict[tuple, str] = {}

    def _key(self, provider: str, model: str) -> tuple:
        return (provider or "unknown", model or "unknown")

    def get_token_parameter(self, provider: str, model: str) -> Optional[str]:
        with self._lock:
            return self._token_params.get(self._key(provider, model))

    def set_token_parameter(self, provider: str, model: str, param: str) -> None:
        with self._lock:
            self._token_params[self._key(provider, model)] = param

    def has(self, provider: str, model: str) -> bool:
        with self._lock:
            return self._key(provider, model) in self._token_params

    def snapshot(self) -> List[Dict[str, str]]:
        """Return all cached capabilities as a JSON-serialisable list."""
        with self._lock:
            return [
                {"provider": p, "model": m, "token_parameter": param}
                for (p, m), param in self._token_params.items()
            ]


# A single shared cache for the process. The evaluator seeds it and the runner
# reads it to populate the manifest / raw stream.
SHARED_CAPABILITY_CACHE = CapabilityCache()


# ---------------------------------------------------------------------------
# Adaptation logging
# ---------------------------------------------------------------------------
@dataclass
class CompatibilityAdaptation:
    """A recorded API compatibility adjustment for a provider/model."""

    provider: str
    model: str
    adjustment: str  # e.g. "max_tokens -> max_completion_tokens"
    reason: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "adjustment": self.adjustment,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return (
            f"API Compatibility:\n"
            f"- Provider: {self.provider}\n"
            f"- Model: {self.model}\n"
            f"- Adjustment: {self.adjustment}"
        )


class AdaptationLog:
    """Collects compatibility adaptations made during a run.

    The runner attaches a callback so each adaptation is also written to the
    raw stream / manifest as it happens. The log itself is a simple ordered,
    thread-safe list that the runner can dump into the manifest and raw stream.
    """

    def __init__(self, on_adapt: Optional[Callable[[CompatibilityAdaptation], None]] = None) -> None:
        self._lock = threading.Lock()
        self._entries: List[CompatibilityAdaptation] = []
        self._on_adapt = on_adapt

    def record(self, adaptation: CompatibilityAdaptation) -> None:
        with self._lock:
            self._entries.append(adaptation)
        logger.info(str(adaptation))
        if self._on_adapt:
            try:
                self._on_adapt(adaptation)
            except Exception:  # pragma: no cover - defensive
                pass

    def all(self) -> List[CompatibilityAdaptation]:
        with self._lock:
            return list(self._entries)

    def to_dict(self) -> List[Dict[str, str]]:
        with self._lock:
            return [e.to_dict() for e in self._entries]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
def detect_provider(api_base: Optional[str], model_name: str) -> str:
    """Best-effort provider slug from the API base URL or model name.

    This is used purely for *logging/attribution* -- it never drives parameter
    selection. Parameter selection is always discovered dynamically.
    """
    if api_base:
        base = api_base.lower()
        if "openai.azure" in base or "azure" in base:
            return "azure"
        if "openrouter" in base:
            return "openrouter"
        if "ai.hackclub" in base:
            return "hackclub"
        if "openai.com" in base or "api.openai" in base:
            return "openai"
        if "anthropic" in base:
            return "anthropic"
        if "googleapis" in base:
            return "google"
    # Fall back to the model-name prefix convention ("provider/model").
    if "/" in model_name:
        return model_name.split("/", 1)[0].lower()
    return "unknown"


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# Substrings that, when present in a provider error message/type, indicate a
# specific category. Order matters: more specific patterns are checked first.
_UNSUPPORTED_PARAM_HINTS = (
    "unsupported parameter",
    "unknown parameter",
    "unrecognized parameter",
    "invalid parameter",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "additional_properties",
    "extra fields",
    "did not expect",
    "not allowed",
    "not supported",
    "unsupported",
)

_AUTH_HINTS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid api_key",
    "incorrect api key",
    "api key",
    "permission denied",
    "forbidden",
    "401",
    "403",
)

_RATE_LIMIT_HINTS = (
    "rate limit",
    "too many requests",
    "quota",
    "429",
)

_DEPLOYMENT_HINTS = (
    "deployment",
    "model not found",
    "the model",
    "does not exist",
    "is not available",
    "currently unavailable",
    "service unavailable",
    "bad gateway",
    "502",
    "503",
    "504",
)

_CONTEXT_LIMIT_HINTS = (
    "context length",
    "context window",
    "maximum context",
    "token limit",
    "too many tokens",
    "exceeds the maximum",
    "maximum length",
)

_TIMEOUT_HINTS = (
    "timeout",
    "timed out",
    "deadline exceeded",
    "connection aborted",
    "connection reset",
    "read timed out",
)

_INVALID_REQUEST_HINTS = (
    "invalid request",
    "bad request",
    "malformed",
    "validation error",
    "400",
)


def classify_error(
    *,
    provider: Optional[str],
    model: Optional[str],
    http_status: Optional[int],
    provider_code: Optional[str],
    message: str,
) -> APIError:
    """Classify a provider error into a structured :class:`APIError`.

    The classification uses both the HTTP status code and the textual content of
    the provider's error (type/code/message). It never collapses distinct
    failures into a single bucket: each category preserves the original
    provider/model/status/code/message and a ``retryable`` flag.
    """
    text = f"{provider_code or ''} {message or ''}".lower()

    # 1. Unsupported parameter -- the key signal for the token-param fallback.
    if http_status in (400, 422, 4000) and _contains_any(text, _UNSUPPORTED_PARAM_HINTS):
        return APIError(
            category=APIErrorCategory.UNSUPPORTED_PARAMETER,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
        )

    # 2. Authentication.
    if http_status in (401, 403) or _contains_any(text, _AUTH_HINTS):
        return APIError(
            category=APIErrorCategory.AUTHENTICATION_ERROR,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
        )

    # 3. Rate limit.
    if http_status == 429 or _contains_any(text, _RATE_LIMIT_HINTS):
        return APIError(
            category=APIErrorCategory.RATE_LIMIT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
        )

    # 4. Deployment / availability.
    if http_status in (502, 503, 504) or _contains_any(text, _DEPLOYMENT_HINTS):
        return APIError(
            category=APIErrorCategory.DEPLOYMENT_ERROR,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
        )

    # 5. Context limit.
    if _contains_any(text, _CONTEXT_LIMIT_HINTS):
        return APIError(
            category=APIErrorCategory.CONTEXT_LIMIT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
        )

    # 6. Timeout (transport-level, no HTTP status).
    if http_status is None and _contains_any(text, _TIMEOUT_HINTS):
        return APIError(
            category=APIErrorCategory.TIMEOUT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
        )

    # 7. Invalid request.
    if http_status == 400 or _contains_any(text, _INVALID_REQUEST_HINTS):
        return APIError(
            category=APIErrorCategory.INVALID_REQUEST,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
        )

    # 8. Fallback.
    return APIError(
        category=APIErrorCategory.UNKNOWN_PROVIDER_ERROR,
        provider=provider,
        model=model,
        http_status=http_status,
        provider_code=provider_code,
        message=message,
        retryable=False,
    )


def _contains_any(text: str, hints) -> bool:
    return any(hint in text for hint in hints)


def is_unsupported_parameter_error(err: APIError) -> bool:
    """Return True if the error indicates an unsupported request parameter."""
    return err.category == APIErrorCategory.UNSUPPORTED_PARAMETER


# ---------------------------------------------------------------------------
# Token-parameter fallback engine
# ---------------------------------------------------------------------------
def build_token_param_payload(
    base_payload: Dict[str, Any],
    param_name: str,
    value: Any,
) -> Dict[str, Any]:
    """Return a copy of ``base_payload`` with the token parameter set.

    Any previously-attempted token parameter key is removed so we never send
    two competing keys (e.g. both ``max_tokens`` and ``max_completion_tokens``)
    in the same request.
    """
    payload = dict(base_payload)
    for key in TOKEN_PARAM_FALLBACK_CHAIN:
        payload.pop(key, None)
    payload[param_name] = value
    return payload


def next_token_param(current: str) -> Optional[str]:
    """Return the next token parameter to try after ``current`` in the chain.

    Returns ``None`` when ``current`` is the last entry (no further fallback).
    """
    try:
        idx = TOKEN_PARAM_FALLBACK_CHAIN.index(current)
    except ValueError:
        # Not in the chain (e.g. a cached provider-specific param); fall back to
        # the first alternative after the default.
        return TOKEN_PARAM_FALLBACK_CHAIN[1] if len(TOKEN_PARAM_FALLBACK_CHAIN) > 1 else None
    if idx + 1 < len(TOKEN_PARAM_FALLBACK_CHAIN):
        return TOKEN_PARAM_FALLBACK_CHAIN[idx + 1]
    return None
