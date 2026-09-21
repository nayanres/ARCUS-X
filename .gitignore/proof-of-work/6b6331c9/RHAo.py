"""Provider/model-agnostic API capability negotiation for ARCUS-X.

This module makes the API execution layer resilient to differences in how
providers name and support their request parameters. It replaces the old
"token-parameter fallback" logic with a *generalized capability negotiation*
system that infers, per ``(provider, model)`` combination, which parameters and
values are accepted.

Design goals (see task spec):

1. **No hardcoded model-name mapping.** Whether a model needs
   ``max_completion_tokens`` (vs ``max_tokens`` vs ``max_output_tokens``), or
   whether it supports ``temperature`` / ``top_p`` / ``frequency_penalty`` /
   ``presence_penalty`` / reasoning / JSON mode, is discovered at runtime from
   provider responses -- never looked up in a table keyed by model name. The
   same model may accept ``max_tokens`` through one provider (OpenRouter) but
   reject it through another (Azure), so every decision is made per
   ``(provider, model)`` combination, dynamically.

2. **General capability negotiation.** The negotiation engine understands a
   *category* of capability (token parameter, temperature, top_p, penalties,
   reasoning, json mode, and an open-ended "future" bucket) rather than a single
   hardcoded parameter. New capabilities can be added by extending the
   capability schema without rewriting the execution layer.

3. **Parameter value fallback.** Beyond unsupported *parameter names*, the
   engine also recovers from unsupported *parameter values*. Example: a provider
   rejects ``temperature=0.0`` with ``unsupported_value``. The engine then
   retries by *omitting* the parameter entirely (preserving the provider's
   default) -- preferred over forcing a value the provider demands.

4. **Separate retryability from recoverability.** ``retryable`` means "the
   identical request may succeed later" (429, timeouts, transient failures).
   ``recoverable`` means "the request can succeed after modifying parameters"
   (unsupported parameter / unsupported value). Authentication is neither.

5. **Capability cache.** Once a ``(provider, model)`` combination is observed,
   its full capability record is cached for the remainder of the process so
   subsequent requests skip failed negotiation attempts.

6. **Adaptation logging.** Every automatic compatibility change is recorded with
   structured metadata (provider, model, adjustment, reason) so it can be
   surfaced in the run manifest, the raw stream, and the post-run analyzer
   output. Nothing is silently changed.
"""

from __future__ import annotations

import logging
import threading
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
# Capability schema
# ---------------------------------------------------------------------------
# The set of capability *categories* the negotiation engine understands. Each
# category maps to one or more request parameters. Adding a new capability is a
# matter of extending this schema and the negotiation engine -- not the
# execution layer.
@dataclass
class Capability:
    """A single negotiated capability for a provider/model.

    Attributes:
        name: Stable capability key (e.g. ``"token_parameter"``,
            ``"temperature"``, ``"top_p"``, ``"frequency_penalty"``,
            ``"presence_penalty"``, ``"reasoning"``, ``"json_mode"``).
        supported: Whether the provider/model accepts this capability at all.
        param: The concrete request parameter name to use (e.g.
            ``"max_completion_tokens"``). ``None`` when unsupported.
        supports_custom_values: Whether a non-default value is accepted. When
            ``False`` the engine omits the parameter to preserve the provider's
            default rather than forcing a value.
        default_value: The provider's default value, if known.
    """

    name: str
    supported: bool = True
    param: Optional[str] = None
    supports_custom_values: bool = True
    default_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "supported": self.supported,
            "param": self.param,
            "supports_custom_values": self.supports_custom_values,
            "default_value": self.default_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capability":
        return cls(
            name=data["name"],
            supported=data.get("supported", True),
            param=data.get("param"),
            supports_custom_values=data.get("supports_custom_values", True),
            default_value=data.get("default_value"),
        )


# Capability category keys (stable identifiers used across the system).
CAP_TOKEN_PARAMETER = "token_parameter"
CAP_TEMPERATURE = "temperature"
CAP_TOP_P = "top_p"
CAP_FREQUENCY_PENALTY = "frequency_penalty"
CAP_PRESENCE_PENALTY = "presence_penalty"
CAP_REASONING = "reasoning"
CAP_JSON_MODE = "json_mode"


# ---------------------------------------------------------------------------
# Structured API error classification
# ---------------------------------------------------------------------------
class APIErrorCategory(str, Enum):
    """Structured categories for provider API failures.

    Each category preserves enough metadata (provider, model, HTTP status,
    provider error code, raw message, retryable/recoverable flags) to support
    debugging and downstream reporting without collapsing everything into a
    single ``[API_ERROR]`` bucket.
    """

    UNSUPPORTED_PARAMETER = "UNSUPPORTED_PARAMETER"
    UNSUPPORTED_VALUE = "UNSUPPORTED_VALUE"
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
        retryable: Whether the *identical* request may succeed later
            (429, timeouts, transient failures).
        recoverable: Whether the request can succeed after *modifying*
            parameters (unsupported parameter / unsupported value).
        offending_param: The request parameter implicated by the error, if the
            engine could infer it (e.g. ``"temperature"``).
    """

    category: APIErrorCategory
    provider: Optional[str]
    model: Optional[str]
    http_status: Optional[int]
    provider_code: Optional[str]
    message: str
    retryable: bool
    recoverable: bool = False
    offending_param: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "provider": self.provider,
            "model": self.model,
            "http_status": self.http_status,
            "provider_code": self.provider_code,
            "message": self.message,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "offending_param": self.offending_param,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"[{self.category.value}] provider={self.provider} model={self.model} "
            f"status={self.http_status} code={self.provider_code} "
            f"retryable={self.retryable} recoverable={self.recoverable} "
            f"param={self.offending_param}: {self.message}"
        )


# ---------------------------------------------------------------------------
# Capability cache
# ---------------------------------------------------------------------------
@dataclass
class CapabilityRecord:
    """A discovered capability set for a ``(provider, model)`` combination."""

    provider: str
    model: str
    capabilities: Dict[str, Capability] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "capabilities": {
                name: cap.to_dict() for name, cap in self.capabilities.items()
            },
        }


class CapabilityCache:
    """Process-wide cache of discovered provider/model capabilities.

    The cache is keyed by ``(provider, model)``. Once a capability set is
    discovered (either because the default worked, or because a fallback was
    required), it is stored so future requests skip the failed attempt.

    Thread-safe so it can be shared across concurrent probe evaluations.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[tuple, CapabilityRecord] = {}

    def _key(self, provider: str, model: str) -> tuple:
        return (provider or "unknown", model or "unknown")

    def get(self, provider: str, model: str) -> Optional[CapabilityRecord]:
        with self._lock:
            return self._records.get(self._key(provider, model))

    def get_capability(self, provider: str, model: str, name: str) -> Optional[Capability]:
        with self._lock:
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
