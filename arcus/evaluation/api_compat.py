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
from datetime import datetime
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
        timestamp: ISO timestamp when the error occurred.
        payload_feature: The payload feature/parameter that caused failure.
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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    payload_feature: Optional[str] = None

    def __post_init__(self) -> None:
        if self.payload_feature is None:
            self.payload_feature = self.offending_param

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
            "timestamp": self.timestamp,
            "payload_feature": self.payload_feature,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"[{self.category.value}] provider={self.provider} model={self.model} "
            f"time={self.timestamp} status={self.http_status} code={self.provider_code} "
            f"retryable={self.retryable} recoverable={self.recoverable} "
            f"feature={self.payload_feature or self.offending_param}: {self.message}"
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
            rec = self._records.get(self._key(provider, model))
            if rec is None:
                return None
            return rec.capabilities.get(name)

    def set(self, record: CapabilityRecord) -> None:
        with self._lock:
            self._records[self._key(record.provider, record.model)] = record

    def set_capability(self, provider: str, model: str, cap: Capability) -> None:
        with self._lock:
            key = self._key(provider, model)
            rec = self._records.get(key)
            if rec is None:
                rec = CapabilityRecord(provider=provider, model=model)
                self._records[key] = rec
            rec.capabilities[cap.name] = cap

    def has(self, provider: str, model: str) -> bool:
        with self._lock:
            return self._key(provider, model) in self._records

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return all cached capabilities as a JSON-serialisable list."""
        with self._lock:
            return [rec.to_dict() for rec in self._records.values()]


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
    kind: str = "parameter_name"  # "parameter_name" | "parameter_value" | "parameter_omitted"

    def to_dict(self) -> Dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "adjustment": self.adjustment,
            "reason": self.reason,
            "kind": self.kind,
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

# Substrings indicating an unsupported *value* for an otherwise-supported
# parameter (e.g. temperature=0.0 rejected, only default supported).
_UNSUPPORTED_VALUE_HINTS = (
    "unsupported value",
    "unsupported_value",
    "only the default",
    "only supports the default",
    "does not support",
    "does not support 0",
    "value is not supported",
    "invalid value",
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


def _contains_any(text: str, hints) -> bool:
    return any(hint in text for hint in hints)


# Mapping of request parameter names to the capability category they belong to.
# Used to infer which capability an unsupported-parameter / unsupported-value
# error refers to, so the engine can adapt the right one.
_PARAM_TO_CAPABILITY = {
    "max_tokens": CAP_TOKEN_PARAMETER,
    "max_completion_tokens": CAP_TOKEN_PARAMETER,
    "max_output_tokens": CAP_TOKEN_PARAMETER,
    "temperature": CAP_TEMPERATURE,
    "top_p": CAP_TOP_P,
    "frequency_penalty": CAP_FREQUENCY_PENALTY,
    "presence_penalty": CAP_PRESENCE_PENALTY,
    "reasoning_effort": CAP_REASONING,
    "include_reasoning": CAP_REASONING,
    "reasoning": CAP_REASONING,
    "response_format": CAP_JSON_MODE,
}


def _infer_offending_param(text: str) -> Optional[str]:
    """Best-effort inference of which request parameter an error refers to."""
    for param in _PARAM_TO_CAPABILITY:
        if param in text:
            return param
    return None


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
    provider/model/status/code/message plus ``retryable`` and ``recoverable``
    flags.

    ``retryable`` means the *identical* request may succeed later (transient).
    ``recoverable`` means the request can succeed after *modifying* parameters.
    """
    text = f"{provider_code or ''} {message or ''}".lower()
    offending = _infer_offending_param(text)

    # 1. Unsupported parameter name -- recoverable (retry with a different name).
    if http_status in (400, 422, 4000) and _contains_any(text, _UNSUPPORTED_PARAM_HINTS):
        # Distinguish an unsupported *value* (recoverable by omission) from an
        # unsupported *parameter name* (recoverable by renaming).
        if _contains_any(text, _UNSUPPORTED_VALUE_HINTS):
            return APIError(
                category=APIErrorCategory.UNSUPPORTED_VALUE,
                provider=provider,
                model=model,
                http_status=http_status,
                provider_code=provider_code,
                message=message,
                retryable=False,
                recoverable=True,
                offending_param=offending,
            )
        return APIError(
            category=APIErrorCategory.UNSUPPORTED_PARAMETER,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
            recoverable=True,
            offending_param=offending,
        )

    # 2. Authentication -- neither retryable nor recoverable.
    if http_status in (401, 403) or _contains_any(text, _AUTH_HINTS):
        return APIError(
            category=APIErrorCategory.AUTHENTICATION_ERROR,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
            recoverable=False,
            offending_param=offending,
        )

    # 3. Rate limit -- retryable, not recoverable.
    if http_status == 429 or _contains_any(text, _RATE_LIMIT_HINTS):
        return APIError(
            category=APIErrorCategory.RATE_LIMIT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
            recoverable=False,
            offending_param=offending,
        )

    # 4. Deployment / availability -- retryable, not recoverable.
    if http_status in (502, 503, 504) or _contains_any(text, _DEPLOYMENT_HINTS):
        return APIError(
            category=APIErrorCategory.DEPLOYMENT_ERROR,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
            recoverable=False,
            offending_param=offending,
        )

    # 5. Context limit -- neither.
    if _contains_any(text, _CONTEXT_LIMIT_HINTS):
        return APIError(
            category=APIErrorCategory.CONTEXT_LIMIT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
            recoverable=False,
            offending_param=offending,
        )

    # 6. Timeout (transport-level, no HTTP status) -- retryable, not recoverable.
    if http_status is None and _contains_any(text, _TIMEOUT_HINTS):
        return APIError(
            category=APIErrorCategory.TIMEOUT,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=True,
            recoverable=False,
            offending_param=offending,
        )

    # 7. Invalid request -- neither (unless it is an unsupported value, which is
    #    already handled above via the unsupported hints).
    if http_status == 400 or _contains_any(text, _INVALID_REQUEST_HINTS):
        return APIError(
            category=APIErrorCategory.INVALID_REQUEST,
            provider=provider,
            model=model,
            http_status=http_status,
            provider_code=provider_code,
            message=message,
            retryable=False,
            recoverable=False,
            offending_param=offending,
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
        recoverable=False,
        offending_param=offending,
    )


def is_unsupported_parameter_error(err: APIError) -> bool:
    """Return True if the error indicates an unsupported request parameter name."""
    return err.category == APIErrorCategory.UNSUPPORTED_PARAMETER


def is_unsupported_value_error(err: APIError) -> bool:
    """Return True if the error indicates an unsupported request parameter value."""
    return err.category == APIErrorCategory.UNSUPPORTED_VALUE


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


# ---------------------------------------------------------------------------
# Capability negotiation engine
# ---------------------------------------------------------------------------
# The set of "optional" sampling parameters the engine negotiates. Each maps to
# a capability category. The engine tries to send these with their default ARCUS
# value; on an unsupported-value error it omits them to preserve the provider's
# default.
OPTIONAL_SAMPLING_PARAMS: Dict[str, str] = {
    "temperature": CAP_TEMPERATURE,
    "top_p": CAP_TOP_P,
    "frequency_penalty": CAP_FREQUENCY_PENALTY,
    "presence_penalty": CAP_PRESENCE_PENALTY,
}

# The default ARCUS value for each optional sampling parameter.
DEFAULT_SAMPLING_VALUES: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


@dataclass
class NegotiationState:
    """Mutable per-request negotiation state used by the execution layer.

    Tracks which token parameter is currently being tried and which optional
    sampling parameters have been *omitted* because the provider rejected their
    value. The execution layer consults this to build each retry payload.
    """

    provider: str
    model: str
    token_param: str = DEFAULT_TOKEN_PARAM
    omitted_params: set = field(default_factory=set)
    cache: CapabilityCache = field(default_factory=lambda: SHARED_CAPABILITY_CACHE)

    def seed_from_cache(self) -> None:
        """Pre-populate negotiation state from the capability cache, if present."""
        rec = self.cache.get(self.provider, self.model)
        if rec is None:
            return
        tok = rec.capabilities.get(CAP_TOKEN_PARAMETER)
        if tok is not None and tok.param:
            self.token_param = tok.param
        for param, cap_name in OPTIONAL_SAMPLING_PARAMS.items():
            cap = rec.capabilities.get(cap_name)
            if cap is not None and not cap.supports_custom_values:
                # Provider only supports the default -> omit to preserve it.
                self.omitted_params.add(param)

    def build_payload(self, base_payload: Dict[str, Any], token_value: Any) -> Dict[str, Any]:
        """Build the request payload for the current negotiation state."""
        payload = build_token_param_payload(base_payload, self.token_param, token_value)
        # Drop any optional sampling params we have learned to omit.
        for param in self.omitted_params:
            payload.pop(param, None)
        return payload

    def record_token_param_success(self) -> None:
        self.cache.set_capability(
            self.provider, self.model,
            Capability(name=CAP_TOKEN_PARAMETER, supported=True,
                       param=self.token_param, supports_custom_values=True),
        )

    def record_token_param_fallback(self, old_param: str, new_param: str) -> None:
        self.token_param = new_param
        self.cache.set_capability(
            self.provider, self.model,
            Capability(name=CAP_TOKEN_PARAMETER, supported=True,
                       param=new_param, supports_custom_values=True),
        )

    def record_value_omission(self, param: str) -> None:
        self.omitted_params.add(param)
        cap_name = OPTIONAL_SAMPLING_PARAMS.get(param, param)
        self.cache.set_capability(
            self.provider, self.model,
            Capability(name=cap_name, supported=True, param=param,
                       supports_custom_values=False),
        )

    def snapshot_record(self) -> CapabilityRecord:
        """Return a CapabilityRecord reflecting the current negotiation state."""
        caps: Dict[str, Capability] = {}
        caps[CAP_TOKEN_PARAMETER] = Capability(
            name=CAP_TOKEN_PARAMETER, supported=True,
            param=self.token_param, supports_custom_values=True,
        )
        for param, cap_name in OPTIONAL_SAMPLING_PARAMS.items():
            if param in self.omitted_params:
                caps[cap_name] = Capability(
                    name=cap_name, supported=True, param=param,
                    supports_custom_values=False,
                )
            else:
                caps[cap_name] = Capability(
                    name=cap_name, supported=True, param=param,
                    supports_custom_values=True,
                    default_value=DEFAULT_SAMPLING_VALUES.get(param),
                )
        return CapabilityRecord(provider=self.provider, model=self.model, capabilities=caps)


# ---------------------------------------------------------------------------
# Model Capability Registry / Config Layer (Extensible)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model Capability Registry / Config Layer (Extensible)
# ---------------------------------------------------------------------------
@dataclass
class ModelCapabilities:
    """A capability set for a model, including reasoning controls and outputs.

    Attributes:
        supports_reasoning_controls: Whether the model supports request parameters for reasoning.
        supports_reasoning_output: Whether the model exposes reasoning output/tokens.
        reasoning_param: The parameter name used for reasoning controls (if supported).
        reasoning_format: Default format/value for reasoning controls.
        supports_temperature: Whether temperature parameter is supported.
        supports_top_p: Whether top_p parameter is supported.
        max_context: Maximum context length.
        token_param: Optional token parameter name.
    """
    supports_reasoning_controls: bool = False
    supports_reasoning_output: bool = False
    reasoning_param: Optional[str] = None
    reasoning_format: Optional[Any] = None
    supports_temperature: bool = True
    supports_top_p: bool = True
    max_context: int = 128000
    token_param: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        if key == "supports_reasoning":
            return self.supports_reasoning_controls
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "supports_reasoning":
            return self.supports_reasoning_controls
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in [
            "supports_reasoning_controls",
            "supports_reasoning_output",
            "supports_reasoning",
            "reasoning_param",
            "reasoning_format",
            "supports_temperature",
            "supports_top_p",
            "max_context",
            "token_param",
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_reasoning_controls": self.supports_reasoning_controls,
            "supports_reasoning_output": self.supports_reasoning_output,
            "supports_reasoning": self.supports_reasoning_controls,
            "reasoning_param": self.reasoning_param,
            "reasoning_format": self.reasoning_format,
            "supports_temperature": self.supports_temperature,
            "supports_top_p": self.supports_top_p,
            "max_context": self.max_context,
            "token_param": self.token_param,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        supports_controls = data.get("supports_reasoning_controls", data.get("supports_reasoning", False))
        supports_output = data.get("supports_reasoning_output", data.get("supports_reasoning", False))
        return cls(
            supports_reasoning_controls=supports_controls,
            supports_reasoning_output=supports_output,
            reasoning_param=data.get("reasoning_param"),
            reasoning_format=data.get("reasoning_format"),
            supports_temperature=data.get("supports_temperature", True),
            supports_top_p=data.get("supports_top_p", True),
            max_context=data.get("max_context", 128000),
            token_param=data.get("token_param"),
        )


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "deepseek-v4-flash": ModelCapabilities(
        supports_reasoning_controls=False,
        supports_reasoning_output=False,
        reasoning_param=None,
        reasoning_format=None,
        supports_temperature=True,
        supports_top_p=True,
        max_context=128000,
    ),
    "deepseek-r1": ModelCapabilities(
        supports_reasoning_controls=True,
        supports_reasoning_output=True,
        reasoning_param="include_reasoning",
        reasoning_format=True,
        supports_temperature=True,
        supports_top_p=True,
        max_context=64000,
    ),
    "gpt-5-mini": ModelCapabilities(
        supports_reasoning_controls=False,
        supports_reasoning_output=False,
        reasoning_param=None,
        reasoning_format=None,
        token_param="max_completion_tokens",
        supports_temperature=True,
        supports_top_p=True,
        max_context=128000,
    ),
    "o3-mini": ModelCapabilities(
        supports_reasoning_controls=True,
        supports_reasoning_output=True,
        reasoning_param="reasoning_effort",
        reasoning_format="medium",
        supports_temperature=False,
        supports_top_p=True,
        max_context=128000,
    ),
    "o1": ModelCapabilities(
        supports_reasoning_controls=True,
        supports_reasoning_output=True,
        reasoning_param="reasoning_effort",
        reasoning_format="medium",
        supports_temperature=False,
        supports_top_p=True,
        max_context=128000,
    ),
    "claude-3.5-sonnet": ModelCapabilities(
        supports_reasoning_controls=True,
        supports_reasoning_output=True,
        reasoning_param="thinking",
        reasoning_format={"type": "enabled", "budget_tokens": 5000},
        supports_temperature=True,
        supports_top_p=True,
        max_context=200000,
    ),
    "gemini-2.5-flash": ModelCapabilities(
        supports_reasoning_controls=False,
        supports_reasoning_output=False,
        reasoning_param=None,
        reasoning_format=None,
        supports_temperature=True,
        supports_top_p=True,
        max_context=1000000,
    ),
    "qwen-3": ModelCapabilities(
        supports_reasoning_controls=False,
        supports_reasoning_output=False,
        reasoning_param=None,
        reasoning_format=None,
        supports_temperature=True,
        supports_top_p=True,
        max_context=131072,
    ),
}


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """Retrieve capabilities for a model using exact or prefix/family matching.

    Extensible: if model_name does not match an exact key, searches for prefix
    or family substrings (e.g. 'deepseek', 'gpt-5', 'claude-3', 'gemini', 'qwen')
    and falls back to standard defaults if unmatched.
    """
    if not model_name:
        return ModelCapabilities(
            supports_reasoning_controls=False,
            supports_reasoning_output=False,
            reasoning_param=None,
            supports_temperature=True,
            max_context=128000,
        )

    key = model_name.lower()
    short_name = key.split("/")[-1]
    if short_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[short_name]
    if key in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[key]

    for pattern, caps in MODEL_CAPABILITIES.items():
        if pattern in short_name or pattern in key:
            return caps

    return ModelCapabilities(
        supports_reasoning_controls=False,
        supports_reasoning_output=False,
        reasoning_param=None,
        supports_temperature=True,
        supports_top_p=True,
        max_context=128000,
    )


def validate_model_capabilities(model_name: str, options: Optional[Dict[str, Any]] = None) -> None:
    """Validate model capabilities before running a benchmark.

    Logs whether reasoning controls are enabled or unsupported -> omitted,
    rather than throwing an exception.
    """
    opts = options or {}
    caps = get_model_capabilities(model_name)
    supports_controls = caps.supports_reasoning_controls if hasattr(caps, "supports_reasoning_controls") else caps.get("supports_reasoning", False)
    wants_reasoning = opts.get("include_reasoning") or opts.get("reasoning_effort") or opts.get("thinking") or opts.get("reasoning")
    
    if wants_reasoning and supports_controls:
        logger.info("[Capability]\nReasoning controls: enabled")
    else:
        logger.info("[Capability]\nReasoning controls: unsupported -> omitted")


def build_capability_aware_payload(
    model_name: str,
    base_payload: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Consult the capability registry to construct an API request payload.

    Only injects supported arguments (reasoning, temperature, etc.) based on
    what the model/provider actually supports. Capability-driven.
    """
    payload = dict(base_payload)
    caps = get_model_capabilities(model_name)
    opts = options or {}

    r_param = caps.reasoning_param if hasattr(caps, "reasoning_param") else caps.get("reasoning_param")
    r_format = caps.reasoning_format if hasattr(caps, "reasoning_format") else caps.get("reasoning_format")
    supports_controls = caps.supports_reasoning_controls if hasattr(caps, "supports_reasoning_controls") else caps.get("supports_reasoning", False)

    if supports_controls and r_param:
        val = opts.get(r_param)
        if val is None:
            val = opts.get("include_reasoning") or opts.get("reasoning_effort") or opts.get("thinking") or opts.get("reasoning") or r_format
        if val is not None:
            payload[r_param] = val
            logger.info("[Capability]\nReasoning controls: enabled")
        else:
            logger.info("[Capability]\nReasoning controls: unsupported -> omitted")
    else:
        logger.info("[Capability]\nReasoning controls: unsupported -> omitted")
        payload.pop("include_reasoning", None)
        payload.pop("reasoning_effort", None)
        payload.pop("thinking", None)
        payload.pop("reasoning", None)

    return payload

