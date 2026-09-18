"""Tests for the provider/model-agnostic API capability negotiation layer.

Covers the required scenarios from the task spec:

   1. Provider rejects ``max_tokens`` but accepts ``max_completion_tokens``
      (dynamic token-parameter fallback succeeds and is recorded as an
      adaptation).
   2. Provider rejects a parameter *value* (e.g. ``temperature=0.0``) and the
      engine recovers by *omitting* the parameter to preserve the provider's
      default.
   3. Parameter-omission recovery is preferred over forcing a value.
   4. Capability cache prevents repeated failed negotiation attempts and stores
      the full capability record (token parameter + value constraints).
   5. ``retryable`` vs ``recoverable`` classification is correct for every
      category (rate limit, unsupported param/value, auth, etc.).
   6. No regression for existing providers (scoring/trajectory untouched).

The evaluator is exercised with a mocked ``requests.post`` so no network or
vLLM dependency is required.
"""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

import arcus.evaluation.evaluator as evaluator_module
from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation import api_compat
from arcus.evaluation.api_compat import (
    APIErrorCategory,
    CapabilityCache,
    Capability,
    AdaptationLog,
    CompatibilityAdaptation,
    NegotiationState,
    classify_error,
    is_unsupported_parameter_error,
    is_unsupported_value_error,
    build_token_param_payload,
    next_token_param,
    TOKEN_PARAM_FALLBACK_CHAIN,
    DEFAULT_TOKEN_PARAM,
    SHARED_CAPABILITY_CACHE,
    CAP_TOKEN_PARAMETER,
    CAP_TEMPERATURE,
)


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: dict, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.reason = "OK" if status_code == 200 else "ERR"

    def json(self):
        return self._payload


def _make_evaluator(model_name: str, api_base: str = "https://azure.example/v1") -> ModelEvaluator:
    """Construct a ModelEvaluator in API mode without network/vLLM."""
    os.environ.setdefault("AZURE_API_KEY", "test-key")
    ev = ModelEvaluator(model_name, api_base=api_base, api_key="test-key")
    return ev


def _success_payload(content: str = "[[0,0]->[0,1]]") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15},
    }


def _unsupported_param_error(param_name: str = "max_tokens") -> dict:
    return {
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_parameter",
            "message": f"Unsupported parameter: '{param_name}' is not supported "
                       f"with this model. Use 'max_completion_tokens' instead.",
        }
    }


def _unsupported_value_error(param_name: str = "temperature") -> dict:
    return {
        "error": {
            "type": "UNSUPPORTED_PARAMETER",
            "code": "unsupported_value",
            "message": f"Unsupported value: '{param_name}' does not support 0.0 "
                       f"with this model. Only the default (1) value is supported.",
        }
    }


def test_fallback_max_tokens_to_max_completion_tokens(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("max_tokens") is not None:
            # First attempt: reject max_tokens as unsupported.
            return _FakeResponse(400, _unsupported_param_error("max_tokens"))
        # Second attempt: max_completion_tokens accepted.
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()

    result = ev.evaluate("move right")

    # The successful response should be returned (fallback succeeded).
    assert result["finish_reason"] == "stop"
    assert "[[0,0]->[0,1]]" in result["output"]

    # Exactly two calls: one with max_tokens, one with max_completion_tokens.
    assert len(calls) == 2
    assert "max_tokens" in calls[0]
    assert "max_completion_tokens" in calls[1]
    # The second request must NOT carry the rejected parameter.
    assert "max_tokens" not in calls[1]

    # An adaptation must have been recorded.
    adaptations = ev.adaptation_log.to_dict()
    assert len(adaptations) == 1
    assert adaptations[0]["provider"] == "azure"
    assert adaptations[0]["model"] == "azure/gpt-5-mini"
    assert adaptations[0]["adjustment"] == "max_tokens -> max_completion_tokens"
    assert adaptations[0]["kind"] == "parameter_name"

    # The capability must be cached for future requests.
    cap = SHARED_CAPABILITY_CACHE.get_capability("azure", "azure/gpt-5-mini", CAP_TOKEN_PARAMETER)
    assert cap is not None
    assert cap.param == "max_completion_tokens"


# ---------------------------------------------------------------------------
# 2. Provider rejects a parameter VALUE (temperature=0.0) -> omit it
# ---------------------------------------------------------------------------
def test_unsupported_value_omits_parameter(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("temperature") is not None:
            # First attempt: reject temperature=0.0 as an unsupported value.
            return _FakeResponse(400, _unsupported_value_error("temperature"))
        # Second attempt: temperature omitted -> accepted.
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "stop"
    # Two calls: first with temperature, second without.
    assert len(calls) == 2
    assert calls[0].get("temperature") is not None
    assert "temperature" not in calls[1]

    # An adaptation recording the omission must be present.
    adaptations = ev.adaptation_log.to_dict()
    assert len(adaptations) == 1
    assert adaptations[0]["kind"] == "parameter_omitted"
    assert "temperature" in adaptations[0]["adjustment"]
    assert "removed" in adaptations[0]["adjustment"].lower()

    # The capability record must mark temperature as not supporting custom
    # values (so future requests omit it directly).
    cap = SHARED_CAPABILITY_CACHE.get_capability("azure", "azure/gpt-5-mini", CAP_TEMPERATURE)
    assert cap is not None
    assert cap.supports_custom_values is False


# ---------------------------------------------------------------------------
# 3. Parameter-omission recovery is preferred over forcing a value
# ---------------------------------------------------------------------------
def test_value_omission_preserves_provider_default(monkeypatch):
    """Omission (not forcing temperature=1) is the recovery path."""
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("temperature") is not None:
            return _FakeResponse(400, _unsupported_value_error("temperature"))
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()

    ev.evaluate("move right")

    # The recovery request must NOT carry a forced temperature value.
    recovery = calls[-1]
    assert "temperature" not in recovery
    # It must still carry the token parameter (negotiation is independent).
    assert recovery.get("max_tokens") is not None or recovery.get("max_completion_tokens") is not None


# ---------------------------------------------------------------------------
# 4. Capability cache prevents repeated failed attempts
# ---------------------------------------------------------------------------
def test_capability_cache_skips_failed_attempt(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        # Always reject max_tokens; but because the cache is pre-seeded with the
        # correct parameter, the evaluator should never send max_tokens.
        if "max_tokens" in json:
            return _FakeResponse(400, _unsupported_param_error("max_tokens"))
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()
    # Pre-seed the cache as if a previous request discovered the capability.
    SHARED_CAPABILITY_CACHE.set_capability(
        "azure", "azure/gpt-5-mini",
        Capability(name=CAP_TOKEN_PARAMETER, supported=True,
                   param="max_completion_tokens", supports_custom_values=True),
    )

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "stop"
    # Only ONE call, and it used the cached parameter (no failed attempt).
    assert len(calls) == 1
    assert "max_tokens" not in calls[0]
    assert calls[0].get("max_completion_tokens") is not None
    # No new adaptation needed because the cache already had the answer.
    assert ev.adaptation_log.to_dict() == []


def test_capability_cache_value_omission_skips_failed_attempt(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("temperature") is not None:
            return _FakeResponse(400, _unsupported_value_error("temperature"))
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()
    # Pre-seed: temperature only supports the default -> omit it.
    SHARED_CAPABILITY_CACHE.set_capability(
        "azure", "azure/gpt-5-mini",
        Capability(name=CAP_TEMPERATURE, supported=True, param="temperature",
                   supports_custom_values=False),
    )

    result = ev.evaluate("move right")
    assert result["finish_reason"] == "stop"
    # Single call, temperature already omitted from the start.
    assert len(calls) == 1
    assert "temperature" not in calls[0]
    assert ev.adaptation_log.to_dict() == []


def test_capability_cache_is_thread_safe_and_snapshot():
    cache = CapabilityCache()
    cache.set_capability("p", "m",
                         Capability(name=CAP_TOKEN_PARAMETER, supported=True,
                                    param="max_completion_tokens"))
    cache.set_capability("p", "m",
                         Capability(name=CAP_TEMPERATURE, supported=True,
                                    param="temperature", supports_custom_values=False))
    snap = cache.snapshot()
    assert len(snap) == 1
    rec = snap[0]
    assert rec["provider"] == "p"
    assert rec["model"] == "m"
    assert rec["capabilities"][CAP_TOKEN_PARAMETER]["param"] == "max_completion_tokens"
    assert rec["capabilities"][CAP_TEMPERATURE]["supports_custom_values"] is False
    assert cache.has("p", "m")
    assert not cache.has("other", "m")


# ---------------------------------------------------------------------------
# 5. retryable vs recoverable classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message,status,expected,retryable,recoverable", [
    # Rate limit: retryable, not recoverable.
    ("Rate limit exceeded", 429, APIErrorCategory.RATE_LIMIT, True, False),
    # Unsupported parameter name: not retryable, recoverable.
    ("Unsupported parameter: 'max_tokens' is not supported", 400,
     APIErrorCategory.UNSUPPORTED_PARAMETER, False, True),
    # Unsupported value: not retryable, recoverable.
    ("Unsupported value: 'temperature' does not support 0.0 with this model. "
     "Only the default (1) value is supported.", 400,
     APIErrorCategory.UNSUPPORTED_VALUE, False, True),
    # Authentication: neither.
    ("Unauthorized", 401, APIErrorCategory.AUTHENTICATION_ERROR, False, False),
    # Deployment: retryable, not recoverable.
    ("The deployment is not available", 503, APIErrorCategory.DEPLOYMENT_ERROR, True, False),
    # Context limit: neither.
    ("context length exceeded", 400, APIErrorCategory.CONTEXT_LIMIT, False, False),
    # Timeout: retryable, not recoverable.
    ("read timeout", None, APIErrorCategory.TIMEOUT, True, False),
    # Invalid request: neither.
    ("bad request", 400, APIErrorCategory.INVALID_REQUEST, False, False),
    # Unknown: neither.
    ("something weird happened", 418, APIErrorCategory.UNKNOWN_PROVIDER_ERROR, False, False),
])
def test_classify_retryable_vs_recoverable(message, status, expected,
                                           retryable, recoverable):
    err = classify_error(
        provider="azure", model="gpt-5-mini",
        http_status=status, provider_code="X", message=message,
    )
    assert err.category == expected
    assert err.retryable is retryable
    assert err.recoverable is recoverable
    # Structured metadata is always preserved.
    assert err.provider == "azure"
    assert err.model == "gpt-5-mini"
    assert err.http_status == status
    assert err.message == message
    d = err.to_dict()
    assert d["category"] == expected.value
    assert d["retryable"] == retryable
    assert d["recoverable"] == recoverable


def test_offending_param_inference():
    err = classify_error(
        provider="azure", model="m", http_status=400,
        provider_code="unsupported_value",
        message="Unsupported value: 'temperature' does not support 0.0",
    )
    assert is_unsupported_value_error(err) is True
    assert err.offending_param == "temperature"

    err2 = classify_error(
        provider="azure", model="m", http_status=400,
        provider_code="invalid_parameter",
        message="Unsupported parameter: 'max_tokens' is not supported",
    )
    assert is_unsupported_parameter_error(err2) is True
    assert err2.offending_param == "max_tokens"


# ---------------------------------------------------------------------------
# 6. No regression for existing providers (scoring/trajectory untouched)
# ---------------------------------------------------------------------------
def test_accepts_max_tokens_normally(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("openrouter/gpt-5-mini", api_base="https://openrouter.ai/api/v1")
    SHARED_CAPABILITY_CACHE._records.clear()

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "stop"
    assert len(calls) == 1
    assert "max_tokens" in calls[0]
    # No fallback -> no adaptation recorded.
    assert ev.adaptation_log.to_dict() == []
    # Default capability cached (provider detected from openrouter api base).
    cap = SHARED_CAPABILITY_CACHE.get_capability("openrouter", "openrouter/gpt-5-mini", CAP_TOKEN_PARAMETER)
    assert cap is not None
    assert cap.param == "max_tokens"


def test_existing_benchmark_run_unchanged(monkeypatch):
    """A normal successful run produces the same raw result shape as before."""
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, _success_payload("[[0,0]->[0,1]->[0,2]]"))

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("openrouter/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()

    probe = {
        "id": "probe_0", "fol_depth": 3, "gravity_target": 0.0,
        "axioms": ["rule"], "question": "apply actions",
    }
    result = ev.evaluate_single_probe(probe, tier=0, depth=3)

    # The result dict still carries the same trajectory-relevant fields used by
    # the runner/metrics; scoring logic is untouched.
    for key in ("probe_id", "prompt_hash", "raw_output", "output_tokens",
                "prompt_tokens", "total_tokens", "fol_depth", "depth", "tier"):
        assert key in result
    assert result["fol_depth"] == 3
    assert result["raw_output"] == "[[0,0]->[0,1]->[0,2]]"
    # No API error key on a successful run.
    assert "api_error" not in result


def test_failure_logs_structured_metadata_with_recoverable(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        # Non-recoverable: authentication error.
        return _FakeResponse(401, {"error": {"message": "Invalid API key", "type": "auth_error"}})

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._records.clear()

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "error"
    # The structured error dict is preserved on the result.
    assert "api_error" in result
    api_err = result["api_error"]
    assert api_err["category"] == APIErrorCategory.AUTHENTICATION_ERROR.value
    assert api_err["provider"] == "azure"
    assert api_err["model"] == "azure/gpt-5-mini"
    assert api_err["http_status"] == 401
    assert api_err["retryable"] is False
    assert api_err["recoverable"] is False
    # The raw output marker also carries structured metadata (not [API_ERROR]).
    raw = result["output"]
    assert isinstance(raw, dict)
    assert raw["Type"] == APIErrorCategory.AUTHENTICATION_ERROR.value
    assert raw["HTTPStatus"] == 401
    assert raw["Recoverable"] is False


# ---------------------------------------------------------------------------
# Negotiation engine unit tests
# ---------------------------------------------------------------------------
def test_negotiation_state_seeds_from_cache():
    cache = CapabilityCache()
    cache.set_capability("p", "m",
                         Capability(name=CAP_TOKEN_PARAMETER, supported=True,
                                    param="max_completion_tokens"))
    cache.set_capability("p", "m",
                         Capability(name=CAP_TEMPERATURE, supported=True,
                                    param="temperature", supports_custom_values=False))
    neg = NegotiationState(provider="p", model="m", cache=cache)
    neg.seed_from_cache()
    assert neg.token_param == "max_completion_tokens"
    assert "temperature" in neg.omitted_params


def test_negotiation_state_build_payload_omits_and_renames():
    neg = NegotiationState(provider="p", model="m")
    neg.token_param = "max_completion_tokens"
    neg.omitted_params.add("temperature")
    base = {"model": "m", "messages": [], "temperature": 0.0}
    payload = neg.build_payload(base, 100)
    assert payload.get("max_completion_tokens") == 100
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    # Original base is not mutated.
    assert base.get("temperature") == 0.0


def test_adaptation_log_records_with_callback():
    recorded = []

    def cb(adaptation: CompatibilityAdaptation):
        recorded.append(adaptation.to_dict())

    log = AdaptationLog(on_adapt=cb)
    log.record(CompatibilityAdaptation(
        provider="azure", model="gpt-5-mini",
        adjustment="max_tokens -> max_completion_tokens",
        kind="parameter_name",
    ))
    assert len(recorded) == 1
    assert recorded[0]["adjustment"] == "max_tokens -> max_completion_tokens"
    assert recorded[0]["kind"] == "parameter_name"
    assert len(log.to_dict()) == 1


def test_fallback_chain_extensibility():
    """Adding a parameter to the chain requires no execution-layer rewrite."""
    original = list(TOKEN_PARAM_FALLBACK_CHAIN)
    try:
        TOKEN_PARAM_FALLBACK_CHAIN.append("max_output_tokens")
        assert next_token_param("max_completion_tokens") == "max_output_tokens"
        assert next_token_param("max_output_tokens") is None
    finally:
        TOKEN_PARAM_FALLBACK_CHAIN[:] = original


def test_build_token_param_payload_strips_competing_keys():
    base = {"model": "m", "max_tokens": 100}
    out = build_token_param_payload(base, "max_completion_tokens", 50)
    assert out.get("max_completion_tokens") == 50
    assert "max_tokens" not in out
    # Original base is not mutated.
    assert base.get("max_tokens") == 100


def test_model_capability_registry_and_validation():
    from arcus.evaluation.api_compat import (
        get_model_capabilities,
        validate_model_capabilities,
        build_capability_aware_payload,
    )

    # 1. DeepSeek-V4-Flash does not support reasoning controls
    caps = get_model_capabilities("DeepSeek-V4-Flash")
    assert caps["supports_reasoning"] is False
    assert caps.supports_reasoning_controls is False
    assert caps.supports_reasoning_output is False
    assert caps["reasoning_param"] is None

    # 2. Validation does not fail fast; gracefully handles unsupported reasoning option for DeepSeek-V4-Flash
    validate_model_capabilities("DeepSeek-V4-Flash", {"include_reasoning": True})

    # 3. Capability-aware payload strips unsupported reasoning for DeepSeek-V4-Flash
    base = {"model": "DeepSeek-V4-Flash", "messages": []}
    payload = build_capability_aware_payload("DeepSeek-V4-Flash", base, {"include_reasoning": True})
    assert "include_reasoning" not in payload

    # 4. DeepSeek-R1 supports reasoning and includes it
    r1_caps = get_model_capabilities("DeepSeek-R1")
    assert r1_caps["supports_reasoning"] is True
    assert r1_caps.supports_reasoning_controls is True
    assert r1_caps.supports_reasoning_output is True
    r1_payload = build_capability_aware_payload("DeepSeek-R1", base, {"include_reasoning": True})
    assert r1_payload.get("include_reasoning") is True


def test_api_error_metadata_preservation():
    from arcus.evaluation.api_compat import classify_error
    err = classify_error(
        provider="azure",
        model="DeepSeek-V4-Flash",
        http_status=400,
        provider_code="unrecognized_request_argument",
        message="Unrecognized request argument supplied: include_reasoning",
    )
    d = err.to_dict()
    assert d["provider"] == "azure"
    assert d["model"] == "DeepSeek-V4-Flash"
    assert d["provider_code"] == "unrecognized_request_argument"
    assert "include_reasoning" in d["message"]
    assert d["timestamp"] is not None
    assert d["retryable"] is False
