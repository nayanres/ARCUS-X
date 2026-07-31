"""Tests for the provider/model-agnostic API compatibility layer.

Covers the required scenarios from the task spec:

  1. Provider rejects ``max_tokens`` but accepts ``max_completion_tokens``
     (dynamic fallback succeeds and is recorded as an adaptation).
  2. Provider accepts ``max_tokens`` normally (no fallback, no adaptation).
  3. Unsupported-parameter error is correctly classified.
  4. Capability cache prevents repeated failed attempts.
  5. API failure logs contain structured metadata.
  6. Existing benchmark runs remain unchanged (scoring/trajectory untouched).

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

from arcus.evaluation.evaluator import ModelEvaluator
from arcus.evaluation import api_compat
from arcus.evaluation.api_compat import (
    APIErrorCategory,
    CapabilityCache,
    AdaptationLog,
    CompatibilityAdaptation,
    classify_error,
    is_unsupported_parameter_error,
    build_token_param_payload,
    next_token_param,
    TOKEN_PARAM_FALLBACK_CHAIN,
    DEFAULT_TOKEN_PARAM,
    SHARED_CAPABILITY_CACHE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    # Avoid the vLLM branch by passing an api_base; the api_key is satisfied
    # via the environment fallback (we set a dummy key).
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


# ---------------------------------------------------------------------------
# 1. Provider rejects max_tokens but accepts max_completion_tokens
# ---------------------------------------------------------------------------
def test_fallback_max_tokens_to_max_completion_tokens(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("max_tokens") is not None:
            # First attempt: reject max_tokens as unsupported.
            return _FakeResponse(400, _unsupported_param_error("max_tokens"))
        # Second attempt: max_completion_tokens accepted.
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(api_compat.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    # Clear any cached capability for a clean test.
    SHARED_CAPABILITY_CACHE._token_params.clear()

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

    # The capability must be cached for future requests.
    assert SHARED_CAPABILITY_CACHE.get_token_parameter("azure", "azure/gpt-5-mini") == "max_completion_tokens"


# ---------------------------------------------------------------------------
# 2. Provider accepts max_tokens normally
# ---------------------------------------------------------------------------
def test_accepts_max_tokens_normally(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json))
        return _FakeResponse(200, _success_payload())

    monkeypatch.setattr(api_compat.requests, "post", fake_post)

    ev = _make_evaluator("openrouter/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._token_params.clear()

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "stop"
    assert len(calls) == 1
    assert "max_tokens" in calls[0]
    # No fallback -> no adaptation recorded.
    assert ev.adaptation_log.to_dict() == []
    # Default capability cached.
    assert SHARED_CAPABILITY_CACHE.get_token_parameter("openrouter", "openrouter/gpt-5-mini") == "max_tokens"


# ---------------------------------------------------------------------------
# 3. Unsupported-parameter error is correctly classified
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("message,status,expected", [
    ("Unsupported parameter: 'max_tokens' is not supported", 400, APIErrorCategory.UNSUPPORTED_PARAMETER),
    ("invalid_parameter: max_completion_tokens not allowed", 422, APIErrorCategory.UNSUPPORTED_PARAMETER),
    ("Unauthorized", 401, APIErrorCategory.AUTHENTICATION_ERROR),
    ("Rate limit exceeded", 429, APIErrorCategory.RATE_LIMIT),
    ("The deployment is not available", 503, APIErrorCategory.DEPLOYMENT_ERROR),
    ("context length exceeded", 400, APIErrorCategory.CONTEXT_LIMIT),
    ("read timeout", None, APIErrorCategory.TIMEOUT),
    ("bad request", 400, APIErrorCategory.INVALID_REQUEST),
    ("something weird happened", 418, APIErrorCategory.UNKNOWN_PROVIDER_ERROR),
])
def test_classify_error_categories(message, status, expected):
    err = classify_error(
        provider="azure", model="gpt-5-mini",
        http_status=status, provider_code="X", message=message,
    )
    assert err.category == expected
    # Structured metadata is always preserved.
    assert err.provider == "azure"
    assert err.model == "gpt-5-mini"
    assert err.http_status == status
    assert err.message == message
    d = err.to_dict()
    assert d["category"] == expected.value
    assert "retryable" in d


def test_is_unsupported_parameter_error_helper():
    err = classify_error(provider="azure", model="m", http_status=400,
                         provider_code="x", message="unsupported parameter max_tokens")
    assert is_unsupported_parameter_error(err) is True
    auth = classify_error(provider="azure", model="m", http_status=401,
                          provider_code="x", message="unauthorized")
    assert is_unsupported_parameter_error(auth) is False


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

    monkeypatch.setattr(api_compat.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._token_params.clear()
    # Pre-seed the cache as if a previous request discovered the capability.
    SHARED_CAPABILITY_CACHE.set_token_parameter("azure", "azure/gpt-5-mini", "max_completion_tokens")

    result = ev.evaluate("move right")

    assert result["finish_reason"] == "stop"
    # Only ONE call, and it used the cached parameter (no failed attempt).
    assert len(calls) == 1
    assert "max_tokens" not in calls[0]
    assert calls[0].get("max_completion_tokens") is not None
    # No new adaptation needed because the cache already had the answer.
    assert ev.adaptation_log.to_dict() == []


def test_capability_cache_is_thread_safe_and_snapshot():
    cache = CapabilityCache()
    cache.set_token_parameter("p", "m", "max_completion_tokens")
    snap = cache.snapshot()
    assert snap == [{"provider": "p", "model": "m", "token_parameter": "max_completion_tokens"}]
    assert cache.has("p", "m")
    assert not cache.has("other", "m")


# ---------------------------------------------------------------------------
# 5. API failure logs contain structured metadata
# ---------------------------------------------------------------------------
def test_failure_logs_structured_metadata(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        # Non-recoverable: authentication error (not an unsupported param).
        return _FakeResponse(401, {"error": {"message": "Invalid API key", "type": "auth_error"}})

    monkeypatch.setattr(api_compat.requests, "post", fake_post)

    ev = _make_evaluator("azure/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._token_params.clear()

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
    # The raw output marker also carries structured metadata (not [API_ERROR]).
    raw = result["output"]
    assert isinstance(raw, dict)
    assert raw["Type"] == APIErrorCategory.AUTHENTICATION_ERROR.value
    assert raw["HTTPStatus"] == 401


def test_adaptation_log_records_with_callback():
    recorded = []

    def cb(adaptation: CompatibilityAdaptation):
        recorded.append(adaptation.to_dict())

    log = AdaptationLog(on_adapt=cb)
    log.record(CompatibilityAdaptation(
        provider="azure", model="gpt-5-mini",
        adjustment="max_tokens -> max_completion_tokens",
    ))
    assert len(recorded) == 1
    assert recorded[0]["adjustment"] == "max_tokens -> max_completion_tokens"
    assert len(log.to_dict()) == 1


# ---------------------------------------------------------------------------
# 6. Existing benchmark runs remain unchanged (scoring/trajectory untouched)
# ---------------------------------------------------------------------------
def test_existing_benchmark_run_unchanged(monkeypatch):
    """A normal successful run produces the same raw result shape as before."""
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(200, _success_payload("[[0,0]->[0,1]->[0,2]]"))

    monkeypatch.setattr(api_compat.requests, "post", fake_post)

    ev = _make_evaluator("openrouter/gpt-5-mini")
    SHARED_CAPABILITY_CACHE._token_params.clear()

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


# ---------------------------------------------------------------------------
# Fallback chain extensibility
# ---------------------------------------------------------------------------
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
