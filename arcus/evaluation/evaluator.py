"""Model evaluation / inference layer for ARCUS-X.

The evaluator is responsible ONLY for turning a prompt into a model response
(and accounting for tokens). It deliberately does **not** score trajectories:
scoring lives in :mod:`arcus.evaluation.metrics` and the runner, so that the
discouraged "checksum / final-answer substring" shortcuts are never the source
of truth.

Two backends are supported:

1. vLLM local (open-weight, tensor-parallel GPU).
2. OpenRouter / Azure compatible chat API (API-only mode, works on Windows).

Reproducibility note: ``prompt_hash`` is now a deterministic ``sha256`` digest
(replacing the non-deterministic builtin :func:`hash`).

Latency note: per-probe ``request_start_time`` / ``response_end_time`` /
``latency_ms`` are recorded as a *systems-level diagnostic only*. Latency is
affected by provider infrastructure, architecture, batching, and deployment
conditions, so it must never be treated as a model capability score.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import requests

from arcus.evaluation.token_accounting import measure_tokens
from arcus.evaluation.api_compat import (
    AdaptationLog,
    CapabilityCache,
    SHARED_CAPABILITY_CACHE,
    APIError,
    APIErrorCategory,
    classify_error,
    detect_provider,
    is_unsupported_parameter_error,
    is_unsupported_value_error,
    build_token_param_payload,
    next_token_param,
    DEFAULT_TOKEN_PARAM,
    TOKEN_PARAM_FALLBACK_CHAIN,
    NegotiationState,
    OPTIONAL_SAMPLING_PARAMS,
    DEFAULT_SAMPLING_VALUES,
    build_capability_aware_payload,
    validate_model_capabilities,
    get_model_capabilities,
    )

try:  # pragma: no cover - optional heavy dependency
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except Exception:  # ImportError or any env issue
    HAS_VLLM = False

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available - tokenizer approximation only")

logger = logging.getLogger(__name__)


class DepthMismatchError(Exception):
    """Raised when the probe depth does not match the requested depth."""


def validate_response(response: Optional[Dict]) -> Dict:
    """Normalize a raw model response so downstream metrics never crash.

    A single malformed/empty provider response (e.g. a ``None`` payload after
    an 85s timeout that returned an exception-like result) must not kill an
    entire multi-hundred-probe experiment. This coerces the response into a
    structurally valid dict with a guaranteed string ``output``.

    Provider failures are tagged with ``provider_failure=True`` and a
    ``failure_reason`` so they can be bucketed separately from model
    capability failures (Transition Rule Failure, Semantic Failure, etc.).

    Args:
        response: The raw response dict from a backend, or ``None``.

    Returns:
        A normalized dict. If the input was ``None`` or had a ``None`` output,
        ``output`` is set to ``""`` and provider failure markers are attached.
    """
    if response is None:
        return {
            "output": "",
            "provider_failure": True,
            "failure_reason": "NULL_RESPONSE",
        }

    if not isinstance(response, dict):
        return {
            "output": "",
            "provider_failure": True,
            "failure_reason": "NON_DICT_RESPONSE",
        }

    if response.get("output") is None:
        response["output"] = ""
        # Only mark as provider failure if not already marked
        if "provider_failure" not in response:
            response["provider_failure"] = True
            response["failure_reason"] = "NULL_OUTPUT"

    return response


class ModelEvaluator:
    """Runs inference for ARCUS-X probes via vLLM (local) or an API."""

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        use_vllm: bool = True,
        max_context: Optional[int] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.tokenizer = None
        self.model = None
        self.max_context = max_context or self._get_model_context_limit(self.model_name)
        # Optional hook the runner uses to surface transient status (e.g.
        # rate-limit retries) on its live progress display.
        self.status_callback = status_callback
        logger.info(f"Model context limit for {self.model_name}: {self.max_context} tokens")

        # --- Provider-agnostic API compatibility layer -------------------
        # Best-effort provider slug (logging/attribution only; never drives
        # parameter selection, which is discovered dynamically per request).
        self.provider = detect_provider(api_base, model_name)
        # Shared, process-wide cache of discovered (provider, model) ->
        # token-parameter capabilities. The runner reads this to populate the
        # manifest / raw stream so adaptations are reproducible.
        self.capability_cache: CapabilityCache = SHARED_CAPABILITY_CACHE
        # Adaptation log: records every compatibility change. The runner can
        # attach an ``on_adapt`` callback so each adaptation is written to the
        # raw stream / manifest as it happens.
        self.adaptation_log = AdaptationLog()

        if api_base:
            self.use_vllm = False
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("AZURE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key must be provided via api_key parameter or "
                    "OPENROUTER_API_KEY/AZURE_API_KEY environment variables"
                )
            if HAS_TRANSFORMERS:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                except Exception:
                    logger.warning(f"Could not load tokenizer for {self.model_name}")
            return

        if not HAS_VLLM:
            raise ImportError(
                "vLLM required for local inference. For API-only mode pass api_base="
                "'https://openrouter.ai/api/v1'."
            )

        self.use_vllm = use_vllm
        if self.use_vllm:
            logger.info(f"Initializing vLLM: {model_name} (tp={tensor_parallel_size})")
            self.model = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size,
                             trust_remote_code=True, max_model_len=8192)
            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=8000,
                                                  logprobs=1, include_stop_str_in_output=True)

    # ------------------------------------------------------------------
    # Context limit discovery (network-resilient)
    # ------------------------------------------------------------------
    def _get_model_context_limit(self, model_id: str) -> int:
        fallback_limits = {
            "step-3.7-flash": 256000, "step-1-flash": 128000,
            "gemini-2.5-flash": 1_000_000, "gemini-2.5-pro": 1_000_000,
            "llama-3.1-405b": 131072, "llama-3.1-70b": 131072, "llama-3.1-8b": 131072,
            "deepseek-r1": 64000, "deepseek-r1-distill": 64000,
            "claude-3.5-sonnet": 200000, "gpt-4-turbo": 128000, "gpt-4o": 128000,
            "o3-mini": 128000, "qwen-2.5-72b": 131072, "mixtral-8x22b": 65536,
        }
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    if m.get("id") == model_id and m.get("context_length"):
                        return int(m["context_length"])
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Could not fetch model metadata: {exc}")
        for key, limit in fallback_limits.items():
            if key.lower() in model_id.lower():
                return limit
        return 128000

    def compute_prompt_margin(self, estimated_prompt_tokens: int) -> int:
        return max(512, min(8192, int(estimated_prompt_tokens * 0.10)))

    # ------------------------------------------------------------------
    # Prompt construction (uses the task's own axioms/question)
    # ------------------------------------------------------------------
    def build_prompt(self, probe: Dict) -> str:
        """Build the model prompt from the probe's axioms and question.

        The prompt must never be empty: an empty ``content`` payload is
        rejected by providers with ``INVALID_REQUEST`` ("Input must have at
        least 1 token"). If a probe carries neither axioms nor a question we
        fall back to the task's own environment description (initial state,
        grid, actions, transition rules) -- legitimate task content, never an
        answer, checksum, or difficulty metadata -- and finally to a neutral
        placeholder so the request is always well-formed.
        """
        parts: List[str] = []
        axioms = probe.get("axioms") or []
        if axioms:
            parts.append("Rules:\n" + "\n".join(f"  - {a}" for a in axioms))
        question = probe.get("question")
        if question:
            parts.append(question)

        prompt = "\n\n".join(parts)
        if prompt.strip():
            return prompt

        # --- Empty-prompt guard: synthesize a minimal, fair task prompt from
        # the probe's own environment fields (no answer/metadata leakage). ---
        fallback: List[str] = []
        grid_w = probe.get("grid_width")
        grid_h = probe.get("grid_height")
        initial = probe.get("initial_state")
        actions = probe.get("actions")
        rules = probe.get("transition_rules")
        if grid_w is not None and grid_h is not None and initial is not None:
            fallback.append(
                f"Grid is {grid_w}x{grid_h}. Start at {tuple(initial)}."
            )
        if actions:
            fallback.append("Actions: " + ", ".join(str(a) for a in actions) + ".")
        if rules:
            rule_lines = "\n".join(
                f"  - {label}: dx={v.get('dx')}, dy={v.get('dy')}"
                for label, v in rules.items()
            )
            fallback.append("Transition rules:\n" + rule_lines)
        fallback_prompt = "\n\n".join(fallback)
        if fallback_prompt.strip():
            return fallback_prompt

        # Last-resort neutral placeholder so the API call is never empty.
        return "Follow the task instructions."

    # ------------------------------------------------------------------
    # Public evaluation entry points
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str, max_tokens: Optional[int] = None) -> Dict:
        """Run inference on ``prompt`` and return a raw result dict."""
        if self.use_vllm:
            response = self._query_vllm(prompt, max_tokens)
        else:
            response = self._query_api(prompt, max_tokens)
        return response

    def evaluate_single_probe(self, probe: Dict, tier: int, depth: int = 1) -> Dict:
        """Evaluate one probe at one tier/depth and return a raw result dict.

        The result contains the raw model output and token accounting, but no
        trajectory score (scoring is done downstream by the runner/metrics).

        Latency is recorded as a *systems-level diagnostic only* (it is affected
        by provider infrastructure, batching, and deployment conditions and must
        never be treated as a model capability score). For every probe we record
        ``request_start_time`` (epoch seconds, just before the request is sent),
        ``response_end_time`` (epoch seconds, just after the response is
        received), and ``latency_ms`` (the wall-clock delta in milliseconds).
        """
        payload_depth = probe.get("fol_depth")
        if payload_depth is None:
            raise DepthMismatchError("Probe payload missing fol_depth.")
        if payload_depth != depth:
            raise DepthMismatchError(
                f"Expected depth {depth}, received {payload_depth}."
            )

        prompt = self.build_prompt(probe)
        request_start_time = time.time()
        response = self.evaluate(prompt)
        response_end_time = time.time()
        latency_ms = (response_end_time - request_start_time) * 1000.0

        # Normalize the response so a single malformed/None provider payload
        # (e.g. an exception-like result after a long timeout) cannot crash
        # the whole experiment. Guarantees a string ``output`` downstream.
        response = validate_response(response)

        if response.get("finish_reason") == "length":
            response["output_truncated"] = True
            response["raw_output"] = (
                f"[OUTPUT_TRUNCATED] Partial: {(response.get('output') or '')[-2000:]}"
            )
        else:
            raw_output = response.get("output", "")
            # API errors now return a structured dict instead of a string marker
            # (e.g. {"Type": ..., "Code": ..., "Message": ...}). Coerce to a
            # string so downstream token accounting never receives a dict.
            if not isinstance(raw_output, str):
                raw_output = str(raw_output)
            response["raw_output"] = raw_output

        token_metrics = measure_tokens(
            prompt, response.get("raw_output", ""),
            provider_response=response.get("provider_response"),
            model_name=self.model_name,
        )

        return {
            "probe_id": probe.get("id"),
            "seed": probe.get("seed", 42),
            "gravity": probe.get("gravity_target", 0.0),
            "depth": depth,
            "tier": tier,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": response.get("raw_output", ""),
            "prompt": prompt,
            "thinking_tokens": response.get("thinking_tokens", 0),
            "output_tokens": token_metrics["completion_tokens"],
            "prompt_tokens": token_metrics["prompt_tokens"],
            "completion_tokens": token_metrics["completion_tokens"],
            "total_tokens": token_metrics["total_tokens"],
            "token_source": token_metrics["token_source"],
            "token_confidence": token_metrics["token_confidence"],
            "token_metadata": token_metrics["token_metadata"],
            "context_exhausted": bool(response.get("context_exhausted", False)),
            "finish_reason": response.get("finish_reason", "unknown"),
            "fol_depth": payload_depth,
            # --- Latency diagnostics (systems-level only; never a capability
            # score). Recorded for every probe so downstream reporting can
            # measure verified benchmark performance per unit time. ---
            "request_start_time": request_start_time,
            "response_end_time": response_end_time,
            "latency_ms": latency_ms,
            "input_tokens": token_metrics["prompt_tokens"],
        }

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _query_vllm(self, prompt: str, max_tokens: Optional[int] = None) -> Dict:
        try:
            outputs = self.model.generate([prompt], self.sampling_params)
            text = outputs[0].outputs[0].text
            return {
                "output": text,
                "finish_reason": "stop",
                "thinking_tokens": 0,
                "output_tokens": measure_tokens("", text, model_name=self.model_name)["completion_tokens"],
            }
        except Exception as exc:
            logger.error(f"vLLM error: {exc}")
            return {"output": "[VLLM_ERROR]", "finish_reason": "error",
                    "thinking_tokens": 0, "output_tokens": 0}

    def _query_api(self, prompt: str, max_tokens: Optional[int] = None) -> Dict:
        estimation = self._estimate_tokens_with_safety_buffer(prompt)
        required = estimation["required_context"]
        if required > self.max_context:
            logger.error(f"CONTEXT_LIMIT_EXCEEDED: {required} > {self.max_context}")
            return {"output": "[CONTEXT_LIMIT_EXCEEDED]", "finish_reason": "context_limit",
                    "thinking_tokens": 0, "output_tokens": 0,
                    "context_exhausted": True, **estimation}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        available = self.max_context - required
        cap = max(512, min(available, max_tokens or 8192))

        # --- Dynamic capability negotiation (provider/model-agnostic) ---
        # Seed negotiation state from any capability already discovered for this
        # (provider, model) in the shared cache; otherwise begin with the
        # universal defaults. The negotiation engine infers token-parameter
        # name, temperature/top_p/penalty support, and value constraints from
        # provider responses -- never from a hardcoded model-name table.
        neg = NegotiationState(
            provider=self.provider,
            model=self.model_name,
            cache=self.capability_cache,
        )
        neg.seed_from_cache()

        base_payload = {"model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": DEFAULT_SAMPLING_VALUES["temperature"]}
        caps = get_model_capabilities(self.model_name)
        options = {}
        if caps.supports_reasoning_controls and caps.reasoning_param:
            if caps.reasoning_format is not None:
                options[caps.reasoning_param] = caps.reasoning_format
            else:
                options[caps.reasoning_param] = True

        # Validate and build capability-aware payload
        validate_model_capabilities(self.model_name, options)
        base_payload = build_capability_aware_payload(self.model_name, base_payload, options)

        retry, max_retries = 0, 8
        while retry <= max_retries:
            # Build the payload for the *current* negotiation state. This always
            # strips any previously-attempted token key and drops any optional
            # sampling params we have learned to omit, so we never send two
            # competing parameters in one request.
            payload = neg.build_payload(base_payload, cap)
            try:
                resp = requests.post(f"{self.api_base}/chat/completions", headers=headers,
                                     json=payload, timeout=60)
                if resp.status_code == 429:
                    retry += 1
                    if retry > max_retries:
                        return {"output": "[RATE_LIMIT_EXCEEDED]", "finish_reason": "rate_limit",
                                "thinking_tokens": 0, "output_tokens": 0}
                    wait = min(12, 2 ** (retry - 1))
                    if resp.headers.get("Retry-After"):
                        try:
                            wait = max(wait, int(resp.headers["Retry-After"]))
                        except ValueError:
                            pass
                    # Surface the retry on the runner's live status line when a
                    # callback is wired; otherwise keep the console warning.
                    if self.status_callback:
                        try:
                            self.status_callback(f"Retrying after rate limit ({retry}/8)")
                        except Exception:  # pragma: no cover - defensive
                            pass
                        logger.debug(f"Rate limited (429) - retry {retry}/8 after {wait}s")
                    else:
                        logger.warning(f"Rate limited (429) - retry {retry}/8 after {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    err = self._parse_api_error(resp)
                    logger.error(f"API error {err}")

                    # --- Capability negotiation: recoverable errors ----------
                    # If the provider rejected the request because of an
                    # unsupported parameter or value, adapt and retry. This is
                    # how Azure's gpt-5-mini (which wants max_completion_tokens,
                    # or only supports the default temperature) is handled
                    # without a hardcoded model-name mapping.
                    if self._try_negotiate(neg, err):
                        continue

                    # Non-recoverable (or no further fallback): preserve the full
                    # structured error. Do NOT collapse into a single [API_ERROR].
                    return {
                        "output": {
                            "Type": err.category.value,
                            "Code": err.provider_code or "Unknown",
                            "Message": err.message,
                            "HTTPStatus": err.http_status,
                            "Provider": err.provider,
                            "Model": err.model,
                            "Retryable": err.retryable,
                            "Recoverable": err.recoverable,
                            "Timestamp": err.timestamp,
                            "PayloadFeature": err.payload_feature,
                        },
                        "finish_reason": "error",
                        "thinking_tokens": 0, "output_tokens": 0,
                        "provider_failure": True,
                        "is_infrastructure_failure": True,
                        "api_error": err.to_dict(),
                    }

                # Success: cache the working capability set (even if defaults)
                # so subsequent requests skip discovery.
                self.capability_cache.set(neg.snapshot_record())
                data = resp.json()
                msg = data["choices"][0]["message"]
                return {
                    "output": msg.get("content", ""),
                    "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                    "thinking_tokens": self._extract_thinking_tokens(data),
                    "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "provider_response": data,
                }
            except Exception as exc:
                logger.error(f"API call failed: {exc}")
                return {"output": "[EXCEPTION]", "finish_reason": "error",
                        "thinking_tokens": 0, "output_tokens": 0}
        return {"output": "[MAX_RETRIES_EXCEEDED]", "finish_reason": "error",
                "thinking_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------
    # API error parsing / adaptation helpers
    # ------------------------------------------------------------------
    def _parse_api_error(self, resp) -> APIError:
        """Parse a non-200 ``requests.Response`` into a structured APIError."""
        status = resp.status_code
        try:
            err_json = resp.json()
            err_body = err_json.get("error", err_json)
            if isinstance(err_body, dict):
                provider_code = (
                    err_body.get("code") or err_body.get("error_code")
                    or err_body.get("type") or err_body.get("error_type") or "Unknown"
                )
                message = err_body.get("message") or resp.text[:200]
            else:
                provider_code = "Unknown"
                message = str(err_body)[:200]
        except Exception:
            provider_code = "Unknown"
            message = resp.text[:200]
        return classify_error(
            provider=self.provider,
            model=self.model_name,
            http_status=status,
            provider_code=provider_code,
            message=message,
        )

    def _record_adaptation(self, from_param: str, to_param: str, reason: str = "",
                           kind: str = "parameter_name") -> None:
        """Record (and cache) a token-parameter compatibility adaptation."""
        from arcus.evaluation.api_compat import CompatibilityAdaptation
        self.adaptation_log.record(
            CompatibilityAdaptation(
                provider=self.provider,
                model=self.model_name,
                adjustment=f"{from_param} -> {to_param}",
                reason=reason,
                kind=kind,
            )
        )

    def _try_negotiate(self, neg: "NegotiationState", err: "APIError") -> bool:
        """Attempt to recover from a recoverable API error by adapting params.

        Returns ``True`` if the request should be retried with an adapted
        payload (the negotiation state has been updated), or ``False`` if the
        error is not recoverable / no further adaptation is possible.
        """
        # 1. Unsupported token-parameter *name*: try the next name in the chain.
        if is_unsupported_parameter_error(err):
            alt = next_token_param(neg.token_param)
            if alt is not None:
                self._record_adaptation(
                    neg.token_param, alt,
                    reason=str(err), kind="parameter_name",
                )
                neg.record_token_param_fallback(neg.token_param, alt)
                return True
            return False

        # 2. Unsupported parameter *value*: prefer omitting the parameter to
        #    preserve the provider's default (rather than forcing a value the
        #    provider demands). This covers e.g. temperature=0.0 rejected.
        if is_unsupported_value_error(err):
            offending = err.offending_param
            if offending is None:
                # Cannot attribute the value error to a parameter; give up.
                return False
            if offending in OPTIONAL_SAMPLING_PARAMS:
                if offending not in neg.omitted_params:
                    self._record_adaptation(
                        offending,
                        f"(removed {offending} parameter - provider only supports default)",
                        reason=str(err), kind="parameter_omitted",
                    )
                    neg.record_value_omission(offending)
                    return True
                return False
            # A token-parameter value error: fall back to the next token name.
            alt = next_token_param(neg.token_param)
            if alt is not None:
                self._record_adaptation(
                    neg.token_param, alt,
                    reason=str(err), kind="parameter_name",
                )
                neg.record_token_param_fallback(neg.token_param, alt)
                return True
            return False

        return False

    def _estimate_tokens_with_safety_buffer(self, text: str) -> Dict:
        tm = measure_tokens(text, "", model_name=self.model_name)
        est = max(1, tm["prompt_tokens"])
        margin = self.compute_prompt_margin(est)
        completion = max(1024, min(8192, int(est * 0.15)))
        return {"estimated_prompt_tokens": est, "prompt_margin": margin,
                "expected_completion": completion, "required_context": est + margin + completion}

    def _extract_thinking_tokens(self, response_json: Dict) -> int:
        try:
            msg = response_json["choices"][0]["message"]
            if msg.get("reasoning"):
                return measure_tokens("", msg["reasoning"], model_name=self.model_name)["completion_tokens"]
        except (KeyError, IndexError, TypeError):
            pass
        return 0
