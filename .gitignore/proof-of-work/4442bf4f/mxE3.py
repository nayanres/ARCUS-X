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
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import requests

from arcus.evaluation.token_accounting import measure_tokens

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
        """Build the model prompt from the probe's axioms and question."""
        parts: List[str] = []
        axioms = probe.get("axioms") or []
        if axioms:
            parts.append("Rules:\n" + "\n".join(f"  - {a}" for a in axioms))
        question = probe.get("question")
        if question:
            parts.append(question)
        return "\n\n".join(parts)

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
        """
        payload_depth = probe.get("fol_depth")
        if payload_depth is None:
            raise DepthMismatchError("Probe payload missing fol_depth.")
        if payload_depth != depth:
            raise DepthMismatchError(
                f"Expected depth {depth}, received {payload_depth}."
            )

        prompt = self.build_prompt(probe)
        response = self.evaluate(prompt)

        if response.get("finish_reason") == "length":
            response["output_truncated"] = True
            response["raw_output"] = (
                f"[OUTPUT_TRUNCATED] Partial: {response.get('output', '')[-2000:]}"
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
        payload = {"model": self.model_name,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": cap}
        if "deepseek" in self.model_name.lower():
            payload["include_reasoning"] = True
        elif "o3" in self.model_name.lower() or "o1" in self.model_name.lower():
            payload["reasoning_effort"] = "medium"
        elif "claude" in self.model_name.lower():
            payload["thinking"] = {"type": "enabled", "budget_tokens": 5000}

        retry, max_retries = 0, 8
        while retry <= max_retries:
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
                    logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                    try:
                        err_json = resp.json()
                        err_body = err_json.get("error", err_json)
                        # Expanded error key pattern matching
                        err_type = err_body.get("type") or err_body.get("error_type") or err_body.get("code") or err_body.get("error_code") or "Unknown"
                        err_code = err_body.get("code") or err_body.get("error_code") or err_body.get("type") or err_body.get("error_type") or "Unknown"
                        err_message = err_body.get("message", resp.text[:200])
                    except Exception:
                        err_type = "Unknown"
                        err_code = "Unknown"
                        err_message = resp.text[:200]
                    return {
                        "output": {
                            "Type": f"{resp.status_code} {resp.reason}",
                            "Code": err_code,
                            "Message": err_message,
                        },
                        "finish_reason": "error",
                        "thinking_tokens": 0, "output_tokens": 0,
                    }
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
