"""Model evaluation and API helpers."""

import os
import json
import requests
import time
import copy
from typing import Dict, List, Optional
import logging
import hashlib

from token_accounting import measure_tokens

# Optional imports - wrapped to work in API-only mode on Windows
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    logging.warning("vLLM not available - API-only mode enabled")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available - tokenizer approximation only")

logger = logging.getLogger(__name__)


class DepthMismatchError(Exception):
    """Exception raised when probe depth does not match expected depth."""
    pass


class ModelEvaluator:
    """
    Evaluates models on probes via vLLM (local) or API (OpenRouter/Azure).
    
    Three execution modes:
    1. vLLM local (open-weight, tensor-parallel GPU)
    2. OpenRouter API (closed-weight & reasoning models)
    3. Azure API (custom endpoints)
    
    Handles context window exhaustion, reasoning token extraction, and model-specific parameters.
    """
    
    def __init__(self, 
                 model_name: str, 
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 use_vllm: bool = True):
        """Initialize evaluator with API or local vLLM backend.
        
        API-Only Mode (Windows):
          If api_base is provided, skips vLLM entirely and uses API backend only.
          This allows the framework to work on Windows with only OpenRouter/Azure.
          
        Args:
            model_name: Model identifier
            api_base: API endpoint URL (enables API-only mode)
            api_key: API authentication key (overrides environment variables)
            tensor_parallel_size: For vLLM: number of GPUs
            use_vllm: Whether to use vLLM (if not using API)
        """
        self.model_name = model_name
        self.api_base = api_base
        self.tokenizer = None
        self.model = None
        
        self.max_context = self._get_model_context_limit(self.model_name)
        logger.info(f"Model context limit for {self.model_name}: {self.max_context} tokens")
        
        if api_base:
            self.use_vllm = False
            logger.info(f"API-Only Mode: Using {api_base}")
            logger.info(f"Model: {model_name}")
            
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("AZURE_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided via api_key parameter or OPENROUTER_API_KEY/AZURE_API_KEY environment variables")
            
            if HAS_TRANSFORMERS:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                except:
                    logger.warning(f"Could not load tokenizer for {model_name}")
            return
        
        if not HAS_VLLM:
            logger.error("vLLM not installed. For local inference, install: pip install vllm torch")
            logger.error("For API-only mode on Windows, use: api_base='https://openrouter.ai/api/v1'")
            raise ImportError("vLLM required for local inference. Use api_base parameter for API-only mode.")
        
        self.use_vllm = use_vllm and not api_base
        
        if self.use_vllm:
            logger.info(f"Initializing vLLM: {model_name} (tensor_parallel_size={tensor_parallel_size})")
            try:
                self.model = LLM(
                    model=model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    trust_remote_code=True,
                    max_model_len=8192
                )
                self.sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=8000,
                    logprobs=1,
                    include_stop_str_in_output=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}")
                raise
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY or AZURE_API_KEY must be set")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except:
                logger.warning(f"Could not load tokenizer for {model_name}, using approximation")
                self.tokenizer = None
    
    def _get_model_context_limit(self, model_id: str) -> int:
        """
        Fetch live model metadata from OpenRouter API to determine context limits.
        Provides resilient fallback for key architectures if network request fails.
        
        This ensures fair evaluation: models with different context windows are 
        evaluated fairly relative to their structural limits.
        
        Args:
            model_id: Model identifier (e.g., 'deepseek/deepseek-r1')
            
        Returns:
            Maximum context limit in tokens (int)
        """
        fallback_limits = {
            'step-3.7-flash': 256000,          # StepFun Flash MoE
            'step-1-flash': 128000,             # StepFun Flash (original)
            'gemini-2.5-flash': 1000000,       # Gemini MoE super-long
            'gemini-2.5-pro': 1000000,         # Gemini Pro
            'llama-3.1-405b': 131072,          # Llama 3.1 biggest
            'llama-3.1-70b': 131072,           # Llama 3.1 70B
            'llama-3.1-8b': 131072,            # Llama 3.1 8B
            'deepseek-r1': 64000,              # DeepSeek R1 reasoning
            'deepseek-r1-distill': 64000,      # DeepSeek R1 distill
            'claude-3.5-sonnet': 200000,       # Claude 3.5 Sonnet
            'gpt-4-turbo': 128000,             # GPT-4 Turbo
            'gpt-4o': 128000,                  # GPT-4o
            'o3-mini': 128000,                 # O3 Mini reasoning
            'qwen-2.5-72b': 131072,            # Qwen 2.5 72B
            'mixtral-8x22b': 65536,            # Mixtral 8x22B
        }
        
        try:
            logger.info(f"Fetching context limit for {model_id} from OpenRouter...")
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=5
            )
            
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                for model in models_data:
                    if model.get("id") == model_id:
                        context_length = model.get("context_length")
                        if context_length:
                            logger.info(f"✓ Fetched context limit: {context_length} tokens")
                            return context_length
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch model metadata from OpenRouter: {e}")
        
        for key, limit in fallback_limits.items():
            if key.lower() in model_id.lower():
                logger.info(f"✓ Using fallback context limit: {limit} tokens (matched '{key}')")
                return limit
        
        default_limit = 128000
        logger.warning(f"Unknown model {model_id}, using conservative default: {default_limit} tokens")
        return default_limit
    
    def compute_prompt_margin(self, estimated_prompt_tokens: int) -> int:
        """
        Margin for tokenizer estimation error.
        """
        return max(
            512,                       # minimum cushion
            min(
                8192,                  # don't waste context
                int(estimated_prompt_tokens * 0.10)
            )
        )

    def _estimate_tokens_with_safety_buffer(self, text: str, model_id: str) -> Dict[str, int]:
        """
        Estimate token requirements for the prompt using the canonical token accounting module.
        
        Returns:
            Dict containing:
                - estimated_prompt_tokens
                - prompt_margin
                - expected_completion
                - required_context
        """
        token_metrics = measure_tokens(text, "", model_name=model_id)
        estimated_prompt_tokens = max(1, token_metrics["prompt_tokens"])

        prompt_margin = self.compute_prompt_margin(estimated_prompt_tokens)

        expected_completion = max(
            1024,
            min(
                8192,
                int(estimated_prompt_tokens * 0.15)
            )
        )

        required_context = estimated_prompt_tokens + prompt_margin + expected_completion

        logger.debug(
            f"Token estimation: prompt={estimated_prompt_tokens} + margin={prompt_margin} + completion={expected_completion} = required={required_context}"
        )

        return {
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "prompt_margin": prompt_margin,
            "expected_completion": expected_completion,
            "required_context": required_context
        }

    def evaluate_single_probe(self, probe: Dict, tier: int, depth: int = 1) -> Dict:
        """
        Evaluate one probe at one OOD tier with isolated payload per depth level.
        
        Args:
            probe: Test case probe
            tier: OOD tier (1, 2, or 3)
            depth: Current logical depth (for context isolation)
            
        Returns: {accuracy, semantic_gravity, thinking_tokens, output_tokens, context_exhausted, ...}
        """
        # Validate depth integrity
        payload_depth = probe.get("fol_depth")
        if payload_depth is None:
            raise DepthMismatchError("Integrity violation: Probe payload missing fol_depth.")
        
        if payload_depth != depth:
            raise DepthMismatchError(
                f"Integrity violation: Expected depth {depth}, received {payload_depth}."
            )

        # Deep copy probe to prevent mutation across depth evaluations
        probe_copy = copy.deepcopy(probe)
        
        # Step 1: Build natural language prompt from probe + tier
        prompt = self._build_prompt(probe_copy, tier)
        
        # --- INJECTED DIAGNOSTIC LOGGING ---
        # Requirement 4: Inject an Invariance Runtime Logger Block
        z = depth
        gravity = probe_copy.get("gravity_target", 0.0)
        
        prompt_snippet = prompt[:400] if isinstance(prompt, str) else "NON-STRING"
        logger.info(f"=== PARAMETER INVARIANCE VERIFICATION ===")
        logger.info(f"Target Config -> Horizon: {z} | Gravity: {gravity} | Tier: {tier}")
        logger.info(f"Prompt Snapshot (First 400 Chars):\n{prompt_snippet}")
        logger.info(f"=========================================")
        
        logger.debug(f"[Depth {depth}] Evaluating probe with prompt length: {len(prompt)} chars")
        
        # Step 2: Query model via appropriate backend
        if self.use_vllm:
            response = self._query_vllm(prompt)
        else:
            response = self._query_api(prompt, tier)
        
        # Ensure schema integrity for failure cases
        response["fol_depth"] = probe_copy.get("fol_depth")
        
        # Step 3: Check for output truncation
        if response.get("finish_reason") == "length":
            logger.warning(f"[Depth {depth}] Output truncated.")
            response["output_truncated"] = True
            response["raw_output"] = f"[OUTPUT_TRUNCATED] Partial: {response.get('output', '')[-2000:]}"
        
        # Step 3b: Check for safety tripwire trigger (fairness validation)
        if response.get("safety_tripwire_triggered"):
            logger.warning(
                f"[Depth {depth}] Safety tripwire triggered - context limit exceeded "
                f"({response.get('estimated_tokens')} > {response.get('model_limit')} tokens)"
            )
            return {
                "accuracy": 0.0,
                "semantic_gravity": 0.0,
                "thinking_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw_output": "[SAFETY_TRIPWIRE_TRIGGERED]",
                "prompt": prompt,
                "context_exhausted": True,
                "failure_reason": "safety_tripwire_context_protection",
                "finish_reason": "context_limit",
                "depth": depth,
                "fol_depth": probe_copy.get("fol_depth"),
                "safety_details": {
                    "estimated_tokens": response.get("estimated_tokens"),
                    "safety_buffer": response.get("safety_buffer"),
                    "model_limit": response.get("model_limit")
                }
            }
        
        # Step 4: Extract output and calculate metrics
        raw_output = response.get("output", "")
        terminal_accuracy = self._extract_accuracy(raw_output, probe_copy)
        semantic_gravity = self._calculate_semantic_gravity(raw_output, tier)

        token_metrics = measure_tokens(
            prompt,
            raw_output,
            provider_response=response.get("provider_response"),
            model_name=self.model_name,
        )

        logger.info("Token Accounting")
        logger.info(f"Prompt Tokens: {token_metrics['prompt_tokens']}")
        logger.info(f"Completion Tokens: {token_metrics['completion_tokens']}")
        logger.info(f"Total Tokens: {token_metrics['total_tokens']}")
        logger.info(f"Source: {token_metrics['token_source']}")
        logger.info(f"Confidence: {token_metrics['token_confidence']}")
        logger.info(f"Metadata: {token_metrics['token_metadata']}")
        
        return {
            "probe_id": probe_copy.get("id"),
            "seed": probe_copy.get("seed", 42),
            "gravity": probe_copy.get("gravity_target", 0.0),
            "depth": depth,
            "tier": tier,
            "prompt_hash": hash(prompt),
            "expected_answer": probe_copy.get("expected_answer"),
            "generated_depth": depth, # Placeholder if not calculated
            "terminal_accuracy": terminal_accuracy,
            "semantic_gravity": semantic_gravity,
            "thinking_tokens": response.get("thinking_tokens", 0),
            "output_tokens": token_metrics["completion_tokens"],
            "prompt_tokens": token_metrics["prompt_tokens"],
            "completion_tokens": token_metrics["completion_tokens"],
            "total_tokens": token_metrics["total_tokens"],
            "token_source": token_metrics["token_source"],
            "token_confidence": token_metrics["token_confidence"],
            "token_metadata": token_metrics["token_metadata"],
            "raw_output": raw_output,
            "prompt": prompt,
            "context_exhausted": False,
            "finish_reason": response.get("finish_reason", "unknown"),
            "fol_depth": probe_copy.get("fol_depth")
        }
    
    def _compile_prompt_buffer(self, probe: Dict) -> str:
        gravity_target = probe.get("gravity_target")
        if gravity_target is not None:
            return f"\n[SYSTEM NOTICE: SEMANTIC ABSTRACTION LAYER ACTIVE]\nEvaluation Abstraction Rigor Index: {float(gravity_target):.2f}.\nNote: Context transformation noise scales continuously with this index.\n"
        return ""

    def _build_prompt(self, probe: Dict, tier: int) -> str:
        """Build natural language prompt from structured probe."""
        # Start with basic reasoning task
        prompt = "Reason step-by-step about the following:\n\n"
        
        # Add probe content (axioms, question)
        if "axioms" in probe:
            prompt += "Axioms:\n"
            for axiom in probe["axioms"]:
                prompt += f"  - {axiom}\n"
        
        if "question" in probe:
            prompt += f"Question: {probe['question']}\n"
        
        # Apply tier-specific mutations (OOD injection)
        if tier == 2:
            prompt += "\n[Attribute Inversion Applied]"
        elif tier == 3:
            prompt += "\n[Axiomatic Contradiction Applied]"
        
        prompt += self._compile_prompt_buffer(probe)
        
        prompt += "\nProvide your reasoning and final answer."
        
        return prompt
    
    def _extract_accuracy(self, output: str, probe: Dict) -> float:
        """Extract accuracy from model output by comparing against expected answer."""
        if not output or "[ERROR]" in output or "[CONTEXT_LIMIT]" in output:
            return 0.0
        
        expected = probe["expected_answer"].lower()
        output_lower = output.lower()
        
        # Simple string matching (can be extended with semantic similarity)
        if expected and expected in output_lower:
            return 1.0
        
        return 0.0
    
    def _calculate_semantic_gravity(self, output: str, tier: int) -> float:
        """Calculate semantic gravity (heuristic without local tokenizer)."""
        # Fallback gravity tiers when tokenizer unavailable
        tier_gravity_map = {
            1: 1.0,      # Tier 1: Light OOD (semantic disruption)
            2: 2.5,      # Tier 2: Medium OOD (attribute inversion)
            3: 4.0       # Tier 3: Heavy OOD (axiomatic contradiction)
        }
        return tier_gravity_map.get(tier, 1.0)
    
    def _extract_thinking_tokens(self, response_json: Dict) -> int:
        """Extracts normalized reasoning tokens from OpenRouter API response."""
        try:
            message = response_json["choices"][0]["message"]
            if "reasoning" in message and message["reasoning"]:
                thinking_text = message["reasoning"]
                return measure_tokens("", thinking_text, model_name=self.model_name)["completion_tokens"]
        except (KeyError, IndexError, TypeError):
            pass
        return 0
    
    def _query_vllm(self, prompt: str) -> Dict:
        """Query local vLLM backend."""
        try:
            outputs = self.model.generate(
                [prompt],
                self.sampling_params
            )
            
            output_text = outputs[0].outputs[0].text
            finish_reason = "stop" if len(outputs) > 0 else "unknown"
            
            # Extract token-level log probabilities (if available)
            logprobs_list = []
            if hasattr(outputs[0].outputs[0], 'logprobs') and outputs[0].outputs[0].logprobs:
                logprobs_list = outputs[0].outputs[0].logprobs
            
            return {
                "output": output_text,
                "finish_reason": finish_reason,
                "logprobs": logprobs_list,
                "thinking_tokens": 0,  # vLLM doesn't expose internal reasoning
                "output_tokens": measure_tokens("", output_text, model_name=self.model_name)["completion_tokens"]
            }
        except Exception as e:
            logger.error(f"vLLM error: {e}")
            return {
                "output": "[VLLM_ERROR]",
                "finish_reason": "error",
                "thinking_tokens": 0,
                "output_tokens": 0
            }

    def _query_api(self, prompt: str, tier: int) -> Dict:
        """
        Query OpenRouter/Azure API with reasoning model support.
        
        Implements:
        - Exponential backoff retry for 429 (rate limit) errors
        - Dynamic context limit validation (fairness across architectures)
        - Deep copy of payloads to prevent context accumulation leaks
        - Token estimation with safety buffers
        
        CRITICAL: Only include parameters supported by OpenRouter/Azure.
        Do NOT include vLLM-specific parameters like logprobs, top_k, top_p, etc.
        """
        # Estimate tokens BEFORE building payload (fairness check)
        estimation = self._estimate_tokens_with_safety_buffer(
            prompt,
            self.model_name
        )
        required_context = estimation["required_context"]
        
        if required_context > self.max_context:
            logger.error(
                f"CONTEXT_LIMIT_EXCEEDED: required_context ({required_context}) > "
                f"model context limit ({self.max_context}). "
                f"Details: {estimation}"
            )
            return {
                "output": "[CONTEXT_LIMIT_EXCEEDED]",
                "finish_reason": "context_limit",
                "thinking_tokens": 0,
                "output_tokens": 0,
                "context_exhausted": True,
                "context_exceeded": True,
                **estimation
            }
        
        logger.info(f"Payload preflight: {estimation}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        available_output = self.max_context - required_context
        max_tokens = max(512, min(available_output, 8192))
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],  # Fresh copy per depth
            "temperature": 0.0,
            "max_tokens": max_tokens
        }
        
        payload = copy.deepcopy(payload)
        
        if "deepseek" in self.model_name.lower():
            payload["include_reasoning"] = True
        elif "o3" in self.model_name.lower() or "o1" in self.model_name.lower():
            payload["reasoning_effort"] = "medium"
        elif "claude" in self.model_name.lower():
            payload["thinking"] = {"type": "enabled", "budget_tokens": 5000}
        
        # ============================================
        # Retry Loop (429 Handling)
        # ============================================
        max_retries = 8
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Handle 429 (Rate Limit) with exponential backoff
                if response.status_code == 429:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Rate limited (429) - max retries ({max_retries}) exceeded")
                        return {
                            "output": "[RATE_LIMIT_EXCEEDED]",
                            "finish_reason": "rate_limit",
                            "thinking_tokens": 0,
                            "output_tokens": 0
                        }
                    
                    wait_time = min(12, 2 ** (retry_count - 1))
                    retry_after = response.headers.get("Retry-After")
                    
                    if retry_after:
                        try:
                            wait_time = max(wait_time, int(retry_after))
                        except ValueError:
                            pass
                    
                    logger.warning(f"Rate limited (429) - Retry {retry_count}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    continue  # Retry the request
                
                # Handle other non-200 status codes
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text[:200]}")
                    if response.status_code == 400:
                        logger.error(f"Bad Request - Check payload compatibility with {self.model_name}")
                    return {
                        "output": "[API_ERROR]",
                        "finish_reason": "error",
                        "thinking_tokens": 0,
                        "output_tokens": 0
                    }
                
                # Success - parse response
                response_json = response.json()
                finish_reason = response_json["choices"][0].get("finish_reason", "stop")
                
                return {
                    "output": response_json["choices"][0]["message"]["content"],
                    "finish_reason": finish_reason,
                    "thinking_tokens": self._extract_thinking_tokens(response_json),
                    "output_tokens": response_json.get("usage", {}).get("completion_tokens", 0),
                    "provider_response": response_json,
                    "reasoning_content": response_json["choices"][0]["message"].get("reasoning", "")
                }
            
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return {
                    "output": "[EXCEPTION]",
                    "finish_reason": "error",
                    "thinking_tokens": 0,
                    "output_tokens": 0
                }
        
        return {
            "output": "[MAX_RETRIES_EXCEEDED]",
            "finish_reason": "error",
            "thinking_tokens": 0,
            "output_tokens": 0
        }
