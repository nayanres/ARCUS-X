"""Semantic gravity utilities."""

import logging
import math
from typing import Optional, Dict, List, Tuple, Any
import warnings

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch/Transformers not available - API-only heuristic mode enabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticGravityCalculator:
    """
    Calculates semantic gravity (NLL) of target substrings conditional on prompt context.
    
    Supports two modes:
    1. Local (vLLM/Transformers) - Exact token-level probability extraction
    2. API-Only Heuristic (Windows) - Tier-based fallback for API-only environments
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "auto",
        dtype: str = "float16",
        cache_dir: Optional[str] = None,
        use_vllm: bool = False,
        api_mode: bool = False
    ):
        """
        Initialize the semantic gravity calculator.
        
        Args:
            model_name: HuggingFace model ID to use for probability extraction
            device: Device to load model on ("cuda", "cpu", or "auto")
            dtype: Data type for model weights ("float16", "bfloat16", "float32")
            cache_dir: Directory to cache model weights
            use_vllm: Whether to use vLLM backend
            api_mode: If True, skip local model loading (API-only / Windows mode)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.use_vllm = use_vllm
        self.api_mode = api_mode
        
        self.model = None
        self.tokenizer = None
        self._device_map = None
        
        if not api_mode and HAS_TORCH:
            self._initialize_model()
        else:
            if api_mode:
                logger.info("API-Only Mode: Skipping local model initialization")
                logger.info("Semantic gravity will use tier-based heuristic")
            else:
                logger.warning("PyTorch not available - using tier-based heuristic for G_s")
    
    def _initialize_model(self):
        """Load model and tokenizer."""
        if not HAS_TORCH:
            logger.error("PyTorch not available - cannot load local model")
            return
        
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                add_eos_token=False,
            )
            
            torch_dtype = self._parse_dtype(self.dtype)
            
            if self.device == "auto":
                self._device_map = "auto"
            else:
                self._device_map = self.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                device_map=self._device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if isinstance(self._device_map, str) and self._device_map != "auto":
                self.model = self.model.to(self._device_map)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self._device_map}")
        
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    @staticmethod
    def _parse_dtype(dtype_str: str):
        """Parse dtype string to torch.dtype"""
        if not HAS_TORCH:
            logger.warning("Torch not available - cannot parse dtype")
            return None
        
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "auto": torch.float16,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def calculate_gs(
        self,
        context_text: str,
        target_substring: str,
        batch_size: int = 32,
        tier: int = 1,
        gravity_target: Optional[float] = None
    ) -> float:
        """
        Calculate Semantic Gravity (Negative Log-Likelihood).
        
        Two modes:
        1. Local (if model loaded): Exact NLL of target_substring conditional on context
        2. API-Only Heuristic (if model unavailable): Tier-based or continuous fallback
        
        Formula (local mode): G_s = -sum(log(P(token_i | context))) / N
        Formula (heuristic continuous): G_s = gravity_target if provided, else fallback to discrete mapping
        
        Args:
            context_text: The prompt/context text
            target_substring: The target substring to compute NLL for
            batch_size: Batch size for token processing
            tier: OOD tier (1, 2, or 3) - used in heuristic fallback if gravity_target is absent
            gravity_target: Continuous gravity target float scale
        
        Returns:
            G_s value (Negative Log-Likelihood). Higher values = lower model confidence.
        """
        # API-Only Heuristic Mode (Windows)
        if not self.model or self.api_mode:
            return self._calculate_gs_heuristic(tier, gravity_target)
        
        # Local Mode with torch
        if not HAS_TORCH:
            logger.warning("Torch unavailable - using heuristic fallback")
            return self._calculate_gs_heuristic(tier, gravity_target)
        
        if not context_text or not target_substring:
            logger.warning("Empty context or target provided")
            return float('inf')
        
        try:
            full_text = context_text + target_substring
            
            context_ids = self.tokenizer.encode(context_text, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            target_token_ids = full_ids[len(context_ids):]
            
            if not target_token_ids:
                logger.warning("No tokens in target substring after encoding")
                return float('inf')
            
            log_probs = self._extract_log_probabilities(
                input_ids=full_ids,
                target_start_idx=len(context_ids)
            )
            
            if not log_probs:
                return float('inf')
            
            g_s = -sum(log_probs) / len(log_probs)
            return float(g_s)
        
        except Exception as e:
            logger.error(f"Error calculating gravity: {e}")
            logger.info("Falling back to heuristic G_s")
            return self._calculate_gs_heuristic(tier, gravity_target)
    
    def _calculate_gs_heuristic(self, tier: int, gravity_target: Optional[float] = None) -> float:
        """
        Tier-based or continuous heuristic for Semantic Gravity when local model unavailable.
        """
        if gravity_target is not None:
            return min(5.0, max(0.0, float(gravity_target)))
        return {1: 1.0, 2: 2.5, 3: 4.0}.get(tier, 1.0)
    
    def _extract_log_probabilities(
        self,
        input_ids: List[int],
        target_start_idx: int,
    ) -> List[float]:
        """
        Extract log-probabilities for target tokens conditional on context.
        
        Replicates lm-evaluation-harness token probability extraction methodology.
        Only works when torch and local model are available.
        
        Args:
            input_ids: Full input token IDs (context + target)
            target_start_idx: Index where target tokens begin
        
        Returns:
            List of log-probabilities for each target token
        """
        if not HAS_TORCH or not self.model:
            logger.warning("Torch or model unavailable - cannot extract log probabilities")
            return []
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        elif isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, return_dict=True)
                logits = outputs.logits
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            return []
        
        log_probs = []
        
        for token_idx in range(target_start_idx, input_ids.shape[1] - 1):
            logits_at_position = logits[0, token_idx - 1, :]
            
            log_softmax = F.log_softmax(logits_at_position, dim=-1)
            
            next_token_id = input_ids[0, token_idx]
            token_log_prob = log_softmax[next_token_id].item()
            
            log_probs.append(token_log_prob)
        
        return log_probs
    
    def calculate_gs_batch(
        self,
        context_target_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Calculate Semantic Gravity for multiple context-target pairs.
        
        Args:
            context_target_pairs: List of (context, target) tuples
        
        Returns:
            List of G_s values, one per pair
        """
        results = []
        for context, target in context_target_pairs:
            try:
                gs_value = self.calculate_gs(context, target)
                results.append(gs_value)
            except Exception as e:
                logger.warning(f"Failed to calculate gravity for pair: {e}")
                results.append(float('inf'))
        
        return results
    
    def get_token_probabilities(
        self,
        context_text: str,
        target_substring: str
    ) -> Dict[str, any]:
        """
        Get detailed token-level probability information.
        
        Returns:
            Dictionary containing:
            - token_ids: List of target token IDs
            - tokens: List of decoded tokens
            - log_probs: List of log-probabilities
            - probs: List of probabilities (exponentiated)
            - mean_log_prob: Average log-probability
            - mean_prob: Average probability
        """
        if not context_text or not target_substring:
            return {}
        
        try:
            full_text = context_text + target_substring
            
            context_ids = self.tokenizer.encode(context_text, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            target_token_ids = full_ids[len(context_ids):]
            
            log_probs = self._extract_log_probabilities(
                input_ids=full_ids,
                target_start_idx=len(context_ids)
            )
            
            if not log_probs:
                return {}
            
            tokens = [self.tokenizer.decode([tid]) for tid in target_token_ids[:-1]]
            probs = [math.exp(lp) for lp in log_probs]
            
            return {
                "token_ids": target_token_ids[:-1],
                "tokens": tokens,
                "log_probs": log_probs,
                "probs": probs,
                "mean_log_prob": sum(log_probs) / len(log_probs) if log_probs else 0.0,
                "mean_prob": sum(probs) / len(probs) if probs else 0.0,
                "semantic_gravity_gs": -sum(log_probs) / len(log_probs) if log_probs else float('inf'),
            }
        
        except Exception as e:
            logger.error(f"Error extracting token probabilities: {e}")
            return {}
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Model memory cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class EscapeVelocityCalculator:
    """
    Calculates Escape Velocity (V_e): minimum reasoning tokens needed to override pre-training priors.
    
    Escape Velocity is the critical threshold where:
      V_e = min{|T_think| | (ΔAcc / ΔG_s) >= -ε}
    
    Measures how many internal reasoning tokens a model needs to escape semantic gravity
    and apply counter-contextual logical rules instead of pre-training priors.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Convergence threshold for accuracy change vs gravity change
        """
        self.epsilon = epsilon
    
    def calculate_ve(
        self,
        tier_results: Dict[int, Dict[str, Any]]
    ) -> Tuple[float, int]:
        """
        Calculate Escape Velocity from tier-based evaluation results.
        
        Args:
            tier_results: Dict mapping tier number to {accuracy, gravity, thinking_tokens}
        
        Returns:
            (escape_velocity, tier_breakthrough): The G_s threshold and tier level where model escapes
        """
        if not tier_results or len(tier_results) < 2:
            return float('inf'), -1
        
        sorted_tiers = sorted(tier_results.keys())
        min_ve = float('inf')
        breakthrough_tier = -1
        
        for i in range(len(sorted_tiers) - 1):
            tier_k = sorted_tiers[i]
            tier_k_plus_1 = sorted_tiers[i + 1]
            
            acc_k = tier_results[tier_k].get('accuracy', 0.0)
            acc_k_plus_1 = tier_results[tier_k_plus_1].get('accuracy', 0.0)
            delta_acc = acc_k_plus_1 - acc_k
            
            g_s_k = tier_results[tier_k].get('semantic_gravity', 1.0)
            g_s_k_plus_1 = tier_results[tier_k_plus_1].get('semantic_gravity', 1.0)
            delta_gs = g_s_k_plus_1 - g_s_k
            
            if abs(delta_gs) < 1e-6:
                continue
            
            ratio = delta_acc / delta_gs
            
            if ratio >= -self.epsilon:
                thinking_tokens = tier_results[tier_k_plus_1].get('thinking_tokens', 0)
                if thinking_tokens < min_ve:
                    min_ve = thinking_tokens
                    breakthrough_tier = tier_k_plus_1
        
        return min_ve if min_ve != float('inf') else 0.0, breakthrough_tier
    
    def analyze_reasoning_hysteresis(
        self,
        token_trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect Reasoning Hysteresis: repeated waffling between abstract rules and pre-training priors.
        
        Args:
            token_trace: List of token generation decisions with their reasoning states
        
        Returns:
            Hysteresis metrics including cycle count and waffling intensity
        """
        if not token_trace or len(token_trace) < 2:
            return {"hysteresis_detected": False, "cycles": 0}
        
        cycles = 0
        previous_commitment = None
        cycle_count = 0
        
        for token_info in token_trace:
            current_commitment = token_info.get('reasoning_mode')  # 'abstract' or 'prior'
            
            if previous_commitment and current_commitment != previous_commitment:
                if current_commitment == previous_commitment:
                    cycles += 1
                    cycle_count += 1
            previous_commitment = current_commitment
        
        return {
            "hysteresis_detected": cycles > 0,
            "cycle_count": cycle_count,
            "waffling_intensity": cycle_count / max(len(token_trace), 1),
            "token_trace_length": len(token_trace)
        }


# Convenience function for single-use calculations
def calculate_semantic_gravity(
    context_text: str,
    target_substring: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device: str = "auto",
    dtype: str = "float16"
) -> float:
    """
    Convenience function to calculate semantic gravity for a single pair.
    
    Args:
        context_text: The prompt/context text
        target_substring: The target substring
        model_name: HuggingFace model ID
        device: Device to load model on
        dtype: Data type for model
    
    Returns:
        Semantic Gravity (G_s) value
    """
    calculator = SemanticGravityCalculator(
        model_name=model_name,
        device=device,
        dtype=dtype
    )
    try:
        return calculator.calculate_gs(context_text, target_substring)
    finally:
        calculator.cleanup()


if __name__ == "__main__":
    context = "The capital of France is"
    target = " Paris"
    
    calculator = SemanticGravityCalculator(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="auto",
        dtype="float16"
    )
    
    gs_value = calculator.calculate_gs(context, target)
    logger.info(f"Semantic Gravity for '{context}{target}': {gs_value:.4f}")
    
    token_info = calculator.get_token_probabilities(context, target)
    logger.info(f"Token details: {token_info}")
    
    calculator.cleanup()
