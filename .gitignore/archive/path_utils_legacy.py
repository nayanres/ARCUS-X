"""Path generation utilities."""
from typing import List, Dict
import re

def extract_model_path(raw_output: str) -> List[str]:
    """Extracts path coordinates from model output using improved regex."""
    matches = re.findall(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', raw_output)
    return [f'[{m[0]},{m[1]}]' for m in matches]

def generate_transition_rule(gravity: float) -> Dict[str, int]:
    """Generates the transition rules based on semantic gravity."""
    stride_noise = int(gravity * 0.1) % 3
    return {
        "even_dx": 2 + stride_noise,
        "even_dy": 1 + stride_noise,
        "odd_dx": 1 + stride_noise,
        "odd_dy": 3 + stride_noise,
        "stride_noise": stride_noise
    }

def generate_expected_path(target_z: int, grid_dim: int, gravity: float = 0.0) -> List[str]:
    """
    Generates the canonical expected path for a given depth, grid dimension, and gravity.
    Must be used by both the prompt builder and the validator.
    """
    variables = []
    cx, cy = 0, 0
    
    rules = generate_transition_rule(gravity)
    
    for i in range(target_z):
        variables.append(f"[{cx},{cy}]")
        
        # State-machine transition matrix based on step parity and gravity noise
        if target_z <= 2:
            # Unit strides (Manhattan distance of 1)
            stride_noise = rules["stride_noise"]
            if i % 2 == 0:
                cx = (cx + 1 + stride_noise) % grid_dim
            else:
                cy = (cy + 1 + stride_noise) % grid_dim
        else:
            # Original logic modified by gravity noise
            if i % 2 == 0:
                cx = (cx + rules["even_dx"]) % grid_dim
                cy = (cy + rules["even_dy"]) % grid_dim
            else:
                cx = (cx + rules["odd_dx"]) % grid_dim
                cy = (cy + rules["odd_dy"]) % grid_dim
                
    return variables

def estimated_optimal_path_length(z: int) -> int:
    """Estimates optimal token count for path of depth z."""
    # Based on: [0,0]->[x,y]... + checksum
    # Rough estimate: ~10 tokens per coordinate pair, z+1 pairs + overhead
    return 10 * (z + 1) + 10
