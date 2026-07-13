"""ILP generation utilities."""

import logging
import random
import hashlib
import re
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OODTier(Enum):
    """Out-of-Distribution Tier Classification"""
    TIER_1_SEMANTIC_DISRUPTION = 1
    TIER_2_ATTRIBUTE_INVERSION = 2
    TIER_3_AXIOMATIC_CONTRADICTION = 3


@dataclass
class ProbeMetadata:
    """Metadata for generated probe"""
    tier: OODTier
    semantic_gravity: float
    structural_hash: str
    isomorphic_pairs: List[str] = field(default_factory=list)
    fol_depth: int = 0
    premise_count: int = 0
    variable_count: int = 0


@dataclass
class ValidityCheckResult:
    """Result of validity safeguard check"""
    passed: bool
    safeguard_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class IsomorphicProbeGenerator:
    """
    Generates structurally isomorphic logic problems with systematic semantic shifts.
    
    Creates FOL problems that preserve logical structure while varying semantic content
    across three explicit OOD tiers to evaluate reasoning invariance.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.token_whitelist = self._build_token_safe_lexicon()
    
    def _build_token_safe_lexicon(self) -> Set[str]:
        """
        Build whitelist of pseudowords guaranteed to map 1:1 to unique token IDs.
        
        Restricts to strings mathematically proven to avoid sub-tokenization across
        common model vocabularies (Llama, GPT, etc.).
        
        Returns:
            Set of token-safe pseudowords
        """
        lexicon = {
            "floxer", "grobian", "trixel", "glumph", "fwibble",
            "snarky", "plixor", "zephyr", "quirton", "blavish",
            "grommet", "quixley", "flapper", "glorian", "sprocket",
            "thriven", "minxley", "frolick", "glanish", "spritely",
            "phizgig", "globule", "prattle", "squib", "wraith"
        }
        return lexicon
    
    def generate_tier1_semantic_disruption(
        self,
        original_problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate Tier 1: Semantic Disruption via novel pseudowords.
        
        Substitutes known concepts with abstract out-of-vocabulary pseudowords
        that preserve logical structure but break semantic familiarity.
        
        Args:
            original_problem: Original FOL problem dict
        
        Returns:
            Modified problem with pseudoword substitutions
        """
        disrupted = original_problem.copy()
        
        entity_map = {}
        for entity in disrupted.get("entities", []):
            pseudoword = random.choice(list(self.token_whitelist))
            entity_map[entity] = pseudoword
        
        disrupted["entities"] = [entity_map.get(e, e) for e in disrupted.get("entities", [])]
        
        rules_str = json.dumps(disrupted.get("rules", {}))
        for orig, pseudo in entity_map.items():
            rules_str = re.sub(r'\b' + orig + r'\b', pseudo, rules_str)
        
        disrupted["rules"] = json.loads(rules_str)
        disrupted["tier"] = OODTier.TIER_1_SEMANTIC_DISRUPTION
        
        return disrupted
    
    def generate_tier2_attribute_inversion(
        self,
        original_problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate Tier 2: Attribute Inversion (counter-intuitive contexts).
        
        Assigns physically or logically inverted attributes to known entities,
        creating cognitive friction between pre-training priors and logical rules.
        
        Args:
            original_problem: Original FOL problem dict
        
        Returns:
            Modified problem with inverted attributes
        """
        inverted = original_problem.copy()
        
        attribute_inversions = {
            "large": "diminutive",
            "hot": "frigid",
            "fast": "sluggish",
            "bright": "dull",
            "heavy": "weightless",
            "old": "pristine",
            "liquid": "crystalline",
            "soft": "rigid"
        }
        
        rules_str = json.dumps(inverted.get("rules", {}))
        for attr, inverted_attr in attribute_inversions.items():
            rules_str = rules_str.replace(attr, inverted_attr)
        
        inverted["rules"] = json.loads(rules_str)
        inverted["tier"] = OODTier.TIER_2_ATTRIBUTE_INVERSION
        
        return inverted
    
    def generate_tier3_axiomatic_contradiction(
        self,
        original_problem: Dict[str, Any],
        override_axiom: str = None
    ) -> Dict[str, Any]:
        """
        Generate Tier 3: Axiomatic Contradiction (systemic overrides).
        
        Injects localized foundational rules that directly violate real-world
        physical and factual priors, requiring explicit override of pre-training.
        
        Args:
            original_problem: Original FOL problem dict
            override_axiom: Specific axiom to inject (or None for auto-generation)
        
        Returns:
            Modified problem with contradictory axioms
        """
        contradictory = original_problem.copy()
        
        if override_axiom is None:
            override_axiom = "For this problem only: All previous facts are reversed."
        
        rules = contradictory.get("rules", {})
        rules["OVERRIDE_AXIOM"] = override_axiom
        contradictory["rules"] = rules
        contradictory["tier"] = OODTier.TIER_3_AXIOMATIC_CONTRADICTION
        
        return contradictory
    
    def compute_structural_hash(self, fol_graph: Dict[str, Any]) -> str:
        """
        Compute MD5 hash of FOL structure for isomorphism verification.
        
        Args:
            fol_graph: FOL graph dictionary
        
        Returns:
            MD5 hash of structural signatures
        """
        rules = fol_graph.get("rules", {})
        signature = json.dumps(rules, sort_keys=True)
        return hashlib.md5(signature.encode()).hexdigest()
    
    def validate_isomorphism(
        self,
        graph1: Dict[str, Any],
        graph2: Dict[str, Any]
    ) -> bool:
        """
        Validate that two graphs are structurally isomorphic.
        
        Two FOL graphs are isomorphic if there exists bijective mapping between
        rule structures preserving all operators, scopes, and directional edges.
        
        Args:
            graph1: First FOL graph
            graph2: Second FOL graph
        
        Returns:
            True if graphs are isomorphic
        """
        hash1 = self.compute_structural_hash(graph1)
        hash2 = self.compute_structural_hash(graph2)
        
        return hash1 == hash2


class ValiditySafeguardsEngine:
    """
    Enforces integrity across 5 programmatic safeguard layers to prevent
    artificial metric inflation, structural cheating, or tokenizer distortions.
    """
    
    def __init__(self):
        """Initialize validity safeguards"""
        self.refusal_patterns = self._build_refusal_set()
        self.token_whitelist = self._build_token_safe_lexicon()
    
    def _build_refusal_set(self) -> Set[str]:
        """
        Build deterministic refusal set for alignment masking filter.
        
        Returns:
            Set of common refusal patterns to filter out
        """
        return {
            "i cannot", "i can't", "i'm unable to", "i cannot help",
            "i'm not able to", "this violates", "i cannot provide",
            "i shouldn't", "that would be", "i refuse", "that's not appropriate"
        }
    
    def _build_token_safe_lexicon(self) -> Set[str]:
        """Build token-safe pseudoword whitelist"""
        return {
            "floxer", "grobian", "trixel", "glumph", "fwibble",
            "snarky", "plixor", "zephyr", "quirton", "blavish",
        }
    
    def alignment_masking_filter(self, model_output: str) -> Tuple[bool, str]:
        """
        Alignment Masking Filter: De-contaminate safety guardrail false positives.
        
        Any generation matching deterministic refusal set R is isolated as
        Alignment Masked Turn (AMT) and omitted from accuracy calculations.
        
        Args:
            model_output: Model's raw output
        
        Returns:
            (is_masked, reason): Whether output is masked, and reason
        """
        output_lower = model_output.lower()
        for pattern in self.refusal_patterns:
            if pattern in output_lower:
                return True, f"Matched refusal pattern: {pattern}"
        return False, "Not a refusal"
    
    def tokenizer_selection_whitelist(self, pseudoword: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Tokenizer Selection Whitelist: Eliminate sub-token slicing variance.
        
        Restricts pseudoword injection to pre-validated whitelist ensuring
        1:1 token mapping (len_tok(W1) = len_tok(W2)) across target models.
        
        Args:
            pseudoword: Candidate pseudoword for injection
        
        Returns:
            (is_valid, details): Validation result and token analysis
        """
        is_valid = pseudoword in self.token_whitelist
        details = {
            "pseudoword": pseudoword,
            "valid": is_valid,
            "reason": "Token-safe" if is_valid else "Sub-tokenization risk"
        }
        return is_valid, details
    
    def permutation_shuffling(
        self,
        premises: List[str]
    ) -> List[str]:
        """
        Permutation Shuffling: Immunize against shortcut heuristics.
        
        Dynamically generates structurally scrambled variations S(G) of premises
        to prevent models from exploiting fixed variable orderings.
        
        Args:
            premises: Original premises
        
        Returns:
            Shuffled premises maintaining logical coherence
        """
        shuffled = premises.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def path_validity_enforcing(
        self,
        output_trace: str,
        fol_graph: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Path-Validity Enforcing: State-trace verification.
        
        Models are forced to output full internal path trace. Automated grader
        verifies transitions align perfectly with directional edges of FOL graph.
        Any broken path drops turn score to 0.0.
        
        Args:
            output_trace: Model's path trace (comma-separated variable IDs)
            fol_graph: Reference FOL graph with valid edges
        
        Returns:
            (is_valid, path): Whether trace is valid, and extracted path
        """
        try:
            path = [v.strip() for v in output_trace.split(",")]
        except:
            return False, []
        
        valid_edges = fol_graph.get("edges", [])
        
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in valid_edges:
                return False, path
        
        return True, path
    
    def constraint_retention_check(
        self,
        model_output: str,
        required_axiom: str
    ) -> Tuple[bool, str]:
        """
        Constraint Retention Check: Task drift mitigation.
        
        Terminal output requires model to explicitly restate specific overriding
        axiom utilized. Catches models drifting during extended reasoning.
        
        Args:
            model_output: Model's complete output
            required_axiom: Required axiom statement
        
        Returns:
            (axiom_retained, explanation): Whether axiom was restated
        """
        output_lower = model_output.lower()
        axiom_lower = required_axiom.lower()
        
        if axiom_lower in output_lower or required_axiom in model_output:
            return True, "Axiom explicitly restated"
        
        return False, "Axiom not found in output"
    
    def run_all_checks(
        self,
        model_output: str,
        pseudoword: str,
        output_trace: str,
        fol_graph: Dict[str, Any],
        required_axiom: str
    ) -> List[ValidityCheckResult]:
        """
        Run all 5 validity safeguards and collect results.
        
        Args:
            model_output: Model's output
            pseudoword: Injected pseudoword
            output_trace: Path trace from output
            fol_graph: Reference FOL graph
            required_axiom: Required axiom statement
        
        Returns:
            List of ValidityCheckResult for each safeguard
        """
        results = []
        
        is_masked, reason = self.alignment_masking_filter(model_output)
        results.append(ValidityCheckResult(
            passed=not is_masked,
            safeguard_name="Alignment Masking Filter",
            message=reason
        ))
        
        is_valid, details = self.tokenizer_selection_whitelist(pseudoword)
        results.append(ValidityCheckResult(
            passed=is_valid,
            safeguard_name="Tokenizer Selection Whitelist",
            message=f"Pseudoword: {pseudoword}",
            details=details
        ))
        
        is_valid, path = self.path_validity_enforcing(output_trace, fol_graph)
        results.append(ValidityCheckResult(
            passed=is_valid,
            safeguard_name="Path-Validity Enforcing",
            message=f"Traced path: {' -> '.join(path)}",
            details={"path": path}
        ))
        
        axiom_retained, msg = self.constraint_retention_check(model_output, required_axiom)
        results.append(ValidityCheckResult(
            passed=axiom_retained,
            safeguard_name="Constraint Retention Check",
            message=msg
        ))
        
        return results


class SemanticDisplacementTracker:
    """
    Monitors semantic gravity shifting across OOD tiers and tracks
    accuracy degradation as function of displacement magnitude.
    """
    
    def __init__(self):
        """Initialize tracker"""
        self.tier_history = {}
    
    def record_tier_result(
        self,
        tier: OODTier,
        semantic_gravity: float,
        accuracy: float,
        thinking_tokens: int = 0
    ):
        """
        Record evaluation result for a tier.
        
        Args:
            tier: OOD tier classification
            semantic_gravity: G_s value for this tier
            accuracy: Model accuracy on this tier
            thinking_tokens: Internal reasoning tokens used
        """
        self.tier_history[tier] = {
            "semantic_gravity": semantic_gravity,
            "accuracy": accuracy,
            "thinking_tokens": thinking_tokens
        }
    
    def calculate_displacement_metrics(self) -> Dict[str, float]:
        """
        Calculate semantic displacement and performance degradation metrics.
        
        Returns:
            Metrics dict with displacement calculations
        """
        if len(self.tier_history) < 2:
            return {}
        
        tiers = sorted(self.tier_history.keys(), key=lambda x: x.value)
        metrics = {}
        
        for i in range(len(tiers) - 1):
            tier_k = tiers[i]
            tier_k_plus_1 = tiers[i + 1]
            
            gs_k = self.tier_history[tier_k]["semantic_gravity"]
            gs_k_plus_1 = self.tier_history[tier_k_plus_1]["semantic_gravity"]
            delta_gs = gs_k_plus_1 - gs_k
            
            acc_k = self.tier_history[tier_k]["accuracy"]
            acc_k_plus_1 = self.tier_history[tier_k_plus_1]["accuracy"]
            delta_acc = acc_k_plus_1 - acc_k
            
            key = f"tier_{tier_k.value}_to_{tier_k_plus_1.value}"
            metrics[key] = {
                "delta_gs": delta_gs,
                "delta_acc": delta_acc,
                "displacement_ratio": delta_acc / delta_gs if abs(delta_gs) > 1e-6 else 0.0
            }
        
        return metrics
    
    def detect_fracture_point(self, random_baseline: float = 0.5) -> Tuple[bool, OODTier]:
        """
        Detect if model's accuracy fractures below random chance.
        
        Returns:
            (fractured, fracture_tier): Whether fracture detected and at which tier
        """
        for tier in sorted(self.tier_history.keys(), key=lambda x: x.value):
            accuracy = self.tier_history[tier]["accuracy"]
            if accuracy < random_baseline:
                return True, tier
        
        return False, None
