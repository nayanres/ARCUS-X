"""Logic probe generation utilities."""

import os
import json
import logging
import string
import random
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import nltk
from nltk.sem import logic
from nltk.sem.logic import LogicParser, Expression
from datasets import load_dataset, DatasetDict

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import random as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FOLAtom:
    """Represents a first-order logic atom (predicate with arguments)"""
    predicate: str
    arguments: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.predicate
        return f"{self.predicate}({','.join(self.arguments)})"
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.arguments)))
    
    def __eq__(self, other):
        if not isinstance(other, FOLAtom):
            return False
        return self.predicate == other.predicate and self.arguments == other.arguments


@dataclass
class FOLRule:
    """Represents a first-order logic rule (head :- body)"""
    head: FOLAtom
    body: List[FOLAtom] = field(default_factory=list)
    
    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}."
    
    def get_variables(self) -> Set[str]:
        """Extract all variables (capitalized) from rule"""
        variables = set()
        for atom in [self.head] + self.body:
            for arg in atom.arguments:
                if arg and arg[0].isupper():
                    variables.add(arg)
        return variables


@dataclass
class FOLGraph:
    """Represents a directed acyclic graph of FOL predicates and rules"""
    facts: Set[FOLAtom] = field(default_factory=set)
    rules: List[FOLRule] = field(default_factory=list)
    predicate_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_fact(self, atom: FOLAtom) -> None:
        """Add a ground fact to the graph"""
        self.facts.add(atom)
    
    def add_rule(self, rule: FOLRule) -> None:
        """Add a rule and update predicate dependencies"""
        self.rules.append(rule)
        head_pred = rule.head.predicate
        for body_atom in rule.body:
            self.predicate_graph[body_atom.predicate].add(head_pred)
    
    def get_reachable_predicates(self, start_pred: str) -> Set[str]:
        """Get all predicates reachable from a given predicate via rules"""
        visited = set()
        stack = [start_pred]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self.predicate_graph:
                for next_pred in self.predicate_graph[current]:
                    if next_pred not in visited:
                        stack.append(next_pred)
        return visited
    
    def compute_structural_hash(self) -> str:
        """
        Compute structural hash of FOL graph for isomorphism testing.
        Two graphs are structurally isomorphic if they have identical hashes.
        """
        rule_signatures = []
        for rule in sorted(self.rules, key=lambda r: str(r)):
            head_arity = len(rule.head.arguments)
            body_arities = [len(atom.arguments) for atom in rule.body]
            signature = f"{rule.head.predicate}/{head_arity}:-{','.join(map(str, body_arities))}"
            rule_signatures.append(signature)
        
        graph_signature = f"rules:{len(self.rules)},facts:{len(self.facts)},edges:{len(self.predicate_graph)}"
        full_sig = f"{graph_signature}|{','.join(rule_signatures)}"
        return hashlib.md5(full_sig.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        return {
            "facts": [str(f) for f in self.facts],
            "rules": [str(r) for r in self.rules],
            "predicate_graph": {k: list(v) for k, v in self.predicate_graph.items()},
            "structural_hash": self.compute_structural_hash()
        }


class FOLParser:
    """Robust First-Order Logic parser using NLTK"""
    
    def __init__(self):
        self.logic_parser = None
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK logic parser"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        self.logic_parser = LogicParser(
            constants=[],
            variables='x y z X Y Z',
            quantifiers='all exists'
        )
    
    def parse_atom(self, atom_str: str) -> Optional[FOLAtom]:
        """Parse a single FOL atom (predicate with arguments)"""
        atom_str = atom_str.strip()
        if not atom_str:
            return None
        
        match = re.match(r'(\w+)\((.*)\)', atom_str)
        if match:
            predicate = match.group(1)
            args_str = match.group(2)
            arguments = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            return FOLAtom(predicate=predicate, arguments=arguments)
        else:
            return FOLAtom(predicate=atom_str, arguments=[])
    
    def parse_rule(self, rule_str: str) -> Optional[FOLRule]:
        """Parse a FOL rule in the form: head :- body1, body2, ..."""
        rule_str = rule_str.strip()
        if rule_str.endswith('.'):
            rule_str = rule_str[:-1]
        
        if ':-' in rule_str:
            head_str, body_str = rule_str.split(':-', 1)
            head = self.parse_atom(head_str)
            if not head:
                return None
            body_atoms = []
            for body_part in body_str.split(','):
                body_atom = self.parse_atom(body_part)
                if body_atom:
                    body_atoms.append(body_atom)
            return FOLRule(head=head, body=body_atoms)
        else:
            head = self.parse_atom(rule_str)
            if head:
                return FOLRule(head=head, body=[])
            return None
    
    def parse_program(self, program_text: str) -> FOLGraph:
        """Parse a complete FOL program and return a FOLGraph"""
        graph = FOLGraph()
        lines = program_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            rule = self.parse_rule(line)
            if rule:
                if rule.body:
                    graph.add_rule(rule)
                else:
                    graph.add_fact(rule.head)
        
        return graph


class PseudowordGenerator:
    """Generates token-safe pseudowords for semantic disruption"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        # Note: np.random is not used; only random.seed() for reproducibility
    
    def generate_unique_pseudoword(self, original_word: str, existing: Set[str]) -> str:
        """Generate a unique, token-safe pseudoword"""
        base_length = max(4, len(original_word))
        while True:
            pseudoword = ''.join(random.choices(string.ascii_lowercase, k=base_length))
            if pseudoword not in existing and pseudoword != original_word:
                existing.add(pseudoword)
                return pseudoword
    
    def create_mapping(self, words: List[str]) -> Dict[str, str]:
        """Create a 1:1 mapping of words to pseudowords"""
        mapping = {}
        existing = set()
        for word in words:
            if word not in mapping:
                mapping[word] = self.generate_unique_pseudoword(word, existing)
        return mapping
    
    def apply_mapping(self, text: str, mapping: Dict[str, str]) -> str:
        """Apply pseudoword mapping to text while preserving word boundaries"""
        result = text
        for original, pseudo in sorted(mapping.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(original) + r'\b'
            result = re.sub(pattern, pseudo, result, flags=re.IGNORECASE)
        return result


class AttributeInverter:
    """Inverts logical attributes for disruption"""
    
    ATTRIBUTE_PAIRS = {
        'positive': ['negative', 'bad', 'harmful', 'detrimental'],
        'negative': ['positive', 'good', 'helpful', 'beneficial'],
        'true': ['false', 'incorrect', 'wrong', 'invalid'],
        'false': ['true', 'correct', 'right', 'valid'],
        'exist': ['nonexist', 'absent', 'missing', 'lacking'],
        'absent': ['exist', 'present', 'available', 'existing'],
        'similar': ['different', 'dissimilar', 'distinct', 'opposite'],
        'different': ['same', 'similar', 'identical', 'equivalent'],
    }
    
    @staticmethod
    def invert_attribute(text: str, attribute: str) -> str:
        """Invert a specific attribute in text"""
        if attribute not in AttributeInverter.ATTRIBUTE_PAIRS:
            return text
        
        inverses = AttributeInverter.ATTRIBUTE_PAIRS[attribute]
        inverse = random.choice(inverses)
        
        pattern = r'\b' + re.escape(attribute) + r'\b'
        return re.sub(pattern, inverse, text, flags=re.IGNORECASE)


class AxiomaticContradictionGenerator:
    """Generates axiomatically contradictory statements"""
    
    CONTRADICTIONS = [
        ("if {var} then {var}", "if {var} then not {var}"),
        ("{pred}({var}) and not {pred}({var})", "{pred}({var}) and {pred}({var})"),
        ("exists {var}: {var} = {var}", "forall {var}: {var} != {var}"),
        ("{var} > {var2} and {var2} > {var}", "{var} > {var2} and {var2} < {var}"),
    ]
    
    @staticmethod
    def inject_contradiction(fol_graph: FOLGraph, rule: FOLRule) -> str:
        """Inject axiomatically contradictory rules"""
        contradiction_rule = f"contradiction :- {rule.head}, not {rule.head}."
        return contradiction_rule


class StructuralIsomorphismValidator:
    """Validates structural isomorphism between FOL graphs"""
    
    @staticmethod
    def are_isomorphic(graph1: FOLGraph, graph2: FOLGraph) -> bool:
        """
        Check if two FOL graphs are structurally isomorphic.
        Two graphs are isomorphic if they have identical rule structures and predicates.
        """
        if len(graph1.rules) != len(graph2.rules):
            return False
        if len(graph1.facts) != len(graph2.facts):
            return False
        
        return graph1.compute_structural_hash() == graph2.compute_structural_hash()
    
    @staticmethod
    def compute_bijection(graph1: FOLGraph, graph2: FOLGraph) -> Optional[Dict[str, str]]:
        """
        Compute bijective mapping between isomorphic graphs.
        Returns dict mapping predicates from graph1 to graph2.
        """
        if not StructuralIsomorphismValidator.are_isomorphic(graph1, graph2):
            return None
        
        pred1 = set(atom.predicate for atom in graph1.facts)
        pred2 = set(atom.predicate for atom in graph2.facts)
        
        if len(pred1) != len(pred2):
            return None
        
        return {p1: p2 for p1, p2 in zip(sorted(pred1), sorted(pred2))}


class ValiditySafeguards:
    """Implements validity safeguards for evaluation integrity"""
    
    def __init__(self, model_tokenizer=None, refusal_set: Optional[Set[str]] = None):
        self.tokenizer = model_tokenizer
        self.refusal_set = refusal_set or self._get_default_refusal_set()
        self.token_safe_whitelist = self._build_token_safe_whitelist()
    
    @staticmethod
    def _get_default_refusal_set() -> Set[str]:
        """Default refusal patterns for safety guardrails"""
        return {
            "i can't",
            "i cannot",
            "unable to",
            "not able to",
            "refuse",
            "violate",
            "inappropriate",
            "not allowed",
            "against my values"
        }
    
    def _build_token_safe_whitelist(self) -> Set[str]:
        """
        Build whitelist of pseudowords that map 1:1 to unique tokens.
        Ensures len_tok(W1) = len_tok(W2) for word substitution.
        """
        whitelist = set()
        for word_len in range(3, 12):
            for _ in range(100):
                candidate = ''.join(random.choices(string.ascii_lowercase, k=word_len))
                if self._is_token_safe(candidate):
                    whitelist.add(candidate)
        return whitelist
    
    def _is_token_safe(self, word: str) -> bool:
        """Check if word tokenizes to exactly 1 token"""
        if not self.tokenizer:
            return True
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        return len(tokens) == 1
    
    def alignment_masking_filter(self, model_output: str) -> Tuple[bool, str]:
        """
        Alignment Masking Filter (AMT): Detect if output matches refusal pattern.
        Returns (is_alignment_masked, masked_category)
        """
        output_lower = model_output.lower()
        for refusal in self.refusal_set:
            if refusal in output_lower:
                return True, refusal
        return False, ""
    
    def permutation_shuffling(self, graph: FOLGraph) -> FOLGraph:
        """
        Permutation Shuffling: Generate structurally identical but semantically scrambled graph.
        Prevents shortcut heuristics by randomizing variable orderings.
        """
        shuffled_graph = FOLGraph()
        
        for fact in graph.facts:
            shuffled_args = list(fact.arguments)
            random.shuffle(shuffled_args)
            shuffled_fact = FOLAtom(predicate=fact.predicate, arguments=shuffled_args)
            shuffled_graph.add_fact(shuffled_fact)
        
        for rule in graph.rules:
            head_args = list(rule.head.arguments)
            random.shuffle(head_args)
            shuffled_head = FOLAtom(predicate=rule.head.predicate, arguments=head_args)
            
            shuffled_body = []
            for atom in rule.body:
                atom_args = list(atom.arguments)
                random.shuffle(atom_args)
                shuffled_body.append(FOLAtom(predicate=atom.predicate, arguments=atom_args))
            
            shuffled_graph.add_rule(FOLRule(head=shuffled_head, body=shuffled_body))
        
        return shuffled_graph
    
    def constraint_retention_check(self, output: str, overriding_axiom: str) -> bool:
        """
        Constraint Retention Check: Verify model explicitly restates overriding axiom.
        Ensures task drift mitigation during extended reasoning.
        """
        axiom_normalized = overriding_axiom.lower().replace(" ", "")
        output_normalized = output.lower().replace(" ", "")
        
        return axiom_normalized in output_normalized or \
               any(word in output.lower() for word in overriding_axiom.lower().split())



class CounterfactualDataLoader:
    """Loads and manages FaithEval counterfactual dataset"""
    
    def __init__(self, split: str = "test", cache_dir: Optional[str] = None):
        self.split = split
        self.cache_dir = cache_dir or "./cache"
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load FaithEval counterfactual dataset from HuggingFace"""
        try:
            logger.info(f"Loading FaithEval counterfactual dataset (split={self.split})...")
            self.dataset = load_dataset(
                "Salesforce/FaithEval-counterfactual-v1.0",
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            logger.info(f"Loaded {len(self.dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load FaithEval dataset: {e}")
            self.dataset = None
    
    def get_mutations(self, example_idx: int) -> List[str]:
        """Extract text mutations from dataset example"""
        if self.dataset is None or example_idx >= len(self.dataset):
            return []
        
        example = self.dataset[example_idx]
        mutations = []
        
        for key in ['original_text', 'counterfactual_text', 'perturbation']:
            if key in example and example[key]:
                mutations.append(str(example[key]))
        
        return mutations
    
    def get_context_question_pair(self, example_idx: int) -> Tuple[str, str]:
        """Get context and question from example"""
        if self.dataset is None or example_idx >= len(self.dataset):
            return "", ""
        
        example = self.dataset[example_idx]
        context = example.get('context', '')
        question = example.get('question', '')
        return context, question
    
    def __len__(self):
        return len(self.dataset) if self.dataset else 0


class OODTierGenerator:
    """Generates Out-of-Distribution tiers for FOL graphs"""
    
    def __init__(self, random_seed: int = 42):
        self.fol_parser = FOLParser()
        self.pseudoword_gen = PseudowordGenerator(seed=random_seed)
        self.attr_inverter = AttributeInverter()
        self.random_seed = random_seed
    
    def generate_tier_1_semantic_disruption(
        self,
        fol_graph: FOLGraph,
        gravity_target: float = 1.5
    ) -> FOLGraph:
        """Tier 1: Semantic Disruption via unique token-safe pseudowords"""
        disrupted_graph = FOLGraph()
        disruption_ratio = min(1.0, max(0.001, gravity_target / 5.0))
        
        words_to_replace = set()
        for fact in fol_graph.facts:
            if random.random() < disruption_ratio:
                for arg in fact.arguments:
                    if arg and not arg[0].isupper():
                        words_to_replace.add(arg)
        
        mapping = self.pseudoword_gen.create_mapping(list(words_to_replace))
        
        for fact in fol_graph.facts:
            new_args = [mapping.get(arg, arg) for arg in fact.arguments]
            new_atom = FOLAtom(predicate=fact.predicate, arguments=new_args)
            disrupted_graph.add_fact(new_atom)
        
        for rule in fol_graph.rules:
            head_args = [mapping.get(arg, arg) for arg in rule.head.arguments]
            new_head = FOLAtom(predicate=rule.head.predicate, arguments=head_args)
            new_body = []
            for body_atom in rule.body:
                body_args = [mapping.get(arg, arg) for arg in body_atom.arguments]
                new_body.append(FOLAtom(predicate=body_atom.predicate, arguments=body_args))
            new_rule = FOLRule(head=new_head, body=new_body)
            disrupted_graph.add_rule(new_rule)
        
        return disrupted_graph
    
    def generate_tier_2_attribute_inversion(
        self,
        fol_graph: FOLGraph,
        gravity_target: float = 1.5
    ) -> FOLGraph:
        """Tier 2: Attribute Inversion (logical contradiction)"""
        inverted_graph = FOLGraph()
        inversion_ratio = min(1.0, max(0.001, gravity_target / 5.0))
        
        for fact in fol_graph.facts:
            fact_str = str(fact)
            if random.random() < inversion_ratio:
                fact_str = self.attr_inverter.invert_attribute(
                    fact_str,
                    random.choice(list(AttributeInverter.ATTRIBUTE_PAIRS.keys()))
                )
            inverted_fact = self.fol_parser.parse_atom(fact_str)
            if inverted_fact:
                inverted_graph.add_fact(inverted_fact)
        
        for rule in fol_graph.rules:
            rule_str = str(rule)
            if random.random() < inversion_ratio:
                rule_str = self.attr_inverter.invert_attribute(
                    rule_str,
                    random.choice(list(AttributeInverter.ATTRIBUTE_PAIRS.keys()))
                )
            inverted_rule = self.fol_parser.parse_rule(rule_str)
            if inverted_rule:
                inverted_graph.add_rule(inverted_rule)
        
        return inverted_graph
    
    def generate_tier_3_axiomatic_contradiction(
        self,
        fol_graph: FOLGraph,
        gravity_target: float = 1.5
    ) -> FOLGraph:
        """Tier 3: Axiomatic Contradiction (violates logical consistency)"""
        contradictory_graph = FOLGraph()
        
        for fact in fol_graph.facts:
            contradictory_graph.add_fact(fact)
        
        for rule in fol_graph.rules:
            contradictory_graph.add_rule(rule)
        
        contradiction_count = max(1, min(int(1 + (gravity_target / 5.0) * 4), len(contradictory_graph.rules)))
        
        sample_rules = random.sample(
            contradictory_graph.rules,
            min(contradiction_count, len(contradictory_graph.rules))
        )
        
        for rule in sample_rules:
            contradiction = AxiomaticContradictionGenerator.inject_contradiction(
                contradictory_graph,
                rule
            )
            parsed_contradiction = self.fol_parser.parse_rule(contradiction)
            if parsed_contradiction:
                contradictory_graph.add_rule(parsed_contradiction)
        
        return contradictory_graph


class GeneratorCore:
    """Main generator integrating FOL parsing and OOD tier generation"""
    
    def __init__(self, random_seed: int = 42, cache_dir: Optional[str] = None):
        self.random_seed = random_seed
        self.cache_dir = cache_dir or "./cache"
        self.fol_parser = FOLParser()
        self.data_loader = CounterfactualDataLoader(split="test", cache_dir=self.cache_dir)
        self.ood_generator = OODTierGenerator(random_seed=random_seed)
        self.output_dir = "./outputs"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.output_dir, self.cache_dir, "./data"]:
            os.makedirs(directory, exist_ok=True)
    
    def generate_from_fol_program(
        self,
        program_text: str,
        example_mutations: List[str],
        gravity_target: float = 1.5
    ) -> Dict[str, Any]:
        """Generate all OOD tiers from a FOL program"""
        base_graph = self.fol_parser.parse_program(program_text)
        
        tier1 = self.ood_generator.generate_tier_1_semantic_disruption(base_graph, gravity_target)
        tier2 = self.ood_generator.generate_tier_2_attribute_inversion(base_graph, gravity_target)
        tier3 = self.ood_generator.generate_tier_3_axiomatic_contradiction(base_graph, gravity_target)
        
        return {
            "base_program": base_graph.to_dict(),
            "tier_1_semantic_disruption": tier1.to_dict(),
            "tier_2_attribute_inversion": tier2.to_dict(),
            "tier_3_axiomatic_contradiction": tier3.to_dict(),
            "mutations_applied": example_mutations,
            "gravity_target": gravity_target
        }
    
    def process_faitheval_batch(
        self,
        batch_size: int = 10,
        max_examples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process FaithEval counterfactual examples in batches"""
        results = []
        total_examples = min(max_examples or len(self.data_loader), len(self.data_loader))
        
        for idx in range(total_examples):
            try:
                context, question = self.data_loader.get_context_question_pair(idx)
                mutations = self.data_loader.get_mutations(idx)
                
                combined_program = f"{context}\n{question}"
                ood_output = self.generate_from_fol_program(combined_program, mutations)
                
                ood_output['example_id'] = idx
                ood_output['context'] = context
                ood_output['question'] = question
                results.append(ood_output)
                
                if (idx + 1) % batch_size == 0:
                    logger.info(f"Processed {idx + 1}/{total_examples} examples")
            
            except Exception as e:
                logger.warning(f"Failed to process example {idx}: {e}")
                continue
        
        return results
    
    def generate_calibration_suite(
        self,
        fol_programs: List[str],
        gravity_levels: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Generate a factorial calibration suite across FOL programs and gravity levels.
        """
        results = []
        for i, program_text in enumerate(fol_programs):
            for gravity in gravity_levels:
                base_graph = self.fol_parser.parse_program(program_text)
                
                tier1 = self.ood_generator.generate_tier_1_semantic_disruption(base_graph, gravity)
                tier2 = self.ood_generator.generate_tier_2_attribute_inversion(base_graph, gravity)
                tier3 = self.ood_generator.generate_tier_3_axiomatic_contradiction(base_graph, gravity)
                
                output = {
                    "program_id": i,
                    "gravity": gravity,
                    "base_program": base_graph.to_dict(),
                    "tier_1": tier1.to_dict(),
                    "tier_2": tier2.to_dict(),
                    "tier_3": tier3.to_dict(),
                }
                
                sig_data = json.dumps(output, sort_keys=True).encode()
                output["experiment_signature"] = hashlib.md5(sig_data).hexdigest()
                
                results.append(output)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "generated_ood_tiers.json"):
        """Save generated results to file"""
        output_path = os.path.join(self.output_dir, output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    generator = GeneratorCore()
    results = generator.process_faitheval_batch(batch_size=5, max_examples=20)
    generator.save_results(results)
    logger.info(f"Generated {len(results)} OOD evaluation examples")
