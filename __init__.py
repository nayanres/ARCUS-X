"""
AI Evaluation Benchmark Framework

A production-grade framework integrating:
- First-Order Logic (FOL) parsing and graph generation
- FaithEval counterfactual dataset integration
- Out-of-Distribution (OOD) tier generation
- Semantic gravity calculation (NLL-based model confidence)
- Path trace validation metrics
- Composite robustness index aggregation

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ML Infrastructure Team"
__description__ = "Production-grade AI evaluation benchmark framework"

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from generator import (
    GeneratorCore,
    FOLParser,
    FOLGraph,
    FOLRule,
    FOLAtom,
    OODTierGenerator,
    CounterfactualDataLoader,
    PseudowordGenerator,
    AttributeInverter,
    AxiomaticContradictionGenerator,
)

from ilp import (
    IsomorphicProbeGenerator,
    ValiditySafeguardsEngine,
    SemanticDisplacementTracker,
    OODTier,
    ProbeMetadata,
    ValidityCheckResult,
)

from gravity import (
    SemanticGravityCalculator,
    calculate_semantic_gravity,
    EscapeVelocityCalculator,
)

from metrics import (
    PathTraceValidityMetric,
    PathTraceExtractor,
    FOLGraphValidator,
    PathTraceResult,
    RiemannSumAggregator,
    calculate_riemann_cri,
    EvaluationMetricsFactory,
    GenerationInefficiencyIndex,
    SyllogisticLeakageScore,
    StructuralYieldPoint,
)

try:
    from model_evaluation import ModelEvaluator, BatchEvaluator, EvaluationResult
    from benchmark_runner import BenchmarkRunner
    from fracture_finder import FracturePointFinder
    from cross_model_validator import CrossModelValidator
    from statistics import StatisticalAnalysis, BenchmarkStatistics
except ImportError:
    # Optional components for full benchmark execution
    pass

__all__ = [
    # Framework components
    "GeneratorCore",
    "FOLParser",
    "FOLGraph",
    "FOLRule",
    "FOLAtom",
    "OODTierGenerator",
    "CounterfactualDataLoader",
    "PseudowordGenerator",
    "AttributeInverter",
    "AxiomaticContradictionGenerator",
    "SemanticGravityCalculator",
    "EscapeVelocityCalculator",
    "calculate_semantic_gravity",
    "PathTraceValidityMetric",
    "PathTraceExtractor",
    "FOLGraphValidator",
    "PathTraceResult",
    "RiemannSumAggregator",
    "calculate_riemann_cri",
    "EvaluationMetricsFactory",
    "GenerationInefficiencyIndex",
    "SyllogisticLeakageScore",
    "StructuralYieldPoint",
    "IsomorphicProbeGenerator",
    "ValiditySafeguardsEngine",
    "SemanticDisplacementTracker",
    "OODTier",
    "ProbeMetadata",
    "ValidityCheckResult",
    # Benchmark execution components
    "ModelEvaluator",
    "BatchEvaluator",
    "EvaluationResult",
    "BenchmarkRunner",
    "FracturePointFinder",
    "CrossModelValidator",
    "StatisticalAnalysis",
    "BenchmarkStatistics",
]

logger.info(f"AI Evaluation Framework v{__version__} loaded")
