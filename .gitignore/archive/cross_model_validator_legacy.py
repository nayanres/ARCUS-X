"""Cross-model validator."""

import logging
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
import numpy as np

from arcus.analysis.gravity import DifficultyConfig
from model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class CrossModelValidator:
    """
    Validates cross-model proxy invariance of semantic gravity.
    """
    
    def validate_proxy_invariance(self,
                                  models: Optional[List[str]] = None,
                                  n_test_cases: int = 50) -> Dict:
        """
        Evaluate whether Llama-3-8B G_s correlates with other models.
        
        Success criteria: Pearson r > 0.8 for all model pairs
        """
        
        if models is None:
            models = ["meta-llama/Meta-Llama-3-8B"]
        
        logger.info(f"Validating cross-model invariance on {len(models)} models")
        logger.info(f"Test cases: {n_test_cases}")
        
        test_probes = self._generate_test_probes(n_test_cases)
        logger.info(f"Generated {len(test_probes)} test probes")
        
        gs_values = {}
        for model_name in models:
            logger.info(f"  Calculating G_s for {model_name}...")
            gs_values[model_name] = []
            
            try:
                calc = DifficultyConfig()
                
                for i, probe in enumerate(test_probes):
                    if (i + 1) % 10 == 0:
                        logger.info(f"    Progress: {i+1}/{len(test_probes)}")
                    
                    gs = calc.calculate_gs(
                        probe.get("context", ""),
                        probe.get("target", "")
                    )
                    gs_values[model_name].append(gs)
            except Exception as e:
                logger.error(f"Failed to calculate G_s for {model_name}: {e}")
                gs_values[model_name] = [0.0] * len(test_probes)
        
        correlations = {}
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                try:
                    # Pearson correlation
                    r_pearson, p_pearson = pearsonr(
                        gs_values[model1],
                        gs_values[model2]
                    )
                    
                    # Spearman correlation (ranks)
                    rho_spearman, p_spearman = spearmanr(
                        gs_values[model1],
                        gs_values[model2]
                    )
                    
                    correlations[f"{model1}_vs_{model2}"] = {
                        "pearson_r": r_pearson,
                        "pearson_p": p_pearson,
                        "spearman_rho": rho_spearman,
                        "spearman_p": p_spearman,
                        "significant": (r_pearson > 0.8 and p_pearson < 0.05)
                    }
                    
                    logger.info(
                        f"  {model1} vs {model2}: "
                        f"r={r_pearson:.3f}, rho={rho_spearman:.3f}"
                    )
                except Exception as e:
                    logger.error(f"Correlation failed: {e}")
                    correlations[f"{model1}_vs_{model2}"] = {
                        "pearson_r": 0.0,
                        "pearson_p": 1.0,
                        "spearman_rho": 0.0,
                        "spearman_p": 1.0,
                        "significant": False
                    }
        
        all_significant = all(
            c.get("significant", False)
            for c in correlations.values()
        ) if correlations else False
        
        return {
            "model_list": models,
            "n_test_cases": n_test_cases,
            "correlations": correlations,
            "all_models_correlated": all_significant,
            "status": "PASS" if all_significant else "FAIL",
            "recommendation": (
                "Proxy invariance validated" if all_significant
                else "Proxy invariance NOT validated - consider alternative reference model"
            )
        }
    
    def _generate_test_probes(self, n: int) -> List[Dict]:
        """Generate diverse test probes."""
        probes = []
        
        # Simple logic probes
        for i in range(n):
            probes.append({
                "context": f"Given: All X are Y. Z is X.",
                "target": f"Therefore, Z is Y."
            })
        
        return probes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = CrossModelValidator()
    
    # Note: This requires model availability
    # results = validator.validate_proxy_invariance(
    #     models=["meta-llama/Meta-Llama-3-8B"],
    #     n_test_cases=10
    # )
    # print(f"Status: {results['status']}")
    
    logger.info("Cross-model validator ready")
