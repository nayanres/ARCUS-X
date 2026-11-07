#!/usr/bin/env python3
"""Prespecified within-trajectory correlation sensitivity, separate from primary results."""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.afs_efficiency_simulation import simulation as sim

CONCENTRATIONS = (5, 30, 200)
CURVES_PER_FAMILY = 50
BUDGETS = (41, 81)


class StressObservations(sim.PotentialObservations):
    def __init__(self, p, validity, identity, concentration):
        super().__init__(p, validity, identity)
        self.concentration = concentration

    def batch(self, z, anchor=False):
        key = (z, anchor)
        if key not in self.cache:
            rng = np.random.default_rng(sim.stable_seed(sim.CONFIG["observation_seed"], "noise-stress", *self.identity, z, anchor))
            n = 1 if anchor else sim.BATCH
            valid = rng.random(n) < self.validity[z - 1]
            mean = self.p[z - 1]
            q = rng.beta(mean * self.concentration, (1 - mean) * self.concentration, size=n)
            scores = rng.binomial(z, q) / z
            self.cache[key] = tuple({"step_accuracy": float(score), "probe_classification": {"valid": bool(ok)}}
                                    for score, ok in zip(scores, valid))
        return self.cache[key]


def main():
    logging.getLogger("arcus.experiments.benchmark_runner").setLevel(logging.ERROR)
    rows = []
    for family in sim.CONFIG["families"]:
        for index in range(CURVES_PER_FAMILY):
            p, v, _ = sim.environment(family, "test", index)
            for replicate in range(sim.CONFIG["replicates_per_curve"]):
                identity = ("test", family, index, replicate)
                for concentration in CONCENTRATIONS:
                    obs = StressObservations(p, v, identity, concentration)
                    for policy in sim.POLICIES:
                        recorder, terminated, native = sim.run_policy(policy, obs, identity, max(BUDGETS))
                        for budget in BUDGETS:
                            trace = [r for r in recorder.trace if r["cost"] <= budget]
                            result = sim.evaluate(trace, p, budget, terminated and recorder.cost <= budget, native)
                            rows.append({"curve_id": f"test:{family}:{index}", "family": family,
                                         "replicate": replicate, "concentration": concentration,
                                         "policy": policy, "budget": budget, **result})
        print(f"[noise stress] completed {family}", flush=True)
    sim.write_rows(sim.HERE / "test" / "noise_stress.csv", rows)
    print(f"Wrote {len(rows)} noise-stress outcomes.")


if __name__ == "__main__":
    main()
