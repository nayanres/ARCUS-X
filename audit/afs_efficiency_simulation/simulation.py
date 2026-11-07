#!/usr/bin/env python3
"""Frozen synthetic experiment using the production AFS control loop."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arcus.experiments.benchmark_runner import BenchmarkRunner

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
POLICIES = ("production_afs", "fixed_grid", "sequential", "uniform_random", "adaptive")
MAX_Z = CONFIG["domain"][1]
BATCH = CONFIG["batch_size"]
THRESHOLD = CONFIG["threshold"]


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def environment(family, split, index):
    """Generate an independent curve; no observations are used to choose parameters."""
    rng = random.Random(stable_seed(CONFIG["split_seeds"][split], family, index))
    r = CONFIG["curve_parameter_ranges"][family]
    draw = lambda key: rng.uniform(*r[key])
    z = np.arange(1, MAX_Z + 1, dtype=float)
    if family == "logistic":
        center, scale = draw("center"), draw("scale")
        p = r["low"] + (r["high"] - r["low"]) / (1 + np.exp((z - center) / scale))
        params = {"center": center, "scale": scale}
    elif family == "exponential":
        half = draw("half_life")
        p = r["low"] + r["amplitude"] * np.exp(-math.log(2) * z / half)
        params = {"half_life": half}
    elif family == "abrupt":
        crossing, before, after = draw("fracture"), draw("before"), draw("after")
        p = np.where(z < crossing, before, after)
        params = {"fracture": crossing, "before": before, "after": after}
    elif family == "gradual":
        initial, slope = draw("initial"), draw("slope")
        p = initial - slope * (z - 1)
        params = {"initial": initial, "slope": slope}
    elif family == "recovery":
        onset, duration, second = draw("onset"), draw("duration"), draw("second_onset")
        p = np.where(((z >= onset) & (z < onset + duration)) | (z >= second), r["low"], r["high"])
        params = {"onset": onset, "duration": duration, "second_onset": second}
    elif family == "oscillatory":
        center, amplitude, period, phase = draw("center"), draw("amplitude"), draw("period"), draw("phase")
        p = center + amplitude * np.sin(2 * np.pi * z / period + phase)
        params = {"center": center, "amplitude": amplitude, "period": period, "phase": phase}
    elif family == "censored":
        if rng.random() < r["near_ceiling_fraction"]:
            crossing = draw("near_ceiling_crossing")
            p = 0.5 + 0.002 * (crossing - z)
            params = {"near_ceiling_crossing": crossing}
        else:
            initial, slope = draw("intercept"), draw("slope")
            p = initial - slope * z
            p = np.maximum(p, 0.505)
            params = {"intercept": initial, "slope": slope}
    elif family == "near_threshold":
        center, slope, amplitude, period = draw("center"), draw("slope"), draw("amplitude"), draw("period")
        p = center + slope * (z - 75) + amplitude * np.sin(2 * np.pi * z / period)
        params = {"center": center, "slope": slope, "amplitude": amplitude, "period": period}
    elif family == "validity_degradation":
        crossing, scale = draw("crossing"), draw("scale")
        p = 0.1 + 0.8 / (1 + np.exp((z - crossing) / scale))
        center, vscale = draw("validity_center"), draw("validity_scale")
        v = 1 / (1 + np.exp((z - center) / vscale))
        cutoff = rng.random() < r["zero_cutoff_fraction"]
        if cutoff:
            v[z >= min(150, center + 2 * vscale)] = 0
        params = {"crossing": crossing, "scale": scale, "validity_center": center,
                  "validity_scale": vscale, "zero_cutoff": cutoff}
        return np.clip(p, 0.001, 0.999), v, params
    else:
        raise ValueError(f"Unknown family {family}")
    return np.clip(p, 0.001, 0.999), np.ones(MAX_Z), params


class PotentialObservations:
    """Order-independent common potential responses, shared across policies."""

    def __init__(self, p, validity, identity):
        self.p, self.validity, self.identity = p, validity, identity
        self.cache = {}

    def batch(self, z, anchor=False):
        key = (z, anchor)
        if key not in self.cache:
            rng = np.random.default_rng(stable_seed(CONFIG["observation_seed"], *self.identity, z, anchor))
            n = 1 if anchor else BATCH
            valid = rng.random(n) < self.validity[z - 1]
            mean = self.p[z - 1]
            concentration = CONFIG["trajectory_beta_concentration"]
            q = rng.beta(mean * concentration, (1 - mean) * concentration, size=n)
            scores = rng.binomial(z, q) / z
            self.cache[key] = tuple(
                {"step_accuracy": float(score), "probe_classification": {"valid": bool(ok)}}
                for score, ok in zip(scores, valid)
            )
        return self.cache[key]


class BudgetExhausted(Exception):
    pass


class Recorder:
    def __init__(self, observations, budget):
        self.observations = observations
        self.budget = budget
        self.cost = 0
        self.cache = {}
        self.trace = []

    def sample(self, z, anchor=False):
        key = (z, anchor)
        if key in self.cache:
            return self.cache[key]
        size = 1 if anchor else BATCH
        if self.cost + size > self.budget:
            raise BudgetExhausted
        results = self.observations.batch(z, anchor=anchor)
        valid = [r["step_accuracy"] for r in results if BenchmarkRunner._is_cognitive_valid(r)]
        mean = sum(valid) / len(valid) if valid else None
        self.cost += size
        self.cache[key] = mean
        if not anchor:
            self.trace.append({"z": z, "mean": mean, "valid": len(valid), "cost": self.cost})
        return mean


def run_afs(observations, budget, threshold=THRESHOLD):
    """Call the actual production controller with only its batch I/O replaced."""
    recorder = Recorder(observations, budget)
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.hard_ceiling = MAX_Z
    runner.gravity_levels = [0.5]
    runner.tiers = [0]
    runner.adaptive_expansion = True
    runner.fracture_cache = {}
    runner.results_matrix = {}
    runner.max_horizon = None
    runner.MIN_ACCURACY_THRESHOLD = threshold

    def batch(self, probes, gravity_target, z, **kwargs):
        anchor = gravity_target == 0.0 and z == 3 and len(probes) == 1
        return recorder.sample(z, anchor=anchor), 0

    runner._evaluate_depth_batch = types.MethodType(batch, runner)
    terminated = True
    try:
        runner._evaluate_all_dimensions([{"id": str(i)} for i in range(BATCH)])
    except BudgetExhausted:
        terminated = False
    native = runner.fracture_cache.get((0.5, "final")) if terminated else None
    return recorder, terminated, native


def adaptive_next(trace):
    sampled = {r["z"] for r in trace}
    for z in (1, MAX_Z):
        if z not in sampled:
            return z
    ordered = sorted(sampled)
    gaps = [(a, b) for a, b in zip(ordered, ordered[1:]) if b - a > 1]
    if not gaps:
        return None
    valid = sorted((abs(r["mean"] - THRESHOLD), r["z"]) for r in trace if r["mean"] is not None)
    for _, anchor in valid:
        adjacent = [(a, b) for a, b in gaps if a == anchor or b == anchor]
        if adjacent:
            a, b = min(adjacent, key=lambda gap: (-(gap[1] - gap[0]), gap[0]))
            return (a + b) // 2
    a, b = min(gaps, key=lambda gap: (-(gap[1] - gap[0]), gap[0]))
    return (a + b) // 2


def run_baseline(policy, observations, budget, identity):
    recorder = Recorder(observations, budget)
    recorder.sample(3, anchor=True)
    if policy == "fixed_grid":
        schedule = iter(CONFIG["fixed_grid_schedule"])
    elif policy == "sequential":
        schedule = iter(range(1, MAX_Z + 1))
    elif policy == "uniform_random":
        points = list(range(1, MAX_Z + 1))
        random.Random(stable_seed(CONFIG["schedule_seed"], *identity)).shuffle(points)
        schedule = iter(points)
    elif policy == "adaptive":
        schedule = None
    else:
        raise ValueError(policy)
    terminated = False
    while True:
        z = adaptive_next(recorder.trace) if schedule is None else next(schedule, None)
        if z is None:
            terminated = True
            break
        try:
            recorder.sample(z)
        except BudgetExhausted:
            break
    return recorder, terminated, None


def true_crossing(p, threshold, window):
    below = p < threshold
    for i in range(len(p) - window + 1):
        if np.all(below[i:i + window]):
            return i + 1
    return None


def estimate(trace, threshold, window):
    below = {r["z"] for r in trace if r["mean"] is not None and r["mean"] < threshold}
    first = min(below) if below else None
    persistent = next((z for z in sorted(below) if all(z + i in below for i in range(window))), None)
    return first, persistent


def evaluate(trace, p, budget, terminated, native):
    trace = [r for r in trace if r["cost"] <= budget]
    first_truth = true_crossing(p, THRESHOLD, 1)
    persistent_truth = true_crossing(p, THRESHOLD, CONFIG["persistent_crossing_window"])
    first, persistent = estimate(trace, THRESHOLD, CONFIG["persistent_crossing_window"])
    band = CONFIG["threshold_region_tolerance"]
    predicted_region = {r["z"] for r in trace if r["mean"] is not None and abs(r["mean"] - THRESHOLD) <= band}
    actual_region = {z for z in range(1, MAX_Z + 1) if abs(p[z - 1] - THRESHOLD) <= band}
    sampled = {r["z"] for r in trace}
    cost_to_3 = cost_to_5 = None
    for i, observation in enumerate(trace):
        prefix_estimate = estimate(trace[:i + 1], THRESHOLD, 1)[0]
        if first_truth is not None and prefix_estimate is not None:
            if cost_to_3 is None and abs(prefix_estimate - first_truth) <= 3:
                cost_to_3 = observation["cost"]
            if cost_to_5 is None and abs(prefix_estimate - first_truth) <= 5:
                cost_to_5 = observation["cost"]
    out = {
        "first_truth": first_truth, "persistent_truth": persistent_truth,
        "first_estimate": first, "persistent_estimate": persistent,
        "first_abs_error": abs(first - first_truth) if first is not None and first_truth is not None else None,
        "persistent_abs_error": abs(persistent - persistent_truth) if persistent is not None and persistent_truth is not None else None,
        "region_coverage": len(sampled & actual_region) / len(actual_region) if actual_region else None,
        "region_precision": len(predicted_region & actual_region) / len(predicted_region) if predicted_region else None,
        "invalid_only_batches": sum(r["mean"] is None for r in trace),
        "valid_responses": sum(r["valid"] for r in trace),
        "cost_to_first_within_3": cost_to_3,
        "cost_to_first_within_5": cost_to_5,
        "distinct_horizons": len(sampled),
        "native_output": native if terminated else None,
        "native_terminated": terminated,
    }
    return out


def run_policy(policy, observations, identity, max_budget):
    if policy == "production_afs":
        return run_afs(observations, max_budget)
    return run_baseline(policy, observations, max_budget, identity)


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation", "test"), default="test")
    args = parser.parse_args()
    logging.getLogger("arcus.experiments.benchmark_runner").setLevel(logging.ERROR)
    size = CONFIG["test_curves_per_family"] if args.split == "test" else CONFIG[f"{args.split}_curves_per_family"]
    output = HERE / args.split
    output.mkdir(exist_ok=True)
    metadata = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "config_sha256": hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256((ROOT / "arcus" / "experiments" / "benchmark_runner.py").read_bytes()).hexdigest(),
        "simulation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": args.split,
    }
    metadata_path = output / "environment.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"{output}: configuration or software fingerprint changed; use a fresh split directory")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for family in CONFIG["families"]:
        marker = output / f"{family}.checkpoint.json"
        names = {kind: output / f"{family}_{kind}.csv" for kind in ("curves", "results", "traces", "threshold_stress")}
        if marker.exists():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            for kind, path in names.items():
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != saved[kind]:
                    raise ValueError(f"Checkpoint integrity mismatch: {path}")
            print(f"[{args.split}] resumed {family}", flush=True)
            continue
        results, curves, traces, threshold_stress = [], [], [], []
        for index in range(size):
            p, v, params = environment(family, args.split, index)
            curve_id = f"{args.split}:{family}:{index}"
            curves.append({"curve_id": curve_id, "family": family, "parameters": json.dumps(params),
                           "accuracy_curve": json.dumps(p.tolist()), "validity_curve": json.dumps(v.tolist())})
            for replicate in range(CONFIG["replicates_per_curve"]):
                identity = (args.split, family, index, replicate)
                observations = PotentialObservations(p, v, identity)
                for policy in POLICIES:
                    recorder, terminated, native = run_policy(policy, observations, identity, max(CONFIG["budgets_responses"]))
                    for row in recorder.trace:
                        traces.append({"curve_id": curve_id, "replicate": replicate, "policy": policy, **row})
                    for budget in CONFIG["budgets_responses"]:
                        available = [r for r in recorder.trace if r["cost"] <= budget]
                        cost = available[-1]["cost"] if available else 1
                        done = terminated and recorder.cost <= budget
                        row = {"curve_id": curve_id, "family": family, "replicate": replicate,
                               "policy": policy, "budget": budget, "cost": cost, "unused_budget": budget - cost,
                               **evaluate(available, p, budget, done, native)}
                        for window in CONFIG["persistence_sensitivity_windows"]:
                            row[f"truth_persistent_{window}"] = true_crossing(p, THRESHOLD, window)
                            row[f"estimate_persistent_{window}"] = estimate(available, THRESHOLD, window)[1]
                        for threshold in CONFIG["threshold_sensitivity_values"]:
                            key = str(threshold).replace(".", "_")
                            row[f"truth_threshold_{key}"] = true_crossing(p, threshold, 1)
                            row[f"estimate_threshold_{key}"] = estimate(available, threshold, 1)[0]
                        results.append(row)
                for threshold in CONFIG["threshold_sensitivity_values"]:
                    recorder, terminated, native = run_afs(observations, max(CONFIG["budgets_responses"]), threshold)
                    for budget in CONFIG["budgets_responses"]:
                        available = [r for r in recorder.trace if r["cost"] <= budget]
                        truth = true_crossing(p, threshold, 1)
                        estimated = estimate(available, threshold, 1)[0]
                        done = terminated and recorder.cost <= budget
                        threshold_stress.append({
                            "curve_id": curve_id, "family": family, "replicate": replicate,
                            "threshold": threshold, "budget": budget,
                            "cost": available[-1]["cost"] if available else 1,
                            "truth": truth, "estimate": estimated,
                            "absolute_error": abs(truth - estimated) if truth is not None and estimated is not None else None,
                            "native_terminated": done, "native_output": native if done else None,
                        })
        for kind, rows in (("curves", curves), ("results", results), ("traces", traces), ("threshold_stress", threshold_stress)):
            write_rows(names[kind], rows)
        marker.write_text(json.dumps({kind: hashlib.sha256(path.read_bytes()).hexdigest() for kind, path in names.items()}, indent=2), encoding="utf-8")
        print(f"[{args.split}] completed {family}: {size} curves", flush=True)
    for kind in ("curves", "results", "traces", "threshold_stress"):
        with (output / f"{kind}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = None
            for family in CONFIG["families"]:
                with (output / f"{family}_{kind}.csv").open(encoding="utf-8", newline="") as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    elif reader.fieldnames != writer.fieldnames:
                        raise ValueError(f"Checkpoint schema mismatch: {family}/{kind}")
                    writer.writerows(reader)
    print(f"Wrote {len(CONFIG['families']) * size} curves and consolidated all checkpoints.", flush=True)


if __name__ == "__main__":
    main()
