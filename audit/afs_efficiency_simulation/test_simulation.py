import numpy as np
import pytest

from audit.afs_efficiency_simulation import simulation as sim


def observations(p=0.8, valid=1.0, identity=("test", "fixed", 0, 0)):
    return sim.PotentialObservations(np.full(150, p), np.full(150, valid), identity)


def test_common_random_numbers_and_reproducibility():
    a = observations()
    b = observations()
    first = a.batch(20)
    a.batch(30)
    b.batch(30)
    assert first == b.batch(20)
    assert first == a.batch(20)


def test_invalidity_excluded_but_charged_and_cached():
    r = sim.Recorder(observations(valid=0), 11)
    assert r.sample(3, anchor=True) is None
    assert r.sample(20) is None
    assert r.cost == 6
    assert r.sample(20) is None
    assert r.cost == 6
    assert r.trace[0]["valid"] == 0
    r.sample(30)
    with pytest.raises(sim.BudgetExhausted):
        r.sample(40)


def test_production_trace_abrupt_and_invalid_only():
    class Fixed:
        def batch(self, z, anchor=False):
            valid = z != 20
            score = 1.0 if z < 30 else 0.0
            return tuple({"step_accuracy": score, "probe_classification": {"valid": valid}}
                         for _ in range(1 if anchor else 5))

    recorder, terminated, native = sim.run_afs(Fixed(), 161)
    assert terminated
    assert [r["z"] for r in recorder.trace] == [20, 30, 25, 27, 28, 29, 44, 36, 32]
    assert recorder.trace[0]["mean"] is None
    assert native == 30
    assert recorder.cost == 46


def test_production_saturation_and_budgets():
    class Fixed:
        def batch(self, z, anchor=False):
            return tuple({"step_accuracy": 1.0, "probe_classification": {"valid": True}}
                         for _ in range(1 if anchor else 5))

    recorder, done, native = sim.run_afs(Fixed(), 161)
    assert done and native == 150
    assert recorder.trace[0]["z"] == 20
    for policy in sim.POLICIES:
        obs = observations()
        r, _, _ = sim.run_policy(policy, obs, ("test", 0), 21)
        assert r.cost <= 21
        assert r.cost == 1 + 5 * len(r.trace)


def test_threshold_stress_uses_production_controller_threshold():
    class Fixed:
        def batch(self, z, anchor=False):
            return tuple({"step_accuracy": 0.45, "probe_classification": {"valid": True}}
                         for _ in range(1 if anchor else 5))

    _, done_low, output_low = sim.run_afs(Fixed(), 161, threshold=0.4)
    _, done_high, output_high = sim.run_afs(Fixed(), 161, threshold=0.6)
    assert done_low and output_low == 150
    assert done_high and output_high == 1


def test_targets_and_policy_neutral_estimator():
    curve = np.full(150, 0.8)
    curve[19:24] = 0.3
    curve[29:] = 0.2
    assert sim.true_crossing(curve, 0.5, 1) == 20
    assert sim.true_crossing(curve, 0.5, 5) == 20
    assert sim.true_crossing(curve, 0.5, 10) == 30
    assert sim.true_crossing(np.full(150, 0.8), 0.5, 1) is None
    trace = [{"z": z, "mean": 0.1, "cost": 6 + 5 * i, "valid": 5}
             for i, z in enumerate([20, 21, 22, 23, 24])]
    assert sim.estimate(trace, 0.5, 5) == (20, 20)
    assert sim.estimate(trace[:4], 0.5, 5) == (20, None)


def test_splits_and_environments_reproducible():
    a, v, params = sim.environment("validity_degradation", "development", 1)
    b, w, params2 = sim.environment("validity_degradation", "development", 1)
    c, _, _ = sim.environment("validity_degradation", "test", 1)
    assert np.array_equal(a, b) and np.array_equal(v, w) and params == params2
    assert not np.array_equal(a, c)


def test_no_ground_truth_required_by_decisions():
    trace = [{"z": 1, "mean": 0.8}, {"z": 150, "mean": 0.2}]
    assert sim.adaptive_next(trace) == 75
    assert sim.estimate(trace, 0.5, 5) == (150, None)


def test_nonadaptive_schedules_do_not_depend_on_response_values():
    for policy in ("fixed_grid", "sequential", "uniform_random"):
        high, _, _ = sim.run_baseline(policy, observations(p=0.9), 41, ("schedule",))
        low, _, _ = sim.run_baseline(policy, observations(p=0.1), 41, ("schedule",))
        assert [x["z"] for x in high.trace] == [x["z"] for x in low.trace]


def test_censored_truth_is_not_assigned_localization_error():
    p = np.full(150, 0.8)
    trace = [{"z": 20, "mean": 0.1, "valid": 5, "cost": 6}]
    result = sim.evaluate(trace, p, 11, False, None)
    assert result["first_truth"] is None
    assert result["first_estimate"] == 20
    assert result["first_abs_error"] is None
    assert result["cost_to_first_within_5"] is None


def test_cost_to_target_distinguishes_transient_from_final_estimate():
    p = np.full(150, 0.8)
    p[19:] = 0.2
    trace = [{"z": 20, "mean": 0.1, "valid": 5, "cost": 6},
             {"z": 1, "mean": 0.1, "valid": 5, "cost": 11}]
    result = sim.evaluate(trace, p, 11, False, None)
    assert result["cost_to_first_within_5"] == 6
    assert result["first_abs_error"] == 19
