import pytest

from arcus.analysis.taxonomy import classify_requested_output, classify_trajectory, ErrorMode


REFERENCE = ["[0,0]", "[1,0]", "[2,0]", "[3,0]", "[4,0]"]
RULES = {"R": {"dx": 1, "dy": 0}}
ACTIONS = ["R"] * 4


def classify(predicted, raw="trajectory"):
    return classify_requested_output(predicted, REFERENCE, 4, RULES, ACTIONS, 0, 10, 10, raw)


def test_requested_and_full_reference_have_different_outcomes():
    assert classify(REFERENCE[:4]).legacy_mode == ErrorMode.NONE
    result = classify(REFERENCE).probe_classification
    assert result.mechanism.value == "Horizon Failure"
    assert "overlong_output" in result.secondary_flags


def test_short_and_overlong_correct_prefix_are_horizon_failures():
    assert classify(REFERENCE[:2]).legacy_mode == ErrorMode.HORIZON_COLLAPSE
    assert classify(REFERENCE + ["[5,0]"]).legacy_mode == ErrorMode.HORIZON_COLLAPSE


def test_overlong_prefix_error_is_not_blindly_attributed_to_horizon():
    assert classify(["[9,9]", *REFERENCE[1:]]).legacy_mode == ErrorMode.SEMANTIC_INTERPRETATION_FAILURE
    assert classify(["[0,0]", "[9,0]", "[8,0]", "[7,0]", "[6,0]"]).legacy_mode != ErrorMode.HORIZON_COLLAPSE
    result = classify(REFERENCE[:4] + ["[99,99]"])
    assert result.legacy_mode == ErrorMode.HORIZON_COLLAPSE
    assert "overlong_output" in result.probe_classification.secondary_flags


def test_trajectory_error_signatures():
    assert classify(["[1,1]", "[2,1]", "[3,1]", "[4,1]"]).legacy_mode == ErrorMode.SEMANTIC_INTERPRETATION_FAILURE
    assert classify(["[0,0]", "[9,0]", "[8,0]", "[7,0]"]).legacy_mode == ErrorMode.TRANSITION_RULE_FAILURE
    assert classify(["[0,0]", "[1,0]", "[3,0]", "[4,0]"]).legacy_mode == ErrorMode.STATE_TRACKING_FAILURE
    assert classify(["[10,0]", "[11,0]", "[12,0]", "[13,0]"]).legacy_mode == ErrorMode.FORMATTING_FAILURE


@pytest.mark.parametrize(("predicted", "raw"), [
    ([], ""), (["not-a-coordinate"], "nonsense"),
    ([], "[EXCEPTION]"), ([], "EXCEPTION: provider runtime failure"),
])
def test_invalidity_is_preserved(predicted, raw):
    assert not classify(predicted, raw).valid


def test_missing_rule_does_not_make_divergence_successful():
    result = classify_requested_output(
        ["[0,0]", "[9,0]", "[8,0]", "[7,0]"], REFERENCE, 4,
        {}, ACTIONS, 0, 10, 10, "raw",
    )
    assert result.legacy_mode == ErrorMode.UNKNOWN_UNMAPPED


def test_unmapped_early_divergence_preserves_historical_fallback():
    predicted = [
        "[1,3]", "[10,3]", "[9,3]", "[9,4]", "[9,3]", "[9,2]", "[9,1]",
        "[8,1]", "[9,1]", "[8,1]", "[8,8]", "[8,7]", "[8,8]", "[8,7]", "[8,8]", "[8,7]",
    ]
    reference = [
        "[1,3]", "[0,3]", "[9,3]", "[9,4]", "[9,3]", "[9,2]", "[9,1]",
        "[8,1]", "[9,1]", "[8,1]", "[8,0]", "[8,7]", "[8,0]", "[8,7]",
        "[8,0]", "[8,1]", "[8,0]", "[8,1]",
    ]
    actions = ["west", "west", "north", "south", "south", "south", "west",
               "east", "west", "south", "south", "north", "south", "north",
               "north", "south", "north"]
    rules = {"east": {"dx": 1, "dy": 0}, "west": {"dx": -1, "dy": 0},
             "north": {"dx": 0, "dy": 1}, "south": {"dx": 0, "dy": -1}}
    historical = classify_trajectory(predicted, reference[:17], rules, actions[:16], 0, 10, 8, "raw")
    corrected = classify_requested_output(predicted, reference, 17, rules, actions, 0, 10, 8, "raw")
    assert historical.legacy_mode == ErrorMode.STATE_TRACKING_FAILURE
    assert historical.explanation == "State synchronization lost after early divergence."
    assert corrected.legacy_mode == ErrorMode.UNKNOWN_UNMAPPED
    assert corrected.explanation == "Early divergence has no explicitly mapped mechanism."
    assert "short_output" in corrected.probe_classification.secondary_flags


def test_unrequested_action_is_not_used_for_requested_errors():
    changed_last_action = ["R", "R", "R", "UNDEFINED"]
    result = classify_requested_output(
        REFERENCE[:4], REFERENCE, 4, RULES, changed_last_action, 0, 10, 10, "raw",
    )
    assert result.legacy_mode == ErrorMode.NONE
    overlong = classify_requested_output(
        REFERENCE, REFERENCE, 4, RULES, changed_last_action, 0, 10, 10, "raw",
    )
    assert overlong.legacy_mode == ErrorMode.HORIZON_COLLAPSE


def test_single_coordinate_and_reference_integrity():
    full = ["[0,0]", "[1,0]"]
    result = classify_requested_output(["[0,0]"], full, 1, RULES, ["R"], 0, 10, 10, "[0,0]")
    assert result.legacy_mode == ErrorMode.NONE
    assert result.probe_classification.step_accuracy == 0.0
    assert classify_requested_output(full, full, 1, RULES, ["R"], 0, 10, 10, "raw").legacy_mode == ErrorMode.HORIZON_COLLAPSE
    assert full == ["[0,0]", "[1,0]"]
    with pytest.raises(ValueError):
        classify_requested_output(full, full, 2, RULES, ["R"])
