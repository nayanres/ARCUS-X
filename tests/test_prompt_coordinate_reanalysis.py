import subprocess
import sys
from pathlib import Path

import pytest

from post_run_analyzer import _score_probe, process_txt_file
from audit.prompt_coordinate_reanalysis.rescore import rescore_entry


def entry(predicted_length, z=50):
    return {
        "z": z, "tier": 0, "seed": 42, "gravity": 0.5,
        "grid": "100x2", "actions": repr(["east"] * z),
        "transition_rules": repr({"east": {"dx": 1, "dy": 0}}),
        "ground_truth": "\n".join(f"[{i},0]" for i in range(z + 1)),
        "raw_output": "\n".join(f"[{i},0]" for i in range(predicted_length)),
        "output_tokens": 0,
    }


def test_prompt_sized_path_passes_and_all_actions_path_does_not():
    scored, detail = rescore_entry(entry(50))
    assert detail["exact_match"]
    assert detail["step_accuracy"] == 1
    assert detail["horizon_compliance"] == 1
    assert detail["reported_taxonomy"] == "None"
    assert len(scored["_actions"]) == 49
    assert scored["ground_truth_length"] == 50

    scored, detail = rescore_entry(entry(51))
    assert not detail["exact_match"]
    assert detail["step_accuracy"] == 1
    assert detail["horizon_compliance"] == pytest.approx(50 / 51)
    assert detail["reported_taxonomy"] != "None"


def test_short_and_invalid_responses():
    _, detail = rescore_entry(entry(49))
    assert detail["horizon_compliance"] == pytest.approx(49 / 50)
    assert detail["step_accuracy"] == pytest.approx(48 / 49)
    _, detail = rescore_entry(entry(0))
    assert not detail["valid"]
    assert detail["horizon_compliance"] == 0


def test_one_coordinate_has_no_scored_transitions():
    _, detail = rescore_entry(entry(1, z=1))
    assert detail["exact_match"]
    assert detail["horizon_compliance"] == 1
    assert detail["step_accuracy"] == 0
    assert _score_probe(entry(1, z=1), coordinate_count=True)["error_mode"] == "None"


def test_unexpected_reference_length_is_rejected():
    data = entry(50)
    data["ground_truth"] = "[0,0]"
    with pytest.raises(ValueError, match="Unexpected reference"):
        rescore_entry(data)


@pytest.mark.parametrize("length", [0, 49, 50, 51])
def test_post_run_option_matches_independent_reanalysis(length):
    data = entry(length)
    expected, _ = rescore_entry(data)
    actual = _score_probe(data, coordinate_count=True)
    for key in ("step_accuracy", "exact_match", "continuity_score",
                "horizon_compliance", "generation_bloat_index",
                "generation_efficiency", "correct_transitions", "ground_truth_path"):
        assert actual[key] == expected[key], key
    assert _score_probe(data)["ground_truth_length"] == 51


def test_txt_parser_and_cli_select_scoring_policy(tmp_path):
    data = entry(3, z=3)
    log = tmp_path / "run.txt"
    log.write_text(
        "Model: example\n=== RAW STREAM ENTRY ===\n"
        "Parameters: z=3, gravity=0.5, tier=0, grid=default\n"
        f"Grid: {data['grid']}\nActions: {data['actions']}\n"
        f"TransitionRules: {data['transition_rules']}\n"
        "[EXACT SENT PROMPT]:\nActions: ['west']\n"
        "EnvironmentMetadata: z=9, gravity=3.0, tier=3\n"
        f"[MODEL OUTPUT]:\n{data['raw_output']}\n"
        f"[GROUND TRUTH EXPECTED]:\n{data['ground_truth']}\n"
        "==============================\n", encoding="utf-8"
    )
    legacy = next(iter(process_txt_file(str(log))["results_matrix"].values()))
    revised = next(iter(process_txt_file(str(log), coordinate_count=True)["results_matrix"].values()))
    assert not legacy["exact_match"]
    assert revised["exact_match"] and revised["horizon_compliance"] == 1
    cli = Path(__file__).resolve().parents[1] / "post_run_analyzer.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--input", str(log), "--coordinate-count"],
        text=True, capture_output=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert "SCORING POLICY: z coordinates" in result.stdout


def test_cli_rejects_coordinate_count_for_json(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("{}", encoding="utf-8")
    cli = Path(__file__).resolve().parents[1] / "post_run_analyzer.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--input", str(path), "--coordinate-count"],
        text=True, capture_output=True, encoding="utf-8"
    )
    assert result.returncode != 0
    assert "requires raw TXT logs" in result.stdout
