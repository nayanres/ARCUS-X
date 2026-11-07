from arcus.experiments.quickstart import _append_minitest_analysis
from arcus.experiments import quickstart
import arcus.evaluation.evaluator as evaluator_module


def test_minitest_report_preserves_prompt_output_and_chain_analysis(tmp_path):
    output_path = tmp_path / "minitests_example.txt"
    expected = [f"[{i},0]" for i in range(51)]
    predicted = expected + ["[51,0]"]
    record = {
        "tier": 2,
        "seed": 3157,
        "z": 50,
        "gravity": 0.5,
        "task_id": "example-task",
        "experiment_hash": "example-hash",
        "prompt": "The exact prompt sent to the model.",
        "raw_output": "\n".join(predicted),
        "ground_truth_path": expected,
        "transition_rules": {"east": {"dx": 1, "dy": 0}},
        "actions": ["east"] * 50,
        "grid_width": 100,
        "grid_height": 2,
    }

    _append_minitest_analysis(output_path, "example-model", [record])

    report = output_path.read_text(encoding="utf-8")
    assert "[EXACT SENT PROMPT]\nThe exact prompt sent to the model." in report
    assert "[EXACT MODEL OUTPUT]\n" + "\n".join(predicted) in report
    assert "[GROUND-TRUTH CHAIN]\n" + "\n".join(expected) in report
    assert "parseable_coordinates=True" in report
    assert "exact_match=False" in report
    assert "step_accuracy=1.000000" in report
    assert "horizon_compliance=0.980392" in report
    assert "taxonomy=Horizon Collapse" in report
    assert "051 transition 51: expected=<missing> actual=[51,0] [MISMATCH]" in report


class _FakeResponse:
    status_code = 200
    headers = {}

    def json(self):
        return {
            "choices": [{"message": {"content": "[0,0]"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
        }


def _mock_api(monkeypatch):
    requests = []

    def fake_post(url, headers=None, json=None, timeout=None):
        requests.append(json)
        return _FakeResponse()

    monkeypatch.setattr(evaluator_module.requests, "post", fake_post)
    return requests


def test_minitest_logs_actual_api_prompt_next_to_each_response(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    requests = _mock_api(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["quickstart_api.py", "--api-base", "https://example.test/v1",
         "--api-key", "test-key", "--model", "test-model", "--minitest"],
    )

    assert quickstart.main() == 0
    report = (tmp_path / "outputs" / "minitests_test-model.txt").read_text(encoding="utf-8")
    assert "MINITEST RUN | model=test-model | seed=3157" in report
    entries = report.split("=== RAW STREAM ENTRY")[1:]
    assert len(entries) == len(requests) == 4
    for tier, (entry, request) in enumerate(zip(entries, requests)):
        prompt = request["messages"][0]["content"]
        assert request["messages"] == [{"role": "user", "content": prompt}]
        assert f"tier={tier}" in entry
        assert "EnvironmentMetadata: seed=3157," in entry
        assert f"[EXACT SENT PROMPT]:\n{prompt}\n\n[MODEL OUTPUT]:\n[0,0]" in entry
        assert "[GROUND TRUTH EXPECTED]:" in entry


def test_minitest_rejects_other_explicit_seeds(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["quickstart_api.py", "--minitest", "--master-seeds", "42"],
    )
    assert quickstart.main() == 2


def test_regular_quickstart_uses_ux_without_logging_prompts(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    requests = _mock_api(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["quickstart_api.py", "--api-base", "https://example.test/v1",
         "--api-key", "test-key", "--model", "test-model", "--dev"],
    )

    assert quickstart.main() == 0
    output = capsys.readouterr().out
    assert "Self-test" in output
    assert "Waiting for API..." in output
    assert "EXECUTION" in output
    raw_log = tmp_path / "outputs" / "absolute_raw_data_local_test_model.txt"
    text = raw_log.read_text(encoding="utf-8")
    assert requests
    assert "[EXACT SENT PROMPT]" not in text
    assert requests[0]["messages"][0]["content"] not in text
