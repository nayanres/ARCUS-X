import pytest

from audit.correction_2026_11 import reconcile_reference_conventions as analysis


def test_distinguishes_requested_and_full_reference(monkeypatch):
    monkeypatch.setattr(analysis.audit, "PUBLICATION_MODELS", ("m",))
    common = {"model": "m", "z": 3, "tier": 0, "gravity": 0.5,
              "valid": True, "ground_truth_path_full": ["a", "b", "c", "d"],
              "coordinate_interpretation": "canonical"}
    rows = [
        {**common, "predicted_path": ["a", "b", "c"], "taxonomy": "Horizon Collapse"},
        {**common, "predicted_path": ["a", "b", "c", "d"], "taxonomy": "None"},
        {**common, "valid": False, "predicted_path": [], "taxonomy": ""},
    ]
    result = analysis.reconcile(rows)[0]
    assert result["requested_exact_valid_count"] == 1
    assert result["requested_exact_pct_completed"] == pytest.approx(100 / 3)
    assert result["requested_exact_label_Horizon Collapse"] == 1
    assert result["none_valid_count"] == result["full_reference_exact_valid_count"] == 1
    assert result["none_length_z"] == 0
    assert result["none_length_z_plus_1"] == 1


def test_rejects_none_without_full_reference_match(monkeypatch):
    monkeypatch.setattr(analysis.audit, "PUBLICATION_MODELS", ("m",))
    row = {"model": "m", "z": 3, "tier": 0, "gravity": 0.5, "valid": True,
           "ground_truth_path_full": ["a", "b", "c", "d"],
           "predicted_path": ["a", "b", "c"], "taxonomy": "None",
           "coordinate_interpretation": "canonical"}
    with pytest.raises(ValueError, match="None/full-reference"):
        analysis.reconcile([row])
